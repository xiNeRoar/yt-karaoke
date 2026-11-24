"""Isolated 3-layer hybrid KTV lyrics engine.

Priority:
1. YouTube human-authored subtitles (yue-HK / zh-HK / zh-TW / en / ja / ko) — instant (~1s),
   unless --force-ai 1 is passed.
2. Exact Studio match on lrclib.net (only if title is NOT live/concert/cover/remix and duration
   matches within +/-2s, and --force-ai 0) — instant (~0.5s), zero typos.
3. Local AI faster-whisper (large-v3-turbo) running on our BS-RoFormer isolated vocals.mp3,
   guided by clean lrclib.net vocabulary prompt + Silero VAD (no instrumental hallucinations)
   + word-level timestamps for KTV sweeping highlight + OpenCC s2hk Traditional Chinese!
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "whisper"
TEMP_DIR = BASE_DIR / "tmp"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
LIVE_KEYWORDS_RE = re.compile(
    r"(live|concert|演唱會|演唱会|現場|现场|cover|翻唱|acoustic|unplugged|remix|plug\s*dj|medley|串燒|ktv)",
    re.IGNORECASE,
)
HALLUCINATION_RE = re.compile(
    r"(訂閱|订阅|按讚|点赞|感謝觀看|感谢观看|字幕組|字幕组|Amara\.org|謝謝收看|谢谢收看|Subscribe|Thanks for watching)",
    re.IGNORECASE,
)


def add_nvidia_dll_dirs() -> None:
    """Expose PyTorch's bundled CUDA/cuBLAS/cuDNN DLLs so CTranslate2 (faster-whisper) can use RTX GPU on Windows."""
    venv_site = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages"
    dll_dirs = []
    torch_lib = venv_site / "torch" / "lib"
    if torch_lib.is_dir():
        dll_dirs.append(torch_lib)
    nvidia_root = venv_site / "nvidia"
    if nvidia_root.exists():
        for bin_dir in nvidia_root.glob("*/bin"):
            if bin_dir.is_dir():
                dll_dirs.append(bin_dir)
    for d in dll_dirs:
        s = str(d)
        if s not in os.environ.get("PATH", ""):
            os.environ["PATH"] = s + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(s)
            except Exception:
                pass


def query_free_vram_mb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
            timeout=2,
            creationflags=CREATE_NO_WINDOW,
        ).strip()
        return float(out.splitlines()[0].strip())
    except Exception:
        return 0.0


def to_hk_traditional(text: str) -> str:
    if not text:
        return text
    try:
        from opencc import OpenCC
        if not hasattr(to_hk_traditional, "_cc"):
            to_hk_traditional._cc = OpenCC("s2hk")
        return to_hk_traditional._cc.convert(text)
    except Exception:
        return text


def clean_search_title(title: str) -> str:
    t = re.sub(r"[\[\(【（][^\]\)】）]*[\]\)】）]", " ", title)
    t = re.sub(r"\b(official|mv|music video|lyrics|lyric video|hd|4k|hq|audio)\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t or title


def parse_lrc(lrc_text: str) -> list[dict]:
    lines = []
    pat = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\](.*)")
    for raw in lrc_text.splitlines():
        m = pat.match(raw.strip())
        if not m:
            continue
        mins = int(m.group(1))
        secs = float(m.group(2))
        text = m.group(3).strip()
        if not text:
            continue
        lines.append({"start": round(mins * 60 + secs, 2), "end": 0.0, "text": to_hk_traditional(text), "words": []})

    lines.sort(key=lambda x: x["start"])
    for i, item in enumerate(lines):
        if i + 1 < len(lines):
            item["end"] = round(min(item["start"] + 8.0, lines[i + 1]["start"]), 2)
        else:
            item["end"] = round(item["start"] + 6.0, 2)
    return lines


def fetch_lrclib(title: str, duration: float) -> tuple[Optional[list[dict]], str]:
    """Returns (synced_lines_if_exact_studio_match, plain_lyrics_prompt_text)."""
    queries = [clean_search_title(title)]
    if queries[0] != title:
        queries.append(title)

    best_synced = None
    best_plain = ""
    is_live_or_alt = bool(LIVE_KEYWORDS_RE.search(title))

    for q in queries:
        if not q:
            continue
        try:
            r = httpx.get("https://lrclib.net/api/search", params={"q": q}, timeout=8)
            if r.status_code != 200:
                continue
            items = r.json()
            if not isinstance(items, list) or not items:
                continue

            # Grab cleanest plain lyrics for Whisper vocabulary prompt
            for it in items[:5]:
                pl = (it.get("plainLyrics") or "").strip()
                if not pl and it.get("syncedLyrics"):
                    pl = "\n".join(x["text"] for x in parse_lrc(it["syncedLyrics"]))
                if pl and not best_plain:
                    best_plain = to_hk_traditional(pl)

                # Only trust lrclib synced timestamps if NOT a live/concert/cover AND duration matches within 2s
                if not is_live_or_alt and it.get("syncedLyrics") and duration > 0:
                    it_dur = float(it.get("duration") or 0)
                    if abs(it_dur - duration) <= 2.0:
                        parsed = parse_lrc(it["syncedLyrics"])
                        if len(parsed) >= 5:
                            best_synced = parsed
                            break
            if best_synced or best_plain:
                break
        except Exception:
            continue

    return best_synced, best_plain


def fetch_youtube_human_subtitles(video_id: str) -> tuple[Optional[list[dict]], str, str, float]:
    """Probe YouTube metadata and download human-authored subtitles if available."""
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(url, download=False)

    title = info.get("title") or video_id
    duration = float(info.get("duration") or 0.0)
    subs = info.get("subtitles") or {}
    if not subs:
        return None, "", title, duration

    preferred_langs = ["yue-HK", "yue", "zh-HK", "zh-Hant", "zh-TW", "zh", "ja", "ko", "en"]
    chosen_lang = None
    for lang in preferred_langs:
        if lang in subs:
            chosen_lang = lang
            break
    if not chosen_lang:
        chosen_lang = next(iter(subs.keys()), None)
    if not chosen_lang:
        return None, "", title, duration

    formats = subs.get(chosen_lang) or []
    json3_url = None
    for fmt in formats:
        if fmt.get("ext") == "json3" and fmt.get("url"):
            json3_url = fmt["url"]
            break
    if not json3_url:
        return None, "", title, duration

    try:
        r = httpx.get(json3_url, timeout=10)
        if r.status_code != 200:
            return None, "", title, duration
        data = r.json()
        lines = []
        for ev in data.get("events") or []:
            segs = ev.get("segs")
            if not segs:
                continue
            text = "".join(s.get("utf8", "") for s in segs).replace("\n", " ").strip()
            text = re.sub(r"\s+", " ", text)
            if not text or text in ("♪", "♫", "[音樂]", "[Music]", "（音樂）"):
                continue
            start_s = round((ev.get("tStartMs") or 0) / 1000.0, 2)
            dur_s = (ev.get("dDurationMs") or 4000) / 1000.0
            end_s = round(start_s + max(0.5, dur_s), 2)
            is_cjk = chosen_lang.startswith(("zh", "yue"))
            lines.append({
                "start": start_s,
                "end": end_s,
                "text": to_hk_traditional(text) if is_cjk else text,
                "words": [],
            })
        if len(lines) >= 4:
            return lines, chosen_lang, title, duration
    except Exception:
        pass

    return None, "", title, duration


def run_whisper_on_vocals(vocals_path: Path, prompt_text: str, device_mode: str) -> tuple[list[dict], str]:
    """Run faster-whisper (large-v3-turbo) on clean isolated vocals.mp3 with word-level timestamps."""
    add_nvidia_dll_dirs()
    from faster_whisper import WhisperModel

    free_mb = query_free_vram_mb()
    use_cuda = (device_mode != "cpu") and (device_mode == "gpu" or free_mb >= 2000)

    model = None
    used_device = "cpu"
    if use_cuda:
        try:
            model = WhisperModel(
                "large-v3-turbo",
                device="cuda",
                compute_type="float16",
                download_root=str(MODEL_DIR),
            )
            used_device = "cuda"
        except Exception as e:
            print(f"[Lyrics] CUDA init fallback to CPU: {e}", file=sys.stderr, flush=True)
            model = None

    if model is None:
        model = WhisperModel(
            "large-v3-turbo",
            device="cpu",
            compute_type="int8",
            cpu_threads=max(4, min(8, os.cpu_count() or 4)),
            download_root=str(MODEL_DIR),
        )
        used_device = "cpu"

    # Keep prompt concise (~220 chars) so Whisper uses it as Hong Kong Traditional / song vocabulary guidance
    if prompt_text:
        vocab_hint = "以下是繁體中文歌詞：" + " ".join(prompt_text.splitlines()[:12])[:260]
    else:
        vocab_hint = "以下是繁體中文粵語或國語歌曲歌詞，請使用標準香港繁體字。"

    def _run_with_model(m):
        segs_iter, inf = m.transcribe(
            str(vocals_path),
            beam_size=3,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 450, "speech_pad_ms": 200},
            initial_prompt=vocab_hint,
            condition_on_previous_text=False,
        )
        return list(segs_iter), inf

    try:
        segments, info = _run_with_model(model)
    except Exception as e:
        if used_device == "cuda":
            print(f"[Lyrics] CUDA transcribe fallback to CPU: {e}", file=sys.stderr, flush=True)
            model = WhisperModel(
                "large-v3-turbo",
                device="cpu",
                compute_type="int8",
                cpu_threads=max(4, min(8, os.cpu_count() or 4)),
                download_root=str(MODEL_DIR),
            )
            used_device = "cpu"
            segments, info = _run_with_model(model)
        else:
            raise

    is_cjk = (info.language or "") in ("zh", "yue", "")
    lines = []
    for seg in segments:
        txt = (seg.text or "").strip()
        if not txt or HALLUCINATION_RE.search(txt):
            continue
        if is_cjk:
            txt = to_hk_traditional(txt)

        words_out = []
        if seg.words:
            for w in seg.words:
                wt = (w.word or "").strip()
                if not wt:
                    continue
                if is_cjk:
                    wt = to_hk_traditional(wt)
                words_out.append({
                    "start": round(float(w.start), 2),
                    "end": round(float(w.end), 2),
                    "word": wt,
                })

        # Smart KTV line splitting: split long Whisper segments on breath pauses (>= 0.35s) or > 14 CJK chars / > 9 words
        if words_out and (len(txt) > 16 or len(words_out) > 10):
            cur_words = []
            cur_chars = 0
            for idx_w, w in enumerate(words_out):
                if cur_words:
                    gap = w["start"] - cur_words[-1]["end"]
                    hit_limit = (cur_chars >= 14) if is_cjk else (len(cur_words) >= 9)
                    if gap >= 0.35 or hit_limit:
                        joiner = "" if is_cjk else " "
                        line_txt = joiner.join(cw["word"] for cw in cur_words).strip()
                        if line_txt:
                            lines.append({
                                "start": cur_words[0]["start"],
                                "end": cur_words[-1]["end"],
                                "text": line_txt,
                                "words": cur_words,
                            })
                        cur_words = []
                        cur_chars = 0
                cur_words.append(w)
                cur_chars += len(w["word"])
            if cur_words:
                joiner = "" if is_cjk else " "
                line_txt = joiner.join(cw["word"] for cw in cur_words).strip()
                if line_txt:
                    lines.append({
                        "start": cur_words[0]["start"],
                        "end": cur_words[-1]["end"],
                        "text": line_txt,
                        "words": cur_words,
                    })
        else:
            lines.append({
                "start": round(float(seg.start), 2),
                "end": round(float(seg.end), 2),
                "text": txt,
                "words": words_out,
            })

    return lines, used_device


def write_status(status_file: Path, payload: dict) -> None:
    payload["updated_at"] = time.time()
    tmp = status_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    try:
        os.replace(tmp, status_file)
    except Exception:
        status_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--vocals-path", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--device-mode", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--force-ai", choices=["0", "1"], default="0")
    args = parser.parse_args()

    status_file = Path(args.status_file)
    out_file = Path(args.out_file)
    vocals_path = Path(args.vocals_path)
    force_ai = args.force_ai == "1"
    started = time.time()

    def update(status: str, progress: int, stage_text: str, error: Optional[str] = None):
        write_status(status_file, {
            "video_id": args.video_id,
            "status": status,
            "progress": progress,
            "stage_text": stage_text,
            "error": error,
            "elapsed_sec": round(time.time() - started, 1),
        })

    try:
        update("running", 10, "正在檢查原片字幕同歌詞庫...")
        yt_lines, yt_lang, title, duration = fetch_youtube_human_subtitles(args.video_id)

        # Layer 1: Human YouTube subtitles (unless user explicitly requested AI re-alignment)
        if yt_lines and not force_ai:
            result = {
                "video_id": args.video_id,
                "title": title,
                "source": "youtube_sub",
                "source_label": f"YouTube 字幕 ({yt_lang})",
                "lines": yt_lines,
                "created_at": time.time(),
            }
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            update("ready", 100, f"已載入 YouTube 字幕 ({yt_lang})")
            return 0

        # Layer 2: Probe lrclib.net (exact studio timing + clean vocabulary prompt for Whisper)
        update("running", 25, "正在搜尋官方歌詞字庫...")
        lrc_synced, plain_prompt = fetch_lrclib(title, duration)

        if lrc_synced and not force_ai:
            result = {
                "video_id": args.video_id,
                "title": title,
                "source": "lrclib_exact",
                "source_label": "精準同步歌詞 (LRCLIB)",
                "lines": lrc_synced,
                "created_at": time.time(),
            }
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            update("ready", 100, "已載入精準同步歌詞")
            return 0

        # Layer 3: Local AI faster-whisper (large-v3-turbo) on our BS-RoFormer clean vocals.mp3!
        if not vocals_path.exists():
            raise RuntimeError("搵唔到純人聲檔 vocals.mp3，請先開始唱K 拆軌。")

        update("running", 45, "AI 正在對準純人聲軌逐字對齊字幕（首次需下載模型）...")
        ai_lines, used_dev = run_whisper_on_vocals(vocals_path, plain_prompt, args.device_mode)
        if not ai_lines:
            raise RuntimeError("AI 未能喺人聲軌識別到歌詞。")

        src = "hybrid_whisper" if plain_prompt else "whisper_ai"
        label = f"AI 純人聲逐字對齊 ({used_dev.upper()})"
        result = {
            "video_id": args.video_id,
            "title": title,
            "source": src,
            "source_label": label,
            "lines": ai_lines,
            "created_at": time.time(),
        }
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        update("ready", 100, f"AI 歌詞對齊完成（耗時 {round(time.time() - started, 1)} 秒）")
        return 0
    except Exception as exc:
        update("error", 0, f"歌詞生成失敗：{exc}", error=str(exc))
        print(f"[LyricsWorker Error] {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
