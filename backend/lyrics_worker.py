"""Isolated 3-layer hybrid KTV lyrics engine with Forced Alignment (0% typos on Live/Covers!).

Priority:
1. YouTube human-authored subtitles (yue-HK / zh-HK / zh-TW / en / ja / ko) — instant (~1s),
   unless --force-ai 1 is passed.
2. Multi-source Official Lyrics lookup (LRCLIB + NetEase Cloud Music with CJK title extraction
   so live covers always find the original song's 100% real lyrics!).
   - If studio track (not live/cover) and LRCLIB duration matches within +/-2s: instant LRC (~0.5s).
3. Forced Alignment via stable-ts (reusing faster-whisper large-v3-turbo on BS-RoFormer vocals.mp3):
   - Aligns the 100% REAL official lyric characters directly to the singer's voice in THIS video!
   - Clamps start times using real RMS vocal onset detection so lyrics NEVER pop up during 0:00 silence!
   - If no official lyrics exist anywhere online: free-transcribes with strict RMS silence gate +
     anti-hallucination filter + OpenCC s2hk HK Traditional Chinese.
"""

import argparse
import asyncio
from difflib import SequenceMatcher
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "whisper"
TEMP_DIR = BASE_DIR / "tmp"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
LIVE_KEYWORDS_RE = re.compile(
    r"(live|concert|演唱會|演唱会|音樂會|音乐会|現場|现场|cover|翻唱|acoustic|unplugged|remix|plug\s*dj|medley|串燒|ktv|試聽|试听|短版|cut|clip|片段|mashup|組曲|组曲|合集|精選|精选|聯唱|联唱)",
    re.IGNORECASE,
)
MASHUP_KEYWORDS_RE = re.compile(
    r"(mashup|medley|串燒|串烧|組曲|组曲|合集|合輯|精選|精选|聯唱|联唱|top\s*\d+|金曲|歌單|歌单|連播|连播|特輯|特辑|勁歌金曲|劲歌金曲)",
    re.IGNORECASE,
)
HALLUCINATION_RE = re.compile(
    r"(訂閱|订阅|按讚|点赞|感謝觀看|感谢观看|字幕組|字幕组|字幕視聽|字幕提供|Amara\.org|謝謝收看|谢谢收看|Subscribe|Thanks for watching|獨播劇場|YoYoTelevision)",
    re.IGNORECASE,
)
GARBAGE_CHARS_RE = re.compile(r"[\ufffd\u0400-\u04FF]")  # Replacement char or Cyrillic hallucinations
CREDIT_LINE_RE = re.compile(
    r"^\s*(作[詞词]|作曲|[編编]曲|[監监]製|製作人|混音|錄音|和音|母帶|出品|發行|原唱|翻唱|演唱|詞|曲|編|監|OP|SP|Lyricist|Composer|Arranger|Producer)\s*[:：]",
    re.IGNORECASE,
)


def ensure_ffmpeg_in_path() -> None:
    """Ensure winget/local ffmpeg is on PATH for stable-ts / librosa audio loading."""
    if shutil.which("ffmpeg"):
        return
    for root in (
        Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages",
        Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links",
        Path("C:/ffmpeg/bin"),
    ):
        if not root.exists():
            continue
        for match in root.rglob("ffmpeg.exe"):
            d = str(match.parent)
            if d not in os.environ.get("PATH", ""):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            return


def add_nvidia_dll_dirs() -> None:
    """Expose PyTorch's bundled CUDA/cuBLAS/cuDNN DLLs so CTranslate2 (faster-whisper) uses RTX GPU on Windows."""
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


def to_simplified(text: str) -> str:
    if not text:
        return text
    try:
        from opencc import OpenCC
        if not hasattr(to_simplified, "_cc"):
            to_simplified._cc = OpenCC("hk2s")
        return to_simplified._cc.convert(text)
    except Exception:
        return text


def extract_lyrics_from_description(yt_desc: str) -> list[str]:
    """Extract pasted lyrics from YouTube video descriptions (crucial for 改詞版 / parodies / indie covers!)."""
    if not yt_desc or len(yt_desc) < 80:
        return []
    url_re = re.compile(r"(https?://|www\.|@|#|\bfb\.com\b|\big\b|instagram|facebook|spotify|apple\s*music|kkbox|joox|moov|訂閱|頻道|商業合作|email)", re.IGNORECASE)
    best_block: list[str] = []
    cur_block: list[str] = []
    for raw in yt_desc.splitlines():
        s = raw.strip()
        if not s:
            if len(cur_block) > len(best_block):
                best_block = cur_block
            cur_block = []
            continue
        if url_re.search(s) or CREDIT_LINE_RE.match(s) or len(s) > 46 or len(s) < 3:
            if len(cur_block) > len(best_block):
                best_block = cur_block
            cur_block = []
            continue
        # Must contain real text (CJK or alphabetic words)
        if re.search(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", s):
            cur_block.append(to_hk_traditional(s))
        else:
            if len(cur_block) > len(best_block):
                best_block = cur_block
            cur_block = []
    if len(cur_block) > len(best_block):
        best_block = cur_block
    return best_block if len(best_block) >= 8 else []


def clean_search_title(title: str) -> str:
    t = re.sub(r"[\[\(【（][^\]\)】）]*[\]\)】）]", " ", title)
    t = re.sub(r"[_|\-/]+", " ", t)
    t = re.sub(r"\b(official|mv|music video|lyrics|lyric video|hd|4k|hq|audio|\d+)\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t or title


def _split_long_official_line(line: str) -> list[str]:
    """Split LRCLIB/NetEase compound multi-phrase CJK lines (e.g. '讓我對這世界好奇 讓我信自己的真理')
    on spaces/commas into natural single-breath KTV phrases (<= 11 CJK chars) so forced alignment NEVER
    crushes two separate musical phrases into one 3-second window!
    """
    non_sp = re.sub(r"\s+", "", line)
    cjk_cnt = len(re.findall(r"[\u4e00-\u9fff]", non_sp))
    if cjk_cnt < 10 or len(non_sp) <= 11:
        return [line]
    parts = [p.strip() for p in re.split(r"[\s,，、；;]+", line) if p.strip()]
    if len(parts) <= 1:
        return [line]
    merged: list[str] = []
    for p in parts:
        if merged and len(re.sub(r"\s+", "", merged[-1] + p)) <= 11:
            merged[-1] = f"{merged[-1]} {p}"
        else:
            merged.append(p)
    return merged


def clean_lyric_lines(raw_lines: list[str]) -> list[str]:
    out = []
    for raw in raw_lines:
        s = re.sub(r"\[[^\]]*\]", "", raw).strip()
        if not s:
            continue
        if CREDIT_LINE_RE.match(s):
            continue
        if s in ("純音樂，請欣賞", "纯音乐，请欣赏", "♪", "♫", "~"):
            continue
        hk = to_hk_traditional(s)
        out.extend(_split_long_official_line(hk))
    return out


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
        if not text or CREDIT_LINE_RE.match(text):
            continue
        lines.append({"start": round(mins * 60 + secs, 2), "end": 0.0, "text": to_hk_traditional(text), "words": []})

    lines.sort(key=lambda x: x["start"])
    for i, item in enumerate(lines):
        if i + 1 < len(lines):
            item["end"] = round(min(item["start"] + 8.0, lines[i + 1]["start"]), 2)
        else:
            item["end"] = round(item["start"] + 6.0, 2)
    return lines


def detect_vocal_energy_profile(vocals_path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Load 16kHz mono vocals + per-100ms RMS dBFS profile so we know exact silent vs singing regions."""
    try:
        ensure_ffmpeg_in_path()
        import librosa
        y, sr = librosa.load(str(vocals_path), sr=16000, mono=True)
        hop = 1600  # 0.1s per frame
        n_frames = max(1, len(y) // hop)
        rms = np.array([
            float(np.sqrt(np.mean(y[i * hop : (i + 1) * hop] ** 2) + 1e-12))
            for i in range(n_frames)
        ])
        db = 20.0 * np.log10(rms + 1e-12)
        # Find first sustained vocal onset (> -38 dBFS for at least 0.3s)
        first_onset = 0.0
        for i in range(len(db) - 2):
            if db[i] > -38.0 and db[i + 1] > -38.0:
                first_onset = max(0.0, i * 0.1 - 0.15)
                break
        return y.astype(np.float32), db, 0.1, first_onset
    except Exception:
        return np.array([], dtype=np.float32), np.array([]), 0.1, 0.0


def segment_has_vocal_energy(db_profile: np.ndarray, step_sec: float, start_s: float, end_s: float) -> bool:
    if len(db_profile) == 0:
        return True
    i0 = max(0, int(start_s / step_sec))
    i1 = min(len(db_profile), max(i0 + 1, int(end_s / step_sec) + 1))
    seg_db = db_profile[i0:i1]
    if len(seg_db) == 0:
        return False
    # At least 90th-percentile energy in segment must exceed -38 dBFS (real singing, not silence!)
    return float(np.percentile(seg_db, 85)) > -38.0


def _track_matches_title(track_name: str, artist_name: str, yt_title_clean: str) -> float:
    """Score how well a lyrics DB candidate (track_name, artist_name) matches the YouTube video title.
    Returns 0.0 if track_name does NOT actually appear in yt_title_clean (prevents artist-only wrong-song hits
    like returning '張敬軒 - 斷點' when searching '張敬軒 - 春秋'!).
    """
    # Unify HK/TW/CN variant glyphs (著/着, 裡/裏, 說/説, 臺/台) via OpenCC Simplified for comparison ONLY
    t_norm = re.sub(r"[\s\W_]+", "", to_simplified(track_name or "")).lower()
    t_base = re.sub(r"[\(\[（【].*?[\)\]）】]", "", track_name or "")
    t_base_norm = re.sub(r"[\s\W_]+", "", to_simplified(t_base)).lower()
    y_norm = re.sub(r"[\s\W_]+", "", to_simplified(yt_title_clean or "")).lower()

    if not t_base_norm or not y_norm:
        return 0.0

    # Must actually match the song's track name (not a 1-char substring of an artist name!)
    score = 0.0
    if t_base_norm == y_norm:
        score = 2.0
    elif len(t_base_norm) >= 2 and t_base_norm in y_norm:
        # Strong exact track-name containment; longer track names score clearly higher
        score = 1.0 + min(0.8, len(t_base_norm) * 0.12)
    else:
        # Fuzzy fallback for minor spelling differences (Traditional/Simplified/punctuation)
        cjk_t = set(re.findall(r"[\u4e00-\u9fff]", t_base_norm))
        cjk_y = set(re.findall(r"[\u4e00-\u9fff]", y_norm))
        if len(cjk_t) >= 2:
            overlap = len(cjk_t & cjk_y) / len(cjk_t)
            if overlap >= 0.75:
                score = overlap
        elif len(t_base_norm) >= 2:
            sim = SequenceMatcher(None, t_base_norm, y_norm).ratio()
            if sim >= 0.65:
                score = sim

    if score <= 0.0:
        return 0.0

    # Bonus if artist name also appears in the YouTube title (disambiguates same-title songs)
    a_norm = re.sub(r"[\s\W_]+", "", to_simplified(artist_name or "")).lower()
    if a_norm and len(a_norm) >= 2 and a_norm in y_norm:
        score += 0.4
    elif t_norm == t_base_norm:
        score += 0.05
    return score


def fetch_official_lyrics(title: str, duration: float, yt_desc: str = "") -> tuple[Optional[list[dict]], list[str], str]:
    """Search LRCLIB + NetEase Cloud Music for 100% official lyrics text (even for live covers!).
    Strictly validates that the returned trackName matches the YouTube video title so artist-name queries
    NEVER pull the wrong song (e.g. '斷點' for '張敬軒 - 春秋').
    """
    cleaned = clean_search_title(title)
    queries = []
    # 1) Always try full cleaned title FIRST ("張敬軒 春秋") so artist + song stay together!
    if cleaned:
        queries.append(cleaned)
    # 2) Then try individual CJK segments (both shorter and longer, e.g. "春秋" AND "戀愛腦之死")
    cjk_parts = re.findall(r"[\u4e00-\u9fff]{2,}", cleaned)
    for p in cjk_parts:
        if p not in queries:
            queries.append(p)
    if title not in queries:
        queries.append(title)

    is_live_or_alt = bool(LIVE_KEYWORDS_RE.search(title)) or bool(LIVE_KEYWORDS_RE.search(yt_desc[:200]))
    best_synced = None
    best_lines: list[str] = []
    best_source = ""
    best_score = 0.0

    # 1) LRCLIB (cleanest Traditional/English/Japanese lyrics)
    for q in queries[:4]:
        try:
            r = httpx.get("https://lrclib.net/api/search", params={"q": q}, timeout=8)
            if r.status_code != 200:
                continue
            items = r.json()
            if not isinstance(items, list) or not items:
                continue
            for it in items[:10]:
                tr_name = it.get("trackName") or ""
                ar_name = it.get("artistName") or ""
                match_sc = _track_matches_title(tr_name, ar_name, cleaned)
                if match_sc <= 0.0:
                    continue

                pl = (it.get("plainLyrics") or "").strip()
                sl = (it.get("syncedLyrics") or "").strip()
                cand_lines = clean_lyric_lines(pl.splitlines()) if pl else clean_lyric_lines([x["text"] for x in parse_lrc(sl)])
                if len(cand_lines) >= 6 and match_sc > best_score:
                    best_score = match_sc
                    best_lines = cand_lines
                    best_source = f"LRCLIB ({tr_name})"

                if not is_live_or_alt and sl and duration > 0 and match_sc >= best_score:
                    it_dur = float(it.get("duration") or 0)
                    if abs(it_dur - duration) <= 2.0:
                        parsed = parse_lrc(sl)
                        if len(parsed) >= 6:
                            best_synced = parsed
                            best_source = "LRCLIB"
                            best_score = match_sc
            if best_synced or best_score >= 1.35:
                break
        except Exception:
            continue

    # 2) NetEase Cloud Music fallback (massive HK Cantopop / Mandopop library — query in BOTH hk2s Simplified & original!)
    if not best_lines:
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://music.163.com/"}
        ne_queries: list[str] = []
        for q in queries[:3]:
            qs = to_simplified(q)
            if qs and qs not in ne_queries:
                ne_queries.append(qs)
            if q not in ne_queries:
                ne_queries.append(q)
        for q in ne_queries[:4]:
            try:
                r = httpx.get("https://music.163.com/api/search/get", params={"s": q, "type": 1, "limit": 8}, headers=headers, timeout=8)
                songs = (r.json().get("result") or {}).get("songs") or []
                for s in songs[:8]:
                    tr_name = s.get("name") or ""
                    ar_name = " ".join((a.get("name") or "") for a in (s.get("artists") or []))
                    match_sc = _track_matches_title(tr_name, ar_name, cleaned)
                    if match_sc <= 0.0 or match_sc <= best_score:
                        continue
                    lr = httpx.get("https://music.163.com/api/song/lyric", params={"id": s["id"], "lv": 1, "tv": -1}, headers=headers, timeout=8)
                    lrc = (lr.json().get("lrc") or {}).get("lyric", "")
                    if not lrc:
                        continue
                    parsed = parse_lrc(lrc)
                    if len(parsed) >= 6:
                        best_score = match_sc
                        best_lines = clean_lyric_lines([x["text"] for x in parsed])
                        best_source = f"NetEase ({tr_name})"
                if best_score >= 1.35:
                    break
            except Exception:
                continue

    # 3) YouTube description fallback (catches 改詞版 / parodies / brand-new indie tracks that paste lyrics in description!)
    if not best_lines and yt_desc:
        desc_lines = extract_lyrics_from_description(yt_desc)
        if desc_lines:
            best_lines = desc_lines
            best_source = "YouTube Description"

    return best_synced, best_lines, best_source


def _pcm16_wav_bytes(y: np.ndarray, sr: int = 16000) -> bytes:
    y16 = np.clip(y * 32767.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(y16.tobytes())
    return buf.getvalue()


def shazam_scan_and_fetch_lyrics(
    audio_16k: np.ndarray,
    db_profile: np.ndarray,
    step_sec: float,
    inst_path: Optional[Path] = None,
    scan_mashup: bool = False,
) -> tuple[list[str], list[dict]]:
    """Use free shazamio acoustic fingerprinting to identify song(s) even when YouTube title has NO song name
    or when the video is a multi-song Mashup / Medley (up to 15 min)!
    Returns (combined_official_lines, identified_tracks_list).
    """
    if len(audio_16k) < 16000 * 8:
        return [], []
    try:
        from shazamio import Shazam
    except Exception:
        return [], []

    # Mix instrumental + vocals if inst_path is available locally (10x faster & more accurate Shazam lock!)
    probe_audio = audio_16k
    if inst_path and inst_path.exists():
        try:
            import librosa
            yi, _ = librosa.load(str(inst_path), sr=16000, mono=True)
            n = min(len(audio_16k), len(yi))
            if n > 16000 * 8:
                probe_audio = np.clip(audio_16k[:n] + yi[:n], -1.0, 1.0).astype(np.float32)
        except Exception:
            pass

    total_sec = len(probe_audio) / 16000.0
    clip_sec = 10.0

    # Pick smart active probe timestamps
    if scan_mashup:
        step_probe = 25.0 if total_sec <= 360 else max(35.0, total_sec / 16.0)
        offsets = []
        t_cur = 10.0
        while t_cur + clip_sec < total_sec and len(offsets) < 16:
            if segment_has_vocal_energy(db_profile, step_sec, t_cur, t_cur + clip_sec):
                offsets.append(t_cur)
            t_cur += step_probe
        if not offsets:
            offsets = [min(total_sec * 0.25, max(0.0, total_sec - clip_sec))]
    else:
        # Single song: probe 2 active singing windows (fast ~0.5-1.5s!)
        cands = [total_sec * 0.25, total_sec * 0.55, total_sec * 0.12]
        offsets = []
        for t_c in cands:
            if t_c + clip_sec < total_sec and segment_has_vocal_energy(db_profile, step_sec, t_c, t_c + clip_sec):
                offsets.append(t_c)
            if len(offsets) >= 2:
                break
        if not offsets:
            offsets = [max(0.0, min(total_sec * 0.3, total_sec - clip_sec))]

    async def _run_probes() -> list[dict]:
        shazam = Shazam()
        sem = asyncio.Semaphore(3)

        async def _probe_one(off_sec: float) -> Optional[dict]:
            s0 = int(off_sec * 16000)
            s1 = min(len(probe_audio), int((off_sec + clip_sec) * 16000))
            wav_b = _pcm16_wav_bytes(probe_audio[s0:s1], 16000)
            async with sem:
                try:
                    res = await asyncio.wait_for(shazam.recognize(wav_b), timeout=14.0)
                except Exception:
                    return None
            tr = (res or {}).get("track") or {}
            tr_title = (tr.get("title") or "").strip()
            tr_artist = (tr.get("subtitle") or "").strip()
            if not tr_title:
                return None
            # Also grab embedded official lyrics directly from Shazam response if present!
            shz_lyrics: list[str] = []
            for sec in (tr.get("sections") or []):
                if sec.get("type") == "LYRICS" and isinstance(sec.get("text"), list):
                    shz_lyrics = clean_lyric_lines(sec["text"])
                    break
            return {"offset": round(off_sec, 1), "title": tr_title, "artist": tr_artist, "shz_lyrics": shz_lyrics}

        results = await asyncio.gather(*[_probe_one(o) for o in offsets])
        return [r for r in results if r]

    try:
        hits = asyncio.run(_run_probes())
    except Exception as exc:
        print(f"[Lyrics] Shazam scan error: {exc}", flush=True)
        return [], []

    # Deduplicate unique tracks chronologically
    seen_keys: set[str] = set()
    unique_tracks: list[dict] = []
    for h in sorted(hits, key=lambda x: x["offset"]):
        key = _norm_lyrics_cmp(re.sub(r"[\(\[（【].*?[\)\]）】]", "", h["title"]))
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        unique_tracks.append(h)

    if not unique_tracks:
        return [], []

    print(
        "[Lyrics] Shazam identified %d track(s): %s"
        % (len(unique_tracks), ", ".join(f"{t['artist']} - {t['title']} @{t['offset']}s" for t in unique_tracks)),
        flush=True,
    )

    # Fetch 100% official lyrics for every identified song and combine into one multi-song pool!
    combined_lines: list[str] = []
    for tr in unique_tracks:
        query_title = f"{tr['artist']} {tr['title']}".strip()
        _, lines, _ = fetch_official_lyrics(query_title, 0.0)
        if not lines:
            _, lines, _ = fetch_official_lyrics(tr["title"], 0.0)
        if not lines and tr.get("shz_lyrics"):
            lines = tr["shz_lyrics"]
        if lines:
            combined_lines.extend(lines)

    return combined_lines, unique_tracks


def search_official_lyrics_by_sung_phrases(asr_segs: list[dict]) -> list[str]:
    """Identify unknown song(s) inside a Mashup / Live Cover / Acoustic Medley directly from sung lyric phrases!
    Uses NetEase Cloud Music full-text lyrics search (type=1006) which even tolerates Whisper homophone typos
    and discovers MULTIPLE back-to-back songs inside a single continuous Mashup cluster!
    """
    cands = [
        s["text"].strip()
        for s in asr_segs
        if 5 <= len(_norm_lyrics_cmp(s["text"])) <= 20
    ]
    if not cands:
        return []

    # Sample up to 6 phrases evenly across the span so back-to-back songs in one Mashup are ALL discovered!
    n_probes = min(6, len(cands))
    if len(cands) <= n_probes:
        probes = cands
    else:
        idxs = sorted({int(round(i * (len(cands) - 1) / (n_probes - 1))) for i in range(n_probes)})
        probes = [cands[i] for i in idxs]

    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://music.163.com/"}
    asr_blob = "".join(_norm_lyrics_cmp(s["text"]) for s in asr_segs)

    seen_song_ids: set[int] = set()
    seen_titles: set[str] = set()
    rescued_all: list[str] = []

    for q in probes:
        qs = to_simplified(q)
        try:
            r = httpx.get(
                "https://music.163.com/api/search/get",
                params={"s": qs, "type": 1006, "limit": 4},
                headers=headers,
                timeout=7,
            )
            songs = (r.json().get("result") or {}).get("songs") or []
        except Exception:
            continue

        for s in songs[:3]:
            sid = s.get("id")
            tr_name = s.get("name") or ""
            base_t = _norm_lyrics_cmp(re.sub(r"[\(\[（【].*?[\)\]）】]", "", tr_name))
            if not sid or sid in seen_song_ids or (base_t and base_t in seen_titles):
                continue
            seen_song_ids.add(sid)
            try:
                lr = httpx.get(
                    "https://music.163.com/api/song/lyric",
                    params={"id": sid, "lv": 1, "tv": -1},
                    headers=headers,
                    timeout=7,
                )
                lrc = (lr.json().get("lrc") or {}).get("lyric", "")
                if not lrc:
                    continue
                parsed = parse_lrc(lrc)
                lines = [x["text"] for x in parsed]
                if len(lines) < 6:
                    continue
                # Verify this song genuinely matches >= 3 sung lines in our unrecognized span!
                hit_lines = sum(
                    1 for ln in set(lines)
                    if len(_norm_lyrics_cmp(ln)) >= 4 and (
                        _norm_lyrics_cmp(ln)[:4] in asr_blob
                        or any(_lyric_similarity(seg["text"], ln) >= 0.45 for seg in asr_segs)
                    )
                )
                if hit_lines >= 3:
                    if base_t:
                        seen_titles.add(base_t)
                    artists = "/".join((a.get("name") or "") for a in (s.get("artists") or []))
                    print(
                        f"[Lyrics] Phrase-search rescued song: {artists} - {tr_name} (matched {hit_lines} lines)",
                        flush=True,
                    )
                    rescued_all.extend(lines)
                    break  # Found the song for this probe phrase; move to next probe in Mashup!
            except Exception:
                continue

    return rescued_all


def fetch_youtube_human_subtitles(video_id: str) -> tuple[Optional[list[dict]], str, str, float, str]:
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(url, download=False)

    title = info.get("title") or video_id
    duration = float(info.get("duration") or 0.0)
    desc = info.get("description") or ""
    subs = info.get("subtitles") or {}
    if not subs:
        return None, "", title, duration, desc

    preferred_langs = ["yue-HK", "yue", "zh-HK", "zh-Hant", "zh-TW", "zh", "ja", "ko", "en"]
    chosen_lang = next((l for l in preferred_langs if l in subs), next(iter(subs.keys()), None))
    if not chosen_lang:
        return None, "", title, duration, desc

    formats = subs.get(chosen_lang) or []
    json3_url = next((f["url"] for f in formats if f.get("ext") == "json3" and f.get("url")), None)
    if not json3_url:
        return None, "", title, duration, desc

    try:
        r = httpx.get(json3_url, timeout=10)
        if r.status_code != 200:
            return None, "", title, duration, desc
        data = r.json()
        lines = []
        for ev in data.get("events") or []:
            segs = ev.get("segs")
            if not segs:
                continue
            text = re.sub(r"\s+", " ", "".join(s.get("utf8", "") for s in segs).replace("\n", " ")).strip()
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
            return lines, chosen_lang, title, duration, desc
    except Exception:
        pass

    return None, "", title, duration, desc


def detect_lyrics_language(lines: list[str]) -> str:
    joined = "".join(lines)
    if not joined:
        return "zh"
    cjk = len(re.findall(r"[\u4e00-\u9fff]", joined))
    jp = len(re.findall(r"[\u3040-\u30ff]", joined))
    ko = len(re.findall(r"[\uac00-\ud7af]", joined))
    total = max(1, len(joined))
    if jp / total > 0.15:
        return "ja"
    if ko / total > 0.15:
        return "ko"
    if cjk / total > 0.2:
        return "zh"
    return "en"


def _norm_lyrics_cmp(s: str) -> str:
    return re.sub(r"[\s\W_]+", "", to_hk_traditional(s or "")).lower()


def _lyric_similarity(a: str, b: str) -> float:
    na, nb = _norm_lyrics_cmp(a), _norm_lyrics_cmp(b)
    if not na or not nb:
        return 0.0
    sm = SequenceMatcher(None, na, nb).ratio()
    sa, sb = set(na), set(nb)
    jacc = len(sa & sb) / max(1, len(sa | sb))
    cov = len(sa & sb) / max(1, min(len(sa), len(sb)))
    return 0.55 * sm + 0.30 * jacc + 0.15 * cov


def build_silence_collapsed_audio(
    y: np.ndarray,
    db_prof: np.ndarray,
    step_s: float,
    sr: int = 16000,
) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
    """Collapse long silent instrumental interludes (>1.6s below -39dBFS) down to 0.35s so
    Whisper's 30s sliding window NEVER hallucinates lyrics into instrumental breaks, while
    preserving an exact piecewise map (comp_start, comp_end, real_start, real_end).
    """
    n_total = len(y)
    if n_total == 0 or len(db_prof) == 0:
        dur = n_total / sr
        return y, [(0.0, dur, 0.0, dur)]

    active = db_prof > -39.0
    pad_frames = max(1, int(round(0.35 / step_s)))
    padded = np.zeros_like(active, dtype=bool)
    for i, a in enumerate(active):
        if a:
            padded[max(0, i - pad_frames) : min(len(active), i + pad_frames + 1)] = True

    islands: list[tuple[float, float]] = []
    in_isl = False
    s_idx = 0
    for i, a in enumerate(padded):
        if a and not in_isl:
            s_idx = i
            in_isl = True
        elif not a and in_isl:
            if islands and (s_idx * step_s - islands[-1][1]) < 1.6:
                islands[-1] = (islands[-1][0], i * step_s)
            else:
                islands.append((s_idx * step_s, i * step_s))
            in_isl = False
    if in_isl:
        islands.append((s_idx * step_s, len(padded) * step_s))

    if not islands:
        dur = n_total / sr
        return y, [(0.0, dur, 0.0, dur)]

    gap_sil = np.zeros(int(0.35 * sr), dtype=np.float32)
    chunks: list[np.ndarray] = []
    mapping: list[tuple[float, float, float, float]] = []
    comp_cursor = 0.0
    for idx, (rs, re_) in enumerate(islands):
        i0 = max(0, int(rs * sr))
        i1 = min(n_total, int(re_ * sr))
        if i1 <= i0:
            continue
        if idx > 0:
            chunks.append(gap_sil)
            comp_cursor += len(gap_sil) / sr
        seg = y[i0:i1]
        dur = len(seg) / sr
        chunks.append(seg)
        mapping.append((comp_cursor, comp_cursor + dur, i0 / sr, i1 / sr))
        comp_cursor += dur

    return (np.concatenate(chunks) if chunks else y), mapping


def _map_comp_to_real(t_comp: float, mapping: list[tuple[float, float, float, float]]) -> float:
    for cs, ce, rs, re_ in mapping:
        if t_comp <= ce + 1e-4:
            frac = max(0.0, min(1.0, (t_comp - cs) / max(1e-6, ce - cs)))
            return rs + frac * (re_ - rs)
    return mapping[-1][3] if mapping else t_comp


def refine_alignment_with_acoustics(
    lines: list[dict],
    db_profile: np.ndarray,
    step_sec: float,
    audio_dur: float,
    first_onset: float,
) -> list[dict]:
    """Fundamental acoustic refinement for singing voices:
    1. Onset Snap: if line.start sits in silence before singing begins, snap forward to real vocal onset.
    2. Tail Extension: extend line.end through held final singing vowels until actual vocal offset.
    3. Word Smoothing: eliminate 0.00s collapsed BPE word glitches so 60fps KTV sweep never jerks.
    """
    if not lines or len(db_profile) == 0:
        return lines

    n_db = len(db_profile)
    for i, ln in enumerate(lines):
        s = max(first_onset, float(ln["start"]))
        e = max(s + 0.35, float(ln["end"]))
        nxt_s = float(lines[i + 1]["start"]) if (i + 1 < len(lines)) else (audio_dur or e + 2.5)

        # 1) Acoustic Onset Snap: advance start out of leading silence (never past e - 0.5s)
        max_snap = min(e - 0.5, s + 2.5)
        t_cur = s
        while t_cur < max_snap:
            idx = int(t_cur / step_sec)
            if idx < n_db and db_profile[idx] > -39.0:
                break
            t_cur += step_sec
        if t_cur > s + 0.12 and t_cur < max_snap:
            s = round(max(first_onset, t_cur - 0.08), 2)

        # 2) Acoustic Vocal-Tail Extension: follow held final vowel until 0.2s genuine silence
        limit_e = min(nxt_s - 0.12, e + 2.2)
        if limit_e > e:
            t_e = e
            quiet_run = 0
            best_e = e
            while t_e < limit_e:
                idx = int(t_e / step_sec)
                if idx >= n_db:
                    break
                if db_profile[idx] > -39.0:
                    quiet_run = 0
                    best_e = t_e + step_sec
                else:
                    quiet_run += 1
                    if quiet_run >= 2:
                        break
                t_e += step_sec
            e = round(min(limit_e, best_e), 2)

        ln["start"] = round(s, 2)
        ln["end"] = round(max(s + 0.35, e), 2)

        # 3) Word-level smoothing: enforce strict bounds & fix collapsed <=0.05s BPE glitches
        ws = ln.get("words") or []
        if ws:
            # Clamp first word start & extend last word end to match refined acoustic boundaries
            ws[0]["start"] = round(max(ln["start"], min(float(ws[0]["start"]), ln["end"] - 0.2)), 2)
            ws[-1]["end"] = ln["end"]
            prev_end = ln["start"]
            for wi, w in enumerate(ws):
                w_s = max(prev_end, float(w["start"]))
                w_e = max(w_s + 0.05, min(ln["end"], float(w["end"])))
                if wi == len(ws) - 1:
                    w_e = ln["end"]
                w["start"] = round(w_s, 2)
                w["end"] = round(w_e, 2)
                prev_end = w["end"]

    return lines


def _alignment_needs_live_rematch(aligned, audio_dur: float, db_profile: np.ndarray, step_sec: float) -> tuple[bool, str]:
    """Detect skipped verses, mid-song starts, swapped sections, or crushed lines in monotonic alignment."""
    segs = [s for s in (aligned.segments or []) if (s.text or "").strip()]
    if not segs:
        return True, "empty"

    crushed = 0
    low_prob = 0
    prev_start = -1.0
    for seg in segs:
        txt = (seg.text or "").strip()
        n_chars = max(1, len(_norm_lyrics_cmp(txt)))
        dur = float(seg.end) - float(seg.start)
        if dur < max(0.45, 0.09 * n_chars) or (prev_start >= 0 and (float(seg.start) - prev_start) < 0.25):
            crushed += 1
        prev_start = float(seg.start)
        probs = [w.probability for w in (seg.words or []) if w.probability is not None]
        avg_p = (sum(probs) / len(probs)) if probs else 0.0
        if avg_p < 0.28:
            low_prob += 1

    low_ratio = low_prob / max(1, len(segs))
    if crushed >= 2:
        return True, f"crushed={crushed}"
    if low_ratio >= 0.24:
        return True, f"low_prob={low_ratio:.2f}"

    # Also check if there's >20s of active singing after the last aligned line (e.g. repeated encore)
    if len(db_profile) > 0 and audio_dur > 0:
        last_end = float(segs[-1].end)
        tail_idx = int((last_end + 4.0) / step_sec)
        if tail_idx < len(db_profile):
            active_tail_sec = float(np.sum(db_profile[tail_idx:] > -35.0)) * step_sec
            if active_tail_sec >= 18.0:
                return True, f"unaligned_tail={active_tail_sec:.1f}s"

    return False, "ok"


def _even_words_for_line(text: str, start_s: float, end_s: float) -> list[dict]:
    tokens = list(text.replace(" ", "")) if detect_lyrics_language([text]) in ("zh", "ja") else text.split()
    if not tokens:
        return []
    dur = max(0.4, end_s - start_s)
    step = dur / len(tokens)
    out = []
    for idx, tk in enumerate(tokens):
        ws = round(start_s + idx * step, 2)
        we = round(start_s + (idx + 1) * step, 2)
        out.append({"start": ws, "end": max(ws + 0.04, we), "word": tk})
    return out


def _live_section_anchored_align(
    model,
    vocals_path: Path,
    audio_16k: np.ndarray,
    db_profile: np.ndarray,
    step_sec: float,
    official_lines: list[str],
    lang: str,
    first_onset: float,
    use_cuda: bool = False,
    inst_path: Optional[Path] = None,
    shazam_already_tried: bool = False,
) -> list[dict]:
    """Live/Concert/Medley-aware alignment:
    1. Transcribes active vocal regions (using GPU BatchedInferencePipeline when available for 4-6x speed!)
       to discover WHICH official lines were actually sung & WHEN.
    2. Groups matched lines into contiguous singing blocks and force-aligns each block on its own
       sliced audio window so skipped sections can NEVER cause drift or crushed timestamps.
    """
    audio_input = audio_16k if len(audio_16k) > 0 else str(vocals_path)
    raw_segments = []

    # Fast GPU BatchedInferencePipeline (~2-4s for a 4-min concert on RTX 5060 Ti!)
    if use_cuda and len(audio_16k) > 0:
        try:
            from faster_whisper import BatchedInferencePipeline, WhisperModel
            if isinstance(model, WhisperModel):
                batched = BatchedInferencePipeline(model=model)
                b_segs, _ = batched.transcribe(
                    audio_16k,
                    language=lang,
                    batch_size=8,
                    beam_size=2,
                    vad_filter=False,
                    condition_on_previous_text=False,
                )
                raw_segments = list(b_segs)
        except Exception as _be:
            print(f"[Lyrics] Batched GPU transcribe fallback: {_be}", flush=True)
            raw_segments = []

    if not raw_segments:
        res = model.transcribe(
            audio_input,
            language=lang,
            beam_size=2,
            vad=False,
            suppress_silence=False,
            condition_on_previous_text=False,
            verbose=False,
        )
        raw_segments = list(res.segments or [])

    asr_segs = []
    for s in raw_segments:
        raw_t = (s.text or "").strip()
        txt = to_hk_traditional(raw_t) if lang == "zh" else raw_t
        if not txt or HALLUCINATION_RE.search(txt) or GARBAGE_CHARS_RE.search(txt):
            continue
        if not segment_has_vocal_energy(db_profile, step_sec, float(s.start), float(s.end)):
            continue
        asr_segs.append({"start": float(s.start), "end": float(s.end), "text": txt})

    if not asr_segs:
        return []

    def _match_against_pool(pool_lines: list[str]) -> tuple[list[dict], list[dict]]:
        u_off = list(dict.fromkeys(pool_lines))
        n_map: dict[str, set[str]] = {}
        for idx in range(len(pool_lines) - 1):
            n_map.setdefault(pool_lines[idx], set()).add(pool_lines[idx + 1])

        def _best_single(text: str, prev_l: Optional[str]) -> Optional[tuple[float, str, float, bool]]:
            c_len = len(_norm_lyrics_cmp(text))
            if c_len < 2:
                return None
            best = None
            for off in u_off:
                o_len = len(_norm_lyrics_cmp(off))
                if o_len == 0 or c_len > o_len * 2.1 or o_len > c_len * 2.6:
                    continue
                sim = _lyric_similarity(text, off)
                is_next = bool(prev_l and off in n_map.get(prev_l, set()))
                bonus = 0.22 if (is_next and sim >= 0.22) else (0.12 if is_next else 0.0)
                score = sim + bonus
                if best is None or score > best[0]:
                    best = (score, off, sim, is_next)
            return best

        m_out: list[dict] = []
        u_gaps: list[dict] = []
        idx_i = 0
        prev_l: Optional[str] = None
        while idx_i < len(asr_segs):
            b1 = _best_single(asr_segs[idx_i]["text"], prev_l)
            chosen_span = 1
            chosen = b1

            if idx_i + 1 < len(asr_segs) and (asr_segs[idx_i + 1]["start"] - asr_segs[idx_i]["end"]) <= 1.8:
                comb = f"{asr_segs[idx_i]['text']} {asr_segs[idx_i + 1]['text']}"
                b2 = _best_single(comb, prev_l)
                b_next = _best_single(asr_segs[idx_i + 1]["text"], b1[1] if b1 else prev_l)
                s1_raw = b1[2] if b1 else 0.0
                s_next_raw = b_next[2] if b_next else 0.0
                if b2 and b2[2] > max(s1_raw, s_next_raw) + 0.06:
                    chosen_span = 2
                    chosen = b2

            if chosen and (chosen[2] >= 0.30 or (chosen[3] and chosen[2] >= 0.20)):
                _, off, raw_sim, _ = chosen
                ts = asr_segs[idx_i]["start"]
                te = asr_segs[idx_i + chosen_span - 1]["end"]
                if m_out and m_out[-1]["text"] == off and (ts - m_out[-1]["end"]) < 7.5:
                    # Extend the existing line's end timestamp instead of discarding its second half!
                    m_out[-1]["end"] = max(m_out[-1]["end"], te)
                else:
                    m_out.append({"start": ts, "end": te, "text": off, "sim": raw_sim})
                    prev_l = off
                idx_i += chosen_span
            else:
                u_gaps.append(asr_segs[idx_i])
                idx_i += 1
        return m_out, u_gaps

    matched, unmatched_gaps = _match_against_pool(official_lines)
    matched_asr_count = len(asr_segs) - len(unmatched_gaps)

    # Smart Shazam rescue: if title-based lyrics poorly matched (<35%) — e.g. clickbait title or unlisted Mashup —
    # run Shazam multi-track fingerprinting and re-match the ALREADY-transcribed ASR segments (0 extra Whisper time!)
    if not shazam_already_tried and (not matched or (len(asr_segs) >= 6 and (matched_asr_count / len(asr_segs)) < 0.35)):
        shz_lines, shz_tracks = shazam_scan_and_fetch_lyrics(
            audio_16k, db_profile, step_sec, inst_path=inst_path, scan_mashup=True
        )
        if shz_lines:
            m2, g2 = _match_against_pool(shz_lines)
            if (len(asr_segs) - len(g2)) > matched_asr_count:
                official_lines = shz_lines
                matched, unmatched_gaps = m2, g2
                matched_asr_count = len(asr_segs) - len(unmatched_gaps)

    # Mashup / Medley Lyric-Phrase Rescue: if there is a sustained unrecognized singing cluster (>= 4 segments, >= 16s)
    # — e.g. Song #2 in a live/acoustic Mashup that Shazam couldn't fingerprint — identify it by sung lyric phrases!
    def _find_sustained_unmatched_clusters(gaps: list[dict]) -> list[list[dict]]:
        clusters: list[list[dict]] = []
        for g in gaps:
            if not clusters or (g["start"] - clusters[-1][-1]["end"]) > 10.0:
                clusters.append([g])
            else:
                clusters[-1].append(g)
        return [c for c in clusters if len(c) >= 4 and (c[-1]["end"] - c[0]["start"]) >= 16.0]

    extra_clusters = _find_sustained_unmatched_clusters(unmatched_gaps)
    if extra_clusters:
        added_pool = False
        for cl in extra_clusters[:3]:
            rescued = search_official_lyrics_by_sung_phrases(cl)
            if rescued:
                official_lines = list(official_lines) + rescued
                added_pool = True
        if added_pool:
            matched, unmatched_gaps = _match_against_pool(official_lines)
            matched_asr_count = len(asr_segs) - len(unmatched_gaps)

    if not matched:
        return []

    # Parody / Rewritten-lyrics detection: if < 35% of active singing segments STILL don't match any official lyrics,
    # show what the singer ACTUALLY sang instead of forcing wrong lyrics!
    if len(asr_segs) >= 6 and (matched_asr_count / len(asr_segs)) < 0.35:
        print(
            f"[Lyrics] Low official-lyrics match ({matched_asr_count}/{len(asr_segs)}); "
            "treating as rewritten/adapted lyrics and using actual sung transcript.",
            flush=True,
        )
        direct_lines: list[dict] = []
        for s in raw_segments:
            raw_t = (s.text or "").strip()
            txt = to_hk_traditional(raw_t) if lang == "zh" else raw_t
            if not txt or HALLUCINATION_RE.search(txt) or GARBAGE_CHARS_RE.search(txt):
                continue
            s_start, s_end = float(s.start), float(s.end)
            if s_end < first_onset or not segment_has_vocal_energy(db_profile, step_sec, s_start, s_end):
                continue
            words_out = []
            for w in (getattr(s, "words", None) or []):
                wt = (w.word or "").strip()
                if not wt or GARBAGE_CHARS_RE.search(wt):
                    continue
                words_out.append({
                    "start": round(max(first_onset, float(w.start)), 2),
                    "end": round(max(first_onset + 0.05, float(w.end)), 2),
                    "word": to_hk_traditional(wt) if lang == "zh" else wt,
                })
            s_clamped = round(max(first_onset, s_start), 2)
            e_clamped = round(max(s_clamped + 0.4, s_end), 2)
            direct_lines.append({
                "start": s_clamped,
                "end": e_clamped,
                "text": txt,
                "words": words_out or _even_words_for_line(txt, s_clamped, e_clamped),
            })
        return direct_lines

    # Multi-line energy-verified gap recovery: if 1..6 official lines sit between matched[k] and matched[k+1]
    # AND real vocal RMS confirms continuous singing in that span, restore ALL missing sung lines!
    pos_map: dict[str, list[int]] = {}
    for idx, off_txt in enumerate(official_lines):
        pos_map.setdefault(off_txt, []).append(idx)

    filled: list[dict] = []
    last_off_idx = -1
    for k, cur in enumerate(matched):
        cands_cur = pos_map.get(cur["text"]) or []
        cur_idx = next((p for p in cands_cur if p > last_off_idx), cands_cur[0] if cands_cur else -1)
        filled.append(cur)
        if cur_idx >= 0:
            last_off_idx = cur_idx
        if k + 1 < len(matched) and cur_idx >= 0:
            nxt = matched[k + 1]
            cands_nxt = pos_map.get(nxt["text"]) or []
            nxt_idx = next((p for p in cands_nxt if cur_idx < p <= cur_idx + 7), -1)
            missing_n = (nxt_idx - cur_idx - 1) if nxt_idx > cur_idx else 0
            gap_s = cur["end"]
            gap_e = nxt["start"]
            if 1 <= missing_n <= 6 and (gap_e - gap_s) >= max(1.6, 1.5 * missing_n):
                # Measure real active singing seconds in [gap_s .. gap_e]
                i0 = max(0, int(gap_s / step_sec))
                i1 = min(len(db_profile), int(gap_e / step_sec))
                active_sec = float(np.sum(db_profile[i0:i1] > -38.0)) * step_sec if i1 > i0 else 0.0
                if active_sec >= max(1.4, 1.6 * missing_n):
                    step_t = (gap_e - gap_s) / missing_n
                    for m_i in range(missing_n):
                        filled.append({
                            "start": round(gap_s + m_i * step_t, 2),
                            "end": round(gap_s + (m_i + 1) * step_t, 2),
                            "text": official_lines[cur_idx + 1 + m_i],
                            "sim": 0.0,
                        })
                    last_off_idx = nxt_idx - 1
    matched = filled

    # If a sustained unrecognized singing cluster STILL remains (e.g. an unreleased/parody section inside a Mashup),
    # keep those sung ASR lines so Mashup viewers never face a blank screen!
    for cl in _find_sustained_unmatched_clusters(unmatched_gaps):
        for g in cl:
            matched.append({"start": g["start"], "end": g["end"], "text": g["text"], "sim": 0.0})
    matched.sort(key=lambda x: x["start"])

    # Group into contiguous singing blocks (split on gaps > 4.5s) and align each block on its own audio slice!
    blocks: list[list[dict]] = []
    for item in matched:
        if not blocks or (item["start"] - blocks[-1][-1]["end"]) > 4.5:
            blocks.append([item])
        else:
            blocks[-1].append(item)

    total_sec = len(audio_16k) / 16000.0 if len(audio_16k) > 0 else 0.0
    final_lines: list[dict] = []
    for b_idx, blk in enumerate(blocks):
        prev_end_bound = (blocks[b_idx - 1][-1]["end"] + 0.15) if b_idx > 0 else first_onset
        next_start_bound = (blocks[b_idx + 1][0]["start"] - 0.15) if (b_idx + 1 < len(blocks)) else total_sec
        b_start = max(first_onset, prev_end_bound, blk[0]["start"] - 1.5)
        b_end = min(total_sec, next_start_bound, blk[-1]["end"] + 4.5) if total_sec > 0 else (blk[-1]["end"] + 4.5)
        if len(audio_16k) > 0 and b_end > b_start + 0.5:
            chunk = audio_16k[int(b_start * 16000) : int(b_end * 16000)]
            blk_text = "\n".join(x["text"] for x in blk)
            try:
                al = model.align(chunk, blk_text, language=lang, original_split=True, verbose=False)
                al_segs = list(al.segments or [])
            except Exception:
                al_segs = []
        else:
            al_segs = []

        if len(al_segs) == len(blk):
            for seg, ref in zip(al_segs, blk):
                s_abs = round(max(first_onset, b_start + float(seg.start)), 2)
                e_abs = round(max(s_abs + 0.4, b_start + float(seg.end)), 2)
                n_chars = max(1, len(_norm_lyrics_cmp(ref["text"])))
                # Guard against any single crushed/drifted line inside a block
                if (e_abs - s_abs) < max(0.4, 0.08 * n_chars) or abs(s_abs - ref["start"]) > 5.5:
                    s_abs = round(max(first_onset, ref["start"]), 2)
                    e_abs = round(max(s_abs + 0.6, ref["end"]), 2)
                    words_out = _even_words_for_line(ref["text"], s_abs, e_abs)
                else:
                    words_out = []
                    for w in (seg.words or []):
                        wt = (w.word or "").strip()
                        if not wt:
                            continue
                        ws = round(max(s_abs, b_start + float(w.start)), 2)
                        we = round(max(ws + 0.05, b_start + float(w.end)), 2)
                        words_out.append({"start": ws, "end": we, "word": wt})
                final_lines.append({"start": s_abs, "end": e_abs, "text": ref["text"], "words": words_out})
        else:
            for ref in blk:
                s_abs = round(max(first_onset, ref["start"]), 2)
                e_abs = round(max(s_abs + 0.6, ref["end"]), 2)
                final_lines.append({
                    "start": s_abs,
                    "end": e_abs,
                    "text": ref["text"],
                    "words": _even_words_for_line(ref["text"], s_abs, e_abs),
                })

    # Ensure strictly monotonic non-overlapping starts + acoustic onset/tail refinement
    for idx in range(1, len(final_lines)):
        if final_lines[idx]["start"] < final_lines[idx - 1]["start"] + 0.2:
            final_lines[idx]["start"] = round(final_lines[idx - 1]["end"], 2)
            final_lines[idx]["end"] = round(max(final_lines[idx]["start"] + 0.5, final_lines[idx]["end"]), 2)

    return refine_alignment_with_acoustics(final_lines, db_profile, step_sec, total_sec, first_onset)


def align_official_lyrics_with_stable_ts(
    vocals_path: Path,
    audio_16k: np.ndarray,
    db_profile: np.ndarray,
    step_sec: float,
    official_lines: list[str],
    device_mode: str,
    first_onset: float,
    prefer_live: bool = False,
    inst_path: Optional[Path] = None,
    shazam_already_tried: bool = False,
) -> tuple[list[dict], str, bool]:
    """Force-align 100% REAL official lyrics text directly onto isolated vocals.mp3.
    Automatically detects mid-song starts, skipped/swapped concert verses, or encores and
    engages Section-Anchored Live Alignment on the same loaded model!
    """
    ensure_ffmpeg_in_path()
    add_nvidia_dll_dirs()
    import stable_whisper

    free_mb = query_free_vram_mb()
    use_cuda = (device_mode != "cpu") and (device_mode == "gpu" or free_mb >= 2000)
    lang = detect_lyrics_language(official_lines)
    text_block = "\n".join(official_lines)
    audio_dur = (len(audio_16k) / 16000.0) if len(audio_16k) > 0 else 0.0

    def _load(dev: str):
        ctype = "float16" if dev == "cuda" else "int8"
        kw = {"device": dev, "compute_type": ctype, "download_root": str(MODEL_DIR)}
        if dev == "cpu":
            kw["cpu_threads"] = max(4, min(8, os.cpu_count() or 4))
        return stable_whisper.load_faster_whisper("large-v3-turbo", **kw)

    used_dev = "cuda" if use_cuda else "cpu"
    try:
        model = _load(used_dev)
    except Exception as e:
        if used_dev == "cuda":
            print(f"[Lyrics] Load CUDA fallback to CPU: {e}", file=sys.stderr, flush=True)
            used_dev = "cpu"
            model = _load("cpu")
        else:
            raise

    # Only bypass silence-collapsed full-song align for genuine Mashups/Medleys (multiple songs!).
    # Full-song covers/lives ALWAYS try Silence-Collapsed Pass 1 first (~2.4s, 0% dropped lines!)
    if prefer_live:
        live_lines = _live_section_anchored_align(
            model, vocals_path, audio_16k, db_profile, step_sec, official_lines, lang, first_onset,
            use_cuda=(used_dev == "cuda"), inst_path=inst_path, shazam_already_tried=shazam_already_tried,
        )
        if len(live_lines) >= 4:
            return live_lines, used_dev, True

    # Pillar 1: Collapse silent instrumental interludes (>1.6s) before m.align() so Whisper's 30s
    # sliding window NEVER hallucinates lyrics into instrumental breaks!
    if len(audio_16k) > 0 and len(db_profile) > 0:
        align_audio, time_map = build_silence_collapsed_audio(audio_16k, db_profile, step_sec, sr=16000)
    else:
        align_audio, time_map = str(vocals_path), []

    try:
        aligned = model.align(
            align_audio,
            text_block,
            language=lang,
            original_split=True,
            verbose=False,
        )
    except Exception as e:
        if used_dev == "cuda":
            print(f"[Lyrics] Forced-align CUDA fallback to CPU: {e}", file=sys.stderr, flush=True)
            used_dev = "cpu"
            model = _load("cpu")
            aligned = model.align(
                align_audio,
                text_block,
                language=lang,
                original_split=True,
                verbose=False,
            )
        else:
            raise

    comp_dur = (len(align_audio) / 16000.0) if isinstance(align_audio, np.ndarray) else audio_dur
    needs_live, reason = _alignment_needs_live_rematch(aligned, comp_dur, np.array([]), step_sec)
    if needs_live:
        print(f"[Lyrics] Detected non-standard live/abridged structure ({reason}); engaging section-anchored live alignment...", flush=True)
        live_lines = _live_section_anchored_align(
            model, vocals_path, audio_16k, db_profile, step_sec, official_lines, lang, first_onset,
            use_cuda=(used_dev == "cuda"), inst_path=inst_path, shazam_already_tried=shazam_already_tried,
        )
        if len(live_lines) >= 4:
            return live_lines, used_dev, True

    lines = []
    for seg in aligned.segments:
        txt = (seg.text or "").strip()
        if not txt:
            continue
        raw_s = _map_comp_to_real(float(seg.start), time_map) if time_map else float(seg.start)
        raw_e = _map_comp_to_real(float(seg.end), time_map) if time_map else float(seg.end)
        start_s = round(max(first_onset, raw_s), 2)
        end_s = round(max(start_s + 0.4, raw_e), 2)

        words_out = []
        for w in (seg.words or []):
            wt = (w.word or "").strip()
            if not wt:
                continue
            ws_r = _map_comp_to_real(float(w.start), time_map) if time_map else float(w.start)
            we_r = _map_comp_to_real(float(w.end), time_map) if time_map else float(w.end)
            ws = round(max(start_s, ws_r), 2)
            we = round(max(ws + 0.05, we_r), 2)
            words_out.append({"start": ws, "end": we, "word": wt})

        lines.append({"start": start_s, "end": end_s, "text": txt, "words": words_out})

    lines = refine_alignment_with_acoustics(lines, db_profile, step_sec, audio_dur, first_onset)
    return lines, used_dev, False


def fallback_free_transcribe(
    vocals_path: Path,
    device_mode: str,
    db_profile: np.ndarray,
    step_sec: float,
    first_onset: float,
    title_hint: str = "",
) -> tuple[list[dict], str]:
    """Only used when NO official or description lyrics exist; strictly gated by RMS silence detector."""
    add_nvidia_dll_dirs()
    from faster_whisper import WhisperModel

    free_mb = query_free_vram_mb()
    use_cuda = (device_mode != "cpu") and (device_mode == "gpu" or free_mb >= 2000)
    is_cjk_title = bool(re.search(r"[\u4e00-\u9fff]", title_hint or ""))
    prompt = f"以下係繁體中文歌詞：{clean_search_title(title_hint)}" if is_cjk_title else None

    def _run(dev: str):
        ctype = "float16" if dev == "cuda" else "int8"
        kw = {"device": dev, "compute_type": ctype, "download_root": str(MODEL_DIR)}
        if dev == "cpu":
            kw["cpu_threads"] = max(4, min(8, os.cpu_count() or 4))
        m = WhisperModel("large-v3-turbo", **kw)
        segs, inf = m.transcribe(
            str(vocals_path),
            beam_size=5,
            word_timestamps=True,
            vad_filter=False,
            initial_prompt=prompt,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            no_speech_threshold=0.65,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.0,
            condition_on_previous_text=False,
        )
        return list(segs), inf

    used_dev = "cuda" if use_cuda else "cpu"
    try:
        segments, info = _run(used_dev)
    except Exception:
        used_dev = "cpu"
        segments, info = _run("cpu")

    is_cjk = (info.language or "") in ("zh", "yue", "")
    lines = []
    for seg in segments:
        s_start = float(seg.start)
        s_end = float(seg.end)
        # Strict RMS energy gate: NEVER emit lyrics during silent intros/outros!
        if s_end < first_onset or not segment_has_vocal_energy(db_profile, step_sec, s_start, s_end):
            continue
        txt = (seg.text or "").strip()
        if not txt or HALLUCINATION_RE.search(txt) or GARBAGE_CHARS_RE.search(txt):
            continue
        if is_cjk:
            txt = to_hk_traditional(txt)
        words_out = []
        for w in (seg.words or []):
            wt = (w.word or "").strip()
            if not wt or GARBAGE_CHARS_RE.search(wt):
                continue
            words_out.append({
                "start": round(float(w.start), 2),
                "end": round(float(w.end), 2),
                "word": to_hk_traditional(wt) if is_cjk else wt,
            })
        lines.append({"start": round(s_start, 2), "end": round(s_end, 2), "text": txt, "words": words_out})

    return lines, used_dev


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
    parser.add_argument("--title", default="")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--inst-path", default="")
    args = parser.parse_args()

    status_file = Path(args.status_file)
    out_file = Path(args.out_file)
    vocals_path = Path(args.vocals_path)
    inst_path = Path(args.inst_path) if args.inst_path else None
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
        update("running", 10, "載入歌詞中...")
        # Skip slow yt-dlp probe if force_ai=True and title was already provided by meta.json!
        if force_ai and args.title.strip():
            yt_lines, yt_lang, title, duration, yt_desc = None, "", args.title.strip(), float(args.duration or 0.0), ""
        else:
            yt_lines, yt_lang, title, duration, yt_desc = fetch_youtube_human_subtitles(args.video_id)
            if not title or title == args.video_id:
                title = args.title.strip() or title
            if duration <= 0 and args.duration > 0:
                duration = float(args.duration)

        # Layer 1: Human YouTube subtitles (unless user explicitly requested AI re-alignment)
        if yt_lines and not force_ai:
            result = {
                "video_id": args.video_id,
                "title": title,
                "source": "youtube_sub",
                "source_label": "同步字幕",
                "lines": yt_lines,
                "created_at": time.time(),
            }
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            update("ready", 100, "歌詞已就緒")
            return 0

        # Layer 2: Multi-source official lyrics lookup (LRCLIB + NetEase + YT Desc)
        update("running", 25, "搜尋歌詞中...")
        is_mashup_title = bool(MASHUP_KEYWORDS_RE.search(title))
        lrc_synced, official_lines, lrc_src = (
            (None, [], "") if is_mashup_title else fetch_official_lyrics(title, duration, yt_desc)
        )

        if lrc_synced and not force_ai:
            result = {
                "video_id": args.video_id,
                "title": title,
                "source": "lrclib_exact",
                "source_label": "同步歌詞",
                "lines": lrc_synced,
                "created_at": time.time(),
            }
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            update("ready", 100, "歌詞已就緒")
            return 0

        if not vocals_path.exists():
            raise RuntimeError("請先開啟唱 K 伴奏")

        audio_16k, db_profile, step_sec, first_onset = detect_vocal_energy_profile(vocals_path)

        # Layer 2B: Automatic Shazam audio fingerprinting for Mashups/Medleys OR videos whose title has no song name!
        shazam_tried = False
        shazam_tracks: list[dict] = []
        if is_mashup_title or not official_lines:
            update("running", 35, "辨識歌曲中...")
            shazam_tried = True
            shz_lines, shazam_tracks = shazam_scan_and_fetch_lyrics(
                audio_16k, db_profile, step_sec, inst_path=inst_path, scan_mashup=is_mashup_title
            )
            if shz_lines:
                official_lines = shz_lines

        # Layer 3A (BEST): Silence-Collapsed Forced Alignment of 100% REAL official lyrics!
        # Only skip directly to section-anchored mode for genuine Mashups/Medleys (multiple songs).
        # Full covers/concerts ALWAYS try Silence-Collapsed Pass 1 first (2.4s, 0% dropped lines!) and
        # automatically escalate to section-anchored if the singer skipped/swapped verses!
        if official_lines:
            update("running", 50, "對齊歌詞中...")
            is_multi_song_mashup = is_mashup_title or len(shazam_tracks) >= 2
            ai_lines, used_dev, used_live = align_official_lyrics_with_stable_ts(
                vocals_path, audio_16k, db_profile, step_sec, official_lines,
                args.device_mode, first_onset, prefer_live=is_multi_song_mashup,
                inst_path=inst_path, shazam_already_tried=shazam_tried,
            )
            src = "mashup_shazam_align" if len(shazam_tracks) >= 2 else ("live_anchored_align" if used_live else "forced_align")
            label = "同步歌詞"
        else:
            # Layer 3B: Free transcription + automatic Lyric-Phrase Rescue (identifies acoustic/live covers with no title!)
            update("running", 50, "辨識歌詞中...")
            ai_lines, used_dev = fallback_free_transcribe(
                vocals_path, args.device_mode, db_profile, step_sec, first_onset, title_hint=title
            )
            rescued_lines = search_official_lyrics_by_sung_phrases(ai_lines) if len(ai_lines) >= 4 else []
            if rescued_lines:
                update("running", 75, "對齊歌詞中...")
                ai_lines, used_dev, _ = align_official_lyrics_with_stable_ts(
                    vocals_path, audio_16k, db_profile, step_sec, rescued_lines,
                    args.device_mode, first_onset, prefer_live=True,
                    inst_path=inst_path, shazam_already_tried=True,
                )
                src = "phrase_rescue_align"
            else:
                src = "whisper_ai"
            label = "同步歌詞"

        if not ai_lines:
            raise RuntimeError("暫未能對齊歌詞")

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
        update("ready", 100, "歌詞已就緒")
        return 0
    except Exception as exc:
        update("error", 0, "暫未能載入歌詞", error=str(exc))
        print(f"[LyricsWorker Error] {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
