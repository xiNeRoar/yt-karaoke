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
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import gc
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
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
    r"(\blive\b|\bconcert\b|演唱會|演唱会|音樂會|音乐会|現場|现场|\bcover\b|翻唱|\bacoustic\b|\bunplugged\b|\bremix\b|plug\s*dj|\bmedley\b|串燒|\bktv\b|試聽|试听|短版|\bcut\b|\bclip\b|片段|\bmashup\b|組曲|组曲|合集|精選|精选|聯唱|联唱)",
    re.IGNORECASE,
)
MASHUP_KEYWORDS_RE = re.compile(
    r"(\bmashup\b|\bmedley\b|串燒|串烧|組曲|组曲|合集|合輯|精選|精选|聯唱|联唱|\btop\s*\d+\b|歌單|歌单|連播|连播|特輯|特辑|勁歌金曲|劲歌金曲)",
    re.IGNORECASE,
)
HALLUCINATION_RE = re.compile(
    r"(訂閱|订阅|按讚|点赞|感謝觀看|感谢观看|字幕組|字幕组|字幕視聽|字幕提供|Amara\.org|謝謝收看|谢谢收看|Subscribe|Thanks for watching|獨播劇場|YoYoTelevision)",
    re.IGNORECASE,
)
GARBAGE_CHARS_RE = re.compile(r"[\ufffd\u0400-\u04FF]")  # Replacement char or Cyrillic hallucinations
CREDIT_LINE_RE = re.compile(
    r"^\s*(?:作[詞词](?:\s*/\s*作曲)?|作曲|[編编]曲|[監监][製制]|[製制]作人|混音|[縮缩]混|[錄录]音|和[音聲声]|母[帶带]|"
    r"出品|[發发]行|原唱|翻唱|演唱|主唱|[詞词]曲|吉他|[貝贝]斯|鼓|[鍵键][盤盘]|弦[樂乐]|企[劃划]|策[劃划]|[統统][籌筹]|[後后]期|"
    r"[詞词]|曲|[編编]|[監监]|OP|SP|"
    r"Lyricist|Lyrics\s*by|Composer|Music\s*(?:&|and)\s*Lyrics|Music\s*by|Written\s*by|"
    r"Arranger|Arranged\s*by|Producer|Produced\s*by|Executive\s*Producer|"
    r"Mixed\s*by|Mixing|Mastered\s*by|Mastering|Vocalist|Backing\s*Vocals?|Vocals?|"
    r"Guitar|Bass|Drums|Piano|Keyboard|Strings|Chorus|Recording|Studio)\s*[:：\-]",
    re.IGNORECASE,
)
SECTION_HEADER_RE = re.compile(
    r"^\s*[\[\(（【]?\s*(?:intro|outro|verse|pre-?chorus|chorus|bridge|hook|interlude|instrumental|solo|rap|break|refrain|coda|end)\s*(?:\d+[a-z]?)?\s*[\]\)）】]?\s*$",
    re.IGNORECASE,
)
VERSION_TAG_RE = re.compile(
    r"(\blive\b|\bremix\b|\bdj\b|\bcover\b|\binstrumental\b|\bkaraoke\b|\boff\s*vocal\b|\bacoustic\b|\bunplugged\b|伴奏|翻唱|[現现][場场]|演[唱会會])",
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


def _whisper_align_lang(lang: str) -> str:
    """Normalize Cantonese 'yue' -> 'zh' for Whisper/stable-ts so split_words_by_space stays False."""
    return "zh" if lang == "yue" else lang


def _release_torch_vram() -> None:
    """Reclaim Python + PyTorch CUDA VRAM cleanly between Stage 1 (Whisper) and Stage 2 (Qwen3)."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _whisper_load_kwargs(dev: str, local_files_only: bool = False) -> dict:
    """Shared faster-whisper/stable-ts model kwargs with multi-core CPU thread scaling."""
    ctype = "float16" if dev == "cuda" else "int8"
    kw: dict = {
        "device": dev,
        "compute_type": ctype,
        "download_root": str(MODEL_DIR),
        "local_files_only": local_files_only,
    }
    if dev == "cpu":
        cores = os.cpu_count() or 4
        kw["cpu_threads"] = max(4, min(16, (cores // 2) if cores >= 12 else cores))
    return kw


def _load_faster_whisper_model(dev: str):
    """Try instant offline cache load first (0 HuggingFace HTTP round-trips), falling back to download."""
    from faster_whisper import WhisperModel
    try:
        return WhisperModel("large-v3-turbo", **_whisper_load_kwargs(dev, local_files_only=True))
    except Exception:
        return WhisperModel("large-v3-turbo", **_whisper_load_kwargs(dev, local_files_only=False))


def _load_stable_whisper_model(dev: str):
    """Try instant offline cache load first for stable-ts faster-whisper, falling back to download."""
    import stable_whisper
    try:
        return stable_whisper.load_faster_whisper("large-v3-turbo", **_whisper_load_kwargs(dev, local_files_only=True))
    except Exception:
        return stable_whisper.load_faster_whisper("large-v3-turbo", **_whisper_load_kwargs(dev, local_files_only=False))



def _is_non_chinese_cjk(text: str) -> bool:
    """True when text contains Japanese Kana or Korean Hangul (must NEVER be passed through Chinese OpenCC!)."""
    return bool(re.search(r"[\u3040-\u30ff\uac00-\ud7af]", text or ""))


def to_hk_traditional(text: str) -> str:
    if not text or _is_non_chinese_cjk(text):
        return text
    if not re.search(r"[\u4e00-\u9fff]", text):
        return text
    try:
        from opencc import OpenCC
        if not hasattr(to_hk_traditional, "_cc"):
            to_hk_traditional._cc = OpenCC("s2hk")
        return to_hk_traditional._cc.convert(text)
    except Exception:
        return text


def estimate_syllable_units(text: str) -> float:
    """Language-agnostic sung syllable estimator (1.0/CJK or Hangul char, 0.7/Kana, vowel-cluster syllables for Latin).
    Prevents raw character counts from breaking English/European/Japanese alignment & Qwen3 padding thresholds!
    """
    if not text:
        return 1.0
    cjk_ko = len(re.findall(r"[\u4e00-\u9fff\uac00-\ud7af]", text))
    kana = len(re.findall(r"[\u3040-\u30ff]", text))
    latin_words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", text)
    latin_syl = 0.0
    for w in latin_words:
        wl = re.sub(r"[^a-zà-öø-ÿ]", "", w.lower())
        if not wl:
            continue
        groups = re.findall(r"[aeiouyà-öø-ÿ]+", wl)
        s_cnt = max(1, len(groups))
        if len(wl) > 3 and wl.endswith("e") and not wl.endswith("le") and s_cnt > 1:
            s_cnt -= 1
        latin_syl += float(s_cnt)
    total = float(cjk_ko) + 0.7 * float(kana) + latin_syl
    return max(1.0, total)


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
    """Extract pasted lyrics from YouTube video descriptions (crucial for 改詞版 / parodies / indie covers!).
    Preserves multi-stanza lyrics separated by blank lines or [Verse]/[Chorus] headers while cleanly isolating
    them from promo paragraphs, URLs, and credit blocks.
    """
    if not yt_desc or len(yt_desc) < 80:
        return []
    url_re = re.compile(r"(https?://|www\.|@|#|\bfb\.com\b|\big\b|instagram|facebook|spotify|apple\s*music|kkbox|joox|moov|訂閱|頻道|商業合作|email)", re.IGNORECASE)
    lyrics_hdr_re = re.compile(r"^\s*[\[\(（【]?\s*(?:歌[詞词]|lyrics?|가사)\s*[\]\)）】]?\s*[:：\-—]*\s*$", re.IGNORECASE)
    best_block: list[str] = []
    cur_block: list[str] = []
    blank_run = 0
    for raw in yt_desc.splitlines():
        s = raw.strip()
        if not s:
            blank_run += 1
            if blank_run >= 3:
                if len(cur_block) > len(best_block):
                    best_block = cur_block
                cur_block = []
            continue
        blank_run = 0
        if lyrics_hdr_re.match(s):
            # Explicit "歌詞:" / "Lyrics:" / "가사" header starts a fresh lyrics block cleanly
            if len(cur_block) > len(best_block):
                best_block = cur_block
            cur_block = []
            continue
        if SECTION_HEADER_RE.match(s):
            # In-song section markers like [Chorus] / [Verse 2] separate stanzas without ending the song
            continue
        max_line_len = 88 if not re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", s) else 46
        if url_re.search(s) or CREDIT_LINE_RE.match(s) or len(s) > max_line_len or len(s) < 2:
            if len(cur_block) > len(best_block):
                best_block = cur_block
            cur_block = []
            continue
        # Must contain real text across Hanzi, Japanese Kana, Korean Hangul, or Latin words
        if re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]{2,}|[A-Za-z]{3,}", s):
            cur_block.append(s)
        else:
            if len(cur_block) > len(best_block):
                best_block = cur_block
            cur_block = []
    if len(cur_block) > len(best_block):
        best_block = cur_block
    if len(best_block) < 8:
        return []
    # Whole-song non-Chinese check so Japanese Kanji-only lines in Japanese songs are NEVER converted by OpenCC!
    is_non_zh = _is_non_chinese_cjk("\n".join(best_block))
    return best_block if is_non_zh else [to_hk_traditional(s) for s in best_block]


def clean_search_title(title: str) -> str:
    """Normalize noise tokens while PRESERVING bracketed song titles (e.g. 【晴天】, [十年], 抽)
    and numeric/year song titles (e.g. 1874, 505, 2002, 1989)!
    """
    # Strip explicit MV/audio/quality tags inside brackets (including full-width Japanese ［...］),
    # while keeping actual song titles inside 【...】/[...]!
    t = re.sub(
        r"[\[\(【（［]\s*(?:official\s*(?:music\s*video|lyric\s*video|video|m/v|mv|pv|audio|visualizer|live\s*clip|stage\s*clip|performance(?:\s*ver\.?)?)|"
        r"music\s*video|lyric\s*video|lyrics?\s*(?:mv|video)|m/v|mv|pv|visualizer|hd|4k|hq|audio|live(?:\s*clip)?|"
        r"performance\s*ver\.?|special\s*clip|stage\s*clip|color\s*coded|han\s*/?\s*rom\s*/?\s*eng|full\s*ver\.?)\s*[\]\)】）］]",
        " ",
        title,
        flags=re.IGNORECASE,
    )
    # Strip standalone K-Pop / J-Pop video tags BEFORE replacing '/' so "M/V" never turns into "M V"!
    t = re.sub(
        r"(?:(?<=\s)|^)(?:official\s*(?:music\s*video|lyric\s*video|m/v|mv|pv|video|audio|visualizer)|"
        r"music\s*video|lyric\s*video|m/v|pv|performance\s*ver\.?|special\s*clip|live\s*clip|stage\s*clip|"
        r"color\s*coded|han\s*/?\s*rom\s*/?\s*eng|visualizer)(?:(?=\s)|$)",
        " ",
        t,
        flags=re.IGNORECASE,
    )
    # Strip broadcast dates/concert years ONLY when in explicit date/bracket/year-suffix context — NEVER strip song titles like 2002 / 1989 / 1979!
    t = re.sub(r"\b(?:19[5-9]\d|20[0-3]\d)[./-]\d{1,2}(?:[./-]\d{1,2})?\b", " ", t)
    t = re.sub(r"[\[\(【（［]\s*(?:19[5-9]\d|20[0-3]\d)\s*[\]\)】）］]", " ", t)
    t = re.sub(r"(?<!\d)(?:19[5-9]\d|20[0-3]\d)\s*年", " ", t)
    t = re.sub(r"[\[\(【（［\]\)】）］_|\-/]+", " ", t)
    t = re.sub(r"\b(official|mv|music video|lyrics|lyric video|hd|4k|hq|audio)\b", " ", t, flags=re.IGNORECASE)
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


def _is_instrumental_placeholder(s: str) -> bool:
    low = re.sub(r"[\s\[\]\(\)（）【】~～♪♫,，。.!！]+", "", (s or "")).lower()
    return low in ("", "純音樂請欣賞", "纯音乐请欣赏", "puremusic", "instrumental", "music", "音樂", "音乐")


def clean_lyric_lines(raw_lines: list[str]) -> list[str]:
    # Whole-song non-Chinese check so Japanese Kanji-only lines in Japanese songs are NEVER corrupted by OpenCC!
    is_non_zh = _is_non_chinese_cjk("\n".join(raw_lines or []))
    out = []
    for raw in raw_lines:
        s = re.sub(r"\[[^\]]*\]", "", raw).strip()
        if not s:
            continue
        if CREDIT_LINE_RE.match(s) or SECTION_HEADER_RE.match(s) or _is_instrumental_placeholder(s):
            continue
        hk = s if is_non_zh else to_hk_traditional(s)
        out.extend(_split_long_official_line(hk))
    return out


def _line_script_kind(text: str) -> str:
    if re.search(r"[\u3040-\u30ff]", text or ""):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", text or ""):
        return "ko"
    if re.search(r"[\u4e00-\u9fff]", text or ""):
        return "zh"
    if re.search(r"[A-Za-z]", text or ""):
        return "en"
    return "other"


def parse_lrc(lrc_text: str) -> list[dict]:
    raw_entries: list[dict] = []
    ts_re = re.compile(r"\[(\d+):(\d+(?:[.:]\d+)?)\]")
    inline_ts_re = re.compile(r"<\d+:\d+(?:[.:]\d+)?>")
    bracket_tag_re = re.compile(r"^\s*\[[^\]]{1,24}\]\s*")
    raw_all = ts_re.sub("", lrc_text or "")
    is_non_zh = _is_non_chinese_cjk(raw_all)
    for raw in (lrc_text or "").splitlines():
        s_raw = raw.strip()
        timestamps = ts_re.findall(s_raw)
        if not timestamps:
            continue
        # Strip ALL [mm:ss.xx] tags, enhanced-LRC <mm:ss.xx> tags, and leading [Chorus]/[Verse] tags
        text = inline_ts_re.sub("", ts_re.sub("", s_raw)).strip()
        text = bracket_tag_re.sub("", text).strip()
        if not text or CREDIT_LINE_RE.match(text) or SECTION_HEADER_RE.match(text) or _is_instrumental_placeholder(text):
            continue
        for mins_s, secs_s in timestamps:
            t_sec = round(int(mins_s) * 60 + float(secs_s.replace(":", ".")), 2)
            raw_entries.append({"start": t_sec, "end": 0.0, "text": text, "words": []})

    raw_entries.sort(key=lambda x: x["start"])
    # Drop opening [00:00.00] "Title - Artist" header line common in NetEase LRC files
    if raw_entries and raw_entries[0]["start"] <= 0.5 and re.search(r"\s[-—–]\s", raw_entries[0]["text"]):
        if len(raw_entries) >= 2 and (raw_entries[1]["start"] - raw_entries[0]["start"]) >= 2.0:
            raw_entries.pop(0)

    # Deduplicate same-timestamp bilingual translation lines (prevents end == start 0.00s lines and EN/JA/KO + ZH pollution!)
    # Determine primary non-ZH script if bilingual pairs exist
    pair_scripts: dict[str, int] = {}
    i_p = 0
    while i_p < len(raw_entries):
        j_p = i_p + 1
        while j_p < len(raw_entries) and abs(raw_entries[j_p]["start"] - raw_entries[i_p]["start"]) < 0.05:
            j_p += 1
        if j_p - i_p >= 2:
            # First line in bilingual LRC is conventionally the original sung lyric
            k0 = _line_script_kind(raw_entries[i_p]["text"])
            pair_scripts[k0] = pair_scripts.get(k0, 0) + 1
        i_p = j_p
    primary_bilingual = max(pair_scripts, key=pair_scripts.get) if pair_scripts else ""

    lines: list[dict] = []
    i_d = 0
    while i_d < len(raw_entries):
        j_d = i_d + 1
        while j_d < len(raw_entries) and abs(raw_entries[j_d]["start"] - raw_entries[i_d]["start"]) < 0.05:
            j_d += 1
        group = raw_entries[i_d:j_d]
        if len(group) == 1:
            chosen = group[0]
        else:
            # Prefer line matching the primary original script (or first non-ZH when bilingual song is EN/JA/KO)
            chosen = group[0]
            if primary_bilingual:
                for g in group:
                    if _line_script_kind(g["text"]) == primary_bilingual:
                        chosen = g
                        break
            elif is_non_zh:
                for g in group:
                    if _line_script_kind(g["text"]) in ("ja", "ko"):
                        chosen = g
                        break
        clean_text = chosen["text"] if is_non_zh else to_hk_traditional(chosen["text"])
        lines.append({"start": chosen["start"], "end": 0.0, "text": clean_text, "words": []})
        i_d = j_d

    for i, item in enumerate(lines):
        if i + 1 < len(lines):
            item["end"] = round(min(item["start"] + 8.0, max(item["start"] + 0.35, lines[i + 1]["start"])), 2)
        else:
            item["end"] = round(item["start"] + 6.0, 2)
    return lines


def _adaptive_vocal_threshold_db(db_profile: np.ndarray, margin_db: float = 0.0) -> float:
    """Compute a mathematically bounded, scale-invariant vocal activity threshold strictly inside (p15, p90).
    Floors digital silence (-120 dB) to -75 dB so studio digital-zero intros never drag threshold down to -60 dB,
    and applies a -52 dB absolute floor so pure-instrumental hiss is never mistaken for human singing!
    """
    if len(db_profile) < 10:
        return -40.0 + margin_db
    clipped = np.maximum(db_profile, -75.0)
    p15 = float(np.percentile(clipped, 15))
    p90 = float(np.percentile(clipped, 90))
    dr = max(0.0, p90 - p15)
    base = max(p15 + 0.30 * dr, p90 - 30.0)
    base = min(base, p15 + 0.55 * dr)
    base = max(base, -52.0)
    return base + margin_db


def _load_audio_16k_mono(audio_path: Path) -> np.ndarray:
    """Ultra-fast 16kHz mono float32 decoder via ffmpeg pipe (~0.15s for 4-min MP3 vs ~2s librosa),
    with automatic librosa fallback if ffmpeg is unavailable.
    """
    ensure_ffmpeg_in_path()
    try:
        res = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", str(audio_path),
                "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", "16000", "pipe:1",
            ],
            capture_output=True,
            timeout=30,
            creationflags=CREATE_NO_WINDOW,
        )
        if res.returncode == 0 and len(res.stdout) >= 4:
            n_bytes = (len(res.stdout) // 4) * 4
            return np.frombuffer(res.stdout[:n_bytes], dtype=np.float32).copy()
    except Exception:
        pass
    import librosa
    y, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    return y.astype(np.float32)


def detect_vocal_energy_profile(vocals_path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Load 16kHz mono vocals + vectorized per-100ms RMS dBFS profile so we know exact silent vs singing regions."""
    try:
        y = _load_audio_16k_mono(vocals_path)
        if len(y) == 0:
            return np.array([], dtype=np.float32), np.array([]), 0.1, 0.0
        hop = 1600  # 0.1s per frame
        n_frames = max(1, len(y) // hop)
        if len(y) >= hop:
            frames = y[: n_frames * hop].reshape(n_frames, hop)
            rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
        else:
            rms = np.array([float(np.sqrt(np.mean(y * y) + 1e-12))], dtype=np.float32)
        db = 20.0 * np.log10(rms + 1e-12)
        thr = _adaptive_vocal_threshold_db(db)
        # Find first sustained vocal onset (>= 0.4s active within a 0.6s window so 0.2s cough/click blips never trigger)
        first_onset = 0.0
        for i in range(max(0, len(db) - 6)):
            if db[i] > thr and int(np.sum(db[i : i + 6] > thr)) >= 4:
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
    # 85th-percentile energy in segment must exceed song-adaptive vocal threshold
    return float(np.percentile(seg_db, 85)) > _adaptive_vocal_threshold_db(db_profile)


def _split_artist_tokens(artist_name: str) -> list[str]:
    """Split multi-artist & bilingual artist credits (e.g. 'MC 张天赋', 'Eason Chan / 陈奕迅', 'IU (아이유)')
    WITHOUT splitting same-script multi-word names like 'Jay Chou' or 'Taylor Swift'!
    """
    if not artist_name:
        return []
    raw = to_simplified(artist_name)
    # 1) Split on explicit multi-artist delimiters & parenthetical aliases
    parts = re.split(r"[/&,、;·×]|(?:\s+(?:feat\.?|ft\.?|and|with|x)\s+)|[\(\)（）\[\]【】]", raw, flags=re.IGNORECASE)
    tokens: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 2) Split ONLY across script boundaries (Latin/digits vs CJK/Kana vs Hangul)
        sub_scripts = re.findall(r"[A-Za-z0-9][A-Za-z0-9\s.'-]*[A-Za-z0-9]|[A-Za-z0-9]|[\u4e00-\u9fff\u3040-\u30ff]+|[\uac00-\ud7af]+", p)
        if not sub_scripts:
            sub_scripts = [p]
        for sub in sub_scripts:
            norm = re.sub(r"[\s\W_]+", "", sub).lower()
            if len(norm) >= 2 and norm not in tokens:
                tokens.append(norm)
    return tokens


def _artist_match_level(artist_name: str, yt_title_clean: str) -> float:
    """Return 1.0 for full artist match, 0.55 for partial bilingual/duet token match, 0.0 for no match.
    Pure-Latin tokens <= 4 chars require word boundaries so 'iu' never matches 'stadium' and 'ai' never matches 'official'!
    """
    toks = _split_artist_tokens(artist_name)
    if not toks:
        return 0.0
    y_simp = to_simplified(yt_title_clean or "")
    y_norm = re.sub(r"[\s\W_]+", "", y_simp).lower()
    y_words = re.sub(r"[^\w\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+", " ", y_simp).lower()
    if not y_norm:
        return 0.0

    # Check whole artist string first (e.g. "jaychou" in "jaychou..."), but never bypass word-boundary guard for <=4-char Latin names
    a_full = re.sub(r"[\s\W_]+", "", to_simplified(artist_name)).lower()
    is_short_latin_full = bool(re.fullmatch(r"[a-z0-9]+", a_full)) and len(a_full) <= 4
    if len(a_full) >= 4 and not is_short_latin_full and a_full in y_norm:
        return 1.0

    hits = 0
    for tk in toks:
        is_latin_short = bool(re.fullmatch(r"[a-z0-9]+", tk)) and len(tk) <= 4
        if is_latin_short:
            if re.search(rf"(?<![a-z0-9]){re.escape(tk)}(?![a-z0-9])", y_words):
                hits += 1
        elif tk in y_norm:
            hits += 1
    if hits == 0:
        return 0.0
    if hits >= len(toks):
        return 1.0
    return 0.55


def _track_matches_title(
    track_name: str,
    artist_name: str,
    yt_title_clean: str,
    cand_dur: float = 0.0,
    yt_dur: float = 0.0,
) -> float:
    """Score how well a lyrics DB candidate (track_name, artist_name) matches the YouTube video title.
    Preserves 1-char CJK titles (e.g. MC 張天賦 - 抽, 張國榮 - 紅), numeric/year titles (1874/505/2002),
    word-boundary Latin titles (prevents 'el' matching 'adelehello'), bilingual parenthetical aliases,
    version-mismatch penalties, and optional duration proximity!
    """
    raw_track = track_name or ""
    t_base = re.sub(r"[\(\[（【［].*?[\)\]）】］]", "", raw_track).strip() or raw_track
    # Also extract non-version parenthetical alias (e.g. "抽 (Inhale)" -> "Inhale", "밤편지 (Through the Night)")
    paren_aliases: list[str] = []
    for pm in re.findall(r"[\(\[（【［]([^()\[\]（）【】［］]{1,48})[\)\]）】］]", raw_track):
        ps = pm.strip()
        if ps and not VERSION_TAG_RE.search(ps) and not re.search(r"\b(?:feat\.?|ft\.?|ver\.?|version|edit|ost|mv|pv)\b", ps, re.IGNORECASE):
            paren_aliases.append(ps)

    y_simp = to_simplified(yt_title_clean or "")
    y_norm = re.sub(r"[\s\W_]+", "", y_simp).lower()
    if not y_norm:
        return 0.0

    artist_lvl = _artist_match_level(artist_name, yt_title_clean)
    artist_matched = artist_lvl > 0.0
    y_words_lower = re.sub(r"[^\w\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+", " ", y_simp).lower()

    def _score_one_variant(variant: str) -> float:
        v_norm = re.sub(r"[\s\W_]+", "", to_simplified(variant)).lower()
        if not v_norm:
            return 0.0
        is_pure_latin_num = bool(re.fullmatch(r"[a-z0-9]+", v_norm))
        v_words = re.sub(r"[^\w\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+", " ", to_simplified(variant)).lower().strip()
        has_cjk_v = bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", v_norm))

        if v_norm == y_norm:
            return 2.0
        if len(v_norm) == 1:
            # Allow famous 1-char CJK song titles (e.g. MC 張天賦-抽, 張國榮-紅, 追, 默, 恋) ONLY when artist matches AND token is isolated!
            if artist_matched and has_cjk_v:
                if re.search(
                    rf"(?:^|[\s《》「」『』【】\[\]()（）［］\-—–/|｜'\"‘’]){re.escape(v_norm)}(?:$|[\s《》「」『』【】\[\]()（）［］\-—–/|｜'\"‘’])",
                    y_simp,
                ):
                    return 1.45
            return 0.0
        if is_pure_latin_num and len(v_norm) <= 6:
            # Short Latin/numeric titles (505, 1874, 2002, Hello, One) MUST match on word boundaries (not 'el' in 'adelehello'!)
            if v_words and re.search(rf"(?<![a-z0-9]){re.escape(v_words)}(?![a-z0-9])", y_words_lower):
                # Script-fair score: single short English word doesn't get runaway ASCII letter bonus over 1-char CJK!
                word_cnt = len(v_words.split())
                return 1.15 + min(0.35, (word_cnt - 1) * 0.15 + (0.12 if len(v_norm) >= 4 else 0.0))
            return 0.0
        if len(v_norm) >= 2 and v_norm in y_norm:
            if has_cjk_v:
                cjk_len = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", v_norm))
                return 1.20 + min(0.55, cjk_len * 0.11)
            word_cnt = max(1, len(v_words.split()))
            # Word-boundary guard for single English words embedded inside another word
            if word_cnt == 1 and v_words and not re.search(rf"(?<![a-z0-9]){re.escape(v_words)}(?![a-z0-9])", y_words_lower):
                return 0.0
            return 1.15 + min(0.55, word_cnt * 0.14)

        # Script-aware fuzzy overlap (Hanzi + Kana + Hangul) or per-segment Latin similarity
        cjk_t = set(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", v_norm))
        cjk_y = set(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", y_norm))
        if len(cjk_t) >= 2:
            overlap = len(cjk_t & cjk_y) / len(cjk_t)
            if overlap >= 0.75:
                return overlap
        elif len(v_norm) >= 3:
            segs = [
                re.sub(r"[\s\W_]+", "", s).lower()
                for s in re.split(r"[｜|—–/《》「」『』【】\[\]()（）［］]+|\s+-\s+", yt_title_clean or "")
                if s.strip()
            ]
            best_sim = max((SequenceMatcher(None, v_norm, sg).ratio() for sg in (segs + [y_norm]) if sg), default=0.0)
            if best_sim >= 0.72:
                return best_sim
        return 0.0

    score = _score_one_variant(t_base)
    for alias in paren_aliases:
        al_sc = _score_one_variant(alias)
        if al_sc > score:
            score = al_sc
    if score <= 0.0:
        return 0.0

    # Tiered artist match bonus (full artist match strictly outranks partial token match & unmatched artists!)
    if artist_lvl >= 0.99:
        score += 0.55
    elif artist_lvl > 0.0:
        score += 0.30

    # Penalize remix/live/cover/instrumental DB tracks UNLESS the YouTube title itself asks for that version
    cand_has_ver = bool(VERSION_TAG_RE.search(raw_track))
    yt_has_ver = bool(VERSION_TAG_RE.search(yt_title_clean or ""))
    if cand_has_ver and not yt_has_ver:
        score -= 0.35
    elif not cand_has_ver and re.sub(r"[\s\W_]+", "", to_simplified(raw_track)).lower() == re.sub(r"[\s\W_]+", "", to_simplified(t_base)).lower():
        score += 0.05

    # Duration proximity bonus/penalty when single-song YouTube duration is known
    if cand_dur > 0.0 and yt_dur > 0.0:
        diff = abs(cand_dur - yt_dur)
        if diff <= 5.0:
            score += 0.15
        elif diff <= 25.0:
            score += 0.08
        elif diff > 60.0:
            score -= 0.25

    return max(0.0, score)


def _split_bilingual_phrase(seg: str) -> list[str]:
    """Split bilingual title segments like '抽 Inhale', '記憶棉 Pillow Talk', '밤편지 Through the Night'
    into separate script-pure parts (preserving 1-char CJK song titles!).
    """
    if not seg:
        return []
    cjk_parts = re.findall(r"[\u4e00-\u9fff\u3040-\u30ff]+", seg)
    ko_parts = re.findall(r"[\uac00-\ud7af]+(?:\s+[\uac00-\ud7af]+)*", seg)
    lat_parts = re.findall(r"[A-Za-z0-9][A-Za-z0-9\s'&.-]*[A-Za-z0-9]|[A-Za-z0-9]{2,}", seg)
    out: list[str] = []
    # Only emit split parts if segment genuinely mixes >= 2 script groups
    present = int(bool(cjk_parts)) + int(bool(ko_parts)) + int(bool(lat_parts))
    if present < 2:
        return []
    if cjk_parts:
        out.append(" ".join(p.strip() for p in cjk_parts if p.strip()).strip())
    if ko_parts:
        out.append(" ".join(p.strip() for p in ko_parts if p.strip()).strip())
    if lat_parts:
        lat_j = " ".join(p.strip() for p in lat_parts if p.strip()).strip()
        if len(lat_j) >= 2:
            out.append(lat_j)
    return [x for x in out if x]


def _extract_query_candidates(
    title: str,
    track: str = "",
    artist: str = "",
    channel: str = "",
) -> list[str]:
    """Language-agnostic structural title decomposition (zero hardcoded word blacklists).

    1) Structured YouTube Content-ID metadata (`artist + track`, `track`) if present.
    2) Quoted title tokens inside universal quotation brackets (`《》`, `「」`, `『』`, `〈〉`, `［］`, `"..."`, `'...'`).
    3) Bilingual script-boundary splits paired with artist (e.g. 'MC 張天賦 - 抽 Inhale' -> 'MC 張天賦 抽' + 'MC 張天賦 Inhale').
    4) Atomic delimiter-split segments (`｜`, `|`, `—`, `–`, ` - `, `/`, `【】`) + cleaned full title.
    """
    out: list[str] = []

    def _add(s: str, allow_single_cjk: bool = False) -> None:
        val = re.sub(r"\s+", " ", (s or "")).strip()
        if not val:
            return
        if len(val) < 2:
            if not (allow_single_cjk and re.fullmatch(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", val)):
                return
        if val not in out:
            out.append(val)

    # 1) Structured metadata from YouTube (highest signal when populated!)
    if track:
        if artist:
            _add(f"{artist} {track}")
        _add(track, allow_single_cjk=True)

    # 2) Explicitly quoted song titles (including 1-char CJK like 《抽》 and K-Pop single quotes 'Hype Boy')
    quote_pat = (
        r'《([^》]{1,48})》|「([^」]{1,48})」|『([^』]{1,48})』|〈([^〉]{1,48})〉|'
        r'"([^"]{1,48})"|(?<![A-Za-z])\'([^\']{2,48})\'(?![A-Za-z])|‘([^’]{1,48})’'
    )
    quoted_items: list[str] = []
    for m in re.finditer(quote_pat, title or ""):
        inner = next((g for g in m.groups() if g), "")
        if inner:
            ci = clean_search_title(inner)
            if ci:
                quoted_items.append(ci)
                _add(ci, allow_single_cjk=bool(artist))

    # 3) Full cleaned title (keeps artist + song together for standard "Artist - Song" MVs)
    cleaned = clean_search_title(title)
    _add(cleaned)

    # 4) Structural delimiter split (works for ANY concert/TV/cover/multi-language title!)
    # Split on pipes, dashes, slashes, and bracket boundaries — detect artist prefix + bilingual song parts!
    raw_segs = re.split(r"[｜|—–/]+|\s+-\s+|[《》「」『』【】\[\]()（）［］]", title or "")
    seg_cands: list[str] = []
    for seg in raw_segs:
        cs = clean_search_title(seg)
        if not cs or re.fullmatch(r"[\d\s.,:-]+", cs):
            # Keep standalone 3-4 digit numeric song titles like 505 / 1874 / 2002!
            if re.fullmatch(r"\d{2,4}", cs or ""):
                seg_cands.append(cs)
            continue
        if len(cs) >= 2 or re.fullmatch(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", cs):
            seg_cands.append(cs)

    # Pair leading artist segment with bilingual-split parts of the song segment (e.g. "MC 張天賦" + "抽" from "抽 Inhale")
    artist_prefix = artist or (seg_cands[0] if len(seg_cands) >= 2 and len(seg_cands[0]) >= 2 else "")
    if artist_prefix:
        for qi in quoted_items:
            if qi != artist_prefix:
                _add(f"{artist_prefix} {qi}")
        for seg in seg_cands[1:] if not artist else seg_cands:
            if seg == artist_prefix:
                continue
            _add(f"{artist_prefix} {seg}")
            for bp in _split_bilingual_phrase(seg):
                _add(f"{artist_prefix} {bp}")
                if len(bp) >= 2:
                    _add(bp)
        # Also pair cross-script artist sub-tokens (e.g. "MC 張天賦" -> "張天賦 抽") for NetEase precision
        for bp_a in _split_bilingual_phrase(artist_prefix):
            if len(bp_a) >= 2:
                for seg in seg_cands[1:3]:
                    for bp_s in (_split_bilingual_phrase(seg) or [seg]):
                        _add(f"{bp_a} {bp_s}")

    for cs in reversed(seg_cands):
        _add(cs)
    for cs in seg_cands:
        _add(cs)

    # 5) Script spans fallback (Hanzi, Japanese Kana+Kanji, Korean Hangul)
    for p in sorted(
        set(re.findall(r"[\u3040-\u30ff\u4e00-\u9fff]{2,}|[\uac00-\ud7af]{2,}", cleaned)),
        key=len,
        reverse=True,
    ):
        _add(p)

    return out[:12]


def fetch_official_lyrics(
    title: str,
    duration: float,
    yt_desc: str = "",
    yt_track: str = "",
    yt_artist: str = "",
    yt_channel: str = "",
) -> tuple[Optional[list[dict]], list[str], str]:
    """Search LRCLIB + NetEase Cloud Music in PARALLEL across all structural title segments.

    Zero hardcoded noise blacklists and zero sequential timeout stalls: LRCLIB & NetEase queries run
    concurrently over a shared HTTP connection pool (~0.8-1.5s total) and strictly score candidates
    against title + track + artist metadata (excluding noisy uploader channel names).
    """
    cleaned = clean_search_title(title)
    # Include RAW title + structured track/artist (NEVER dump noisy yt_channel names into title matching!)
    match_context = " ".join(x for x in (title, cleaned, yt_track, yt_artist) if x).strip()
    queries = _extract_query_candidates(title, track=yt_track, artist=yt_artist, channel=yt_channel)

    is_live_or_alt = bool(LIVE_KEYWORDS_RE.search(title)) or bool(LIVE_KEYWORDS_RE.search(yt_desc[:200]))
    best_synced = None
    best_synced_score = 0.0
    best_lines: list[str] = []
    best_source = ""
    best_score = 0.0

    ne_headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://music.163.com/"}
    ne_queries: list[str] = []
    for q in queries[:8]:
        qs = to_simplified(q)
        if qs and qs not in ne_queries:
            ne_queries.append(qs)
        if q not in ne_queries:
            ne_queries.append(q)

    lrc_pool: list[dict] = []
    ne_raw_songs: list[dict] = []

    if queries:
        with httpx.Client(timeout=5.0, follow_redirects=True) as client:
            def _query_lrclib(q: str) -> list[dict]:
                try:
                    r = client.get("https://lrclib.net/api/search", params={"q": q})
                    if r.status_code == 200:
                        items = r.json()
                        return items[:10] if isinstance(items, list) else []
                except Exception:
                    pass
                return []

            def _query_netease(q: str) -> list[dict]:
                try:
                    r = client.get("https://music.163.com/api/search/get", params={"s": q, "type": 1, "limit": 6}, headers=ne_headers)
                    return ((r.json().get("result") or {}).get("songs") or [])[:6]
                except Exception:
                    return []

            # Run LRCLIB + NetEase searches concurrently in one shared pool!
            with ThreadPoolExecutor(max_workers=8) as ex:
                lrc_futs = [ex.submit(_query_lrclib, q) for q in queries[:8]]
                ne_futs = [ex.submit(_query_netease, q) for q in ne_queries[:8]]
                for f in as_completed(lrc_futs):
                    lrc_pool.extend(f.result())
                for f in as_completed(ne_futs):
                    ne_raw_songs.extend(f.result())

            # 1) Score LRCLIB candidates
            seen_lrc_ids: set = set()
            lrc_scored: list[tuple[float, float, dict]] = []
            for it in lrc_pool:
                lid = it.get("id")
                if lid is not None:
                    if lid in seen_lrc_ids:
                        continue
                    seen_lrc_ids.add(lid)
                tr_name = it.get("trackName") or ""
                ar_name = it.get("artistName") or ""
                it_dur = float(it.get("duration") or 0.0)
                match_sc = _track_matches_title(tr_name, ar_name, match_context, cand_dur=it_dur, yt_dur=duration)
                if match_sc > 0.0:
                    has_sync_bonus = 0.02 if (it.get("syncedLyrics") or "").strip() else 0.0
                    lrc_scored.append((match_sc + has_sync_bonus, match_sc, it))

            lrc_scored.sort(key=lambda x: x[0], reverse=True)
            for _, match_sc, it in lrc_scored:
                tr_name = it.get("trackName") or ""
                pl = (it.get("plainLyrics") or "").strip()
                sl = (it.get("syncedLyrics") or "").strip()
                cand_lines = clean_lyric_lines(pl.splitlines()) if pl else clean_lyric_lines([x["text"] for x in parse_lrc(sl)])
                if len(cand_lines) >= 6 and match_sc > best_score:
                    best_score = match_sc
                    best_lines = cand_lines
                    best_source = f"LRCLIB ({tr_name})"
                    if best_synced_score < best_score - 0.05:
                        best_synced = None
                        best_synced_score = 0.0

                if not is_live_or_alt and sl and duration > 0 and match_sc >= best_score - 0.05:
                    it_dur = float(it.get("duration") or 0.0)
                    # Allow up to +/-25s for Official MVs with intro/outro logos; main() enforces strict 3-layer acoustic span checks!
                    if it_dur > 0.0 and abs(it_dur - duration) <= 25.0:
                        parsed = parse_lrc(sl)
                        if len(parsed) >= 6 and (best_synced is None or match_sc > best_synced_score):
                            best_synced = parsed
                            best_synced_score = match_sc
                            if match_sc >= best_score:
                                best_source = "LRCLIB"
                                best_score = match_sc
                                if not best_lines:
                                    best_lines = clean_lyric_lines([x["text"] for x in parsed])

            # 2) Score NetEase candidates and fetch top lyrics (also allows upgrading synced LRC on equal score!)
            ne_candidates: list[tuple[float, dict]] = []
            seen_ne_ids: set = set()
            for s in ne_raw_songs:
                sid = s.get("id")
                if not sid or sid in seen_ne_ids:
                    continue
                seen_ne_ids.add(sid)
                tr_name = s.get("name") or ""
                ar_name = " / ".join((a.get("name") or "") for a in (s.get("artists") or []) if a.get("name"))
                ne_dur = float(s.get("duration") or 0.0) / 1000.0
                sc = _track_matches_title(tr_name, ar_name, match_context, cand_dur=ne_dur, yt_dur=duration)
                if sc > 0.0:
                    ne_candidates.append((sc, s))

            ne_candidates.sort(key=lambda x: x[0], reverse=True)
            for sc, s in ne_candidates[:5]:
                can_upgrade_sync = (not is_live_or_alt) and (best_synced is None) and (sc >= best_score - 0.05)
                if sc <= best_score and not can_upgrade_sync:
                    break
                try:
                    lr = client.get("https://music.163.com/api/song/lyric", params={"id": s["id"], "lv": 1, "tv": -1}, headers=ne_headers)
                    lrc = (lr.json().get("lrc") or {}).get("lyric", "")
                    if not lrc:
                        continue
                    parsed = parse_lrc(lrc)
                    if len(parsed) >= 6:
                        ne_dur = float(s.get("duration") or 0.0) / 1000.0
                        dur_ok = (duration > 0.0 and ne_dur > 0.0 and abs(ne_dur - duration) <= 25.0)
                        if sc > best_score:
                            best_score = sc
                            best_lines = clean_lyric_lines([x["text"] for x in parsed])
                            best_source = f"NetEase ({s.get('name') or ''})"
                            if not is_live_or_alt and dur_ok:
                                best_synced = parsed
                                best_synced_score = sc
                            elif best_synced_score < best_score - 0.05:
                                best_synced = None
                                best_synced_score = 0.0
                            break
                        elif can_upgrade_sync and dur_ok:
                            best_synced = parsed
                            best_synced_score = sc
                            if not best_lines:
                                best_lines = clean_lyric_lines([x["text"] for x in parsed])
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


def _extract_bracketed_titles(title: str) -> list[str]:
    """Extract quoted/bracketed segments using paired delimiters so nested quotes
    (e.g. 《唯獨不可敗給名為「我」的你》《戀愛腦之死》) and 1-char CJK titles (《抽》) are never truncated!
    """
    if not title:
        return []
    found: list[str] = []
    # Prefer outer book-title brackets 《...》 first; fall back to 「」/『』/""/'' if fewer than 2 《》 exist
    outer_books = [m.strip() for m in re.findall(r"《([^《》]{1,48})》", title) if m.strip()]
    if len(outer_books) >= 2:
        found.extend(outer_books)
    else:
        for pat in (
            r"《([^《》]{1,48})》",
            r"「([^「」]{1,48})」",
            r"『([^『』]{1,48})』",
            r"\"([^\"]{2,48})\"",
            r"(?<![A-Za-z])'([^']{2,48})'(?![A-Za-z])",
            r"‘([^’]{1,48})’",
        ):
            for m in re.findall(pat, title):
                s = m.strip()
                if s:
                    found.append(s)
    return list(dict.fromkeys(found))


def _extract_artist_hint_outside_brackets(title: str) -> str:
    """Extract non-bracketed artist context from multi-song titles like '《CHILL CLUB》 晚安莉莉《Song A》《Song B》'."""
    stripped = re.sub(r"《[^《》]*》|「[^「」]*」|『[^『』]*』|\"[^\"]*\"|'[^']*'|‘[^’]*’|[\[\(【（［][^\]\)】）］]*[\]\)】）］]", " ", title or "")
    stripped = re.sub(r"[^\w\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\s.'-]+", " ", stripped)
    stripped = clean_search_title(stripped)
    return stripped if 2 <= len(stripped) <= 32 else ""


def fetch_multi_song_lyrics_from_title(title: str, duration: float, yt_artist: str = "") -> list[list[str]]:
    """Generic multi-song lookup for long videos containing >=2 songs in the title —
    works for BOTH bracketed titles ('《Song A》《Song B》', '「A」「B」') AND unbracketed multi-song titles
    (e.g. '陳奕迅 今日 黑擇明', 'Adele - Hello / Someone Like You').
    Zero hardcoded blacklists: preserves artist hint, queries each piece in parallel, and deduplicates by lyric content.
    """
    if not title:
        return []
    pieces = _extract_bracketed_titles(title)
    artist_hint = (yt_artist or "").strip() or _extract_artist_hint_outside_brackets(title)
    if len(pieces) >= 2:
        if duration > 0.0 and duration < 240.0 and not MASHUP_KEYWORDS_RE.search(title):
            return []
        query_pieces = pieces[:6]
    else:
        # Unbracketed multi-song video (e.g. 8-min 2-song mashup/medley)
        if duration < 320.0 and not MASHUP_KEYWORDS_RE.search(title):
            return []
        cleaned = clean_search_title(title)
        has_cjk_only = bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", cleaned)) and not bool(re.search(r"[A-Za-z]{3,}", cleaned))
        # Only split on spaces for CJK-only titles (e.g. '陳奕迅 今日 黑擇明'); English/Korean split ONLY on explicit separators!
        sep_pat = r"[\s/|｜+＋、,&]+|\s*[-—–]\s*" if has_cjk_only else r"[/|｜+＋、,&]+|\s+[-—–x×]\s+"
        raw_toks = [
            t.strip()
            for t in re.split(sep_pat, cleaned)
            if len(t.strip()) >= 2 and not re.fullmatch(r"[\d.:-]+", t.strip())
        ]
        if len(raw_toks) < 2:
            return []
        query_pieces = []
        prefix_artist = raw_toks[0] if len(raw_toks) >= 3 else ""
        if prefix_artist and not artist_hint:
            artist_hint = prefix_artist
        for tok in raw_toks[1:] if prefix_artist else raw_toks:
            if tok not in query_pieces:
                query_pieces.append(tok)
        query_pieces = query_pieces[:6]

    def _lookup_one(piece: str) -> list[str]:
        try:
            _, lines, _ = fetch_official_lyrics(piece, 0.0, "", yt_artist=artist_hint)
            return lines if len(lines) >= 8 else []
        except Exception:
            return []

    pools: list[list[str]] = []
    seen_fps: set[str] = set()
    with ThreadPoolExecutor(max_workers=min(4, len(query_pieces))) as ex:
        results = list(ex.map(_lookup_one, query_pieces))
    for lines in results:
        if not lines:
            continue
        fp = "".join(_norm_lyrics_cmp(x) for x in lines[:4])[:40]
        if fp and fp not in seen_fps:
            seen_fps.add(fp)
            pools.append(lines)
    return pools if len(pools) >= 2 else []


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
            yi = _load_audio_16k_mono(inst_path)
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
        sem = asyncio.Semaphore(3)

        async with Shazam() as shazam:
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

    # Fetch 100% official lyrics for every identified song and preserve per-track pools (zero split-brain)!
    combined_lines: list[str] = []
    for tr in unique_tracks:
        query_title = f"{tr['artist']} {tr['title']}".strip()
        _, lines, _ = fetch_official_lyrics(query_title, 0.0)
        if not lines:
            _, lines, _ = fetch_official_lyrics(tr["title"], 0.0)
        if not lines and tr.get("shz_lyrics"):
            lines = tr["shz_lyrics"]
        if lines:
            tr["official_lines"] = list(lines)
            combined_lines.extend(lines)

    return combined_lines, unique_tracks


def search_official_lyrics_pools_by_sung_phrases(asr_segs: list[dict]) -> list[list[str]]:
    """Identify unknown song(s) inside a Mashup / Live Cover / Acoustic Medley directly from sung lyric phrases!
    Parallelized across probes using shared httpx.Client; handles CJK, Japanese, Korean, and English safely,
    and returns distinct per-song pools (`list[list[str]]`) so multi-song discoveries never smash together!
    """
    cands = []
    for s in asr_segs:
        txt = (s.get("text") or "").strip()
        n_txt = _norm_lyrics_cmp(txt)
        is_cjk_seg = bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", txt))
        max_len = 22 if is_cjk_seg else 48
        if 5 <= len(n_txt) <= max_len:
            cands.append(txt)
    if not cands:
        return []

    n_probes = min(6, len(cands))
    if len(cands) <= n_probes:
        probes = cands
    else:
        idxs = sorted({int(round(i * (len(cands) - 1) / (n_probes - 1))) for i in range(n_probes)})
        probes = [cands[i] for i in idxs]

    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://music.163.com/"}
    asr_blob = "".join(_norm_lyrics_cmp(s["text"]) for s in asr_segs)

    def _probe_search(client: httpx.Client, q: str) -> list[dict]:
        qs = to_simplified(q) if not _is_non_chinese_cjk(q) else q
        try:
            r = client.get(
                "https://music.163.com/api/search/get",
                params={"s": qs, "type": 1006, "limit": 4},
                headers=headers,
            )
            return ((r.json().get("result") or {}).get("songs") or [])[:3]
        except Exception:
            return []

    seen_song_ids: set[int] = set()
    seen_titles: set[str] = set()
    rescued_pools: list[list[str]] = []

    with httpx.Client(timeout=7.0, follow_redirects=True) as client:
        with ThreadPoolExecutor(max_workers=min(6, len(probes))) as ex:
            probe_results = list(ex.map(lambda q: _probe_search(client, q), probes))

        for songs in probe_results:
            for s in songs:
                sid = s.get("id")
                tr_name = s.get("name") or ""
                base_t = _norm_lyrics_cmp(re.sub(r"[\(\[（【［].*?[\)\]）】］]", "", tr_name))
                if not sid or sid in seen_song_ids or (base_t and base_t in seen_titles):
                    continue
                seen_song_ids.add(sid)
                try:
                    lr = client.get(
                        "https://music.163.com/api/song/lyric",
                        params={"id": sid, "lv": 1, "tv": -1},
                        headers=headers,
                    )
                    lrc = (lr.json().get("lrc") or {}).get("lyric", "")
                    if not lrc:
                        continue
                    parsed = parse_lrc(lrc)
                    lines = clean_lyric_lines([x["text"] for x in parsed])
                    if len(lines) < 6:
                        continue
                    # Script-aware verification: CJK prefix check ONLY for CJK/JP lines; EN/KO require real word-level similarity!
                    hit_lines = 0
                    for ln in set(lines):
                        nln = _norm_lyrics_cmp(ln)
                        if len(nln) < 4:
                            continue
                        is_cjk_ln = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff]", ln)) >= 4
                        if (is_cjk_ln and nln[:4] in asr_blob) or any(
                            _lyric_similarity(seg["text"], ln) >= (0.45 if is_cjk_ln else 0.52) for seg in asr_segs
                        ):
                            hit_lines += 1
                    if hit_lines >= 3:
                        if base_t:
                            seen_titles.add(base_t)
                        artists = "/".join((a.get("name") or "") for a in (s.get("artists") or []))
                        print(
                            f"[Lyrics] Phrase-search rescued song: {artists} - {tr_name} (matched {hit_lines} lines)",
                            flush=True,
                        )
                        rescued_pools.append(lines)
                        break
                except Exception:
                    continue

    return rescued_pools


def search_official_lyrics_by_sung_phrases(asr_segs: list[dict]) -> list[str]:
    pools = search_official_lyrics_pools_by_sung_phrases(asr_segs)
    return [ln for p in pools for ln in p]


def _get_yt_js_runtimes() -> dict:
    runtimes: dict = {}
    for name, fallbacks in (
        ("deno", [Path.home() / ".deno" / "bin" / "deno.exe"]),
        ("node", [Path("C:/Program Files/nodejs/node.exe"), Path("C:/Program Files (x86)/nodejs/node.exe")]),
        ("bun", [Path.home() / ".bun" / "bin" / "bun.exe"]),
    ):
        found = shutil.which(name)
        if not found:
            for fb in fallbacks:
                if fb.exists():
                    found = str(fb)
                    break
        if found:
            runtimes[name] = {"path": found}
    return {"js_runtimes": runtimes} if runtimes else {}


def _get_yt_cookie_opts() -> dict:
    cookies_file = BASE_DIR / "cookies.txt"
    if cookies_file.exists() and cookies_file.stat().st_size > 0:
        return {"cookiefile": str(cookies_file)}
    ff_prof = BASE_DIR / ".yt_firefox_profile"
    if (ff_prof / "cookies.sqlite").exists():
        return {"cookiesfrombrowser": ("firefox", str(ff_prof), None, None)}
    return {}


def _pick_youtube_subtitle_lang(subs: dict, title: str = "", video_lang: str = "", description: str = "") -> Optional[str]:
    """Select the YouTube human subtitle track that actually matches the SONG'S language (exact or BCP-47 prefix).
    Never forces Chinese fan translations onto English/Japanese/Korean MVs and never falls back to arbitrary languages!
    """
    if not isinstance(subs, dict) or not subs:
        return None
    keys = list(subs.keys())

    def _find_prefix(prefixes: list[str]) -> Optional[str]:
        for pref in prefixes:
            pl = pref.lower()
            for k in keys:
                if k.lower() == pl:
                    return k
            for k in keys:
                kl = k.lower()
                if kl.startswith(pl + "-") or kl.startswith(pl + "_"):
                    return k
        return None

    v_low = (video_lang or "").strip().lower()
    t = title or ""
    has_kana = bool(re.search(r"[\u3040-\u30ff]", t))
    has_hangul = bool(re.search(r"[\uac00-\ud7af]", t))
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", t))

    # When video_lang is missing and title is pure ASCII (e.g. 'YOASOBI - Idol' or 'BLACKPINK - Pink Venom'),
    # check the script ratio of the opening description (first 300 non-space chars) to detect JA/KO/ZH originals!
    desc_lang_hint = ""
    if not v_low and not (has_kana or has_hangul or has_cjk) and description:
        d_head = re.sub(r"https?://\S+", "", description[:450])
        d_clean = re.sub(r"[\s\W_]+", "", d_head)[:300]
        if len(d_clean) >= 20:
            d_total = len(d_clean)
            if len(re.findall(r"[\u3040-\u30ff]", d_clean)) / d_total >= 0.15:
                desc_lang_hint = "ja"
            elif len(re.findall(r"[\uac00-\ud7af]", d_clean)) / d_total >= 0.15:
                desc_lang_hint = "ko"
            elif len(re.findall(r"[\u4e00-\u9fff]", d_clean)) / d_total >= 0.25:
                desc_lang_hint = "zh"

    if v_low.startswith("yue") or (v_low.startswith("zh") and ("hk" in v_low)):
        prefs = ["yue-HK", "yue", "zh-HK", "zh-Hant", "zh-TW", "zh"]
    elif v_low.startswith("zh"):
        prefs = ["zh-Hant", "zh-TW", "zh-HK", "zh-Hans", "zh-CN", "zh", "yue"]
    elif v_low.startswith("ja") or has_kana or desc_lang_hint == "ja":
        prefs = ["ja", "ja-JP"]
    elif v_low.startswith("ko") or has_hangul or desc_lang_hint == "ko":
        prefs = ["ko", "ko-KR"]
    elif desc_lang_hint == "zh":
        prefs = ["yue-HK", "yue", "zh-HK", "zh-Hant", "zh-TW", "zh"]
    elif v_low.startswith("en") or (not has_cjk and re.search(r"[A-Za-z]{3,}", t)):
        prefs = ["en", "en-US", "en-GB"]
    elif v_low and len(v_low) >= 2:
        prefs = [v_low.split("-")[0]]
    elif has_cjk:
        prefs = ["yue-HK", "yue", "zh-HK", "zh-Hant", "zh-TW", "zh"]
    else:
        prefs = ["en", "yue-HK", "zh-HK", "zh-Hant", "zh-TW", "zh", "ja", "ko"]

    return _find_prefix(prefs)


def _download_json3_subtitles(json3_url: str, chosen_lang: str) -> Optional[list[dict]]:
    if not json3_url:
        return None
    try:
        r = httpx.get(json3_url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        lines = []
        cl = (chosen_lang or "").lower()
        is_zh = cl.startswith(("zh", "yue"))
        is_ja = cl.startswith("ja")
        is_ko = cl.startswith("ko")
        non_lyric_cue_re = re.compile(r"^[\[\(（【][^\]\)）】]{1,32}[\]\)）】]$")
        for ev in data.get("events") or []:
            segs = ev.get("segs")
            if not segs:
                continue
            raw_joined = "".join(s.get("utf8", "") for s in segs).strip()
            if not raw_joined:
                continue
            # Bilingual 2-line CC cleanup (keeps ONLY the line matching chosen_lang script; NEVER drops Japanese Kanji lines!)
            sub_lines = [ln.strip() for ln in raw_joined.splitlines() if ln.strip()]
            if len(sub_lines) == 2:
                k0, k1 = _line_script_kind(sub_lines[0]), _line_script_kind(sub_lines[1])
                if is_ja:
                    # Keep Japanese line (Kana or Kanji); only drop an English/Korean translation line!
                    if (k0 in ("ja", "zh")) != (k1 in ("ja", "zh")):
                        raw_joined = sub_lines[0] if k0 in ("ja", "zh") else sub_lines[1]
                elif is_ko:
                    if (k0 == "ko") != (k1 == "ko"):
                        raw_joined = sub_lines[0] if k0 == "ko" else sub_lines[1]
                elif is_zh:
                    if (k0 == "zh") != (k1 == "zh"):
                        raw_joined = sub_lines[0] if k0 == "zh" else sub_lines[1]
                else:
                    if (k0 == "en") != (k1 == "en"):
                        raw_joined = sub_lines[0] if k0 == "en" else sub_lines[1]
            text = re.sub(r"\s+", " ", raw_joined.replace("\n", " ")).strip()
            text = re.sub(r"^[♪♫\s]+|[♪♫\s]+$", "", text).strip()
            if not text or non_lyric_cue_re.match(text) or CREDIT_LINE_RE.match(text) or SECTION_HEADER_RE.match(text) or _is_instrumental_placeholder(text):
                continue
            start_s = round((ev.get("tStartMs") or 0) / 1000.0, 2)
            dur_s = (ev.get("dDurationMs") or 4000) / 1000.0
            end_s = round(start_s + max(0.5, dur_s), 2)
            lines.append({
                "start": start_s,
                "end": end_s,
                "text": to_hk_traditional(text) if is_zh else text,
                "words": [],
            })
        # Clamp overlapping dDurationMs spans cleanly
        lines.sort(key=lambda x: x["start"])
        for i in range(len(lines) - 1):
            if lines[i]["end"] > lines[i + 1]["start"]:
                lines[i]["end"] = round(max(lines[i]["start"] + 0.35, lines[i + 1]["start"]), 2)
        if len(lines) >= 4:
            return lines
    except Exception:
        pass
    return None


def fetch_youtube_human_subtitles(video_id: str) -> tuple[Optional[list[dict]], str, str, float, str, str, str, str]:
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, **_get_yt_cookie_opts(), **_get_yt_js_runtimes()}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        print(f"[Lyrics] YouTube subtitle probe skipped ({video_id}): {str(exc)[:140]}", flush=True)
        return None, "", video_id, 0.0, "", "", "", ""
    if not isinstance(info, dict):
        return None, "", video_id, 0.0, "", "", "", ""

    title = info.get("title") or video_id
    duration = float(info.get("duration") or 0.0)
    desc = info.get("description") or ""
    yt_track = (info.get("track") or "").strip()
    yt_artist = (info.get("artist") or info.get("creator") or "").strip()
    yt_channel = (info.get("channel") or info.get("uploader") or "").strip()
    video_lang = (info.get("language") or "").strip()
    subs = info.get("subtitles") or {}
    if not subs:
        return None, "", title, duration, desc, yt_track, yt_artist, yt_channel

    chosen_lang = _pick_youtube_subtitle_lang(subs, title=title, video_lang=video_lang, description=desc)
    if not chosen_lang:
        return None, "", title, duration, desc, yt_track, yt_artist, yt_channel

    formats = subs.get(chosen_lang) or []
    json3_url = next((f["url"] for f in formats if f.get("ext") == "json3" and f.get("url")), None)
    if not json3_url:
        return None, "", title, duration, desc, yt_track, yt_artist, yt_channel

    lines = _download_json3_subtitles(json3_url, chosen_lang)
    return lines, chosen_lang, title, duration, desc, yt_track, yt_artist, yt_channel


# Colloquial Cantonese markers (excludes Mandarin particle '呢' so Mandopop is never misclassified as Cantonese!)
CANTONESE_MARKERS_RE = re.compile(r"[嘅唔係喺咗啲冇佢咁哋睇諗嗰乜嘢嚟㗎咩]")


def detect_lyrics_language(lines: list[str], sub_lang: str = "") -> str:
    """Detect lyrics language over non-whitespace/non-punctuation script ratios (correctly handles mixed K-pop/J-pop/EN
    and distinguishes Cantonese 'yue' from Mandarin 'zh' when markers or subtitle metadata exist).
    """
    joined = re.sub(r"[\s\W_]+", "", "".join(lines or ""))
    if not joined:
        return "zh"
    cjk = len(re.findall(r"[\u4e00-\u9fff]", joined))
    jp = len(re.findall(r"[\u3040-\u30ff]", joined))
    ko = len(re.findall(r"[\uac00-\ud7af]", joined))
    total = max(1, len(joined))
    if jp / total >= 0.03:
        return "ja"
    if ko / total >= 0.05:
        return "ko"
    if cjk / total >= 0.22:
        sl = (sub_lang or "").lower()
        if sl.startswith("yue") or "hk" in sl:
            return "yue"
        if len(CANTONESE_MARKERS_RE.findall(joined)) >= 2:
            return "yue"
        return "zh"
    return "en"


def _norm_lyrics_cmp(s: str) -> str:
    return re.sub(r"[\s\W_]+", "", to_hk_traditional(s or "")).lower()


def _lyric_similarity(a: str, b: str) -> float:
    na, nb = _norm_lyrics_cmp(a), _norm_lyrics_cmp(b)
    if not na or not nb:
        return 0.0
    # Use word-level comparison for space-delimited Latin/Korean text so 26-letter alphabet overlap never inflates English scores!
    is_cjk = bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", a or "")) or bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", b or ""))
    if not is_cjk:
        wa = re.findall(r"[a-z0-9'\uac00-\ud7af]+", (a or "").lower())
        wb = re.findall(r"[a-z0-9'\uac00-\ud7af]+", (b or "").lower())
        if wa and wb:
            sm_w = SequenceMatcher(None, wa, wb).ratio()
            sa_w, sb_w = set(wa), set(wb)
            jacc_w = len(sa_w & sb_w) / max(1, len(sa_w | sb_w))
            cov_w = len(sa_w & sb_w) / max(1, min(len(sa_w), len(sb_w)))
            return 0.55 * sm_w + 0.30 * jacc_w + 0.15 * cov_w
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
    min_silence_sec: float = 1.6,
    keep_pad_sec: float = 0.35,
    gap_sec: float = 0.35,
    thr_db: Optional[float] = None,
) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
    """Collapse long silent instrumental interludes down to `gap_sec` so Whisper/ForcedAligner
    never stretches or hallucinates lyrics across instrumental solos, while preserving an exact
    piecewise map (comp_start, comp_end, real_start, real_end).
    """
    n_total = len(y)
    if n_total == 0 or len(db_prof) == 0 or step_s <= 0:
        dur = n_total / sr
        return y, [(0.0, dur, 0.0, dur)]

    eff_thr_db = thr_db if thr_db is not None else _adaptive_vocal_threshold_db(db_prof, margin_db=-1.0)
    active = db_prof > eff_thr_db
    pad_frames = max(1, int(round(keep_pad_sec / step_s)))
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
            if islands and (s_idx * step_s - islands[-1][1]) < min_silence_sec:
                islands[-1] = (islands[-1][0], i * step_s)
            else:
                islands.append((s_idx * step_s, i * step_s))
            in_isl = False
    if in_isl:
        if islands and (s_idx * step_s - islands[-1][1]) < min_silence_sec:
            islands[-1] = (islands[-1][0], len(padded) * step_s)
        else:
            islands.append((s_idx * step_s, len(padded) * step_s))

    if not islands:
        dur = n_total / sr
        return y, [(0.0, dur, 0.0, dur)]

    gap_sil = np.zeros(max(0, int(gap_sec * sr)), dtype=np.float32)
    chunks: list[np.ndarray] = []
    mapping: list[tuple[float, float, float, float]] = []
    comp_cursor = 0.0
    for idx, (rs, re_) in enumerate(islands):
        i0 = max(0, int(rs * sr))
        i1 = min(n_total, int(re_ * sr))
        if i1 <= i0:
            continue
        if chunks and len(gap_sil) > 0:
            chunks.append(gap_sil)
            comp_cursor += len(gap_sil) / sr
        seg = y[i0:i1]
        dur = len(seg) / sr
        chunks.append(seg)
        mapping.append((comp_cursor, comp_cursor + dur, i0 / sr, i1 / sr))
        comp_cursor += dur

    if not chunks:
        dur = n_total / sr
        return y, [(0.0, dur, 0.0, dur)]
    return np.concatenate(chunks), mapping


def _map_comp_to_real(
    t_comp: float,
    mapping: list[tuple[float, float, float, float]],
    is_end: bool = False,
) -> float:
    """Map compressed timestamp back to real audio time WITHOUT teleporting line ends across collapsed solos!
    When t_comp lands inside the 0.35s synthetic gap between island k and island k+1, snap `is_end=True`
    backward to island k's real end (`re_k`) and `is_end=False` forward to island k+1's real start (`rs_{k+1}`).
    """
    if not mapping:
        return t_comp
    for idx, (cs, ce, rs, re_) in enumerate(mapping):
        if t_comp < cs - 1e-4:
            # t_comp sits in the synthetic silence gap between mapping[idx-1] and mapping[idx]
            if idx > 0 and is_end:
                return mapping[idx - 1][3]
            return rs
        if t_comp <= ce + 1e-4:
            frac = max(0.0, min(1.0, (t_comp - cs) / max(1e-6, ce - cs)))
            return rs + frac * (re_ - rs)
    return mapping[-1][3]


def _distribute_by_vocal_energy_cdf(
    ws_slice: list[dict],
    span_s: float,
    span_e: float,
    db_profile: np.ndarray,
    step_sec: float,
) -> None:
    """Distribute collapsed zero-duration words across [span_s, span_e] using the cumulative vocal energy CDF.
    Where the singer is silent inside the span, energy ~ 0 so the CDF stays flat (natural pause);
    where the singer is vocalizing, the CDF rises smoothly so characters track real singing energy.
    """
    if not ws_slice or span_e <= span_s:
        return

    weights = [max(1, len(re.sub(r"\s+", "", str(w.get("word") or "")))) for w in ws_slice]
    total_w = float(sum(weights))

    # Build piecewise-linear time(u) inverse CDF for u in [0, 1]
    use_cdf = False
    if len(db_profile) >= 10 and step_sec > 0:
        thr_db = _adaptive_vocal_threshold_db(db_profile)
        f0 = max(0, int(span_s / step_sec))
        f1 = min(len(db_profile), max(f0 + 1, int(np.ceil(span_e / step_sec))))
        seg_db = db_profile[f0:f1]
        if len(seg_db) >= 2:
            # Linear excess vocal energy above adaptive floor (0 in silence, >0 when singing)
            energy = np.maximum(0.0, 10.0 ** ((seg_db - thr_db) / 20.0) - 1.0)
            if float(np.sum(energy)) > 1e-4:
                # Tiny baseline epsilon prevents division issues while keeping silent gaps nearly flat
                energy = energy + 1e-3 * float(np.mean(energy))
                cdf = np.concatenate(([0.0], np.cumsum(energy)))
                cdf = cdf / float(cdf[-1])
                times = np.linspace(span_s, span_e, len(cdf))
                use_cdf = True

    def _u_to_time(u: float) -> float:
        u = max(0.0, min(1.0, u))
        if use_cdf:
            return float(np.interp(u, cdf, times))
        return span_s + (span_e - span_s) * u

    acc = 0.0
    for w, wt in zip(ws_slice, weights):
        u0 = acc / total_w
        acc += wt
        u1 = acc / total_w
        w["start"] = round(_u_to_time(u0), 2)
        w["end"] = round(_u_to_time(u1), 2)


def _smooth_bunched_words(
    ws: list[dict],
    line_start: float,
    line_end: float,
    db_profile: np.ndarray,
    step_sec: float,
) -> list[dict]:
    """Generic, first-principles recovery for degenerate zero-duration aligner collapses (`raw_start == raw_end`).

    Unlike hardcoded speaking-rate heuristics (which can distort fast rap), this triggers ONLY on tokens
    where the aligner mathematically failed to assign a distinct frame span (raw_dur <= 0.02s before clamping),
    and redistributes the collapsed run across its bounded acoustic span via vocal-energy CDF integration.
    """
    if not ws:
        return ws

    n = len(ws)
    # Record exact pre-clamp raw durations BEFORE normalizing so real fast speech (e.g. 0.07s rap) is never confused
    # with aligner degeneracy (0.00s - 0.02s collapsed tokens).
    raw_collapsed = [
        (float(w.get("end", 0.0)) - float(w.get("start", 0.0))) <= 0.021
        for w in ws
    ]

    # Pass 1: Normalize bounds monotonically without inflating collapsed tokens yet
    ws[0]["start"] = round(max(line_start, min(float(ws[0]["start"]), line_end - 0.15)), 2)
    ws[-1]["end"] = round(line_end, 2)
    prev_e = line_start
    for idx, w in enumerate(ws):
        s_i = max(prev_e, min(line_end, float(w["start"])))
        e_i = max(s_i, min(line_end, float(w["end"])))
        if idx == n - 1:
            e_i = line_end
        w["start"] = round(s_i, 2)
        w["end"] = round(e_i, 2)
        prev_e = e_i

    # Also detect post-clamp zero-duration tokens (e.g. when onset/tail snap squeezed boundary tokens to 0.00s)
    is_collapsed = [
        raw_collapsed[idx] or ((float(ws[idx]["end"]) - float(ws[idx]["start"])) <= 0.021)
        for idx in range(n)
    ]

    # Pass 2: Redistribute contiguous collapsed runs across their true bounded acoustic span via Energy CDF
    # Also rescue classic [hog, 0s, 0s, 0s] / [0s, 0s, 0s, hog] Whisper collapse where an adjacent token swallowed the span!
    i = 0
    while i < n:
        if is_collapsed[i]:
            j = i + 1
            while j < n and is_collapsed[j]:
                j += 1
            run_s, run_e = i, j
            span_s = ws[run_s - 1]["end"] if run_s > 0 else line_start
            span_e = ws[run_e]["start"] if run_e < n else line_end
            if span_e <= span_s + 0.05:
                # Expand to include adjacent duration-hogging token so Energy CDF redistributes the full acoustic span
                left_dur = (ws[run_s - 1]["end"] - ws[run_s - 1]["start"]) if run_s > 0 else 0.0
                right_dur = (ws[run_e]["end"] - ws[run_e]["start"]) if run_e < n else 0.0
                if left_dur >= right_dur and left_dur > 0.35 and run_s > 0:
                    run_s -= 1
                    span_s = ws[run_s - 1]["end"] if run_s > 0 else line_start
                elif right_dur > 0.35 and run_e < n:
                    run_e += 1
                    span_e = ws[run_e]["start"] if run_e < n else line_end
            if span_e > span_s + 0.05:
                _distribute_by_vocal_energy_cdf(ws[run_s:run_e], span_s, span_e, db_profile, step_sec)
            i = max(j, run_e)
        else:
            i += 1

    # Pass 3: Enforce strict monotonic continuity strictly INSIDE [line_start, line_end] (never overflow on fast rap!)
    min_step = min(0.04, max(0.01, (line_end - line_start) / max(1, n)))
    prev_e = line_start
    for idx, w in enumerate(ws):
        remaining_min = min_step * (n - 1 - idx)
        w_s = max(prev_e, min(line_end - remaining_min, float(w["start"])))
        w_e = max(w_s + min_step, min(line_end - remaining_min, float(w["end"])))
        if idx == n - 1:
            w_e = max(w_s, line_end)
        w["start"] = round(w_s, 2)
        w["end"] = round(min(line_end, w_e), 2)
        prev_e = w["end"]
    return ws


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
    thr_db = _adaptive_vocal_threshold_db(db_profile, margin_db=-1.0)
    for i, ln in enumerate(lines):
        s = max(first_onset, float(ln["start"]))
        e = max(s + 0.35, float(ln["end"]))
        nxt_s = float(lines[i + 1]["start"]) if (i + 1 < len(lines)) else (audio_dur or e + 2.5)

        # 1) Acoustic Onset Snap + Full Dead-Space Trim:
        # Never let a line sit lit across >1.5s of pure silence/solo before or after the actual vocal burst!
        # Filter isolated 1-frame (0.1s) cymbal/breath blips so a stray click never holds a 20s silent span open.
        i0_ln = max(0, int(s / step_sec))
        i1_ln = min(n_db, int(e / step_sec))
        if (e - s) > 3.5 and i1_ln > i0_ln:
            seg_slice = db_profile[i0_ln:i1_ln]
            # For anomalously stretched lines (> 8.0s), use a local dynamic floor (max of global thr and line_p90 - 16 dB)
            # so faint reverb / crowd cheer / backing bleed during an interlude can NEVER hold a line open for 15-36s!
            eff_ln_thr = thr_db
            if (e - s) > 8.0 and len(seg_slice) >= 20:
                eff_ln_thr = max(thr_db, float(np.percentile(seg_slice, 90)) - 16.0)
            raw_act = seg_slice > eff_ln_thr
            # Keep only frames with at least one active neighbor (eliminates 100ms stray blips)
            if len(raw_act) >= 3:
                sustained = raw_act.copy()
                sustained[0] = raw_act[0] and raw_act[1]
                sustained[-1] = raw_act[-1] and raw_act[-2]
                sustained[1:-1] = raw_act[1:-1] & (raw_act[:-2] | raw_act[2:])
                act_frames = np.flatnonzero(sustained)
                if len(act_frames) == 0:
                    act_frames = np.flatnonzero(raw_act)
            else:
                act_frames = np.flatnonzero(raw_act)
            if len(act_frames) > 0:
                # If line still spans >8s with a >=2.5s internal dead gap, snap to the dominant active vocal island!
                if (e - s) > 8.0 and len(act_frames) >= 2:
                    islands = []
                    st_f = int(act_frames[0])
                    pv_f = st_f
                    for af in act_frames[1:]:
                        af_i = int(af)
                        if (af_i - pv_f) * step_sec >= 2.5:
                            islands.append((st_f, pv_f))
                            st_f = af_i
                        pv_f = af_i
                    islands.append((st_f, pv_f))
                    if len(islands) >= 2:
                        # Pick island with the most active vocal frames
                        best_isl = max(islands, key=lambda pair: int(np.sum(raw_act[pair[0] : pair[1] + 1])))
                        v_first = (i0_ln + best_isl[0]) * step_sec
                        v_last = (i0_ln + best_isl[1] + 1) * step_sec
                    else:
                        v_first = (i0_ln + int(act_frames[0])) * step_sec
                        v_last = (i0_ln + int(act_frames[-1]) + 1) * step_sec
                else:
                    v_first = (i0_ln + int(act_frames[0])) * step_sec
                    v_last = (i0_ln + int(act_frames[-1]) + 1) * step_sec
                if (v_first - s) > 1.5 and (e - v_first) >= 0.45:
                    s = round(max(first_onset, v_first - 0.15), 2)
                if (e - v_last) > 1.5 and (v_last - s) >= 0.45:
                    e = round(max(s + 0.5, min(e, v_last + 0.30)), 2)

        max_snap = min(e - 0.5, s + 2.5)
        t_cur = s
        while t_cur < max_snap:
            idx = int(t_cur / step_sec)
            if idx < n_db and db_profile[idx] > thr_db:
                break
            t_cur += step_sec
        if t_cur > s + 0.12 and t_cur < max_snap:
            s = round(max(first_onset, t_cur - 0.08), 2)

        # 2) Acoustic Vocal-Tail Extension: follow held final vowel until 0.2s genuine silence
        # Use margin_db=+1.0 so genuine held singing vowels extend cleanly while faint live crowd/guitar bleed does not!
        tail_thr_db = _adaptive_vocal_threshold_db(db_profile, margin_db=1.0)
        limit_e = min(nxt_s - 0.12, e + 2.2)
        if limit_e > e:
            t_e = e
            quiet_run = 0
            best_e = e
            while t_e < limit_e:
                idx = int(t_e / step_sec)
                if idx >= n_db:
                    break
                if db_profile[idx] > tail_thr_db:
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

        # 3) Word-level smoothing: eliminate bunched/collapsed word glitches so 60fps KTV sweep never shoots ahead
        ws = ln.get("words") or []
        if ws:
            ln["words"] = _smooth_bunched_words(ws, ln["start"], ln["end"], db_profile, step_sec)

    # 4) Global structural post-invariants: ZERO line overlaps and ZERO out-of-bounds word timestamps!
    for i in range(len(lines)):
        if i + 1 < len(lines) and lines[i]["end"] > lines[i + 1]["start"]:
            if lines[i + 1]["start"] >= lines[i]["start"] + 0.20:
                lines[i]["end"] = round(lines[i + 1]["start"], 2)
            else:
                lines[i + 1]["start"] = round(lines[i]["end"], 2)
                if lines[i + 1]["end"] < lines[i + 1]["start"] + 0.25:
                    lines[i + 1]["end"] = round(lines[i + 1]["start"] + 0.25, 2)
        ls, le = float(lines[i]["start"]), float(lines[i]["end"])
        ws = lines[i].get("words") or []
        if ws:
            prev_we = ls
            for wi, w in enumerate(ws):
                ws_i = max(prev_we, min(le, float(w.get("start", ls))))
                we_i = max(ws_i, min(le, float(w.get("end", ws_i))))
                if wi == len(ws) - 1:
                    we_i = le
                w["start"] = round(ws_i, 2)
                w["end"] = round(we_i, 2)
                prev_we = we_i

    return lines


def _contiguous_vocal_span(
    db_profile: np.ndarray,
    step_sec: float,
    gap_thr: float,
    t_start: float,
    t_end: float,
    direction: str = "backward",
    max_silent_gap: float = 8.0,
) -> tuple[float, float, float]:
    """Return (active_vocal_seconds, first_active_t, last_active_t) inside [t_start, t_end]
    restricted to vocal activity connected to the anchor edge without crossing a dead-silent gap > max_silent_gap.
    Prevents TV host intros/applause or post-song MC banter from pulling song lines across long silences.
    """
    if len(db_profile) == 0 or t_end <= t_start or step_sec <= 0:
        return (0.0, t_start, t_end)
    i0 = max(0, int(t_start / step_sec))
    i1 = min(len(db_profile), int(t_end / step_sec))
    if i1 <= i0:
        return (0.0, t_start, t_end)

    raw_act = db_profile[i0:i1] > gap_thr
    n = len(raw_act)
    # Require >= 2 active frames in a 3-frame window (0.3s) to filter isolated 1-frame cymbal/breath clicks
    sustained = np.zeros(n, dtype=bool)
    for k in range(n):
        if raw_act[k]:
            lo = max(0, k - 1)
            hi = min(n, k + 2)
            if int(np.sum(raw_act[lo:hi])) >= 2:
                sustained[k] = True

    max_gap_frames = max(1, int(round(max_silent_gap / step_sec)))
    kept_indices: list[int] = []
    if direction == "backward":
        silent_run = 0
        for k in range(n - 1, -1, -1):
            if sustained[k]:
                kept_indices.append(k)
                silent_run = 0
            else:
                silent_run += 1
                if silent_run > max_gap_frames:
                    break
        kept_indices.reverse()
    elif direction == "forward":
        silent_run = 0
        for k in range(n):
            if sustained[k]:
                kept_indices.append(k)
                silent_run = 0
            else:
                silent_run += 1
                if silent_run > max_gap_frames:
                    break
    else:
        kept_indices = [int(k) for k in np.flatnonzero(sustained)]

    if not kept_indices:
        return (0.0, t_start, t_end)
    return (
        float(len(kept_indices)) * step_sec,
        (i0 + kept_indices[0]) * step_sec,
        (i0 + kept_indices[-1] + 1) * step_sec,
    )


def _select_monotonic_run_anchors(run_items: list[dict], pool_lines: list[str]) -> list[list[dict]]:
    """Assign optimal strictly-monotonic official-lyrics indices within a single song run via Weighted DP.

    Eliminates isolated out-of-order ASR false positives (e.g. matching Verse 1 English rap to an Outro line,
    which would otherwise poison Leading Recovery and drop the opening verse), while preserving genuine
    repeated live encore blocks (>= 3 consecutive monotonically-progressing anchors).
    """
    if not run_items or not pool_lines:
        return [run_items] if run_items else []

    exact_pos: dict[str, list[int]] = {}
    norm_pos: dict[str, list[int]] = {}
    for idx, txt in enumerate(pool_lines):
        exact_pos.setdefault(txt, []).append(idx)
        nk = _norm_lyrics_cmp(txt)
        if nk:
            norm_pos.setdefault(nk, []).append(idx)

    cands_per_item: list[list[int]] = []
    for item in run_items:
        txt = item["text"]
        cs = exact_pos.get(txt) or norm_pos.get(_norm_lyrics_cmp(txt)) or []
        cands_per_item.append(cs)

    def _dp_longest_monotonic(item_indices: list[int]) -> list[tuple[int, int]]:
        nodes: list[tuple[int, int, float]] = []
        for ii in item_indices:
            sim_w = 1.0 + max(0.0, float(run_items[ii].get("sim") or 0.0))
            for pos in cands_per_item[ii]:
                nodes.append((ii, pos, sim_w))
        if not nodes:
            return []
        m = len(nodes)
        dp = [nd[2] for nd in nodes]
        parent = [-1] * m
        for j in range(m):
            ij, pj, wj = nodes[j]
            best_val = wj
            best_p = -1
            for i in range(j):
                ii, pi, _ = nodes[i]
                if ii < ij and pi < pj:
                    diff = pj - pi
                    # Gentle smoothness tie-breaker favors consecutive lyric progression over wild skips
                    trans = 0.18 if diff <= 6 else (0.06 if diff <= 16 else 0.0)
                    cand_v = dp[i] + wj + trans
                    if cand_v > best_val:
                        best_val = cand_v
                        best_p = i
            dp[j] = best_val
            parent[j] = best_p
        best_end = max(range(m), key=lambda k: dp[k])
        chain: list[tuple[int, int]] = []
        cur = best_end
        while cur != -1:
            chain.append((nodes[cur][0], nodes[cur][1]))
            cur = parent[cur]
        chain.reverse()
        return chain

    all_idxs = [i for i in range(len(run_items)) if cands_per_item[i]]
    if not all_idxs:
        return []

    primary_chain = _dp_longest_monotonic(all_idxs)
    if not primary_chain:
        return []

    used_items = {ii for ii, _ in primary_chain}
    sub_runs: list[list[dict]] = [
        [{**run_items[ii], "_idx": pos} for ii, pos in primary_chain]
    ]

    # Iteratively preserve ALL genuine live encore / mid-song repeated-chorus blocks (>= 3 monotonically advancing
    # unused anchors in any contiguous temporal span — before, between, OR after the primary backbone!)
    # while rejecting isolated 1-2 line out-of-order ASR outliers!
    unused = [i for i in all_idxs if i not in used_items]
    if len(unused) >= 3:
        # Partition unused indices into contiguous temporal gaps separated by used_items
        gap_groups: list[list[int]] = []
        for ui in unused:
            if not gap_groups or (ui - gap_groups[-1][-1]) > 2:
                gap_groups.append([ui])
            else:
                gap_groups[-1].append(ui)
        for grp in gap_groups:
            if len(grp) >= 3:
                repeat_chain = _dp_longest_monotonic(grp)
                if len(repeat_chain) >= 3:
                    sub_runs.append([{**run_items[ii], "_idx": pos} for ii, pos in repeat_chain])

    sub_runs.sort(key=lambda r: float(r[0]["start"]))
    return sub_runs


def _alignment_needs_live_rematch(
    aligned,
    audio_dur: float,
    db_profile: np.ndarray,
    step_sec: float,
    time_map: Optional[list[tuple[float, float, float, float]]] = None,
    lang: str = "",
) -> tuple[bool, str]:
    """Detect skipped verses, mid-song starts, swapped sections, crushed lines, OR uncovered singing
    (e.g. Song #2 / encore in a multi-song video) in monotonic single-song alignment.
    """
    segs = [s for s in (aligned.segments or []) if (s.text or "").strip()]
    if not segs:
        return True, "empty"

    # Mandarin Whisper produces naturally lower token probabilities on Cantonese singing — avoid false live-rematch kicks!
    prob_floor = 0.18 if lang == "yue" else 0.28
    low_ratio_limit = 0.40 if lang == "yue" else 0.24

    crushed = 0
    tight_run = 0
    max_tight_run = 0
    low_prob = 0
    prev_start = -1.0
    prev_was_zero = False
    for seg in segs:
        txt = (seg.text or "").strip()
        n_syl = estimate_syllable_units(txt)
        rs = _map_comp_to_real(float(seg.start), time_map, is_end=False) if time_map else float(seg.start)
        re_ = _map_comp_to_real(float(seg.end), time_map, is_end=True) if time_map else float(seg.end)
        dur = re_ - rs
        is_zero = dur <= 0.02
        if dur < max(0.45, 0.09 * n_syl) or (prev_start >= 0 and not prev_was_zero and (rs - prev_start) < 0.25):
            crushed += 1
        if n_syl >= 4.0 and dur < 0.55:
            tight_run += 1
            max_tight_run = max(max_tight_run, tight_run)
        else:
            tight_run = 0
        prev_start = rs
        prev_was_zero = is_zero
        probs = [w.probability for w in (seg.words or []) if w.probability is not None]
        avg_p = (sum(probs) / len(probs)) if probs else 0.0
        if avg_p < prob_floor:
            low_prob += 1

    low_ratio = low_prob / max(1, len(segs))
    min_crushed = max(3, int(round(len(segs) * 0.08)))
    if crushed >= min_crushed or max_tight_run >= 3:
        return True, f"crushed={crushed},tight_run={max_tight_run}"
    if low_ratio >= low_ratio_limit:
        return True, f"low_prob={low_ratio:.2f}"

    # Check BOTH uncovered head AND uncovered tail against REAL vocal energy (mapped back from compressed time!)
    if len(db_profile) > 0 and audio_dur > 0:
        real_first = _map_comp_to_real(float(segs[0].start), time_map, is_end=False) if time_map else float(segs[0].start)
        real_last = _map_comp_to_real(float(segs[-1].end), time_map, is_end=True) if time_map else float(segs[-1].end)
        cov_thr = _adaptive_vocal_threshold_db(db_profile, margin_db=2.0)

        tail_idx = int((real_last + 4.0) / step_sec)
        if tail_idx < len(db_profile):
            active_tail_sec = float(np.sum(db_profile[tail_idx:] > cov_thr)) * step_sec
            if active_tail_sec >= 18.0:
                return True, f"unaligned_tail={active_tail_sec:.1f}s"

        head_idx = int(max(0.0, real_first - 4.0) / step_sec)
        if head_idx > 0:
            active_head_sec = float(np.sum(db_profile[:head_idx] > cov_thr)) * step_sec
            if active_head_sec >= 18.0:
                return True, f"unaligned_head={active_head_sec:.1f}s"

    return False, "ok"


def _join_asr_word_tokens(words: list[dict], is_cjk: bool) -> str:
    """Join ASR word tokens cleanly: spaces between Latin/Korean words, no extra spaces between CJK chars,
    and automatic single space between adjacent ASCII words inside mixed CJK+English lines!
    """
    toks = [(x.get("word") or "").strip() for x in words if (x.get("word") or "").strip()]
    if not toks:
        return ""
    if not is_cjk:
        return " ".join(toks).strip()
    out = toks[0]
    for tk in toks[1:]:
        if out and tk and re.search(r"[A-Za-z0-9]$", out) and re.search(r"^[A-Za-z0-9]", tk):
            out = f"{out} {tk}"
        else:
            out = f"{out}{tk}"
    return out.strip()


def _split_long_asr_segment(
    s_real: float,
    e_real: float,
    txt: str,
    words_mapped: list[dict],
    lang: str,
) -> list[dict]:
    """Break overly long Whisper run-on segments (e.g. 25-40s / 60-char blobs from silence-collapsed audio)
    into natural single-line ASR chunks using word-level pauses, punctuation, or breath ceilings.
    Prevents giant ASR paragraphs from failing length-ratio checks against official lyric lines!
    """
    norm_len = len(_norm_lyrics_cmp(txt))
    is_cjk = lang in ("zh", "yue", "ja")
    max_chars = 14 if is_cjk else 42
    if norm_len <= max_chars + 4 and (e_real - s_real) <= 6.5:
        return [{"start": s_real, "end": e_real, "text": txt, "words": words_mapped}]

    # 1) Preferred: split using Whisper word timestamps on natural pauses (>=0.28s), punctuation, or length ceiling
    if words_mapped and len(words_mapped) >= 3:
        chunks: list[dict] = []
        cur_ws: list[dict] = []
        cur_chars = 0
        for w in words_mapped:
            wt = (w.get("word") or "").strip()
            if not wt:
                continue
            w_norm = len(_norm_lyrics_cmp(wt))
            gap = (float(w["start"]) - float(cur_ws[-1]["end"])) if cur_ws else 0.0
            cur_dur = (float(w["end"]) - float(cur_ws[0]["start"])) if cur_ws else 0.0
            ends_punct = bool(cur_ws and re.search(r"[，。！？、,;!?]$", (cur_ws[-1].get("word") or "").strip()))
            should_cut = bool(cur_ws) and (
                (gap >= 0.28 and cur_chars >= 4)
                or (ends_punct and cur_chars >= 5)
                or (cur_chars + w_norm > max_chars and cur_chars >= 6)
                or (cur_dur >= 5.2 and cur_chars >= 5)
            )
            if should_cut:
                ct = _join_asr_word_tokens(cur_ws, is_cjk)
                ct = to_hk_traditional(ct) if lang in ("zh", "yue") else ct
                if ct:
                    chunks.append({
                        "start": float(cur_ws[0]["start"]),
                        "end": max(float(cur_ws[0]["start"]) + 0.2, float(cur_ws[-1]["end"])),
                        "text": ct,
                        "words": list(cur_ws),
                    })
                cur_ws = [w]
                cur_chars = w_norm
            else:
                cur_ws.append(w)
                cur_chars += w_norm
        if cur_ws:
            ct = _join_asr_word_tokens(cur_ws, is_cjk)
            ct = to_hk_traditional(ct) if lang in ("zh", "yue") else ct
            if ct:
                chunks.append({
                    "start": float(cur_ws[0]["start"]),
                    "end": max(float(cur_ws[0]["start"]) + 0.2, float(cur_ws[-1]["end"])),
                    "text": ct,
                    "words": list(cur_ws),
                })
        if chunks:
            return chunks

    # 2) Fallback when word timestamps are unavailable:
    # For space-delimited EN/KO, split on punctuation then wrap by word boundaries (NEVER 1-word lines!).
    parts: list[str] = []
    if is_cjk:
        raw_parts = [p.strip() for p in re.split(r"[\s，。！？、,;!?]+", txt) if p.strip()]
        for p in raw_parts:
            if len(p) <= max_chars:
                parts.append(p)
            else:
                for k in range(0, len(p), max_chars):
                    parts.append(p[k : k + max_chars])
    else:
        clauses = [p.strip() for p in re.split(r"[，。！？、,;!?]+", txt) if p.strip()]
        for cl in clauses:
            if len(cl) <= max_chars:
                parts.append(cl)
            else:
                cur_line = ""
                for wk in cl.split():
                    cand = f"{cur_line} {wk}".strip() if cur_line else wk
                    if cur_line and len(cand) > max_chars:
                        parts.append(cur_line)
                        cur_line = wk
                    else:
                        cur_line = cand
                if cur_line:
                    parts.append(cur_line)

    if len(parts) <= 1:
        return [{"start": s_real, "end": e_real, "text": txt, "words": words_mapped}]
    total_c = max(1, sum(len(p) for p in parts))
    dur = max(0.4, e_real - s_real)
    cur_t = s_real
    out: list[dict] = []
    for p in parts:
        dt = dur * (len(p) / total_c)
        out.append({"start": round(cur_t, 2), "end": round(cur_t + dt, 2), "text": p, "words": []})
        cur_t += dt
    return out


def _even_words_for_line(text: str, start_s: float, end_s: float) -> list[dict]:
    """Generate clean token timestamps across Chinese (Mandarin & Cantonese 'yue'), Japanese, Korean, English,
    and mixed-script lines (splits CJK/Kana per character while preserving Latin/Hangul words intact!).
    """
    lang = detect_lyrics_language([text])
    if lang in ("zh", "yue", "ja"):
        tokens = re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\u3400-\u4dbf]|[A-Za-z0-9']+|[\uac00-\ud7af]+", text or "")
    else:
        tokens = (text or "").split()
    if not tokens:
        return []
    dur = max(0.4, end_s - start_s)
    weights = [max(1.0, estimate_syllable_units(tk)) for tk in tokens]
    total_w = sum(weights)
    out = []
    acc = 0.0
    for tk, wt in zip(tokens, weights):
        ws = round(start_s + dur * (acc / total_w), 2)
        acc += wt
        we = round(start_s + dur * (acc / total_w), 2)
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
    initial_song_pools: Optional[list[list[str]]] = None,
) -> list[dict]:
    """Live/Concert/Medley-aware alignment:
    1. Transcribes active vocal regions (using GPU BatchedInferencePipeline when available for 4-6x speed!)
       to discover WHICH official lines were actually sung & WHEN.
    2. Groups matched lines into contiguous singing blocks and force-aligns each block on its own
       sliced audio window so skipped sections can NEVER cause drift or crushed timestamps.
    """
    audio_input = audio_16k if len(audio_16k) > 0 else str(vocals_path)
    raw_segments = []
    comp_time_map: list[tuple[float, float, float, float]] = []

    # Fast GPU BatchedInferencePipeline (~2-4s for a 4-min concert on RTX 5060 Ti!)
    # Run on silence-collapsed audio with vad_filter=False & word_timestamps=True so falsetto/reverberant singing
    # is NEVER dropped by speech-trained Silero VAD, and multi-song mashups auto-detect mixed languages!
    is_multi_pool_in = bool(initial_song_pools and len(initial_song_pools) >= 2)
    whisper_lang = _whisper_align_lang(lang)
    transcribe_lang = None if is_multi_pool_in else whisper_lang
    if len(audio_16k) > 0 and len(db_profile) > 0:
        comp_audio, comp_time_map = build_silence_collapsed_audio(audio_16k, db_profile, step_sec, sr=16000)
        if len(comp_audio) == 0 or not comp_time_map:
            comp_audio, comp_time_map = audio_input, []
    else:
        comp_audio, comp_time_map = audio_input, []

    if use_cuda and len(audio_16k) > 0:
        try:
            from faster_whisper import BatchedInferencePipeline, WhisperModel
            if isinstance(model, WhisperModel):
                batched = BatchedInferencePipeline(model=model)
                b_segs, _ = batched.transcribe(
                    comp_audio,
                    language=transcribe_lang,
                    batch_size=8,
                    beam_size=2,
                    vad_filter=False,
                    word_timestamps=True,
                    condition_on_previous_text=False,
                )
                raw_segments = list(b_segs)
        except Exception as _be:
            print(f"[Lyrics] Batched GPU transcribe fallback: {_be}", flush=True)
            raw_segments = []

    if not raw_segments:
        res = model.transcribe(
            comp_audio,
            language=transcribe_lang,
            beam_size=2 if use_cuda else 1,
            vad=False,
            suppress_silence=False,
            condition_on_previous_text=False,
            verbose=False,
        )
        raw_segments = list(res.segments or [])

    norm_segments: list[dict] = []
    asr_segs = []
    for s in raw_segments:
        raw_t = (s.text or "").strip()
        s_raw, e_raw = float(s.start), float(s.end)
        s_real = _map_comp_to_real(s_raw, comp_time_map, is_end=False) if comp_time_map else s_raw
        e_real = _map_comp_to_real(e_raw, comp_time_map, is_end=True) if comp_time_map else e_raw
        e_real = max(s_real + 0.05, e_real)
        words_mapped = []
        for w in (getattr(s, "words", None) or []):
            ws_r, we_r = float(w.start), float(w.end)
            ws_real = _map_comp_to_real(ws_r, comp_time_map, is_end=False) if comp_time_map else ws_r
            we_real = _map_comp_to_real(we_r, comp_time_map, is_end=True) if comp_time_map else we_r
            words_mapped.append({"word": getattr(w, "word", "") or "", "start": ws_real, "end": max(ws_real, we_real)})
        norm_segments.append({"start": s_real, "end": e_real, "text": raw_t, "words": words_mapped})

        txt = to_hk_traditional(raw_t) if lang in ("zh", "yue") else raw_t
        if not txt or HALLUCINATION_RE.search(txt) or GARBAGE_CHARS_RE.search(txt):
            continue
        if not segment_has_vocal_energy(db_profile, step_sec, s_real, e_real):
            continue
        for sub in _split_long_asr_segment(s_real, e_real, txt, words_mapped, lang):
            sub_t = (sub.get("text") or "").strip()
            if not sub_t or HALLUCINATION_RE.search(sub_t) or GARBAGE_CHARS_RE.search(sub_t):
                continue
            if segment_has_vocal_energy(db_profile, step_sec, sub["start"], sub["end"]):
                asr_segs.append({"start": sub["start"], "end": sub["end"], "text": sub_t})

    if not asr_segs:
        return []

    def _match_against_pool(pool_lines: list[str], pools_list: Optional[list[list[str]]] = None) -> tuple[list[dict], list[dict]]:
        u_off = list(dict.fromkeys(pool_lines))
        n_map: dict[str, set[str]] = {}
        for idx in range(len(pool_lines) - 1):
            n_map.setdefault(pool_lines[idx], set()).add(pool_lines[idx + 1])

        # Track which song pool(s) each lyric line belongs to so multi-song medleys keep song continuity
        line_to_pools: dict[str, set[int]] = {}
        if pools_list and len(pools_list) >= 2:
            for pi, pl in enumerate(pools_list):
                for ln in pl:
                    line_to_pools.setdefault(ln, set()).add(pi)

        def _best_single(
            text: str,
            prev_l: Optional[str],
            cur_pool: int = -1,
            restrict_pool: Optional[int] = None,
        ) -> Optional[tuple[float, str, float, bool]]:
            c_len = len(_norm_lyrics_cmp(text))
            if c_len < 2:
                return None
            best = None
            for off in u_off:
                if restrict_pool is not None and restrict_pool not in line_to_pools.get(off, set()):
                    continue
                o_len = len(_norm_lyrics_cmp(off))
                if o_len == 0 or c_len > o_len * 2.1 or o_len > c_len * 2.6:
                    continue
                sim = _lyric_similarity(text, off)
                is_next = bool(prev_l and off in n_map.get(prev_l, set()))
                bonus = 0.22 if (is_next and sim >= 0.22) else (0.12 if is_next else 0.0)
                if cur_pool >= 0 and cur_pool in line_to_pools.get(off, set()) and sim >= 0.22:
                    bonus += 0.08
                score = sim + bonus
                if best is None or score > best[0]:
                    best = (score, off, sim, is_next)
            return best

        m_out: list[dict] = []
        u_gaps: list[dict] = []
        idx_i = 0
        prev_l: Optional[str] = None
        cur_pool = -1
        while idx_i < len(asr_segs):
            b1 = _best_single(asr_segs[idx_i]["text"], prev_l, cur_pool=cur_pool)
            chosen_span = 1
            chosen = b1

            if idx_i + 1 < len(asr_segs) and (asr_segs[idx_i + 1]["start"] - asr_segs[idx_i]["end"]) <= 1.8:
                comb = f"{asr_segs[idx_i]['text']} {asr_segs[idx_i + 1]['text']}"
                b2 = _best_single(comb, prev_l, cur_pool=cur_pool)
                b_next = _best_single(asr_segs[idx_i + 1]["text"], b1[1] if b1 else prev_l, cur_pool=cur_pool)
                s1_raw = b1[2] if b1 else 0.0
                s_next_raw = b_next[2] if b_next else 0.0
                if b2 and b2[2] > max(s1_raw, s_next_raw) + 0.06:
                    chosen_span = 2
                    chosen = b2

            if chosen and (chosen[2] >= 0.30 or (chosen[3] and chosen[2] >= 0.20)):
                _, off, raw_sim, _ = chosen
                ts = asr_segs[idx_i]["start"]
                te = asr_segs[idx_i + chosen_span - 1]["end"]
                off_pools = line_to_pools.get(off, set())
                if off_pools and cur_pool not in off_pools:
                    cur_pool = min(off_pools)
                # Preserve legitimate back-to-back identical lines (e.g. "Let it go" / "Let it go") when
                # the official lyrics genuinely repeat that line consecutively!
                is_true_repeat = off in n_map.get(off, set())
                if m_out and m_out[-1]["text"] == off and (ts - m_out[-1]["end"]) < 7.5 and not is_true_repeat:
                    # Extend the existing line's end timestamp instead of discarding its second half!
                    m_out[-1]["end"] = max(m_out[-1]["end"], te)
                else:
                    m_out.append({
                        "start": ts,
                        "end": te,
                        "text": off,
                        "sim": raw_sim,
                        "_asr_text": " ".join(asr_segs[k]["text"] for k in range(idx_i, idx_i + chosen_span)),
                    })
                    prev_l = off
                idx_i += chosen_span
            else:
                u_gaps.append(asr_segs[idx_i])
                idx_i += 1

        # De-intrude isolated 1-line foreign-song jumps (A -> B -> A) in multi-song medleys!
        if pools_list and len(pools_list) >= 2 and len(m_out) >= 3:
            def _primary_pool(txt: str) -> int:
                ps = line_to_pools.get(txt, set())
                return min(ps) if len(ps) == 1 else -1

            cleaned_m: list[dict] = []
            for i, cur in enumerate(m_out):
                p_cur = _primary_pool(cur["text"])
                p_prev = _primary_pool(m_out[i - 1]["text"]) if i > 0 else -1
                p_next = _primary_pool(m_out[i + 1]["text"]) if i + 1 < len(m_out) else -1
                if p_cur >= 0 and p_prev >= 0 and p_prev == p_next and p_cur != p_prev:
                    # Isolated 1-line foreign intrusion -> re-match strictly inside surrounding song pool p_prev!
                    alt = _best_single(cur.get("_asr_text", ""), m_out[i - 1]["text"], cur_pool=p_prev, restrict_pool=p_prev)
                    if alt and alt[2] >= 0.26:
                        cleaned_m.append({"start": cur["start"], "end": cur["end"], "text": alt[1], "sim": alt[2]})
                    # Otherwise drop the foreign intrusion cleanly so internal gap recovery restores the real line!
                else:
                    cleaned_m.append({k: v for k, v in cur.items() if k != "_asr_text"})
            m_out = cleaned_m
        else:
            m_out = [{k: v for k, v in cur.items() if k != "_asr_text"} for cur in m_out]

        return m_out, u_gaps

    song_pools: list[list[str]] = (
        [list(p) for p in initial_song_pools if p]
        if initial_song_pools
        else ([list(official_lines)] if official_lines else [])
    )
    # Structural invariant: official_lines and song_pools MUST share identical line strings (zero split-brain!)
    if song_pools:
        official_lines = [ln for p in song_pools for ln in p]
    matched, unmatched_gaps = _match_against_pool(official_lines, song_pools)
    matched_asr_count = len(asr_segs) - len(unmatched_gaps)

    # Smart Shazam rescue: if title-based lyrics poorly matched (<35%) — e.g. clickbait title or unlisted Mashup —
    # run Shazam multi-track fingerprinting and re-match the ALREADY-transcribed ASR segments (0 extra Whisper time!)
    if not shazam_already_tried and (not matched or (len(asr_segs) >= 6 and (matched_asr_count / len(asr_segs)) < 0.35)):
        shz_lines, shz_tracks = shazam_scan_and_fetch_lyrics(
            audio_16k, db_profile, step_sec, inst_path=inst_path, scan_mashup=True
        )
        if shz_lines:
            shz_pools = [tr["official_lines"] for tr in shz_tracks if tr.get("official_lines")] or [list(shz_lines)]
            shz_combined = [ln for p in shz_pools for ln in p]
            m2, g2 = _match_against_pool(shz_combined, shz_pools)
            if (len(asr_segs) - len(g2)) > matched_asr_count:
                official_lines = shz_combined
                song_pools = shz_pools
                matched, unmatched_gaps = m2, g2
                matched_asr_count = len(asr_segs) - len(unmatched_gaps)

    # Mashup / Medley Lyric-Phrase Rescue: if there is a sustained unrecognized singing cluster (>= 4 segments, >= 16s)
    # — e.g. Song #2 in a live/acoustic Mashup that Shazam couldn't fingerprint — identify it by sung lyric phrases!
    # Strict verification gate: ONLY accept a rescued song if it genuinely matches >=3 (>=4 when title already gave multi-songs)
    # distinct ASR segments in that cluster at high similarity, so noisy rock/English bridges NEVER inject random songs!
    def _find_sustained_unmatched_clusters(gaps: list[dict]) -> list[list[dict]]:
        clusters: list[list[dict]] = []
        for g in gaps:
            if not clusters or (g["start"] - clusters[-1][-1]["end"]) > 10.0:
                clusters.append([g])
            else:
                clusters[-1].append(g)
        return [c for c in clusters if len(c) >= 4 and (c[-1]["end"] - c[0]["start"]) >= 16.0]

    def _verify_rescued_pool(cl: list[dict], rescued_lines: list[str], strict_multi: bool) -> bool:
        min_hits = 4 if strict_multi else 3
        min_sim = 0.48 if strict_multi else 0.42
        hits = 0
        for g in cl:
            g_txt = g.get("text", "")
            if len(_norm_lyrics_cmp(g_txt)) < 3:
                continue
            best_s = max((_lyric_similarity(g_txt, rl) for rl in rescued_lines), default=0.0)
            if best_s >= min_sim:
                hits += 1
        return hits >= min_hits

    extra_clusters = _find_sustained_unmatched_clusters(unmatched_gaps)
    if extra_clusters:
        added_pool = False
        has_multi_title = bool(initial_song_pools and len(initial_song_pools) >= 2)
        for cl in extra_clusters[:3]:
            rescued_pools = search_official_lyrics_pools_by_sung_phrases(cl)
            for r_pool in rescued_pools:
                if r_pool and _verify_rescued_pool(cl, r_pool, strict_multi=has_multi_title):
                    official_lines = list(official_lines) + r_pool
                    song_pools.append(list(r_pool))
                    added_pool = True
        if added_pool:
            matched, unmatched_gaps = _match_against_pool(official_lines, song_pools)
            matched_asr_count = len(asr_segs) - len(unmatched_gaps)

    if not matched:
        return []

    # Script-aware Parody / Rewritten-lyrics detection:
    # - For space-delimited EN/KO: use word-set overlap (prevents 45% false char-3gram overlap on English parodies!)
    # - For CJK/Cantonese: use char 3-gram overlap with homophone-tolerant threshold (prevents Cantonese live songs from being falsely discarded!)
    is_cjk_song = lang in ("zh", "yue", "ja")
    if is_cjk_song:
        asr_blob = "".join(_norm_lyrics_cmp(s["text"]) for s in asr_segs)
        off_blob = "".join(_norm_lyrics_cmp(ln) for ln in official_lines)
        overlap_cov = 0.0
        if len(asr_blob) >= 12 and len(off_blob) >= 12:
            asr_3g = {asr_blob[k : k + 3] for k in range(len(asr_blob) - 2)}
            off_3g = {off_blob[k : k + 3] for k in range(len(off_blob) - 2)}
            if asr_3g:
                overlap_cov = len(asr_3g & off_3g) / len(asr_3g)
        match_thr = 0.25 if lang == "yue" else 0.35
        cov_thr = 0.22 if lang == "yue" else 0.35
    else:
        asr_words = [w for s in asr_segs for w in re.findall(r"[a-z0-9'\uac00-\ud7af]{2,}", (s["text"] or "").lower())]
        off_words = {w for ln in official_lines for w in re.findall(r"[a-z0-9'\uac00-\ud7af]{2,}", (ln or "").lower())}
        overlap_cov = (sum(1 for w in asr_words if w in off_words) / max(1, len(asr_words))) if len(asr_words) >= 8 else 1.0
        match_thr = 0.35
        cov_thr = 0.38

    if len(asr_segs) >= 6 and (matched_asr_count / len(asr_segs)) < match_thr and overlap_cov < cov_thr:
        print(
            f"[Lyrics] Low official-lyrics match ({matched_asr_count}/{len(asr_segs)}, cov={overlap_cov:.2f}); "
            "treating as rewritten/adapted lyrics and using actual sung transcript.",
            flush=True,
        )
        direct_lines: list[dict] = []
        for s in norm_segments:
            raw_t = (s["text"] or "").strip()
            txt = to_hk_traditional(raw_t) if lang in ("zh", "yue") else raw_t
            if not txt or HALLUCINATION_RE.search(txt) or GARBAGE_CHARS_RE.search(txt):
                continue
            s_start, s_end = float(s["start"]), float(s["end"])
            if s_end < first_onset or not segment_has_vocal_energy(db_profile, step_sec, s_start, s_end):
                continue
            for sub in _split_long_asr_segment(s_start, s_end, txt, s.get("words") or [], lang):
                stxt = (sub.get("text") or "").strip()
                if not stxt or HALLUCINATION_RE.search(stxt) or GARBAGE_CHARS_RE.search(stxt):
                    continue
                words_out = []
                for w in (sub.get("words") or []):
                    wt = (w["word"] or "").strip()
                    if not wt or GARBAGE_CHARS_RE.search(wt):
                        continue
                    ws = round(max(first_onset, float(w["start"])), 3)
                    we = round(max(ws, float(w["end"])), 3)
                    words_out.append({
                        "start": ws,
                        "end": we,
                        "word": to_hk_traditional(wt) if lang in ("zh", "yue") else wt,
                    })
                s_clamped = round(max(first_onset, float(sub["start"])), 2)
                e_clamped = round(max(s_clamped + 0.4, float(sub["end"])), 2)
                direct_lines.append({
                    "start": s_clamped,
                    "end": e_clamped,
                    "text": stxt,
                    "words": words_out or _even_words_for_line(stxt, s_clamped, e_clamped),
                })
        audio_dur = (len(audio_16k) / 16000.0) if len(audio_16k) > 0 else 0.0
        return refine_alignment_with_acoustics(direct_lines, db_profile, step_sec, audio_dur, first_onset)

    total_sec_est = (len(audio_16k) / 16000.0) if len(audio_16k) > 0 else (len(db_profile) * step_sec)
    gap_thr = _adaptive_vocal_threshold_db(db_profile)

    def _active_vocal_span(t_start: float, t_end: float) -> tuple[float, float, float]:
        """Returns (active_vocal_seconds, first_active_t, last_active_t) inside [t_start, t_end]."""
        if len(db_profile) == 0 or t_end <= t_start:
            return (0.0, t_start, t_end)
        i0 = max(0, int(t_start / step_sec))
        i1 = min(len(db_profile), int(t_end / step_sec))
        if i1 <= i0:
            return (0.0, t_start, t_end)
        mask = db_profile[i0:i1] > gap_thr
        active_idx = np.flatnonzero(mask)
        if len(active_idx) == 0:
            return (0.0, t_start, t_end)
        return (
            float(len(active_idx)) * step_sec,
            (i0 + int(active_idx[0])) * step_sec,
            (i0 + int(active_idx[-1]) + 1) * step_sec,
        )

    # Assign each matched anchor to its originating song pool (with normalized fallback so punctuation never splits pools),
    # partition into contiguous song runs, and resolve strictly-monotonic lyric indices via Weighted DP!
    pool_exact_sets: list[set[str]] = [set(pl) for pl in song_pools]
    pool_norm_sets: list[set[str]] = [{_norm_lyrics_cmp(ln) for ln in pl if _norm_lyrics_cmp(ln)} for pl in song_pools]

    raw_song_runs: list[tuple[int, list[dict]]] = []
    cur_pool_i = -1
    for cur in matched:
        txt = cur["text"]
        ntxt = _norm_lyrics_cmp(txt)
        chosen_p = -1
        if 0 <= cur_pool_i < len(song_pools) and (txt in pool_exact_sets[cur_pool_i] or (ntxt and ntxt in pool_norm_sets[cur_pool_i])):
            chosen_p = cur_pool_i
        else:
            # Prefer a pool where this text uniquely belongs before shared lines
            matching_ps = [
                pi for pi in range(len(song_pools))
                if txt in pool_exact_sets[pi] or (ntxt and ntxt in pool_norm_sets[pi])
            ]
            if matching_ps:
                chosen_p = matching_ps[0]
        if chosen_p >= 0:
            cur_pool_i = chosen_p
        if not raw_song_runs or chosen_p != raw_song_runs[-1][0] or chosen_p < 0:
            raw_song_runs.append((chosen_p, [cur]))
        else:
            raw_song_runs[-1][1].append(cur)

    # Apply Weighted Monotonic Anchor Consensus per song run (eliminates isolated out-of-order ASR outliers!)
    song_runs: list[list[dict]] = []
    for pi, r_items in raw_song_runs:
        if pi < 0 or pi >= len(song_pools):
            song_runs.append([{**it, "_pool": -1, "_idx": -1} for it in r_items])
        else:
            for sub_r in _select_monotonic_run_anchors(r_items, song_pools[pi]):
                if sub_r:
                    song_runs.append([{**it, "_pool": pi} for it in sub_r])

    # MACRO-BOUNDARY -> PER-SONG FULL SILENCE-COLLAPSED ALIGNMENT:
    # For sequential multi-song live sets (e.g. 2-song TV live / concert setlists where each song is sung in full),
    # slice the audio cleanly at the inter-song acoustic boundary and run full silence-collapsed model.align()
    # per song (0% dropped rock/fast lines, chronological order even if sung reversed, 0-anchor show brackets pruned!).
    # Genuine interleaved/abridged medleys automatically fall through to section-anchored alignment!
    valid_pool_runs = [r for r in song_runs if r and r[0].get("_pool", -1) >= 0 and len(r) >= 4]
    pool_ids_seq = [r[0]["_pool"] for r in valid_pool_runs]
    is_sequential_distinct_songs = (
        len(valid_pool_runs) >= 2
        and len(valid_pool_runs) == len(set(pool_ids_seq))
        and len(audio_16k) > 0
        and len(db_profile) > 0
    )

    def _find_inter_song_split(t_left: float, t_right: float, rem_left: int, lead_right: int) -> float:
        if t_right <= t_left + 0.4:
            return 0.5 * (t_left + t_right)
        i0 = max(0, int(t_left / step_sec))
        i1 = min(len(db_profile), int(t_right / step_sec))
        if i1 <= i0 + 4:
            frac = max(0.15, min(0.85, (rem_left + 0.5) / max(1.0, rem_left + lead_right + 1.0)))
            return t_left + frac * (t_right - t_left)
        seg_quiet = db_profile[i0:i1] <= gap_thr
        # Find longest quiet gap >= 1.0s inside [t_left, t_right]
        best_len = 0
        best_mid = -1
        cur_s = -1
        for k, q in enumerate(seg_quiet):
            if q and cur_s < 0:
                cur_s = k
            elif not q and cur_s >= 0:
                if (k - cur_s) > best_len:
                    best_len = k - cur_s
                    best_mid = cur_s + (k - cur_s) // 2
                cur_s = -1
        if cur_s >= 0 and (len(seg_quiet) - cur_s) > best_len:
            best_len = len(seg_quiet) - cur_s
            best_mid = cur_s + (len(seg_quiet) - cur_s) // 2
        if best_len * step_sec >= 1.0 and best_mid >= 0:
            return (i0 + best_mid) * step_sec
        # Seamless transition: split proportionally by active vocal frames
        act_idx = np.flatnonzero(~seg_quiet)
        frac = max(0.15, min(0.85, (rem_left + 0.5) / max(1.0, rem_left + lead_right + 1.0)))
        if len(act_idx) >= 2:
            pick = act_idx[min(len(act_idx) - 1, int(round(frac * (len(act_idx) - 1))))]
            return (i0 + int(pick)) * step_sec
        return t_left + frac * (t_right - t_left)

    if is_sequential_distinct_songs:
        # Compute macro boundaries [s_i, e_i] for each sequential song run
        macro_slices: list[tuple[float, float, int, list[dict]]] = []
        split_points: list[float] = [0.0]
        for idx_r in range(len(valid_pool_runs) - 1):
            r_cur = valid_pool_runs[idx_r]
            r_nxt = valid_pool_runs[idx_r + 1]
            pl_cur = song_pools[r_cur[0]["_pool"]]
            rem_cur = max(0, len(pl_cur) - 1 - r_cur[-1]["_idx"])
            lead_nxt = max(0, r_nxt[0]["_idx"])
            sp = _find_inter_song_split(float(r_cur[-1]["end"]), float(r_nxt[0]["start"]), rem_cur, lead_nxt)
            split_points.append(sp)
        split_points.append(total_sec_est)

        # Verify each song looks like a full performance (not a tiny 30s abridged snippet!)
        all_full_songs = True
        for idx_r, r_cur in enumerate(valid_pool_runs):
            pi = r_cur[0]["_pool"]
            pl_cur = song_pools[pi]
            sl_s, sl_e = split_points[idx_r], split_points[idx_r + 1]
            span_cov = (r_cur[-1]["_idx"] - r_cur[0]["_idx"] + 1) / max(1, len(pl_cur))
            if (sl_e - sl_s) < 90.0 and span_cov < 0.55:
                all_full_songs = False
                break
            macro_slices.append((sl_s, sl_e, pi, r_cur))

        if all_full_songs and macro_slices:
            macro_out: list[dict] = []
            macro_ok = True
            for sl_s, sl_e, pi, r_cur in macro_slices:
                pl_cur = song_pools[pi]
                s_lang = detect_lyrics_language(pl_cur)
                di0 = max(0, int(sl_s / step_sec))
                di1 = min(len(db_profile), int(sl_e / step_sec))
                a0 = max(0, int(sl_s * 16000))
                a1 = min(len(audio_16k), int(sl_e * 16000))
                slice_audio = audio_16k[a0:a1]
                slice_db = db_profile[di0:di1]
                if len(slice_audio) < 16000 * 5 or len(slice_db) < 10:
                    macro_ok = False
                    break
                # Find local onset inside this song's slice
                sl_thr = _adaptive_vocal_threshold_db(slice_db)
                sl_onset = 0.0
                for k_o in range(max(0, len(slice_db) - 6)):
                    if slice_db[k_o] > sl_thr and int(np.sum(slice_db[k_o : k_o + 6] > sl_thr)) >= 4:
                        sl_onset = max(0.0, k_o * step_sec - 0.15)
                        break
                comp_sl, sl_tmap = build_silence_collapsed_audio(slice_audio, slice_db, step_sec, sr=16000)
                try:
                    al_sl = model.align(
                        comp_sl,
                        "\n".join(pl_cur),
                        language=_whisper_align_lang(s_lang),
                        original_split=True,
                        fast_mode=(not use_cuda),
                        verbose=False,
                    )
                except Exception:
                    macro_ok = False
                    break
                sl_needs_live, sl_reason = _alignment_needs_live_rematch(
                    al_sl, len(slice_audio) / 16000.0, slice_db, step_sec, time_map=sl_tmap, lang=s_lang
                )
                if sl_needs_live:
                    print(f"[Lyrics] Macro-slice song #{pi + 1} needs live rematch ({sl_reason}); falling back to section-anchored.", flush=True)
                    macro_ok = False
                    break
                sl_lines: list[dict] = []
                for seg in (al_sl.segments or []):
                    stxt = (seg.text or "").strip()
                    if not stxt:
                        continue
                    rs_l = _map_comp_to_real(float(seg.start), sl_tmap, is_end=False) if sl_tmap else float(seg.start)
                    re_l = _map_comp_to_real(float(seg.end), sl_tmap, is_end=True) if sl_tmap else float(seg.end)
                    abs_s = round(max(first_onset, sl_s + max(sl_onset, rs_l)), 2)
                    abs_e = round(max(abs_s + 0.35, sl_s + re_l), 2)
                    ws_out = []
                    for w in (seg.words or []):
                        wt = (w.word or "").strip()
                        if not wt:
                            continue
                        ws_r = _map_comp_to_real(float(w.start), sl_tmap, is_end=False) if sl_tmap else float(w.start)
                        we_r = _map_comp_to_real(float(w.end), sl_tmap, is_end=True) if sl_tmap else float(w.end)
                        ws_a = round(max(abs_s, sl_s + ws_r), 3)
                        we_a = round(max(ws_a, sl_s + we_r), 3)
                        ws_out.append({"start": ws_a, "end": we_a, "word": wt})
                    sl_lines.append({"start": abs_s, "end": abs_e, "text": stxt, "words": ws_out})
                if len(sl_lines) < 4:
                    macro_ok = False
                    break
                macro_out.extend(sl_lines)
            if macro_ok and len(macro_out) >= 8:
                print(f"[Lyrics] Macro-boundary per-song full alignment succeeded ({len(macro_slices)} songs, {len(macro_out)} lines)!", flush=True)
                return refine_alignment_with_acoustics(macro_out, db_profile, step_sec, total_sec_est, first_onset)

    filled: list[dict] = []
    for r_i, run in enumerate(song_runs):
        pi = run[0]["_pool"]
        if pi < 0 or pi >= len(song_pools):
            filled.extend({k2: v2 for k2, v2 in it.items() if not k2.startswith("_")} for it in run)
            continue
        pl = song_pools[pi]
        prev_bound = (song_runs[r_i - 1][-1]["end"] + 0.5) if r_i > 0 else first_onset
        next_bound = (song_runs[r_i + 1][0]["start"] - 0.5) if (r_i + 1 < len(song_runs)) else total_sec_est

        # 1) Leading lines recovery (0 .. first_idx - 1) walking BACKWARD across contiguous singing
        first_item = run[0]
        first_idx = first_item["_idx"]
        if first_idx > 0 and first_item["start"] - prev_bound >= 1.5:
            lead_n = first_idx
            max_lead_window = min(first_item["start"] - prev_bound, max(12.0, lead_n * 5.2))
            win_s = max(prev_bound, first_item["start"] - max_lead_window)
            act_sec, v_first, _ = _contiguous_vocal_span(
                db_profile, step_sec, gap_thr, win_s, first_item["start"], direction="backward", max_silent_gap=8.0
            )
            if act_sec >= max(0.8, 0.55 * lead_n):
                span_s = max(prev_bound, v_first)
                span_e = first_item["start"]
                if span_e > span_s + 0.5:
                    step_t = (span_e - span_s) / lead_n
                    for m_i in range(lead_n):
                        filled.append({
                            "start": round(span_s + m_i * step_t, 2),
                            "end": round(span_s + (m_i + 1) * step_t, 2),
                            "text": pl[m_i],
                            "sim": 0.0,
                        })

        # 2) Anchors + internal gap recovery (up to 30 missing lines within the same song!)
        # Lowered floor (0.65s/line, 0.45s vocal/line) so fast rock/rap lines are NEVER dropped before alignment!
        for k, cur in enumerate(run):
            filled.append({k2: v2 for k2, v2 in cur.items() if not k2.startswith("_")})
            if k + 1 < len(run):
                nxt = run[k + 1]
                cur_idx, nxt_idx = cur["_idx"], nxt["_idx"]
                missing_n = (nxt_idx - cur_idx - 1) if (0 <= cur_idx < nxt_idx <= cur_idx + 31) else 0
                gap_s, gap_e = cur["end"], nxt["start"]
                if 1 <= missing_n <= 30 and (gap_e - gap_s) >= max(0.75, 0.65 * missing_n):
                    act_sec, v_first, v_last = _active_vocal_span(gap_s, gap_e)
                    if act_sec >= max(0.50, 0.45 * missing_n):
                        span_s = max(gap_s, v_first)
                        span_e = min(gap_e, max(span_s + 0.4, v_last))
                        step_t = (span_e - span_s) / missing_n
                        for m_i in range(missing_n):
                            filled.append({
                                "start": round(span_s + m_i * step_t, 2),
                                "end": round(span_s + (m_i + 1) * step_t, 2),
                                "text": pl[cur_idx + 1 + m_i],
                                "sim": 0.0,
                            })

        # 3) Trailing lines recovery (last_idx + 1 .. len(pl) - 1) walking FORWARD across contiguous singing
        last_item = run[-1]
        last_idx = last_item["_idx"]
        if 0 <= last_idx < len(pl) - 1 and next_bound - last_item["end"] >= 1.5:
            trail_n = len(pl) - 1 - last_idx
            max_trail_window = min(next_bound - last_item["end"], max(12.0, trail_n * 5.2))
            win_e = min(next_bound, last_item["end"] + max_trail_window)
            act_sec, _, v_last = _contiguous_vocal_span(
                db_profile, step_sec, gap_thr, last_item["end"], win_e, direction="forward", max_silent_gap=8.0
            )
            if act_sec >= max(0.8, 0.55 * trail_n):
                span_s = last_item["end"]
                span_e = min(next_bound, v_last)
                if span_e > span_s + 0.5:
                    step_t = (span_e - span_s) / trail_n
                    for m_i in range(trail_n):
                        filled.append({
                            "start": round(span_s + m_i * step_t, 2),
                            "end": round(span_s + (m_i + 1) * step_t, 2),
                            "text": pl[last_idx + 1 + m_i],
                            "sim": 0.0,
                        })

    matched = filled

    # Exclude any unmatched ASR gap whose timestamp was ALREADY covered by recovered official lyrics!
    def _covered_by_matched(t_mid: float) -> bool:
        for m in matched:
            if (m["start"] - 1.0) <= t_mid <= (m["end"] + 1.0):
                return True
        return False

    remaining_gaps = [g for g in unmatched_gaps if not _covered_by_matched(0.5 * (g["start"] + g["end"]))]

    for cl in _find_sustained_unmatched_clusters(remaining_gaps):
        for g in cl:
            matched.append({"start": g["start"], "end": g["end"], "text": g["text"], "sim": 0.5})
    matched.sort(key=lambda x: x["start"])

    # Group into contiguous singing blocks (split on gaps > 4.5s) and align each block on its own audio slice!
    blocks: list[list[dict]] = []
    for item in matched:
        if not blocks or (item["start"] - blocks[-1][-1]["end"]) > 4.5:
            blocks.append([item])
        else:
            blocks[-1].append(item)

    total_sec = len(audio_16k) / 16000.0 if len(audio_16k) > 0 else 0.0
    blk_thr_db = _adaptive_vocal_threshold_db(db_profile, margin_db=-1.0) if len(db_profile) > 0 else None
    final_lines: list[dict] = []
    for b_idx, blk in enumerate(blocks):
        prev_end_bound = (blocks[b_idx - 1][-1]["end"] + 0.15) if b_idx > 0 else first_onset
        next_start_bound = (blocks[b_idx + 1][0]["start"] - 0.15) if (b_idx + 1 < len(blocks)) else total_sec
        b_start = max(first_onset, prev_end_bound, blk[0]["start"] - 1.5)
        b_end = min(total_sec, next_start_bound, blk[-1]["end"] + 4.5) if total_sec > 0 else (blk[-1]["end"] + 4.5)
        if len(audio_16k) > 0 and b_end > b_start + 0.5:
            chunk = audio_16k[int(b_start * 16000) : int(b_end * 16000)]
            blk_text = "\n".join(x["text"] for x in blk)
            blk_time_map: list[tuple[float, float, float, float]] = []
            align_chunk = chunk
            if (b_end - b_start) >= 20.0 and len(db_profile) > 0:
                di0 = max(0, int(b_start / step_sec))
                di1 = min(len(db_profile), int(b_end / step_sec))
                if di1 > di0 + 20:
                    align_chunk, blk_time_map = build_silence_collapsed_audio(
                        chunk, db_profile[di0:di1], step_sec, thr_db=blk_thr_db
                    )
            try:
                al = model.align(
                    align_chunk,
                    blk_text,
                    language=_whisper_align_lang(lang),
                    original_split=True,
                    fast_mode=(not use_cuda),
                    verbose=False,
                )
                al_segs = list(al.segments or [])
            except Exception:
                al_segs = []
        else:
            al_segs = []
            blk_time_map = []

        if len(al_segs) == len(blk):
            for seg, ref in zip(al_segs, blk):
                seg_s = _map_comp_to_real(float(seg.start), blk_time_map, is_end=False) if blk_time_map else float(seg.start)
                seg_e = _map_comp_to_real(float(seg.end), blk_time_map, is_end=True) if blk_time_map else float(seg.end)
                raw_s = b_start + seg_s
                raw_e = b_start + seg_e
                raw_dur = raw_e - raw_s
                n_syl = estimate_syllable_units(ref["text"])
                is_real_anchor = float(ref.get("sim") or 0.0) > 0.0
                # Only drop a collapsed recovered line when its reference span ALSO lacks vocal energy!
                # If the singer IS singing inside [ref.start, ref.end], keep the bounded scaffold for Qwen3 refinement!
                if (not is_real_anchor) and (raw_dur <= 0.12 or (n_syl >= 4.0 and (raw_dur / n_syl) < 0.035)):
                    if not segment_has_vocal_energy(db_profile, step_sec, float(ref["start"]), float(ref["end"])):
                        continue
                    s_abs = round(max(first_onset, float(ref["start"])), 2)
                    e_abs = round(max(s_abs + 0.5, float(ref["end"])), 2)
                    final_lines.append({
                        "start": s_abs,
                        "end": e_abs,
                        "text": ref["text"],
                        "words": _even_words_for_line(ref["text"], s_abs, e_abs),
                    })
                    continue
                s_abs = round(max(first_onset, raw_s), 2)
                e_abs = round(max(s_abs + 0.35, raw_e), 2)
                anchor_drifted = is_real_anchor and abs(s_abs - ref["start"]) > 6.0
                totally_collapsed = raw_dur <= 0.05
                if anchor_drifted or (is_real_anchor and totally_collapsed):
                    s_abs = round(max(first_onset, ref["start"]), 2)
                    e_abs = round(max(s_abs + 0.6, ref["end"]), 2)
                    words_out = _even_words_for_line(ref["text"], s_abs, e_abs)
                else:
                    words_out = []
                    for w in (seg.words or []):
                        wt = (w.word or "").strip()
                        if not wt:
                            continue
                        ws_l = _map_comp_to_real(float(w.start), blk_time_map, is_end=False) if blk_time_map else float(w.start)
                        we_l = _map_comp_to_real(float(w.end), blk_time_map, is_end=True) if blk_time_map else float(w.end)
                        ws = round(max(s_abs, b_start + ws_l), 3)
                        we = round(max(ws, b_start + we_l), 3)
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

    # Ensure strictly monotonic non-overlapping line boundaries before acoustic refinement
    for idx in range(1, len(final_lines)):
        if final_lines[idx]["start"] < final_lines[idx - 1]["start"] + 0.2:
            final_lines[idx]["start"] = round(final_lines[idx - 1]["end"], 2)
            final_lines[idx]["end"] = round(max(final_lines[idx]["start"] + 0.4, final_lines[idx]["end"]), 2)
        elif final_lines[idx]["start"] < final_lines[idx - 1]["end"]:
            final_lines[idx - 1]["end"] = round(max(final_lines[idx - 1]["start"] + 0.3, final_lines[idx]["start"]), 2)

    return refine_alignment_with_acoustics(final_lines, db_profile, step_sec, total_sec, first_onset)


QWEN3_LANG_MAP = {
    "zh": "Chinese",
    "yue": "Cantonese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
}


def _detect_cantonese_vs_mandarin_audio(
    audio_16k: np.ndarray,
    first_onset: float,
    lines: list[dict],
    sub_lang: str = "",
    meta_lang: str = "",
    preloaded_model=None,
) -> str:
    """Resolve Chinese lyrics ('zh' vs 'yue') so Mandarin songs use Qwen3 'Chinese' and Cantonese songs use 'Cantonese'!
    Samples actual CJK singing (skipping intro banter/English intros) and reuses any already-loaded Whisper model.
    """
    lang_hint = sub_lang or meta_lang
    base = detect_lyrics_language([ln.get("text", "") for ln in lines], sub_lang=lang_hint)
    if base != "zh":
        return base
    # If meta_lang explicitly indicates Cantonese (yue / zh-HK), resolve immediately (0s)
    ml = (meta_lang or "").strip().lower()
    if ml.startswith("yue") or (ml.startswith("zh") and "hk" in ml):
        return "yue"

    # For literary Chinese (e.g. 林夕/黃偉文 Cantopop vs Mandopop), run a fast 24s Whisper language-ID check.
    if len(audio_16k) >= 16000 * 10:
        m_owned = None
        try:
            # Sample from actual CJK singing (25% into CJK lines) to avoid spoken MC banter or English intros!
            cjk_starts = [
                float(ln["start"])
                for ln in lines
                if isinstance(ln, dict) and ln.get("start") is not None and re.search(r"[\u4e00-\u9fff]", ln.get("text", ""))
            ]
            sample_start = cjk_starts[len(cjk_starts) // 4] if len(cjk_starts) >= 4 else first_onset
            s0 = int(max(0.0, sample_start) * 16000)
            if len(audio_16k) - s0 < 16000 * 8:
                s0 = int(max(0.0, first_onset) * 16000)
            s1 = min(len(audio_16k), s0 + 16000 * 24)
            clip = audio_16k[s0:s1]
            if len(clip) >= 16000 * 5:
                det = ""
                if preloaded_model is not None:
                    if hasattr(preloaded_model, "transcribe_original"):
                        _, info = preloaded_model.transcribe_original(
                            clip, beam_size=1, vad_filter=False, condition_on_previous_text=False
                        )
                        det = (getattr(info, "language", "") or "").lower()
                    else:
                        res_id = preloaded_model.transcribe(
                            clip, beam_size=1, vad=False, condition_on_previous_text=False, verbose=False
                        )
                        det = (getattr(res_id, "language", "") or "").lower()
                else:
                    add_nvidia_dll_dirs()
                    free_mb = query_free_vram_mb()
                    dev = "cuda" if free_mb >= 2000 else "cpu"
                    m_owned = _load_faster_whisper_model(dev)
                    _, info = m_owned.transcribe(clip, beam_size=1, vad_filter=False, condition_on_previous_text=False)
                    det = (getattr(info, "language", "") or "").lower()
                if det == "yue":
                    return "yue"
        except Exception:
            pass
        finally:
            if m_owned is not None:
                del m_owned
                _release_torch_vram()
    return "zh"


def refine_lines_with_qwen3(
    lines: list[dict],
    audio_16k: np.ndarray,
    db_profile: np.ndarray,
    step_sec: float,
    device_mode: str,
    first_onset: float,
    max_pad_sec: float = 1.5,
    require_all_windows: bool = False,
    sub_lang: str = "",
    meta_lang: str = "",
    detected_lang: str = "",
) -> tuple[list[dict], bool]:
    """Windowed batched Cantonese/multilingual word-level timestamp refinement using Qwen3-ForcedAligner-0.6B.

    Takes globally scaffolded lines (from stable-ts, which already resolves repeated choruses & live cuts),
    groups them into clean ~50s windows (no repeated chorus inside a window), and runs Qwen3-ForcedAligner
    across each window to produce ultra-accurate per-character KTV sweep timestamps (~2-3s total on GPU).
    Falls back gracefully per-window or globally to original lines on low VRAM / CPU / any error.
    """
    if not lines or len(lines) < 2 or device_mode == "cpu" or len(audio_16k) == 0:
        return lines, False

    free_mb = query_free_vram_mb()
    if device_mode != "gpu" and free_mb < 2200:
        return lines, False

    if detected_lang and detected_lang in QWEN3_LANG_MAP:
        lang = detected_lang
    else:
        lang = _detect_cantonese_vs_mandarin_audio(
            audio_16k, first_onset, lines, sub_lang=sub_lang, meta_lang=meta_lang
        )
    qwen_lang = QWEN3_LANG_MAP.get(lang)
    if not qwen_lang:
        return lines, False

    audio_dur = len(audio_16k) / 16000.0

    # Group lines into clean ~45-55s windows AND ALWAYS split at instrumental interludes (>= 4.0s gap)
    # so Qwen3-ForcedAligner NEVER sees a 20-35s instrumental break inside a window (eliminates stretched ghost lines!)
    windows: list[list[int]] = []
    cur_win: list[int] = []
    win_start = float(lines[0]["start"])
    for idx, ln in enumerate(lines):
        prev_gap = (float(ln["start"]) - float(lines[cur_win[-1]]["end"])) if cur_win else 0.0
        if cur_win and (prev_gap >= 4.0 or float(ln["end"]) - win_start > 55.0 or len(cur_win) >= 14):
            windows.append(cur_win)
            cur_win = [idx]
            win_start = float(ln["start"])
        else:
            cur_win.append(idx)
    if cur_win:
        windows.append(cur_win)
    # Merge tiny trailing 1-line window into prior window when no interlude separates them (prevents <1s skip aborts!)
    if len(windows) >= 2 and len(windows[-1]) == 1:
        gap_last = float(lines[windows[-1][0]]["start"]) - float(lines[windows[-2][-1]]["end"])
        span_merged = float(lines[windows[-1][-1]]["end"]) - float(lines[windows[-2][0]]["start"])
        if gap_last < 4.0 and span_merged <= 65.0:
            windows[-2].extend(windows.pop())

    aligner = None
    try:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        import torch
        if not torch.cuda.is_available():
            return lines, False
        from qwen_asr import Qwen3ForcedAligner

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        try:
            aligner = Qwen3ForcedAligner.from_pretrained(
                "Qwen/Qwen3-ForcedAligner-0.6B",
                dtype=dtype,
                device_map="cuda:0",
                local_files_only=True,
            )
        except Exception:
            aligner = Qwen3ForcedAligner.from_pretrained(
                "Qwen/Qwen3-ForcedAligner-0.6B",
                dtype=dtype,
                device_map="cuda:0",
            )

        refined = [dict(ln) for ln in lines]
        refined_windows = 0

        for wi, idx_list in enumerate(windows):
            # Use ALREADY-REFINED previous window end so adjacent windows NEVER expand into the same vocal region!
            prev_end = float(refined[idx_list[0] - 1]["end"]) if idx_list[0] > 0 else 0.0
            next_start = float(lines[idx_list[-1] + 1]["start"]) if (idx_list[-1] + 1) < len(lines) else audio_dur
            # Structural drift rescue using SYLLABLE UNITS (0.07s/syl protects fast rap while still catching crushed lines!)
            has_compressed = any(
                (float(lines[i]["end"]) - float(lines[i]["start"])) / estimate_syllable_units(lines[i].get("text", "")) < 0.07
                for i in idx_list
            )
            eff_pad = 12.0 if has_compressed else max_pad_sec
            pad_left = min(eff_pad, max(0.25, (float(lines[idx_list[0]]["start"]) - prev_end) * (0.85 if has_compressed else 0.45)))
            pad_right = min(eff_pad, max(0.25, (next_start - float(lines[idx_list[-1]]["end"])) * (0.85 if has_compressed else 0.45)))
            w_t0 = max(prev_end, float(lines[idx_list[0]]["start"]) - pad_left)
            w_t1 = min(audio_dur, float(lines[idx_list[-1]]["end"]) + pad_right)
            if w_t1 - w_t0 < 0.35 or w_t1 - w_t0 > 120.0:
                continue

            s0 = int(round(w_t0 * 16000))
            s1 = int(round(w_t1 * 16000))
            chunk_audio = audio_16k[s0:s1]
            if len(chunk_audio) < 8000:
                continue

            # Per-window language override for bilingual songs / multi-language medleys (e.g., EN chorus in CJK song)
            win_texts = [(lines[i].get("text") or "").strip() for i in idx_list]
            win_lang = detect_lyrics_language(win_texts)
            if win_lang == "zh" and lang == "yue":
                win_lang = "yue"
            win_qwen_lang = QWEN3_LANG_MAP.get(win_lang, qwen_lang)

            valid_pairs: list[tuple[int, str, int]] = []
            for idx in idx_list:
                txt = (lines[idx].get("text") or "").strip()
                wlist, _ = aligner.aligner_processor.encode_timestamp(txt, win_qwen_lang)
                if wlist:
                    valid_pairs.append((idx, txt, len(wlist)))
            if not valid_pairs:
                continue

            win_text = "\n".join(p[1] for p in valid_pairs)
            res = aligner.align(audio=(chunk_audio, 16000), text=win_text, language=win_qwen_lang)
            items = list(res[0].items) if (res and len(res) > 0) else []
            expected_n = sum(p[2] for p in valid_pairs)
            if not items or len(items) != expected_n:
                continue

            # Build candidate refined lines for this window with strict per-line collapse & stretch guards
            cand_updates: list[tuple[int, dict]] = []
            cursor = 0
            squished = 0
            exploded = 0
            for idx, txt, n_tok in valid_pairs:
                tok_slice = items[cursor: cursor + n_tok]
                cursor += n_tok
                if not tok_slice:
                    continue
                local_s = float(tok_slice[0].start_time)
                local_e = float(tok_slice[-1].end_time)
                q_dur = local_e - local_s
                n_syl = estimate_syllable_units(txt)
                orig_dur = max(0.35, float(lines[idx]["end"]) - float(lines[idx]["start"]))
                is_squished = q_dur <= max(0.18, 0.04 * n_syl)
                is_exploded = q_dur > max(9.5, orig_dur * 2.2)
                if is_squished:
                    squished += 1
                if is_exploded:
                    exploded += 1
                # Per-line protection: never replace a healthy Stage-1 line with a Qwen3 collapse or interlude blow-up!
                if (is_squished and orig_dur >= 0.65) or is_exploded:
                    continue
                abs_s = round(max(first_onset, w_t0 + local_s), 2)
                abs_e = round(max(abs_s + 0.35, w_t0 + local_e), 2)
                words_out = []
                for it in tok_slice:
                    wt = (it.text or "").strip()
                    if not wt:
                        continue
                    ws = round(max(abs_s, min(abs_e, w_t0 + float(it.start_time))), 3)
                    we = round(max(ws, min(abs_e, w_t0 + float(it.end_time))), 3)
                    words_out.append({"start": ws, "end": we, "word": wt})
                cand_updates.append((idx, {"start": abs_s, "end": abs_e, "text": txt, "words": words_out}))

            # Accept healthy lines in this window even if 1 outlier line exploded/squished (<=15% anomaly rate)
            if cand_updates and ((squished + exploded) / max(1, len(valid_pairs))) <= 0.15:
                for idx, upd in cand_updates:
                    refined[idx] = upd
                refined_windows += 1

        if refined_windows == 0 or (require_all_windows and refined_windows < len(windows)):
            return lines, False

        # Ensure strict monotonic NON-OVERLAPPING line boundaries across window edges AND clamp inner words!
        for i in range(1, len(refined)):
            if refined[i]["start"] < refined[i - 1]["end"]:
                mid = round((refined[i - 1]["end"] + refined[i]["start"]) / 2.0, 2)
                new_prev_e = max(round(refined[i - 1]["start"] + 0.25, 2), mid)
                refined[i - 1]["end"] = new_prev_e
                if refined[i - 1].get("words"):
                    ps = refined[i - 1]["start"]
                    for w in refined[i - 1]["words"]:
                        w["start"] = round(max(ps, min(new_prev_e, float(w["start"]))), 3)
                        w["end"] = round(max(w["start"], min(new_prev_e, float(w["end"]))), 3)
                refined[i]["start"] = max(new_prev_e, refined[i]["start"])
                refined[i]["end"] = max(round(refined[i]["start"] + 0.30, 2), refined[i]["end"])
                if refined[i].get("words"):
                    cs, ce = refined[i]["start"], refined[i]["end"]
                    for w in refined[i]["words"]:
                        w["start"] = round(max(cs, min(ce, float(w["start"]))), 3)
                        w["end"] = round(max(w["start"], min(ce, float(w["end"]))), 3)

        refined = refine_alignment_with_acoustics(refined, db_profile, step_sec, audio_dur, first_onset)
        print(f"[Lyrics] Qwen3-ForcedAligner refined {refined_windows}/{len(windows)} windows ({len(refined)} lines)", file=sys.stderr, flush=True)
        return refined, True
    except Exception as exc:
        print(f"[Lyrics] Qwen3-ForcedAligner refinement skipped: {exc}", file=sys.stderr, flush=True)
        return lines, False
    finally:
        if aligner is not None:
            del aligner
        _release_torch_vram()


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
    initial_song_pools: Optional[list[list[str]]] = None,
    sub_lang: str = "",
    meta_lang: str = "",
    lang_out: Optional[dict] = None,
) -> tuple[list[dict], str, bool]:
    """Force-align 100% REAL official lyrics text directly onto isolated vocals.mp3.
    Automatically detects mid-song starts, skipped/swapped concert verses, or encores and
    engages Section-Anchored Live Alignment on the same loaded model!
    Always releases Whisper VRAM before returning so Qwen3 Stage-2 has full GPU headroom.
    """
    ensure_ffmpeg_in_path()
    add_nvidia_dll_dirs()

    free_mb = query_free_vram_mb()
    use_cuda = (device_mode != "cpu") and (device_mode == "gpu" or free_mb >= 2000)
    lang = detect_lyrics_language(official_lines, sub_lang=sub_lang or meta_lang)
    whisper_lang = _whisper_align_lang(lang)
    text_block = "\n".join(official_lines)
    audio_dur = (len(audio_16k) / 16000.0) if len(audio_16k) > 0 else 0.0

    used_dev = "cuda" if use_cuda else "cpu"
    model = None
    try:
        try:
            model = _load_stable_whisper_model(used_dev)
        except Exception as e:
            if used_dev == "cuda":
                print(f"[Lyrics] Load CUDA fallback to CPU: {e}", file=sys.stderr, flush=True)
                used_dev = "cpu"
                model = _load_stable_whisper_model("cpu")
            else:
                raise

        # Reuse the already-loaded Stage-1 Whisper model for instant ~0.15s Cantonese vs Mandarin ID
        # (saves ~3s duplicate model load inside Qwen3 AND calibrates Cantonese rematch thresholds!)
        if lang == "zh" and len(audio_16k) >= 16000 * 10:
            probe_lines = [{"start": first_onset, "text": t} for t in official_lines[:20]]
            lang = _detect_cantonese_vs_mandarin_audio(
                audio_16k, first_onset, probe_lines,
                sub_lang=sub_lang, meta_lang=meta_lang, preloaded_model=model,
            )
        if lang_out is not None:
            lang_out["lang"] = lang

        def _run_live_safe() -> list[dict]:
            nonlocal model, used_dev
            try:
                return _live_section_anchored_align(
                    model, vocals_path, audio_16k, db_profile, step_sec, official_lines, lang, first_onset,
                    use_cuda=(used_dev == "cuda"), inst_path=inst_path, shazam_already_tried=shazam_already_tried,
                    initial_song_pools=initial_song_pools,
                )
            except Exception as e:
                if used_dev == "cuda":
                    print(f"[Lyrics] Live-align CUDA fallback to CPU: {e}", file=sys.stderr, flush=True)
                    used_dev = "cpu"
                    model = None
                    _release_torch_vram()
                    model = _load_stable_whisper_model("cpu")
                    return _live_section_anchored_align(
                        model, vocals_path, audio_16k, db_profile, step_sec, official_lines, lang, first_onset,
                        use_cuda=False, inst_path=inst_path, shazam_already_tried=shazam_already_tried,
                        initial_song_pools=initial_song_pools,
                    )
                raise

        # Only bypass silence-collapsed full-song align for genuine Mashups/Medleys (multiple songs!).
        # Full-song covers/lives ALWAYS try Silence-Collapsed Pass 1 first (~2.4s, 0% dropped lines!)
        if prefer_live:
            live_lines = _run_live_safe()
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
                language=whisper_lang,
                original_split=True,
                fast_mode=(used_dev == "cpu"),
                verbose=False,
            )
        except Exception as e:
            if used_dev == "cuda":
                print(f"[Lyrics] Forced-align CUDA fallback to CPU: {e}", file=sys.stderr, flush=True)
                used_dev = "cpu"
                model = None
                _release_torch_vram()
                model = _load_stable_whisper_model("cpu")
                aligned = model.align(
                    align_audio,
                    text_block,
                    language=whisper_lang,
                    original_split=True,
                    fast_mode=True,
                    verbose=False,
                )
            else:
                raise

        needs_live, reason = _alignment_needs_live_rematch(aligned, audio_dur, db_profile, step_sec, time_map=time_map, lang=lang)
        if needs_live:
            print(f"[Lyrics] Detected non-standard live/abridged structure ({reason}); engaging section-anchored live alignment...", flush=True)
            live_lines = _run_live_safe()
            if len(live_lines) >= 4:
                return live_lines, used_dev, True

        lines = []
        for seg in aligned.segments:
            txt = (seg.text or "").strip()
            if not txt:
                continue
            raw_s = _map_comp_to_real(float(seg.start), time_map, is_end=False) if time_map else float(seg.start)
            raw_e = _map_comp_to_real(float(seg.end), time_map, is_end=True) if time_map else float(seg.end)
            start_s = round(max(first_onset, raw_s), 2)
            end_s = round(max(start_s + 0.4, raw_e), 2)

            # Preserve raw aligner timestamps (including 0.00s collapsed tokens!) so refine_alignment_with_acoustics
            # can detect exact aligner degeneracy (raw_dur <= 0.02s) without confusing it with fast singing/rap!
            words_out = []
            for w in (seg.words or []):
                wt = (w.word or "").strip()
                if not wt:
                    continue
                ws_r = _map_comp_to_real(float(w.start), time_map, is_end=False) if time_map else float(w.start)
                we_r = _map_comp_to_real(float(w.end), time_map, is_end=True) if time_map else float(w.end)
                ws = round(max(start_s, ws_r), 3)
                we = round(max(ws, we_r), 3)
                words_out.append({"start": ws, "end": we, "word": wt})

            lines.append({"start": start_s, "end": end_s, "text": txt, "words": words_out})

        lines = refine_alignment_with_acoustics(lines, db_profile, step_sec, audio_dur, first_onset)
        return lines, used_dev, False
    finally:
        if model is not None:
            del model
        _release_torch_vram()


def fast_probe_transcribe_for_rescue(
    audio_16k: np.ndarray,
    first_onset: float,
    device_mode: str,
    title_hint: str = "",
    probe_sec: float = 48.0,
) -> list[dict]:
    """Transcribe ONLY the first ~48s of singing (~8s on CPU / ~0.5s on GPU) to rescue official lyrics
    via NetEase/LRCLIB full-text search without wasting 80s transcribing an entire unknown song on CPU!
    """
    if len(audio_16k) < 16000 * 6:
        return []
    add_nvidia_dll_dirs()

    s0 = int(max(0.0, first_onset) * 16000)
    s1 = min(len(audio_16k), s0 + int(probe_sec * 16000))
    clip = audio_16k[s0:s1]
    if len(clip) < 16000 * 5:
        return []

    free_mb = query_free_vram_mb()
    use_cuda = (device_mode != "cpu") and (device_mode == "gpu" or free_mb >= 2000)
    # Never inject Cantonese prompt on Japanese Kanji or Korean titles!
    is_zh_title = bool(re.search(r"[\u4e00-\u9fff]", title_hint or "")) and not _is_non_chinese_cjk(title_hint or "")
    prompt = f"以下係繁體中文歌詞：{clean_search_title(title_hint)}" if is_zh_title else None

    def _run(dev: str):
        m = _load_faster_whisper_model(dev)
        try:
            segs, inf = m.transcribe(
                clip,
                beam_size=1 if dev == "cpu" else 2,
                word_timestamps=False,
                vad_filter=True,
                initial_prompt=prompt,
                condition_on_previous_text=False,
            )
            return list(segs), inf
        finally:
            del m
            _release_torch_vram()

    try:
        segments, info = _run("cuda" if use_cuda else "cpu")
    except Exception:
        segments, info = _run("cpu")

    det_lang = (getattr(info, "language", "") or "").lower()
    is_cjk = det_lang in ("zh", "yue")
    out = []
    for seg in segments:
        txt = (seg.text or "").strip()
        if not txt or HALLUCINATION_RE.search(txt) or GARBAGE_CHARS_RE.search(txt):
            continue
        out.append({"text": to_hk_traditional(txt) if is_cjk else txt})
    return out


def fallback_free_transcribe(
    vocals_path: Path,
    device_mode: str,
    db_profile: np.ndarray,
    step_sec: float,
    first_onset: float,
    title_hint: str = "",
    audio_16k: Optional[np.ndarray] = None,
) -> tuple[list[dict], str]:
    """Only used when NO official or description lyrics exist; uses silence-collapsed vocals + RMS gate."""
    add_nvidia_dll_dirs()

    free_mb = query_free_vram_mb()
    use_cuda = (device_mode != "cpu") and (device_mode == "gpu" or free_mb >= 2000)
    # Never inject Cantonese prompt on Japanese Kanji or Korean titles!
    is_zh_title = bool(re.search(r"[\u4e00-\u9fff]", title_hint or "")) and not _is_non_chinese_cjk(title_hint or "")
    prompt = f"以下係繁體中文歌詞：{clean_search_title(title_hint)}" if is_zh_title else None

    comp_time_map: list[tuple[float, float, float, float]] = []
    if audio_16k is not None and len(audio_16k) > 0 and len(db_profile) > 0:
        transcribe_input, comp_time_map = build_silence_collapsed_audio(audio_16k, db_profile, step_sec, sr=16000)
        if len(transcribe_input) == 0 or not comp_time_map:
            transcribe_input, comp_time_map = str(vocals_path), []
    else:
        transcribe_input = str(vocals_path)

    def _run(dev: str):
        m = _load_faster_whisper_model(dev)
        try:
            segs, inf = m.transcribe(
                transcribe_input,
                beam_size=2 if dev == "cpu" else 5,
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
        finally:
            del m
            _release_torch_vram()

    used_dev = "cuda" if use_cuda else "cpu"
    try:
        segments, info = _run(used_dev)
    except Exception:
        used_dev = "cpu"
        segments, info = _run("cpu")

    det_lang = (getattr(info, "language", "") or "").lower()
    is_cjk = det_lang in ("zh", "yue")
    split_lang = det_lang if det_lang in ("zh", "yue", "ja", "ko", "en") else ("zh" if is_cjk else "en")
    lines = []
    for seg in segments:
        s_raw = float(seg.start)
        e_raw = float(seg.end)
        s_start = _map_comp_to_real(s_raw, comp_time_map, is_end=False) if comp_time_map else s_raw
        s_end = _map_comp_to_real(e_raw, comp_time_map, is_end=True) if comp_time_map else e_raw
        s_end = max(s_start + 0.05, s_end)
        # Strict RMS energy gate: NEVER emit lyrics during silent intros/outros!
        if s_end < first_onset or not segment_has_vocal_energy(db_profile, step_sec, s_start, s_end):
            continue
        txt = (seg.text or "").strip()
        if not txt or HALLUCINATION_RE.search(txt) or GARBAGE_CHARS_RE.search(txt):
            continue
        if is_cjk:
            txt = to_hk_traditional(txt)
        words_mapped = []
        for w in (seg.words or []):
            wt = (w.word or "").strip()
            if not wt or GARBAGE_CHARS_RE.search(wt):
                continue
            ws_r = _map_comp_to_real(float(w.start), comp_time_map, is_end=False) if comp_time_map else float(w.start)
            we_r = _map_comp_to_real(float(w.end), comp_time_map, is_end=True) if comp_time_map else float(w.end)
            ws = round(max(first_onset, ws_r), 3)
            we = round(max(ws, we_r), 3)
            words_mapped.append({
                "start": ws,
                "end": we,
                "word": to_hk_traditional(wt) if is_cjk else wt,
            })

        # Split long run-on ASR segments cleanly into singable KTV lines
        sub_chunks = _split_long_asr_segment(s_start, s_end, txt, words_mapped, split_lang)
        for sub in sub_chunks:
            stxt = (sub.get("text") or "").strip()
            if not stxt or HALLUCINATION_RE.search(stxt) or GARBAGE_CHARS_RE.search(stxt):
                continue
            s_clamped = round(max(first_onset, float(sub["start"])), 2)
            e_clamped = round(max(s_clamped + 0.4, float(sub["end"])), 2)
            sub_words = [
                w for w in words_mapped
                if w["start"] >= s_clamped - 0.12 and w["end"] <= e_clamped + 0.12
            ]
            lines.append({
                "start": s_clamped,
                "end": e_clamped,
                "text": stxt,
                "words": sub_words or _even_words_for_line(stxt, s_clamped, e_clamped),
            })

    audio_dur = (len(db_profile) * step_sec) if len(db_profile) > 0 else 0.0
    lines = refine_alignment_with_acoustics(lines, db_profile, step_sec, audio_dur, first_onset)
    return lines, used_dev


from atomic_io import write_json_atomic


def write_status(status_file: Path, payload: dict) -> None:
    write_json_atomic(status_file, payload, stamp_updated_at=True)


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
    parser.add_argument("--meta-file", default="")
    args = parser.parse_args()

    status_file = Path(args.status_file)
    out_file = Path(args.out_file)
    vocals_path = Path(args.vocals_path)
    inst_path = Path(args.inst_path) if args.inst_path else None
    meta_file = Path(args.meta_file) if args.meta_file else None
    force_ai = args.force_ai == "1"
    started = time.time()

    meta_data: dict = {}
    if meta_file and meta_file.exists():
        try:
            meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            meta_data = {}

    last_state = {"status": "running", "progress": 10, "stage_text": "載入歌詞中...", "error": None}
    state_lock = threading.Lock()
    hb_stop = threading.Event()

    def update(status: str, progress: int, stage_text: str, error: Optional[str] = None):
        with state_lock:
            last_state.update({"status": status, "progress": progress, "stage_text": stage_text, "error": error})
            if status in ("ready", "error"):
                hb_stop.set()
            write_status(status_file, {
                "video_id": args.video_id,
                "status": status,
                "progress": progress,
                "stage_text": stage_text,
                "error": error,
                "elapsed_sec": round(time.time() - started, 1),
            })

    def _heartbeat_loop() -> None:
        """Keep updated_at fresh during long CPU alignments (e.g. 8-10 min medleys) so monitors never false-alarm."""
        while not hb_stop.wait(25.0):
            with state_lock:
                if hb_stop.is_set() or last_state["status"] != "running":
                    return
                write_status(status_file, {
                    "video_id": args.video_id,
                    "status": last_state["status"],
                    "progress": last_state["progress"],
                    "stage_text": last_state["stage_text"],
                    "error": last_state["error"],
                    "elapsed_sec": round(time.time() - started, 1),
                })

    threading.Thread(target=_heartbeat_loop, daemon=True, name="lyrics-heartbeat").start()

    try:
        update("running", 10, "載入歌詞中...")
        yt_track = (meta_data.get("track") or "").strip()
        yt_artist = (meta_data.get("artist") or "").strip()
        yt_channel = (meta_data.get("channel") or "").strip()
        yt_desc = meta_data.get("description") or ""
        yt_video_lang = (meta_data.get("language") or "").strip()
        title = (meta_data.get("title") or args.title or "").strip()
        duration = float(meta_data.get("duration") or args.duration or 0.0)
        yt_lines = None
        yt_lang = ""

        # Zero-duplicate yt-dlp: reuse metadata + cached subtitle lines captured during worker.py's single YouTube fetch!
        if meta_data.get("yt_meta_present"):
            yt_lang = meta_data.get("subtitles_lang") or ""
            if not force_ai:
                cached_subs = meta_data.get("yt_sub_lines")
                if isinstance(cached_subs, list) and len(cached_subs) >= 4:
                    yt_lines = cached_subs
                elif meta_data.get("subtitles_url"):
                    yt_lines = _download_json3_subtitles(meta_data["subtitles_url"], yt_lang)
        elif force_ai and title:
            pass
        else:
            yt_lines, yt_lang, fetched_title, fetched_dur, yt_desc, yt_track, yt_artist, yt_channel = fetch_youtube_human_subtitles(args.video_id)
            if fetched_title and fetched_title != args.video_id:
                title = fetched_title
            if fetched_dur > 0:
                duration = fetched_dur

        # Layer 1: Human YouTube subtitles — NEVER ship 0-word linear-sweep lines when vocals.mp3 exists!
        # Also verify CC covers >=55% of active vocal singing (blocks 5-line spoken-intro CC from hijacking a song!).
        yt_human_official_lines: list[str] = []
        cached_vocal_profile = None
        if yt_lines and not force_ai:
            if not vocals_path.exists():
                result = {
                    "video_id": args.video_id,
                    "title": title,
                    "source": "youtube_sub",
                    "source_label": "同步字幕",
                    "lines": yt_lines,
                    "created_at": time.time(),
                }
                out_file.parent.mkdir(parents=True, exist_ok=True)
                write_json_atomic(out_file, result, stamp_updated_at=False)
                update("finalizing", 98, "載入歌詞中...")
                return 0

            cached_vocal_profile = detect_vocal_energy_profile(vocals_path)
            audio_16k_yt, db_prof_yt, step_yt, onset_yt = cached_vocal_profile
            cc_covers_vocals = True
            if len(db_prof_yt) > 0 and step_yt > 0:
                cov_thr = _adaptive_vocal_threshold_db(db_prof_yt, margin_db=2.0)
                active_mask = db_prof_yt > cov_thr
                total_active_frames = int(np.sum(active_mask))
                if total_active_frames >= int(15.0 / step_yt):
                    covered_mask = np.zeros(len(db_prof_yt), dtype=bool)
                    for ln in yt_lines:
                        i0 = max(0, int((float(ln["start"]) - 1.0) / step_yt))
                        i1 = min(len(db_prof_yt), int((float(ln["end"]) + 1.0) / step_yt) + 1)
                        covered_mask[i0:i1] = True
                    cov_ratio = float(np.sum(active_mask & covered_mask)) / max(1, total_active_frames)
                    if cov_ratio < 0.55:
                        cc_covers_vocals = False
                        print(f"[Lyrics] Skipping partial YouTube CC (vocal coverage={cov_ratio:.2f} < 0.55)", file=sys.stderr, flush=True)

            if cc_covers_vocals and len(audio_16k_yt) > 0:
                update("running", 65, "精修歌詞時間軸...")
                yt_ref, q_ok = refine_lines_with_qwen3(
                    yt_lines, audio_16k_yt, db_prof_yt, step_yt, args.device_mode, onset_yt,
                    max_pad_sec=1.5, require_all_windows=True,
                    sub_lang=yt_lang, meta_lang=yt_video_lang,
                )
                if q_ok and len(yt_ref) >= 4:
                    result = {
                        "video_id": args.video_id,
                        "title": title,
                        "source": "youtube_sub_qwen3",
                        "source_label": "同步字幕",
                        "lines": yt_ref,
                        "created_at": time.time(),
                    }
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    write_json_atomic(out_file, result, stamp_updated_at=False)
                    update("finalizing", 98, "載入歌詞中...")
                    return 0
            # GPU busy / Qwen3 skipped -> keep these exact human YouTube lines ONLY if they cover the vocals!
            if cc_covers_vocals:
                yt_human_official_lines = clean_lyric_lines([ln["text"] for ln in yt_lines])

        # Layer 2: Generic parallel official lyrics lookup (Multi-Song Title Set -> LRCLIB + NetEase + YT Desc)
        update("running", 25, "搜尋歌詞中...")
        has_mashup_kw = bool(MASHUP_KEYWORDS_RE.search(title))
        bracket_count_title = len(re.findall(r"《[^》]{1,48}》|「[^」]{1,48}」|『[^』]{1,48}』", title or ""))
        is_mashup_title = has_mashup_kw or (bracket_count_title >= 3) or (
            (duration == 0.0 or duration >= 330.0) and bool(re.search(r"》\s*《|」\s*「|』\s*『", title or ""))
        )
        multi_song_pools: list[list[str]] = []
        if yt_human_official_lines:
            lrc_synced, official_lines, lrc_src = None, yt_human_official_lines, "YouTube Human Subtitles"
        else:
            multi_song_pools = fetch_multi_song_lyrics_from_title(title, duration, yt_artist=yt_artist)
            if multi_song_pools:
                lrc_synced = None
                official_lines = [ln for pool in multi_song_pools for ln in pool]
                lrc_src = f"Multi-Song Official ({len(multi_song_pools)} songs)"
            else:
                # Only skip single-song lookup when explicit mashup/medley keywords exist (not mere 3-bracket show titles!)
                lrc_synced, official_lines, lrc_src = (
                    (None, [], "")
                    if has_mashup_kw
                    else fetch_official_lyrics(
                        title, duration, yt_desc,
                        yt_track=yt_track, yt_artist=yt_artist, yt_channel=yt_channel,
                    )
                )

        # Only return raw line-only LRC if vocals.mp3 is unavailable; otherwise ALWAYS acoustically align
        # against THIS video's actual vocals (eliminates MV-vs-Spotify intro offsets and fake 8s 0-word sweeps!).
        if lrc_synced and not force_ai and not vocals_path.exists():
            result = {
                "video_id": args.video_id,
                "title": title,
                "source": "lrclib_exact",
                "source_label": "同步歌詞",
                "lines": lrc_synced,
                "created_at": time.time(),
            }
            out_file.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(out_file, result, stamp_updated_at=False)
            update("finalizing", 98, "載入歌詞中...")
            return 0

        if not vocals_path.exists():
            raise RuntimeError("請先開啟唱 K 伴奏")

        audio_16k, db_profile, step_sec, first_onset = (
            cached_vocal_profile if cached_vocal_profile is not None else detect_vocal_energy_profile(vocals_path)
        )

        # Fast-path acoustic alignment when we have studio LRC scaffold + free GPU VRAM (~2.5s total with Qwen3!):
        # Auto-compensate MV-vs-Spotify intro offset AND enforce 3 acoustic gates:
        # (1) tail uncovered vocals < 10s, (2) overall vocal span matches LRC span within 5s (blocks Story MVs!),
        # (3) active vocal hit ratio >= 85% (blocks edits/medleys/wrong-version LRC!).
        if lrc_synced and not force_ai and len(lrc_synced) >= 4 and len(audio_16k) > 0:
            shift = first_onset - float(lrc_synced[0]["start"])
            eff_shift = shift if abs(shift) > 0.6 else 0.0
            audio_dur = len(audio_16k) / 16000.0
            lrc_tail_end = float(lrc_synced[-1]["end"]) + eff_shift
            lrc_covers_audio = True
            if len(db_profile) > 0 and step_sec > 0:
                cov_thr = _adaptive_vocal_threshold_db(db_profile, margin_db=2.0)
                active_mask = db_profile > cov_thr
                active_idxs = np.where(active_mask)[0]
                t_idx = int((lrc_tail_end + 4.0) / step_sec)
                if t_idx < len(db_profile):
                    uncov_tail = float(np.sum(active_mask[t_idx:])) * step_sec
                    if uncov_tail >= 10.0:
                        lrc_covers_audio = False
                if lrc_covers_audio and len(active_idxs) > 0:
                    last_active_sec = float(active_idxs[-1]) * step_sec
                    audio_vocal_span = max(0.0, last_active_sec - first_onset)
                    lrc_span = max(0.0, float(lrc_synced[-1]["end"]) - float(lrc_synced[0]["start"]))
                    if abs(audio_vocal_span - lrc_span) > 5.0:
                        lrc_covers_audio = False
                if lrc_covers_audio and len(active_idxs) >= int(15.0 / step_sec):
                    covered_mask = np.zeros(len(db_profile), dtype=bool)
                    for ln in lrc_synced:
                        ns_m = max(0.0, float(ln["start"]) + eff_shift - 1.2)
                        ne_m = min(audio_dur, float(ln["end"]) + eff_shift + 1.2)
                        covered_mask[int(ns_m / step_sec): int(ne_m / step_sec) + 1] = True
                    hit_ratio = float(np.sum(active_mask & covered_mask)) / max(1, int(np.sum(active_mask)))
                    if hit_ratio < 0.85:
                        lrc_covers_audio = False
            if abs(shift) <= 12.0 and lrc_covers_audio:
                shifted_lrc = []
                for ln in lrc_synced:
                    ns = max(0.0, round(float(ln["start"]) + eff_shift, 2))
                    ne = min(audio_dur, max(ns + 0.5, round(float(ln["end"]) + eff_shift, 2)))
                    shifted_lrc.append({"start": ns, "end": ne, "text": ln["text"], "words": []})
                update("running", 65, "精修歌詞時間軸...")
                q_lines, q_ok = refine_lines_with_qwen3(
                    shifted_lrc, audio_16k, db_profile, step_sec, args.device_mode, first_onset,
                    max_pad_sec=2.5, require_all_windows=True,
                    sub_lang=yt_lang, meta_lang=yt_video_lang,
                )
                if q_ok and len(q_lines) >= 4:
                    result = {
                        "video_id": args.video_id,
                        "title": title,
                        "source": "lrclib_qwen3_align",
                        "source_label": "同步歌詞",
                        "lines": q_lines,
                        "created_at": time.time(),
                    }
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    write_json_atomic(out_file, result, stamp_updated_at=False)
                    update("finalizing", 98, "載入歌詞中...")
                    return 0

        # Layer 2B: Automatic Shazam audio fingerprinting when title did NOT resolve >=2 multi-song pools or found nothing.
        # Merges deduplicated Shazam pools with any existing title pool instead of wiping out already-found lyrics!
        shazam_tried = False
        shazam_tracks: list[dict] = []
        if (is_mashup_title and len(multi_song_pools) < 2) or not official_lines:
            update("running", 35, "辨識歌曲中...")
            shazam_tried = True
            shz_lines, shazam_tracks = shazam_scan_and_fetch_lyrics(
                audio_16k, db_profile, step_sec, inst_path=inst_path, scan_mashup=is_mashup_title
            )
            if shz_lines:
                shz_pools = [tr["official_lines"] for tr in shazam_tracks if tr.get("official_lines")]
                if shz_pools:
                    merged_pools = list(multi_song_pools)
                    seen_sigs = {"|".join(p[:3]) for p in merged_pools if p}
                    for sp in shz_pools:
                        sig = "|".join(sp[:3])
                        if sig and sig not in seen_sigs:
                            seen_sigs.add(sig)
                            merged_pools.append(sp)
                    multi_song_pools = merged_pools
                    official_lines = [ln for p in multi_song_pools for ln in p]
                elif not multi_song_pools:
                    official_lines = shz_lines

        # Layer 3A (BEST): 2-Stage Forced Alignment of 100% REAL official lyrics!
        # Stage 1: stable-ts global scaffold (handles repeated choruses, live skipped verses, mashups).
        # Stage 2: Qwen3-ForcedAligner-0.6B windowed refinement (~2-3s GPU, native Cantonese/CJK/EN/JP/KR precision!).
        if official_lines:
            update("running", 50, "對齊歌詞中...")
            is_multi_song_mashup = is_mashup_title or len(shazam_tracks) >= 2 or len(multi_song_pools) >= 2
            lang_out: dict = {}
            ai_lines, used_dev, used_live = align_official_lyrics_with_stable_ts(
                vocals_path, audio_16k, db_profile, step_sec, official_lines,
                args.device_mode, first_onset, prefer_live=is_multi_song_mashup,
                inst_path=inst_path, shazam_already_tried=(shazam_tried or len(multi_song_pools) >= 2),
                initial_song_pools=(multi_song_pools if multi_song_pools else None),
                sub_lang=yt_lang, meta_lang=yt_video_lang, lang_out=lang_out,
            )
            qwen_ok = False
            if ai_lines:
                update("running", 82, "精修歌詞時間軸...")
                ai_lines, qwen_ok = refine_lines_with_qwen3(
                    ai_lines, audio_16k, db_profile, step_sec, args.device_mode, first_onset,
                    sub_lang=yt_lang, meta_lang=yt_video_lang, detected_lang=lang_out.get("lang", ""),
                )
            if len(shazam_tracks) >= 2:
                src = "mashup_qwen3_align" if qwen_ok else "mashup_shazam_align"
            elif used_live:
                src = "live_qwen3_align" if qwen_ok else "live_anchored_align"
            else:
                src = "qwen3_forced_align" if qwen_ok else "forced_align"
            label = "同步歌詞"
        else:
            # Layer 3B: Fast 48s vocal probe -> Phrase Rescue BEFORE ever paying for full-song CPU free transcription!
            update("running", 45, "辨識歌曲中...")
            probe_segs = fast_probe_transcribe_for_rescue(audio_16k, first_onset, args.device_mode, title_hint=title)
            rescued_pools = search_official_lyrics_pools_by_sung_phrases(probe_segs) if len(probe_segs) >= 2 else []
            fallback_ai_lines: list[dict] = []
            if not rescued_pools:
                update("running", 55, "辨識歌詞中...")
                fallback_ai_lines, used_dev = fallback_free_transcribe(
                    vocals_path, args.device_mode, db_profile, step_sec, first_onset,
                    title_hint=title, audio_16k=audio_16k,
                )
                rescued_pools = search_official_lyrics_pools_by_sung_phrases(fallback_ai_lines) if len(fallback_ai_lines) >= 4 else []
            rescued_lines = [ln for p in rescued_pools for ln in p]
            ai_lines = []
            if rescued_lines:
                update("running", 75, "對齊歌詞中...")
                lang_out_r: dict = {}
                rescued_aligned, used_dev, _ = align_official_lyrics_with_stable_ts(
                    vocals_path, audio_16k, db_profile, step_sec, rescued_lines,
                    args.device_mode, first_onset, prefer_live=(len(rescued_pools) >= 2),
                    inst_path=inst_path, shazam_already_tried=True,
                    initial_song_pools=(rescued_pools if len(rescued_pools) >= 2 else None),
                    sub_lang=yt_lang, meta_lang=yt_video_lang, lang_out=lang_out_r,
                )
                if rescued_aligned:
                    ai_lines = rescued_aligned
                    qwen_ok = False
                    update("running", 88, "精修歌詞時間軸...")
                    ai_lines, qwen_ok = refine_lines_with_qwen3(
                        ai_lines, audio_16k, db_profile, step_sec, args.device_mode, first_onset,
                        sub_lang=yt_lang, meta_lang=yt_video_lang, detected_lang=lang_out_r.get("lang", ""),
                    )
                    src = "qwen3_phrase_rescue_align" if qwen_ok else "phrase_rescue_align"
                else:
                    ai_lines = fallback_ai_lines
            if not ai_lines and fallback_ai_lines:
                ai_lines = fallback_ai_lines
            if not rescued_lines or (ai_lines is fallback_ai_lines):
                qwen_ok = False
                if ai_lines:
                    ai_lines, qwen_ok = refine_lines_with_qwen3(
                        ai_lines, audio_16k, db_profile, step_sec, args.device_mode, first_onset,
                        sub_lang=yt_lang, meta_lang=yt_video_lang,
                    )
                src = "whisper_qwen3_ai" if qwen_ok else "whisper_ai"
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
        write_json_atomic(out_file, result, stamp_updated_at=False)
        update("finalizing", 98, "載入歌詞中...")
        return 0
    except Exception as exc:
        update("error", 0, "暫未能載入歌詞", error=str(exc))
        print(f"[LyricsWorker Error] {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        hb_stop.set()


if __name__ == "__main__":
    raise SystemExit(main())
