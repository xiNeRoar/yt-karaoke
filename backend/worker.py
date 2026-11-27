"""Isolated AI vocal separation worker process with real-time dynamic GPU<->CPU switching.

Features:
1. Main FastAPI server never imports PyTorch or touches CUDA (0 MB VRAM idle).
2. In 'auto' mode, checks live free VRAM before EVERY ~8s audio chunk:
   - On GPU and free VRAM drops < 900 MB (or hits OOM): immediately moves model to CPU
     and frees 100% VRAM back to local video gen (MiniMax) / games without failing!
   - On CPU and free VRAM recovers >= 2200 MB: immediately moves model back to RTX GPU!
3. Smooth real-time chunk-by-chunk progress written to JSON status file.
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
MODEL_DIR = BASE_DIR / "models"
TEMP_DIR = BASE_DIR / "tmp"
YT_COOKIES_FILE = BASE_DIR / "cookies.txt"
YT_FIREFOX_PROFILE = BASE_DIR / ".yt_firefox_profile"


def compute_cpu_thread_cap() -> int:
    """Cap CPU worker threads to ~25% of logical cores (max 6) so CPU fallback never starves desktop UI."""
    env_override = os.environ.get("KARAOKE_CPU_THREADS", "").strip()
    if env_override.isdigit() and int(env_override) >= 1:
        return min(16, int(env_override))
    cores = os.cpu_count() or 4
    if cores >= 16:
        return min(6, max(4, cores // 4))
    if cores >= 8:
        return 4
    return max(2, cores // 2)


class _HQYieldInterrupt(RuntimeError):
    """Raised when idle --force-hq background upgrade detects GPU contention and yields cleanly (rc=2)."""



def get_yt_js_runtimes() -> dict:
    """Detect local JS runtime (node/deno/bun) so yt-dlp 2026+ solves YouTube player formats cleanly."""
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


def get_yt_cookie_opts() -> dict:
    """Use optional owner-configured local cookies.txt or dedicated Firefox profile (never stored in git)."""
    if YT_COOKIES_FILE.exists() and YT_COOKIES_FILE.stat().st_size > 0:
        return {"cookiefile": str(YT_COOKIES_FILE)}
    if (YT_FIREFOX_PROFILE / "cookies.sqlite").exists():
        return {"cookiesfrombrowser": ("firefox", str(YT_FIREFOX_PROFILE), None, None)}
    return {}

MODELS = {
    "pure": {
        "label": "純伴奏 (MelBand RoFormer Inst FVX)",
        "filename": "mel_band_roformer_instrumental_fvx_gabox.ckpt",
    },
}
CPU_FAST_MODEL = "UVR-MDX-NET-Inst_HQ_3.onnx"


def find_ffmpeg_dir() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return str(Path(found).parent)
    candidates = [
        Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages",
        Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links",
        Path("C:/ffmpeg/bin"),
        Path("C:/Program Files/ffmpeg/bin"),
    ]
    for root in candidates:
        if not root.exists():
            continue
        if (root / "ffmpeg.exe").exists():
            return str(root)
        for match in root.rglob("ffmpeg.exe"):
            return str(match.parent)
    return None


def friendly_error_message(exc: Exception) -> str:
    """Convert raw yt-dlp / PyTorch / OS exceptions into clean user-facing Cantonese messages."""
    raw = str(exc or "").strip()
    low = raw.lower()
    if any(k in low for k in ("sign in to confirm", "not a bot", "use --cookies")):
        return "此影片需要 YouTube 驗證，暫時無法處理（請稍後再試或轉唱其他版本）"
    if "age" in low and ("confirm" in low or "restrict" in low or "sign in" in low):
        return "此影片有年齡限制，暫時無法處理"
    if any(k in low for k in ("video unavailable", "private video", "members-only", "copyright")):
        return "此影片無法讀取或已被設為私人／受地區限制"
    if "live" in low and ("唔支援" in raw or "stream" in low):
        return "唔支援直播影片，請揀選一般 MV 或歌曲"
    if "超過 15 分鐘" in raw or "搵唔到 ffmpeg" in low or "ai 輸出音軌不完整" in low:
        return raw
    if "out of memory" in low:
        return "伺服器記憶體暫時緊張，請稍後再試"
    return "暫時無法處理此影片，請稍後再試"


from atomic_io import write_json_atomic


def write_status(status_file: Path, payload: dict) -> None:
    write_json_atomic(status_file, payload, stamp_updated_at=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--mode", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--url", required=True)
    parser.add_argument("--device-mode", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--force-hq", action="store_true", help="Require BS-RoFormer HQ on GPU; exit 2 if VRAM < 3000MB")
    args = parser.parse_args()

    device_mode = args.device_mode
    target_cache_root = Path(args.cache_dir)
    status_file = Path(args.status_file)
    started = time.time()

    state = {
        "video_id": args.video_id,
        "mode": args.mode,
        "status": "downloading",
        "progress": 6,
        "stage_text": "準備音訊中...",
        "error": None,
        "title": "",
        "duration": 0.0,
        "device_used": device_mode.upper(),
        "elapsed_sec": 0.0,
    }

    def update(**kwargs):
        state.update(kwargs)
        state["elapsed_sec"] = round(time.time() - started, 1)
        write_status(status_file, state)

    work_dir = TEMP_DIR / f"{args.video_id}_{args.mode}_{int(started)}"
    work_dir.mkdir(parents=True, exist_ok=True)
    update()

    try:
        ffmpeg_dir = find_ffmpeg_dir()
        if not ffmpeg_dir:
            raise RuntimeError("搵唔到 FFmpeg，請先安裝 FFmpeg。")
        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

        # Step 1: Download audio via yt-dlp
        import yt_dlp

        out_template = str(work_dir / "source.%(ext)s")

        def dl_hook(d: dict):
            if d.get("status") == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                downloaded = d.get("downloaded_bytes") or 0
                if total > 0:
                    ratio = min(1.0, downloaded / total)
                    pct = int(8 + ratio * 18)
                    update(progress=pct, stage_text=f"下載音訊中... {int(ratio * 100)}%")

        def check_video_limits(info_dict: dict, *, incomplete: bool = False):
            if info_dict.get("is_live"):
                return "唔支援直播影片，請揀選一般 MV 或歌曲。"
            dur = float(info_dict.get("duration") or 0.0)
            if dur > 900:
                return f"影片長度超過 15 分鐘（{int(dur // 60)} 分鐘），為保護磁碟同資源已暫停處理。"
            return None

        cookie_opts = get_yt_cookie_opts()
        js_opts = get_yt_js_runtimes()
        print(
            f"[Worker] yt-dlp init vid={args.video_id} auth={'firefox' if 'cookiesfrombrowser' in cookie_opts else ('cookies.txt' if 'cookiefile' in cookie_opts else 'none')} js={list(js_opts.get('js_runtimes', {}).keys())}",
            file=sys.stderr,
            flush=True,
        )

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "ffmpeg_location": ffmpeg_dir,
            "match_filter": check_video_limits,
            "progress_hooks": [dl_hook],
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "0",
                }
            ],
            **cookie_opts,
            **js_opts,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Single YouTube request (halves rate-limit/bot-check exposure!) with pre-download limit guard
            info = ydl.extract_info(args.url, download=True)
            if not info:
                raise RuntimeError("無法讀取影片資料。")
            duration = float(info.get("duration") or 0.0)
            if info.get("is_live"):
                raise RuntimeError("唔支援直播影片，請揀選一般 MV 或歌曲。")
            if duration > 900:
                raise RuntimeError(f"影片長度超過 15 分鐘（{int(duration // 60)} 分鐘），為保護磁碟同資源已暫停處理。")
            title = info.get("title") or args.video_id
            yt_track = (info.get("track") or "").strip()
            yt_artist = (info.get("artist") or info.get("creator") or "").strip()
            yt_channel = (info.get("channel") or info.get("uploader") or "").strip()
            yt_desc = (info.get("description") or "")[:4000]
            yt_video_lang = (info.get("language") or "").strip()
            yt_sub_lang = ""
            yt_sub_url = ""
            yt_sub_lines: list[dict] = []
            _subs = info.get("subtitles") or {}
            if isinstance(_subs, dict) and _subs:
                try:
                    from lyrics_worker import _pick_youtube_subtitle_lang, _download_json3_subtitles
                    _chosen = _pick_youtube_subtitle_lang(
                        _subs, title=title, video_lang=yt_video_lang, description=yt_desc
                    )
                    if _chosen:
                        _fmts = _subs.get(_chosen) or []
                        _j3 = next((f.get("url") for f in _fmts if f.get("ext") == "json3" and f.get("url")), "")
                        if _j3:
                            yt_sub_lang = _chosen
                            yt_sub_url = _j3
                            # Download & cache json3 lines immediately while YouTube's signed URL is fresh!
                            yt_sub_lines = _download_json3_subtitles(_j3, _chosen) or []
                except Exception:
                    pass

        wav_path = work_dir / "source.wav"
        if not wav_path.exists():
            candidates = list(work_dir.glob("source.*"))
            if not candidates:
                raise RuntimeError("下載音訊失敗，搵唔到音訊檔案。")
            wav_path = candidates[0]

        update(title=title, duration=duration, status="separating", progress=28, stage_text="準備去人聲中...")

        # Step 2: Setup PyTorch, ONNXRuntime & audio-separator with strict CPU thread cap (prevents 100% CPU lockups!)
        cpu_cap = compute_cpu_thread_cap()
        for _env_k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[_env_k] = str(cpu_cap)
        if device_mode == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        import torch
        if device_mode == "cpu":
            torch.cuda.is_available = lambda: False
        try:
            torch.set_num_threads(cpu_cap)
            torch.set_num_interop_threads(min(2, cpu_cap))
        except Exception:
            pass

        # Cap ONNXRuntime CPU threads too (MDXSeparator creates default SessionOptions with intra_op_num_threads=0 = ALL cores!)
        try:
            import onnxruntime as _ort
            _OrigSessionOptions = _ort.SessionOptions

            class _CappedSessionOptions(_OrigSessionOptions):
                def __init__(self, *a, **kw):
                    super().__init__(*a, **kw)
                    self.intra_op_num_threads = cpu_cap
                    self.inter_op_num_threads = 1

            _ort.SessionOptions = _CappedSessionOptions
        except Exception:
            pass

        cuda_ok = (device_mode != "cpu") and torch.cuda.is_available()
        gpu_name = "GPU" if cuda_ok else "CPU"
        if cuda_ok:
            try:
                _gn = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    text=True,
                    timeout=2,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                ).strip().splitlines()
                if _gn:
                    gpu_name = _gn[0].strip()
            except Exception:
                pass

        from audio_separator.separator import Separator
        import tqdm as tqdm_mod
        import audio_separator.separator.architectures.mdxc_separator as mdxc_mod
        import audio_separator.separator.architectures.mdx_separator as mdx_mod

        # Keep overlap-add accumulation buffers on CPU RAM so dynamic GPU<->CPU switching is seamless and zero-VRAM
        mdxc_mod.should_accumulate_on_device = lambda *a, **kw: False

        # Real-time chunk progress reporter (never disables update() and never floods stdout pipes)
        class StatusTqdm:
            def __init__(self, iterable=None, total=None, *a, **kw):
                self.iterable = iterable
                if total is not None:
                    self.total = total
                elif iterable is not None and hasattr(iterable, "__len__"):
                    self.total = len(iterable)
                else:
                    self.total = 0
                self.n = 0
                self._last_report = 0.0

            def __iter__(self):
                if self.iterable is None:
                    return
                for item in self.iterable:
                    yield item
                    self.update(1)

            def update(self, n=1):
                self.n += n
                now = time.time()
                if self.total > 0 and (now - self._last_report >= 0.35 or self.n >= self.total):
                    self._last_report = now
                    frac = min(1.0, self.n / self.total)
                    pct_disp = max(1, int(frac * 100)) if self.n > 0 else 0
                    pct = int(42 + frac * 48)  # 42% -> 90%
                    update(progress=pct, stage_text=f"去人聲中... {pct_disp}%")

            def set_description(self, *a, **kw):
                pass

            def close(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        tqdm_mod.tqdm = StatusTqdm
        mdxc_mod.tqdm = StatusTqdm
        mdx_mod.tqdm = StatusTqdm

        out_dir = work_dir / "stems"
        out_dir.mkdir(parents=True, exist_ok=True)

        def query_nvidia_free_vram_mb() -> float:
            """Query free VRAM via nvidia-smi without creating a ~300MB PyTorch CUDA context."""
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=2,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                ).strip().splitlines()
                if out:
                    return float(out[0].strip())
            except Exception:
                pass
            return 0.0

        initial_free_vram = query_nvidia_free_vram_mb() if cuda_ok else 0.0
        if args.force_hq:
            if (not cuda_ok) or (initial_free_vram < 3000):
                print(
                    f"[Worker] Skipping HQ upgrade: insufficient free VRAM ({initial_free_vram:.0f}MB < 3000MB)",
                    file=sys.stderr,
                    flush=True,
                )
                return 2
            use_cpu_fast_model = False
        else:
            use_cpu_fast_model = (not cuda_ok) or (device_mode == "cpu") or (device_mode == "auto" and initial_free_vram < 1800)
        model_filename = CPU_FAST_MODEL if use_cpu_fast_model else MODELS[args.mode]["filename"]
        print(
            f"[Worker] Selected model={model_filename} mode={device_mode} initial_free_vram={initial_free_vram:.0f}MB",
            file=sys.stderr,
            flush=True,
        )

        separator = Separator(
            model_file_dir=str(MODEL_DIR),
            output_dir=str(out_dir),
            output_format="MP3",
            output_bitrate="320k",
            normalization_threshold=0.9,
            use_autocast=(cuda_ok and not use_cpu_fast_model),
            use_soundfile=False,
            mdx_params={
                "hop_length": 1024,
                "segment_size": 256,
                "overlap": 0.25,
                "batch_size": 1,
                "enable_denoise": False,
            },
            mdxc_params={
                "segment_size": 256,
                "override_model_segment_size": False,
                "batch_size": 1,
                "overlap": 2,  # MelBand Inst FVX: 2x faster on GPU (~12-18s) with fuller untouched instrumental
                "pitch_shift": 0,
            },
        )
        # In "auto" (dynamic) or CPU-fast mode, load onto CPU RAM first (0.0 MB VRAM!) so low VRAM NEVER OOMs!
        if use_cpu_fast_model or (device_mode == "auto"):
            separator.torch_device = torch.device("cpu")
            separator.onnx_execution_provider = ["CPUExecutionProvider"]

        update(progress=34, stage_text="準備去人聲中...")
        separator.load_model(model_filename=model_filename)
        if use_cpu_fast_model:
            state["device_used"] = "CPU Fast AI (MDX-Net HQ3 · 0 MB VRAM)"

        # Install Real-Time Dynamic GPU <-> CPU Scheduler onto the loaded RoFormer instance!
        mi = separator.model_instance
        if (not use_cpu_fast_model) and cuda_ok and hasattr(mi, "model_run") and hasattr(mi, "_run_roformer_model"):
            if device_mode == "gpu":
                mi.model_run.to("cpu")
                torch.cuda.empty_cache()

            dyn_state = {
                "dev": "cpu",
                "gpu_chunks": 0,
                "cpu_chunks": 0,
                "switches": 0,
                "last_smi_check": 0.0,
                "last_smi_free_mb": 0.0,
            }
            cuda_dev = torch.device("cuda")
            cpu_dev = torch.device("cpu")

            def free_vram_mb() -> float:
                # While on CPU, query nvidia-smi (throttled to 1/s) so we touch 0.0 MB VRAM until >=2200 MB is free!
                if dyn_state["dev"] == "cpu":
                    now_t = time.time()
                    if now_t - dyn_state["last_smi_check"] >= 1.0:
                        dyn_state["last_smi_free_mb"] = query_nvidia_free_vram_mb()
                        dyn_state["last_smi_check"] = now_t
                    return dyn_state["last_smi_free_mb"]
                try:
                    free_b, _ = torch.cuda.mem_get_info(0)
                    return free_b / (1024 * 1024)
                except Exception:
                    return 0.0

            def move_model(target: str):
                if dyn_state["dev"] == target:
                    return
                if target == "cuda":
                    mi.model_run.to(cuda_dev)
                    dyn_state["dev"] = "cuda"
                    dyn_state["switches"] += 1
                    state["device_used"] = f"Dynamic GPU ({gpu_name})"
                else:
                    mi.model_run.to(cpu_dev)
                    gc.collect()
                    torch.cuda.empty_cache()
                    dyn_state["dev"] = "cpu"
                    dyn_state["switches"] += 1
                    state["device_used"] = "Dynamic CPU (讓出 VRAM)"

            def dynamic_run_roformer_model(part):
                # Decide optimal device BEFORE EVERY CHUNK (~0.05ms check)
                if device_mode == "gpu":
                    want = "cuda"
                elif args.force_hq:
                    # Idle background HQ upgrade MUST NEVER grind the huge RoFormer transformer on CPU!
                    # If another GPU workload starts mid-song, release 100% VRAM and yield cleanly (rc=2).
                    fmb = free_vram_mb()
                    min_needed = 1200 if dyn_state["dev"] == "cuda" else 3000
                    if fmb < min_needed:
                        move_model("cpu")
                        raise _HQYieldInterrupt(
                            f"Idle HQ upgrade yielded mid-song (free_vram={fmb:.0f}MB < {min_needed}MB)"
                        )
                    want = "cuda"
                else:
                    # Hysteresis: stay on GPU while >= 900 MB free; hop back onto GPU once >= 2200 MB free
                    fmb = free_vram_mb()
                    if dyn_state["dev"] == "cuda":
                        want = "cpu" if fmb < 900 else "cuda"
                    else:
                        want = "cuda" if fmb >= 2200 else "cpu"

                if want == "cuda":
                    try:
                        move_model("cuda")
                        inp = part.to(cuda_dev, non_blocking=True).unsqueeze(0)
                        with torch.amp.autocast("cuda"):
                            out = mi.model_run(inp)[0]
                        dyn_state["gpu_chunks"] += 1
                        return out.detach().cpu()
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                            # Instant OOM recovery: yield 100% VRAM!
                            move_model("cpu")
                            if args.force_hq:
                                raise _HQYieldInterrupt(f"Idle HQ upgrade yielded on CUDA OOM: {e}")
                        else:
                            raise

                # CPU execution path (0 MB VRAM!)
                move_model("cpu")
                inp = part.to(cpu_dev).unsqueeze(0)
                out = mi.model_run(inp)[0]
                dyn_state["cpu_chunks"] += 1
                return out.detach()

            mi._run_roformer_model = dynamic_run_roformer_model
            state["device_used"] = f"Dynamic ({gpu_name} / CPU)"

        update(progress=42, stage_text="去人聲中... 0%")
        output_files = separator.separate(str(wav_path))

        update(status="finalizing", progress=93, stage_text="最後整理中...")

        inst_path = None
        voc_path = None
        resolved = [Path(f) if Path(f).is_absolute() else out_dir / f for f in output_files]
        for p in resolved:
            nl = p.name.lower()
            if "(instrumental)" in nl or "no_vocals" in nl or "accompaniment" in nl or "karaoke" in nl:
                inst_path = p
            elif "(vocals)" in nl or "(vocal)" in nl:
                voc_path = p
        if len(resolved) == 2 and (not inst_path or not voc_path):
            inst_path = inst_path or resolved[0]
            voc_path = voc_path or (resolved[1] if resolved[0] == inst_path else resolved[0])

        if not inst_path or not voc_path or not inst_path.exists() or not voc_path.exists():
            raise RuntimeError(f"AI 輸出音軌不完整：{output_files}")

        cache_dir = target_cache_root / args.video_id / args.mode
        cache_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(inst_path), str(cache_dir / "instrumental.mp3"))
        shutil.move(str(voc_path), str(cache_dir / "vocals.mp3"))

        elapsed = round(time.time() - started, 1)
        meta = {
            "video_id": args.video_id,
            "mode": args.mode,
            "model": model_filename,
            "quality_tier": "cpu_fast" if use_cpu_fast_model else "gpu_hq",
            "title": title,
            "duration": duration,
            "track": yt_track,
            "artist": yt_artist,
            "channel": yt_channel,
            "description": yt_desc,
            "language": yt_video_lang,
            "subtitles_lang": yt_sub_lang,
            "subtitles_url": yt_sub_url,
            "yt_sub_lines": yt_sub_lines,
            "yt_meta_present": True,
            "device_used": state["device_used"],
            "elapsed_sec": elapsed,
            "created_at": time.time(),
        }
        write_json_atomic(cache_dir / "meta.json", meta, stamp_updated_at=False)

        # Leave status at finalizing (95%) — separator.py atomically flips to "ready" (100%)
        # AFTER Google Drive upload & local cleanup finish, preventing premature /api/lyrics races!
        update(
            status="finalizing",
            progress=95,
            elapsed_sec=elapsed,
            stage_text="最後整理中...",
        )
        return 0
    except _HQYieldInterrupt as y_exc:
        print(f"[Worker] {y_exc}", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        clean_msg = friendly_error_message(exc)
        update(
            status="error",
            error=clean_msg,
            stage_text=f"處理失敗：{clean_msg}",
        )
        print(f"[Worker Error] {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
