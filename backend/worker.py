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
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
MODEL_DIR = BASE_DIR / "models"
TEMP_DIR = BASE_DIR / "tmp"

MODELS = {
    "pure": {
        "label": "純伴奏 (BS-RoFormer 1297)",
        "filename": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    },
}


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
    parser.add_argument("--mode", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--url", required=True)
    parser.add_argument("--device-mode", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--status-file", required=True)
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
        "stage_text": "正在從 YouTube 下載音訊...",
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
                    update(progress=pct, stage_text=f"正在下載歌曲音訊... {int(ratio * 100)}%")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "ffmpeg_location": ffmpeg_dir,
            "progress_hooks": [dl_hook],
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "0",
                }
            ],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # First inspect metadata without downloading to block 10-hour streams / oversized videos (> 15 min)
            probe = ydl.extract_info(args.url, download=False)
            duration = float(probe.get("duration") or 0.0)
            if probe.get("is_live"):
                raise RuntimeError("唔支援直播影片，請揀選一般 MV 或歌曲。")
            if duration > 900:
                raise RuntimeError(f"影片長度超過 15 分鐘（{int(duration // 60)} 分鐘），為保護磁碟同資源已暫停處理。")
            info = ydl.extract_info(args.url, download=True)
            title = info.get("title") or args.video_id
            duration = float(info.get("duration") or duration)

        wav_path = work_dir / "source.wav"
        if not wav_path.exists():
            candidates = list(work_dir.glob("source.*"))
            if not candidates:
                raise RuntimeError("下載音訊失敗，搵唔到音訊檔案。")
            wav_path = candidates[0]

        update(title=title, duration=duration, status="separating", progress=28, stage_text="正在載入 AI 去人聲模型...")

        # Step 2: Setup PyTorch & audio-separator
        if device_mode == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        import torch
        if device_mode == "cpu":
            torch.cuda.is_available = lambda: False

        cuda_ok = (device_mode != "cpu") and torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU"

        from audio_separator.separator import Separator
        import tqdm as tqdm_mod
        import audio_separator.separator.architectures.mdxc_separator as mdxc_mod

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
                    pct = int(42 + frac * 48)  # 42% -> 90%
                    update(progress=pct, stage_text=f"AI 正在精準去除人聲... {int(frac * 100)}%")

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

        out_dir = work_dir / "stems"
        out_dir.mkdir(parents=True, exist_ok=True)
        model_filename = MODELS[args.mode]["filename"]

        separator = Separator(
            model_file_dir=str(MODEL_DIR),
            output_dir=str(out_dir),
            output_format="MP3",
            output_bitrate="256k",
            normalization_threshold=0.9,
            use_autocast=cuda_ok,
            use_soundfile=False,
            mdxc_params={
                "segment_size": 256,
                "override_model_segment_size": False,
                "batch_size": 1,
                "overlap": 2,  # 4x faster than overlap=8 while keeping ~12.8dB SDR
                "pitch_shift": 0,
            },
        )
        if not cuda_ok:
            separator.torch_device = torch.device("cpu")
            separator.onnx_execution_provider = ["CPUExecutionProvider"]

        update(progress=34, stage_text="正在準備 AI 去人聲引擎...")
        separator.load_model(model_filename=model_filename)

        # Install Real-Time Dynamic GPU <-> CPU Scheduler onto the loaded RoFormer instance!
        mi = separator.model_instance
        if cuda_ok and hasattr(mi, "model_run") and hasattr(mi, "_run_roformer_model"):
            # Move model initially to CPU so demix() keeps input/accumulation tensors on CPU RAM;
            # our dynamic wrapper moves model onto CUDA on chunk 1 whenever VRAM permits,
            # and smoothly yields VRAM back to CPU mid-song if MiniMax/games need it!
            mi.model_run.to("cpu")
            torch.cuda.empty_cache()

            dyn_state = {
                "dev": "cpu",
                "gpu_chunks": 0,
                "cpu_chunks": 0,
                "switches": 0,
            }
            cuda_dev = torch.device("cuda")
            cpu_dev = torch.device("cpu")

            def free_vram_mb() -> float:
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
                            # Instant OOM recovery: yield 100% VRAM and retry this exact chunk on CPU!
                            move_model("cpu")
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

        update(progress=42, stage_text="AI 正在精準去除人聲... 0%")
        output_files = separator.separate(str(wav_path))

        update(status="finalizing", progress=93, stage_text="正在整理伴奏與人聲軌...")

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
            "title": title,
            "duration": duration,
            "device_used": state["device_used"],
            "elapsed_sec": elapsed,
            "created_at": time.time(),
        }
        (cache_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        update(
            status="ready",
            progress=100,
            elapsed_sec=elapsed,
            stage_text=f"伴奏準備完成！（耗時 {elapsed} 秒）",
        )
        return 0
    except Exception as exc:
        update(
            status="error",
            error=str(exc),
            stage_text=f"處理失敗：{exc}",
        )
        print(f"[Worker Error] {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
