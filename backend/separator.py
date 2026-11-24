import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from logger import CloudSyncedLogger
from storage import GoogleDriveStorage, find_rclone_exe

MODELS: Dict[str, Dict[str, str]] = {
    "pure": {
        "label": "純伴奏 (MelBand RoFormer Inst FVX)",
        "filename": "mel_band_roformer_instrumental_fvx_gabox.ckpt",
        "description": "伴奏優先旗艦模型，完整保留原曲厚度並乾淨去人聲",
    },
}

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.json"
GDRIVE_INDEX_FILE = BASE_DIR / "gdrive_index.json"
DEFAULT_CACHE_DIR = BASE_DIR / "cache"
MODEL_DIR = BASE_DIR / "models"
TEMP_DIR = BASE_DIR / "tmp"
JOBS_DIR = TEMP_DIR / "jobs"
WORKER_SCRIPT = BASE_DIR / "worker.py"
LYRICS_WORKER_SCRIPT = BASE_DIR / "lyrics_worker.py"

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000

for d in (DEFAULT_CACHE_DIR, MODEL_DIR, TEMP_DIR, JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def find_ffmpeg_dir() -> Optional[str]:
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


def query_nvidia_gpu() -> tuple[bool, str, float, float]:
    """Query GPU name and live free/total VRAM via nvidia-smi without importing PyTorch (0 MB VRAM!)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
            creationflags=CREATE_NO_WINDOW,
        ).strip()
        first = out.splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) >= 3:
            return (True, parts[0], float(parts[1]), float(parts[2]))
    except Exception:
        pass
    return (False, "CPU Only", 0.0, 0.0)


class KaraokeSeparatorManager:
    """Zero-VRAM manager with multi-user anti-overload queue, rate limits, and Google Drive cloud cache."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._worker_lock = threading.Lock()
        self._lyrics_lock = threading.Lock()
        self._active_procs: Dict[str, subprocess.Popen] = {}
        self._queued_keys: set[str] = set()
        self._rate_history: Dict[str, list[float]] = {}
        self._lyrics_mem_cache: Dict[str, dict] = {}
        self._lyrics_gdrive_missing: set[str] = set()
        self._lyrics_in_progress: set[str] = set()
        self._hq_upgrade_backoff: Dict[str, float] = {}
        self._hq_upgrade_min_vram_mb = int(os.environ.get("KARAOKE_HQ_UPGRADE_VRAM_MB", "3500"))

        self.cuda_available, self.gpu_name, _, _ = query_nvidia_gpu()
        env_mode = os.environ.get("KARAOKE_DEVICE", "auto").strip().lower()
        if env_mode not in ("auto", "gpu", "cpu"):
            env_mode = "auto"
        self.device_mode = env_mode if self.cuda_available else "cpu"
        self.min_free_vram_mb = int(os.environ.get("KARAOKE_MIN_FREE_VRAM_MB", "1800"))
        self.ffmpeg_dir = find_ffmpeg_dir()

        # Storage & Overload Protection defaults (persisted in config.json)
        self.cache_dir = DEFAULT_CACHE_DIR
        self.max_cache_mb = 1000
        self.storage_mode = "local"  # "local" | "gdrive"
        self.pause_new_jobs = False
        self.max_queue_size = 3
        self.rate_limit_per_10min = 5
        self.max_clients = 100

        self.gdrive = GoogleDriveStorage(GDRIVE_INDEX_FILE)
        self.log = CloudSyncedLogger(TEMP_DIR, find_rclone_exe, self.gdrive.is_configured)
        self._migration_state = {
            "running": False,
            "done": 0,
            "total": 0,
            "freed_mb": 0.0,
            "message": "",
            "error": None,
        }
        self._load_config()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Always prefer Google Drive cloud mode automatically once connected!
        if self.gdrive.is_configured():
            self.storage_mode = "gdrive"
            self.save_config()
            # Only scan remote on boot if local index is empty; otherwise use instant gdrive_index.json!
            if not self.gdrive.list_items():
                threading.Thread(target=self.gdrive.sync_index_from_remote, daemon=True, name="gdrive-boot-sync").start()
        elif self.storage_mode == "gdrive":
            self.storage_mode = "local"
        self.log.info(
            "Startup",
            f"Backend ready | gpu={self.gpu_name} device_mode={self.device_mode} storage={self.storage_mode} gdrive={self.gdrive.is_configured()}",
        )

        # Clean stale temp jobs from prior session
        for p in TEMP_DIR.iterdir():
            if p.name != "jobs" and p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        for p in JOBS_DIR.glob("*.json"):
            try:
                p.unlink()
            except Exception:
                pass

        # Background zero-local-disk reaper: automatically retries & migrates any song stranded on local disk
        # (e.g. after a transient Google Drive 403 rate-limit spike) so local disk ALWAYS returns to 0.0 MB!
        threading.Thread(target=self._bg_gdrive_reaper_loop, daemon=True, name="gdrive-auto-reaper").start()

        # Background idle HQ upgrader: when GPU has ample free VRAM (>= 3.5 GB) and no user jobs are active,
        # silently re-separates any 'cpu_fast' (MDX-Net HQ3) songs with BS-RoFormer 1297 and replaces them in cache!
        threading.Thread(target=self._bg_gpu_hq_upgrader_loop, daemon=True, name="hq-idle-upgrader").start()

    def count_pending_hq_upgrades(self) -> int:
        return sum(1 for item in self.list_cache() if item.get("quality_tier") == "cpu_fast")

    def _bg_gpu_hq_upgrader_loop(self) -> None:
        time.sleep(45)
        while True:
            try:
                self._maybe_upgrade_one_cpu_song()
            except Exception as exc:
                self.log.warning("HQUpgrade", f"Idle upgrader loop warning: {str(exc)[:240]}")
            time.sleep(120)

    def _maybe_upgrade_one_cpu_song(self) -> None:
        if self.pause_new_jobs or self.device_mode == "cpu":
            return
        with self._lock:
            if self._queued_keys or self._lyrics_in_progress or self._migration_state.get("running"):
                return
        has_gpu, _, free_mb, _ = query_nvidia_gpu()
        if (not has_gpu) or (free_mb < self._hq_upgrade_min_vram_mb):
            return

        now = time.time()
        candidate = None
        for item in self.list_cache():
            if item.get("quality_tier") != "cpu_fast":
                continue
            key = self._job_key(item["video_id"], item["mode"])
            if self._hq_upgrade_backoff.get(key, 0.0) > now:
                continue
            candidate = item
            break
        if not candidate:
            return

        if not self._worker_lock.acquire(blocking=False):
            return

        vid = candidate["video_id"]
        mode = candidate["mode"]
        key = self._job_key(vid, mode)
        staging_root = TEMP_DIR / f"hq_upgrade_{vid}_{int(now)}"
        staging_status = TEMP_DIR / f"hq_upgrade_{vid}_{int(now)}.json"
        try:
            # Re-verify idle state & VRAM right after acquiring lock
            with self._lock:
                if self.pause_new_jobs or self._queued_keys or self._lyrics_in_progress:
                    return
            has_gpu, _, free_mb, _ = query_nvidia_gpu()
            if (not has_gpu) or (free_mb < self._hq_upgrade_min_vram_mb):
                return

            staging_root.mkdir(parents=True, exist_ok=True)
            self.log.info(
                "HQUpgrade",
                f"Start idle MelBand Inst FVX upgrade for {vid} ({mode}) | free_vram={free_mb:.0f}MB",
            )
            target_url = f"https://www.youtube.com/watch?v={vid}"
            cmd = [
                sys.executable,
                "-u",
                str(WORKER_SCRIPT),
                f"--video-id={vid}",
                f"--mode={mode}",
                f"--url={target_url}",
                "--device-mode=auto",
                "--force-hq",
                f"--cache-dir={staging_root}",
                f"--status-file={staging_status}",
            ]
            proc = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                env=os.environ.copy(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS,
            )
            try:
                _, stderr_text = proc.communicate(timeout=360)
            except subprocess.TimeoutExpired:
                proc.kill()
                _, stderr_text = proc.communicate()
                self._hq_upgrade_backoff[key] = time.time() + 21600
                self.log.warning("HQUpgrade", f"Timed out upgrading {vid} ({mode}); will retry in 6h")
                return

            if proc.returncode == 2:
                # VRAM dipped before model load — skip cleanly without backoff
                return
            if proc.returncode != 0:
                self._hq_upgrade_backoff[key] = time.time() + 21600
                self.log.warning(
                    "HQUpgrade",
                    f"Upgrade skipped for {vid} ({mode}, rc={proc.returncode}), retry in 6h: {(stderr_text or '')[-400:]}",
                )
                return

            staged_song_dir = staging_root / vid / mode
            if not (staged_song_dir / "instrumental.mp3").exists() or not (staged_song_dir / "vocals.mp3").exists():
                self._hq_upgrade_backoff[key] = time.time() + 21600
                return

            # Preserve original created_at so library ordering stays intact
            meta_file = staged_song_dir / "meta.json"
            upgraded_meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
            if candidate.get("created_at"):
                upgraded_meta["created_at"] = candidate["created_at"]
            upgraded_meta["quality_tier"] = "gpu_hq"
            upgraded_meta["upgraded_at"] = time.time()
            meta_file.write_text(json.dumps(upgraded_meta, ensure_ascii=False, indent=2), encoding="utf-8")

            if self.storage_mode == "gdrive" and self.gdrive.is_configured():
                self.gdrive.upload_song(vid, mode, staged_song_dir, upgraded_meta)
                # Remove any stale local copy if present
                local_dir = self.get_cache_dir(vid, mode)
                if local_dir.exists():
                    shutil.rmtree(local_dir, ignore_errors=True)
            else:
                local_dir = self.get_cache_dir(vid, mode)
                local_dir.mkdir(parents=True, exist_ok=True)
                for fname in ("instrumental.mp3", "vocals.mp3", "meta.json"):
                    src = staged_song_dir / fname
                    if src.exists():
                        os.replace(src, local_dir / fname)

            self._hq_upgrade_backoff.pop(key, None)
            self.log.info(
                "HQUpgrade",
                f"Upgraded {vid} ({mode}) cpu_fast -> gpu_hq (MelBand Inst FVX) in {upgraded_meta.get('elapsed_sec', '?')}s",
            )
            self.log.trigger_sync_async()
        except Exception as exc:
            self._hq_upgrade_backoff[key] = time.time() + 21600
            self.log.warning("HQUpgrade", f"Failed upgrading {vid} ({mode}), retry in 6h: {str(exc)[:240]}")
        finally:
            self._worker_lock.release()
            shutil.rmtree(staging_root, ignore_errors=True)
            try:
                staging_status.unlink(missing_ok=True)
            except Exception:
                pass

    def _bg_gdrive_reaper_loop(self) -> None:
        time.sleep(20)
        while True:
            try:
                if self.storage_mode == "gdrive" and self.gdrive.is_configured():
                    with self._lock:
                        busy_mig = bool(self._migration_state.get("running"))
                        queued = set(self._queued_keys)
                        busy_lyrics = set(self._lyrics_in_progress)
                    if not busy_mig:
                        local_items = self._list_local_cache()
                        for it in local_items:
                            vid, mode = it["video_id"], it["mode"]
                            if self._job_key(vid, mode) in queued or vid in busy_lyrics:
                                continue
                            local_dir = self.get_cache_dir(vid, mode)
                            if not local_dir.exists():
                                continue
                            meta = self.get_meta(vid, mode)
                            try:
                                self.gdrive.upload_song(vid, mode, local_dir, meta)
                                ly = self._lyrics_file(vid)
                                if ly.exists():
                                    self.gdrive.upload_single_file(vid, mode, ly)
                                    try:
                                        ly.unlink()
                                    except Exception:
                                        pass
                                shutil.rmtree(local_dir, ignore_errors=True)
                                parent = self.cache_dir / vid
                                if parent.exists() and not any(parent.iterdir()):
                                    try:
                                        parent.rmdir()
                                    except Exception:
                                        pass
                                self.log.info("GDrive", f"Auto-reaper uploaded {vid} ({mode}) & freed local disk")
                                self.log.trigger_sync_async()
                            except Exception as exc:
                                self.log.warning("GDrive", f"Auto-reaper will retry {vid} later: {str(exc)[:180]}")
                                time.sleep(15)
            except Exception:
                pass
            time.sleep(90)

    def _load_config(self) -> None:
        if CONFIG_FILE.exists():
            try:
                cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                if cfg.get("cache_dir"):
                    p = Path(cfg["cache_dir"])
                    if p.exists() or p.parent.exists():
                        self.cache_dir = p
                if cfg.get("max_cache_mb"):
                    self.max_cache_mb = max(100, int(cfg["max_cache_mb"]))
                if cfg.get("storage_mode") in ("local", "gdrive"):
                    self.storage_mode = cfg["storage_mode"]
                if "pause_new_jobs" in cfg:
                    self.pause_new_jobs = bool(cfg["pause_new_jobs"])
                if cfg.get("max_queue_size"):
                    self.max_queue_size = max(1, min(20, int(cfg["max_queue_size"])))
                if cfg.get("rate_limit_per_10min"):
                    self.rate_limit_per_10min = max(1, min(50, int(cfg["rate_limit_per_10min"])))
                if cfg.get("max_clients"):
                    self.max_clients = max(5, min(500, int(cfg["max_clients"])))
            except Exception:
                pass

    def save_config(self) -> None:
        try:
            CONFIG_FILE.write_text(
                json.dumps(
                    {
                        "cache_dir": str(self.cache_dir),
                        "max_cache_mb": self.max_cache_mb,
                        "storage_mode": self.storage_mode,
                        "pause_new_jobs": self.pause_new_jobs,
                        "max_queue_size": self.max_queue_size,
                        "rate_limit_per_10min": self.rate_limit_per_10min,
                        "max_clients": self.max_clients,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def set_cache_dir(self, new_dir: Path) -> None:
        new_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = new_dir
        self.save_config()

    def enforce_cache_limit(self) -> None:
        items = self._list_local_cache()
        total_mb = sum(i["size_mb"] for i in items)
        if total_mb <= self.max_cache_mb:
            return
        for old in reversed(items):
            if total_mb <= self.max_cache_mb * 0.85:
                break
            target = self.get_cache_dir(old["video_id"], old["mode"])
            shutil.rmtree(target, ignore_errors=True)
            total_mb -= old["size_mb"]
            parent = self.cache_dir / old["video_id"]
            if parent.exists() and not any(parent.iterdir()):
                try:
                    parent.rmdir()
                except Exception:
                    pass

    def _job_key(self, video_id: str, mode: str) -> str:
        return f"{video_id}__{mode}"

    def _status_file(self, video_id: str, mode: str) -> Path:
        return JOBS_DIR / f"{video_id}__{mode}.json"

    def get_cache_dir(self, video_id: str, mode: str) -> Path:
        return self.cache_dir / video_id / mode

    def is_local_cached(self, video_id: str, mode: str) -> bool:
        cdir = self.get_cache_dir(video_id, mode)
        return (cdir / "instrumental.mp3").exists() and (cdir / "vocals.mp3").exists()

    def is_cached(self, video_id: str, mode: str) -> bool:
        return self.is_local_cached(video_id, mode) or self.gdrive.has_song(video_id, mode)

    def get_cached_modes(self, video_id: str) -> list[str]:
        return [m for m in MODELS if self.is_cached(video_id, m)]

    def get_meta(self, video_id: str, mode: str) -> dict:
        meta_path = self.get_cache_dir(video_id, mode) / "meta.json"
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return self.gdrive.get_meta(video_id, mode)

    def _read_job_status(self, video_id: str, mode: str) -> Optional[dict]:
        sf = self._status_file(video_id, mode)
        if not sf.exists():
            return None
        try:
            return json.loads(sf.read_text(encoding="utf-8"))
        except Exception:
            return None

    def get_queue_count(self) -> int:
        with self._lock:
            return len(self._queued_keys)

    def check_rate_limit(self, client_ip: str) -> bool:
        """Returns True if allowed, False if IP exceeded rate_limit_per_10min."""
        now = time.time()
        window = 600.0
        with self._lock:
            hist = [t for t in self._rate_history.get(client_ip, []) if now - t < window]
            if len(hist) >= self.rate_limit_per_10min:
                self._rate_history[client_ip] = hist
                return False
            hist.append(now)
            self._rate_history[client_ip] = hist
            return True

    def get_status(self, video_id: str, mode: str) -> dict:
        cached_modes = self.get_cached_modes(video_id)
        if self.is_cached(video_id, mode):
            meta = self.get_meta(video_id, mode)
            return {
                "video_id": video_id,
                "mode": mode,
                "status": "ready",
                "progress": 100,
                "stage_text": "已就緒，隨時可以開唱",
                "error": None,
                "title": meta.get("title", ""),
                "duration": meta.get("duration", 0),
                "device_used": meta.get("device_used", ""),
                "cached": True,
                "cached_modes": cached_modes,
            }

        job_data = self._read_job_status(video_id, mode)
        if job_data:
            job_data["cached"] = False
            job_data["cached_modes"] = cached_modes
            return job_data

        return {
            "video_id": video_id,
            "mode": mode,
            "status": "idle",
            "progress": 0,
            "stage_text": "未處理",
            "error": None,
            "title": "",
            "duration": 0,
            "cached": False,
            "cached_modes": cached_modes,
        }

    def start_separation(self, video_id: str, mode: str, client_ip: str = "127.0.0.1") -> dict:
        if mode not in MODELS:
            raise ValueError(f"Unsupported mode: {mode}")

        if self.is_cached(video_id, mode):
            return self.get_status(video_id, mode)

        key = self._job_key(video_id, mode)
        with self._lock:
            if key in self._queued_keys:
                return self.get_status(video_id, mode)

            if self.pause_new_jobs:
                raise PermissionError("管理員已暫停新歌拆軌（僅供播放現有快取歌曲），請稍後再試。")

            if len(self._queued_keys) >= self.max_queue_size:
                raise OverflowError(f"目前已有 {len(self._queued_keys)} 首歌曲排隊處理中，請等幾十秒再試。")

        if not self.check_rate_limit(client_ip):
            raise OverflowError(f"請求過於頻繁（每 10 分鐘上限 {self.rate_limit_per_10min} 首新歌），請稍後再試。")

        sf = self._status_file(video_id, mode)
        with self._lock:
            self._queued_keys.add(key)
            q_pos = len(self._queued_keys)
            init_payload = {
                "video_id": video_id,
                "mode": mode,
                "status": "queued",
                "progress": 4,
                "stage_text": f"排隊處理中（第 {q_pos} 位）..." if q_pos > 1 else "準備處理中...",
                "error": None,
                "title": "",
                "duration": 0,
                "updated_at": time.time(),
            }
            sf.write_text(json.dumps(init_payload, ensure_ascii=False), encoding="utf-8")

        target_url = f"https://www.youtube.com/watch?v={video_id}"
        t = threading.Thread(
            target=self._run_worker_subprocess,
            args=(video_id, mode, target_url, sf),
            daemon=True,
            name=f"worker-mgr-{video_id}-{mode}",
        )
        t.start()
        return self.get_status(video_id, mode)

    def _run_worker_subprocess(self, video_id: str, mode: str, target_url: str, status_file: Path) -> None:
        key = self._job_key(video_id, mode)
        try:
            with self._worker_lock:
                has_gpu, gpu_name, free_mb, _ = query_nvidia_gpu()
                effective_mode = self.device_mode if has_gpu else "cpu"
                self.log.info(
                    "Separate",
                    f"Start {video_id} ({mode}) mode={effective_mode} free_vram={free_mb:.0f}MB",
                )

                # Use --flag=value syntax so YouTube video IDs starting with '-' (e.g. -8HPO9tXnZU) never break argparse!
                cmd = [
                    sys.executable,
                    "-u",
                    str(WORKER_SCRIPT),
                    f"--video-id={video_id}",
                    f"--mode={mode}",
                    f"--url={target_url}",
                    f"--device-mode={effective_mode}",
                    f"--cache-dir={self.cache_dir}",
                    f"--status-file={status_file}",
                ]

                env = os.environ.copy()
                if effective_mode == "cpu":
                    env["CUDA_VISIBLE_DEVICES"] = "-1"

                # BELOW_NORMAL_PRIORITY_CLASS guarantees zero lag for games/desktop even on CPU!
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(BASE_DIR),
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS,
                )
                with self._lock:
                    self._active_procs[key] = proc

                try:
                    _, stderr_text = proc.communicate(timeout=600)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    _, stderr_text = proc.communicate()
                    self.log.error(
                        "Separate",
                        f"Worker timeout ({video_id}, {mode}) after 600s | stderr: {(stderr_text or '')[-1500:]}",
                    )
                    raise RuntimeError("處理逾時（超過 10 分鐘），已自動中止保護系統資源。")
                finally:
                    with self._lock:
                        self._active_procs.pop(key, None)

                if proc.returncode != 0:
                    cur = self._read_job_status(video_id, mode) or {}
                    if cur.get("status") != "error":
                        cur.update(
                            {
                                "video_id": video_id,
                                "mode": mode,
                                "status": "error",
                                "error": "暫時無法處理此影片，請稍後再試",
                                "stage_text": "處理失敗：暫時無法處理此影片，請稍後再試",
                                "updated_at": time.time(),
                            }
                        )
                        status_file.write_text(json.dumps(cur, ensure_ascii=False), encoding="utf-8")
                    self.log.error(
                        "Separate",
                        f"Worker failed ({video_id}, {mode}, rc={proc.returncode}): {(stderr_text or '')[-1500:]}",
                    )
                    return

                # If Google Drive storage is active, upload immediately and delete local MP3s (0 MB local disk!)
                pre_voc_hint: Optional[Path] = None
                pre_inst_hint: Optional[Path] = None
                if self.storage_mode == "gdrive" and self.gdrive.is_configured():
                    local_song_dir = self.get_cache_dir(video_id, mode)
                    meta = self.get_meta(video_id, mode)
                    try:
                        cur = self._read_job_status(video_id, mode) or {}
                        cur.update({"status": "finalizing", "progress": 96, "stage_text": "最後整理中..."})
                        status_file.write_text(json.dumps(cur, ensure_ascii=False), encoding="utf-8")

                        # Keep tiny temporary copies of vocals & instrumental so background lyrics & Shazam run at 0s download!
                        local_voc = local_song_dir / "vocals.mp3"
                        local_inst = local_song_dir / "instrumental.mp3"
                        if local_voc.exists():
                            try:
                                pre_voc_hint = TEMP_DIR / f"pre_voc_{video_id}_{int(time.time())}.mp3"
                                shutil.copy2(local_voc, pre_voc_hint)
                            except Exception:
                                pre_voc_hint = None
                        if local_inst.exists():
                            try:
                                pre_inst_hint = TEMP_DIR / f"pre_inst_{video_id}_{int(time.time())}.mp3"
                                shutil.copy2(local_inst, pre_inst_hint)
                            except Exception:
                                pre_inst_hint = None

                        self.gdrive.upload_song(video_id, mode, local_song_dir, meta)
                        shutil.rmtree(local_song_dir, ignore_errors=True)
                        parent = self.cache_dir / video_id
                        if parent.exists() and not any(parent.iterdir()):
                            try:
                                parent.rmdir()
                            except Exception:
                                pass

                        cur.update({"status": "ready", "progress": 100, "stage_text": "伴奏已就緒"})
                        status_file.write_text(json.dumps(cur, ensure_ascii=False), encoding="utf-8")
                        self.log.info("GDrive", f"Uploaded {video_id} ({mode}) & freed local disk")
                        self.log.trigger_sync_async()
                    except Exception as gerr:
                        self.log.error("GDrive", f"Upload fallback to local ({video_id}): {gerr}")
                        self.enforce_cache_limit()
                else:
                    self.enforce_cache_limit()
                    self.log.info("Separate", f"Done {video_id} ({mode}) — 100% VRAM reclaimed")
                    self.log.trigger_sync_async()

                # Auto pre-generate lyrics in background while stems are hot on SSD -> 0.05s instant lyrics!
                try:
                    self.start_lyrics_generation(
                        video_id, mode, force_ai=False,
                        local_vocals_hint=pre_voc_hint, local_inst_hint=pre_inst_hint,
                    )
                except Exception:
                    for _p in (pre_voc_hint, pre_inst_hint):
                        if _p and _p.exists():
                            try:
                                _p.unlink()
                            except Exception:
                                pass
        except Exception as exc:
            self.log.error("Separate", f"Exception ({video_id}, {mode}): {exc}")
            cur = self._read_job_status(video_id, mode) or {}
            cur.update(
                {
                    "video_id": video_id,
                    "mode": mode,
                    "status": "error",
                    "error": str(exc),
                    "stage_text": f"處理失敗：{exc}",
                    "updated_at": time.time(),
                }
            )
            try:
                status_file.write_text(json.dumps(cur, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
        finally:
            with self._lock:
                self._queued_keys.discard(key)

    def get_migration_state(self) -> dict:
        with self._lock:
            return dict(self._migration_state)

    def start_migrate_local_to_gdrive(self) -> dict:
        """Start non-blocking background migration of all local cached songs to Google Drive."""
        if not self.gdrive.is_configured():
            raise RuntimeError("尚未連接 Google Drive，請先喺控制台登入 Google Drive。")

        with self._lock:
            if self._migration_state["running"]:
                return dict(self._migration_state)
            local_items = self._list_local_cache()
            # Switch to gdrive mode immediately so any new songs go straight to Google Drive too!
            self.storage_mode = "gdrive"
            self.save_config()

            if not local_items:
                self._migration_state = {
                    "running": False,
                    "done": 0,
                    "total": 0,
                    "freed_mb": 0.0,
                    "message": "已切換至 Google Drive 雲端模式（無本地檔需遷移）",
                    "error": None,
                }
                return dict(self._migration_state)

            self._migration_state = {
                "running": True,
                "done": 0,
                "total": len(local_items),
                "freed_mb": 0.0,
                "message": f"遷移中 0/{len(local_items)} 首...",
                "error": None,
            }

        self.log.info("Migrate", f"Starting background Google Drive migration for {len(local_items)} local songs")
        t = threading.Thread(target=self._run_migration, args=(local_items,), daemon=True, name="gdrive-migrate")
        t.start()
        return self.get_migration_state()

    def _run_migration(self, local_items: list[dict]) -> None:
        done = 0
        freed_mb = 0.0
        errors = []
        total = len(local_items)

        for item in local_items:
            vid = item["video_id"]
            mode = item["mode"]
            title = item.get("title", vid)
            local_dir = self.get_cache_dir(vid, mode)
            if not local_dir.exists():
                done += 1
                continue

            with self._lock:
                self._migration_state["message"] = f"上傳中 ({done + 1}/{total})：{title[:16]}"
            try:
                meta = self.get_meta(vid, mode)
                self.gdrive.upload_song(vid, mode, local_dir, meta)
                # Also upload lyrics.json if present for this video
                lf = self._lyrics_file(vid)
                if lf.exists():
                    self.gdrive.upload_single_file(vid, mode, lf)
                    try:
                        lf.unlink()
                    except Exception:
                        pass
                shutil.rmtree(local_dir, ignore_errors=True)
                parent = self.cache_dir / vid
                if parent.exists() and not any(parent.iterdir()):
                    try:
                        parent.rmdir()
                    except Exception:
                        pass
                done += 1
                freed_mb += float(item.get("size_mb", 0))
                self.log.info("Migrate", f"Uploaded ({done}/{total}) {vid} ({mode}) {item.get('size_mb', 0)}MB")
                with self._lock:
                    self._migration_state["done"] = done
                    self._migration_state["freed_mb"] = round(freed_mb, 2)
                    self._migration_state["message"] = f"已完成 {done}/{total} 首（釋放 {freed_mb:.1f} MB）"
            except Exception as exc:
                err_txt = f"{vid}: {exc}"
                errors.append(err_txt)
                self.log.error("Migrate", f"Failed migrating {err_txt}")

        with self._lock:
            self._migration_state["running"] = False
            self._migration_state["done"] = done
            self._migration_state["freed_mb"] = round(freed_mb, 2)
            if errors:
                self._migration_state["error"] = "; ".join(errors[:2])
                self._migration_state["message"] = f"遷移部分完成 ({done}/{total})，有 {len(errors)} 個錯誤"
            else:
                self._migration_state["error"] = None
                self._migration_state["message"] = f"全部 {done} 首已上 Google Drive（釋放 {freed_mb:.1f} MB）"

        self.log.info("Migrate", self._migration_state["message"])
        self.log.trigger_sync_async()

    def get_vram_mb(self) -> float:
        return 0.0

    def get_vram_free_total_mb(self) -> tuple[float, float]:
        has_gpu, gpu_name, free_mb, total_mb = query_nvidia_gpu()
        if has_gpu:
            self.cuda_available = True
            self.gpu_name = gpu_name
            return (round(free_mb, 1), round(total_mb, 1))
        return (0.0, 0.0)

    def _list_local_cache(self) -> list[dict]:
        items = []
        if not self.cache_dir.exists():
            return items
        for vid_dir in sorted(self.cache_dir.iterdir()):
            if not vid_dir.is_dir():
                continue
            for mode_dir in sorted(vid_dir.iterdir()):
                if not mode_dir.is_dir():
                    continue
                inst = mode_dir / "instrumental.mp3"
                voc = mode_dir / "vocals.mp3"
                if inst.exists() and voc.exists():
                    meta = self.get_meta(vid_dir.name, mode_dir.name)
                    size_bytes = sum(f.stat().st_size for f in mode_dir.glob("*") if f.is_file())
                    model_name = meta.get("model", "")
                    quality_tier = meta.get("quality_tier") or (
                        "cpu_fast" if model_name == "UVR-MDX-NET-Inst_HQ_3.onnx" else "gpu_hq"
                    )
                    items.append(
                        {
                            "video_id": vid_dir.name,
                            "mode": mode_dir.name,
                            "title": meta.get("title", vid_dir.name),
                            "duration": meta.get("duration", 0),
                            "size_mb": round(size_bytes / (1024 * 1024), 2),
                            "created_at": meta.get("created_at", 0),
                            "quality_tier": quality_tier,
                            "model": model_name,
                            "storage": "local",
                            "url": f"https://www.youtube.com/watch?v={vid_dir.name}",
                        }
                    )
        items.sort(key=lambda x: x["created_at"], reverse=True)
        return items

    def list_cache(self) -> list[dict]:
        by_key: Dict[str, dict] = {}
        for item in self.gdrive.list_items():
            by_key[self._job_key(item["video_id"], item["mode"])] = item
        for item in self._list_local_cache():
            by_key[self._job_key(item["video_id"], item["mode"])] = item
        items = list(by_key.values())
        items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return items

    def delete_cache(self, video_id: Optional[str] = None, mode: Optional[str] = None) -> int:
        removed = 0
        if video_id:
            target = (self.cache_dir / video_id / mode) if mode else (self.cache_dir / video_id)
            if target.exists() and target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
                removed += 1
            if self.gdrive.is_configured():
                removed += self.gdrive.delete_song(video_id, mode)
        else:
            if self.cache_dir.exists():
                for child in self.cache_dir.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                        removed += 1
            if self.gdrive.is_configured():
                removed += self.gdrive.purge_all()
        return removed

    # -------------------- 3-Layer Hybrid Lyrics (On-Demand) --------------------
    def _lyrics_file(self, video_id: str) -> Path:
        return self.cache_dir / video_id / "lyrics.json"

    def _lyrics_status_file(self, video_id: str) -> Path:
        return JOBS_DIR / f"lyrics__{video_id}.json"

    def get_lyrics(self, video_id: str, mode: str = "pure") -> dict:
        # 1) Ultra-fast in-memory cache (~4KB/song, 0 MB local disk, <1ms response!)
        with self._lock:
            is_running = video_id in self._lyrics_in_progress
            if not is_running:
                mem = self._lyrics_mem_cache.get(video_id)
                if mem and mem.get("lines"):
                    return {"status": "ready", "progress": 100, "stage_text": mem.get("source_label", "歌詞已就緒"), **mem}

        # 2) If worker is currently running, return status immediately without hitting Google Drive!
        sf = self._lyrics_status_file(video_id)
        if is_running and sf.exists():
            try:
                return json.loads(sf.read_text(encoding="utf-8"))
            except Exception:
                pass

        # 3) Check local temp lyrics file if present
        lf = self._lyrics_file(video_id)
        if lf.exists():
            try:
                data = json.loads(lf.read_text(encoding="utf-8"))
                if data.get("lines"):
                    with self._lock:
                        self._lyrics_mem_cache[video_id] = data
                    return {"status": "ready", "progress": 100, "stage_text": data.get("source_label", "歌詞已就緒"), **data}
            except Exception:
                pass

        # 4) Fetch once from Google Drive (skip if already known absent this session)
        if (
            not is_running
            and video_id not in self._lyrics_gdrive_missing
            and self.gdrive.is_configured()
            and self.gdrive.has_song(video_id, mode)
        ):
            remote_data = self.gdrive.read_json(video_id, mode, "lyrics.json")
            if remote_data and remote_data.get("lines"):
                with self._lock:
                    self._lyrics_mem_cache[video_id] = remote_data
                return {"status": "ready", "progress": 100, "stage_text": remote_data.get("source_label", "歌詞已就緒"), **remote_data}
            else:
                self._lyrics_gdrive_missing.add(video_id)

        if sf.exists():
            try:
                return json.loads(sf.read_text(encoding="utf-8"))
            except Exception:
                pass

        return {"video_id": video_id, "status": "idle", "progress": 0, "stage_text": "未生成歌詞", "lines": []}

    def start_lyrics_generation(
        self,
        video_id: str,
        mode: str = "pure",
        force_ai: bool = False,
        local_vocals_hint: Optional[Path] = None,
        local_inst_hint: Optional[Path] = None,
    ) -> dict:
        def _cleanup_hints():
            for _p in (local_vocals_hint, local_inst_hint):
                if _p and _p.exists():
                    try:
                        _p.unlink()
                    except Exception:
                        pass

        if not force_ai:
            existing = self.get_lyrics(video_id, mode)
            if existing.get("status") == "ready":
                _cleanup_hints()
                return existing
        else:
            with self._lock:
                self._lyrics_mem_cache.pop(video_id, None)
                self._lyrics_gdrive_missing.discard(video_id)

        with self._lock:
            if video_id in self._lyrics_in_progress:
                _cleanup_hints()
                return self.get_lyrics(video_id, mode)
            self._lyrics_in_progress.add(video_id)

            sf = self._lyrics_status_file(video_id)
            init = {
                "video_id": video_id,
                "status": "running",
                "progress": 8,
                "stage_text": "對齊歌詞中..." if force_ai else "載入歌詞中...",
                "error": None,
                "updated_at": time.time(),
            }
            sf.write_text(json.dumps(init, ensure_ascii=False), encoding="utf-8")

        t = threading.Thread(
            target=self._run_lyrics_worker,
            args=(video_id, mode, force_ai, sf, local_vocals_hint, local_inst_hint),
            daemon=True,
            name=f"lyrics-{video_id}",
        )
        t.start()
        return init

    def _run_lyrics_worker(
        self,
        video_id: str,
        mode: str,
        force_ai: bool,
        status_file: Path,
        local_vocals_hint: Optional[Path] = None,
        local_inst_hint: Optional[Path] = None,
    ) -> None:
        key = f"lyrics__{video_id}"
        temp_vocals: Optional[Path] = local_vocals_hint
        temp_inst: Optional[Path] = local_inst_hint
        try:
            with self._lyrics_lock:
                if local_vocals_hint and local_vocals_hint.exists():
                    vocals_path = local_vocals_hint
                else:
                    vocals_path = self.get_cache_dir(video_id, mode) / "vocals.mp3"
                    if not vocals_path.exists() and self.gdrive.is_configured() and self.gdrive.has_song(video_id, mode):
                        temp_vocals = TEMP_DIR / f"voc_{video_id}_{int(time.time())}.mp3"
                        if self.gdrive.download_file(video_id, mode, "vocals.mp3", temp_vocals):
                            vocals_path = temp_vocals

                inst_path = local_inst_hint if (local_inst_hint and local_inst_hint.exists()) else (self.get_cache_dir(video_id, mode) / "instrumental.mp3")
                out_file = self._lyrics_file(video_id)
                has_gpu, _, _, _ = query_nvidia_gpu()
                eff_mode = self.device_mode if has_gpu else "cpu"
                meta = self.get_meta(video_id, mode)

                # Use --flag=value syntax so video IDs or song titles starting with '-' (e.g. -8HPO9tXnZU) never break argparse!
                cmd = [
                    sys.executable,
                    "-u",
                    str(LYRICS_WORKER_SCRIPT),
                    f"--video-id={video_id}",
                    f"--vocals-path={vocals_path}",
                    f"--out-file={out_file}",
                    f"--status-file={status_file}",
                    f"--device-mode={eff_mode}",
                    f"--force-ai={'1' if force_ai else '0'}",
                    f"--title={meta.get('title') or ''}",
                    f"--duration={float(meta.get('duration') or 0.0)}",
                ]
                if inst_path and inst_path.exists():
                    cmd.append(f"--inst-path={inst_path}")
                ly_env = os.environ.copy()
                if eff_mode == "cpu":
                    ly_env["CUDA_VISIBLE_DEVICES"] = "-1"
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(BASE_DIR),
                    env=ly_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS,
                )
                with self._lock:
                    self._active_procs[key] = proc

                try:
                    _, stderr_text = proc.communicate(timeout=240)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    _, stderr_text = proc.communicate()
                    self.log.error(
                        "Lyrics",
                        f"Worker timeout ({video_id}) after 240s | stderr: {(stderr_text or '')[-1500:]}",
                    )
                    raise RuntimeError("歌詞生成逾時")
                finally:
                    with self._lock:
                        self._active_procs.pop(key, None)

                if proc.returncode == 0 and out_file.exists():
                    # Load into RAM cache immediately so Extension gets lyrics in <1ms!
                    try:
                        data = json.loads(out_file.read_text(encoding="utf-8"))
                        if data.get("lines"):
                            with self._lock:
                                self._lyrics_mem_cache[video_id] = data
                                self._lyrics_gdrive_missing.discard(video_id)
                    except Exception:
                        pass

                    # Sync tiny lyrics.json (~4KB) to Google Drive in background and remove local copy (0 MB disk!)
                    if self.storage_mode == "gdrive" and self.gdrive.is_configured() and self.gdrive.has_song(video_id, mode):
                        def _bg_upload_lyrics(vid=video_id, m=mode, path=out_file):
                            if self.gdrive.upload_single_file(vid, m, path):
                                try:
                                    path.unlink()
                                    parent = self.cache_dir / vid
                                    if parent.exists() and not any(parent.iterdir()):
                                        parent.rmdir()
                                except Exception:
                                    pass
                        threading.Thread(target=_bg_upload_lyrics, daemon=True, name=f"ul-lyrics-{video_id}").start()
                    self.log.info("Lyrics", f"Ready for {video_id} (force_ai={force_ai})")
                    self.log.trigger_sync_async()
                elif proc.returncode != 0:
                    self.log.error(
                        "Lyrics",
                        f"Worker failed ({video_id}, rc={proc.returncode}): {(stderr_text or '')[-1500:]}",
                    )
        except Exception as exc:
            self.log.error("Lyrics", f"Exception ({video_id}): {exc}")
            try:
                status_file.write_text(
                    json.dumps(
                        {
                            "video_id": video_id,
                            "status": "error",
                            "progress": 0,
                            "stage_text": f"歌詞生成失敗：{exc}",
                            "error": str(exc),
                            "updated_at": time.time(),
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass
        finally:
            with self._lock:
                self._lyrics_in_progress.discard(video_id)
            for _tf in (temp_vocals, temp_inst):
                if _tf and _tf.exists():
                    try:
                        _tf.unlink()
                    except Exception:
                        pass


manager = KaraokeSeparatorManager()
