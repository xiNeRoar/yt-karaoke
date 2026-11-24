import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from storage import GoogleDriveStorage

MODELS: Dict[str, Dict[str, str]] = {
    "pure": {
        "label": "純伴奏 (BS-RoFormer 1297)",
        "filename": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "description": "完全去除所有主唱及和音，最高準確率純伴奏",
    },
    "karaoke": {
        "label": "KTV 保留和音 (Mel-Band RoFormer Karaoke)",
        "filename": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        "description": "去除主唱人聲，保留副歌和音，還原卡拉OK伴奏感",
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
        self._lock = threading.Lock()
        self._worker_lock = threading.Lock()
        self._active_procs: Dict[str, subprocess.Popen] = {}
        self._queued_keys: set[str] = set()
        self._rate_history: Dict[str, list[float]] = {}

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
        self._load_config()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Auto-enable gdrive if configured and user chose it
        if self.storage_mode == "gdrive" and not self.gdrive.is_configured():
            self.storage_mode = "local"

        # Clean stale temp jobs from prior session
        for p in TEMP_DIR.iterdir():
            if p.name != "jobs" and p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        for p in JOBS_DIR.glob("*.json"):
            try:
                p.unlink()
            except Exception:
                pass

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
                print(
                    f"[Karaoke] Launching isolated worker for {video_id} ({mode}) mode={effective_mode} (free_vram={free_mb:.0f}MB)",
                    flush=True,
                )

                cmd = [
                    sys.executable,
                    "-u",
                    str(WORKER_SCRIPT),
                    "--video-id",
                    video_id,
                    "--mode",
                    mode,
                    "--url",
                    target_url,
                    "--device-mode",
                    effective_mode,
                    "--cache-dir",
                    str(self.cache_dir),
                    "--status-file",
                    str(status_file),
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
                    _, stderr_text = proc.communicate(timeout=360)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    _, stderr_text = proc.communicate()
                    raise RuntimeError("處理逾時（超過 6 分鐘），已自動中止保護系統資源。")
                finally:
                    with self._lock:
                        self._active_procs.pop(key, None)

                if proc.returncode != 0:
                    cur = self._read_job_status(video_id, mode) or {}
                    if cur.get("status") != "error":
                        err_msg = (stderr_text or "").strip().splitlines()[-1] if stderr_text else f"Worker exit {proc.returncode}"
                        cur.update(
                            {
                                "video_id": video_id,
                                "mode": mode,
                                "status": "error",
                                "error": err_msg,
                                "stage_text": f"處理失敗：{err_msg}",
                                "updated_at": time.time(),
                            }
                        )
                        status_file.write_text(json.dumps(cur, ensure_ascii=False), encoding="utf-8")
                    print(f"[Karaoke] Worker failed ({video_id}): {(stderr_text or '')[-400:]}", flush=True)
                    return

                # If Google Drive storage is active, upload immediately and delete local MP3s (0 MB local disk!)
                if self.storage_mode == "gdrive" and self.gdrive.is_configured():
                    local_song_dir = self.get_cache_dir(video_id, mode)
                    meta = self.get_meta(video_id, mode)
                    try:
                        cur = self._read_job_status(video_id, mode) or {}
                        cur.update({"status": "finalizing", "progress": 96, "stage_text": "正在同步伴奏到 Google Drive..."})
                        status_file.write_text(json.dumps(cur, ensure_ascii=False), encoding="utf-8")

                        self.gdrive.upload_song(video_id, mode, local_song_dir, meta)
                        shutil.rmtree(local_song_dir, ignore_errors=True)
                        parent = self.cache_dir / video_id
                        if parent.exists() and not any(parent.iterdir()):
                            try:
                                parent.rmdir()
                            except Exception:
                                pass

                        cur.update({"status": "ready", "progress": 100, "stage_text": "已存上 Google Drive，隨時開唱！"})
                        status_file.write_text(json.dumps(cur, ensure_ascii=False), encoding="utf-8")
                        print(f"[Karaoke] Uploaded {video_id} ({mode}) to Google Drive and freed local disk!", flush=True)
                    except Exception as gerr:
                        print(f"[Karaoke] Google Drive upload fallback to local ({video_id}): {gerr}", flush=True)
                        self.enforce_cache_limit()
                else:
                    self.enforce_cache_limit()
                    print(f"[Karaoke] Worker finished cleanly ({video_id}, {mode}) — 100% VRAM reclaimed!", flush=True)
        except Exception as exc:
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

    def migrate_local_to_gdrive(self) -> dict:
        """Upload all existing local cached songs to Google Drive and remove local copies."""
        if not self.gdrive.is_configured():
            raise RuntimeError("尚未連接 Google Drive，請先喺控制台登入 Google Drive。")
        local_items = self._list_local_cache()
        migrated = 0
        freed_mb = 0.0
        for item in local_items:
            vid = item["video_id"]
            mode = item["mode"]
            local_dir = self.get_cache_dir(vid, mode)
            meta = self.get_meta(vid, mode)
            self.gdrive.upload_song(vid, mode, local_dir, meta)
            shutil.rmtree(local_dir, ignore_errors=True)
            parent = self.cache_dir / vid
            if parent.exists() and not any(parent.iterdir()):
                try:
                    parent.rmdir()
                except Exception:
                    pass
            migrated += 1
            freed_mb += item["size_mb"]
        self.storage_mode = "gdrive"
        self.save_config()
        return {"migrated": migrated, "freed_mb": round(freed_mb, 2)}

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
                    items.append(
                        {
                            "video_id": vid_dir.name,
                            "mode": mode_dir.name,
                            "title": meta.get("title", vid_dir.name),
                            "duration": meta.get("duration", 0),
                            "size_mb": round(size_bytes / (1024 * 1024), 2),
                            "created_at": meta.get("created_at", 0),
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


manager = KaraokeSeparatorManager()
