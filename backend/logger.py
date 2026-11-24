"""Lightweight structured logger that auto-syncs to Google Drive (YT-Karaoke-Cache/_logs/)
so local disk stays near 0 MB while keeping full diagnostics for debugging.
"""

import logging
import subprocess
import threading
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Optional

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
RCLONE_REMOTE = "ytkaraoke_gdrive"
RCLONE_LOG_FOLDER = "YT-Karaoke-Cache/_logs"


class CloudSyncedLogger:
    def __init__(self, temp_dir: Path, rclone_finder: Callable[[], Optional[str]], gdrive_check: Callable[[], bool]) -> None:
        self.temp_dir = temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.temp_dir / "karaoke_live.log"
        self._find_rclone = rclone_finder
        self._is_gdrive_ready = gdrive_check
        self._sync_lock = threading.Lock()
        self._last_sync_ts = 0.0
        self._dirty = False

        self.logger = logging.getLogger("yt_karaoke")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.handlers:
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

            # Tiny local rolling buffer (max 1 MB, 0 backups -> never bloats local disk!)
            fh = RotatingFileHandler(self.log_file, maxBytes=1_048_576, backupCount=0, encoding="utf-8")
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

            # Mirror to stdout so Tray App live log drawer sees everything immediately
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            self.logger.addHandler(sh)

        # Gentle background sync every 90s when new logs arrive
        t = threading.Thread(target=self._bg_sync_loop, daemon=True, name="cloud-log-sync")
        t.start()

    def info(self, component: str, msg: str) -> None:
        self.logger.info(f"[{component}] {msg}")
        self._dirty = True

    def warning(self, component: str, msg: str) -> None:
        self.logger.warning(f"[{component}] {msg}")
        self._dirty = True

    def error(self, component: str, msg: str) -> None:
        self.logger.error(f"[{component}] {msg}")
        self._dirty = True
        self.trigger_sync_async()

    def trigger_sync_async(self) -> None:
        threading.Thread(target=self.sync_to_gdrive, daemon=True, name="cloud-log-flush").start()

    def _bg_sync_loop(self) -> None:
        while True:
            time.sleep(90)
            if self._dirty:
                self.sync_to_gdrive()

    def sync_to_gdrive(self) -> bool:
        if not self.log_file.exists() or self.log_file.stat().st_size == 0:
            return False
        if not self._is_gdrive_ready():
            return False
        exe = self._find_rclone()
        if not exe:
            return False

        with self._sync_lock:
            try:
                day = datetime.now().strftime("%Y-%m-%d")
                remote_path = f"{RCLONE_REMOTE}:{RCLONE_LOG_FOLDER}/karaoke-{day}.log"

                # Snapshot current log into a tiny temp copy to avoid file locking during upload
                snap = self.temp_dir / "karaoke_upload_snap.log"
                snap.write_bytes(self.log_file.read_bytes())

                # Append to today's cloud log if it already exists or overwrite with merged daily file
                res = subprocess.run(
                    [exe, "copyto", str(snap), remote_path, "--quiet"],
                    capture_output=True,
                    timeout=30,
                    creationflags=CREATE_NO_WINDOW,
                )
                try:
                    snap.unlink()
                except Exception:
                    pass

                if res.returncode == 0:
                    self._last_sync_ts = time.time()
                    self._dirty = False
                    return True
            except Exception:
                pass
            return False

    def tail(self, max_lines: int = 120) -> str:
        if not self.log_file.exists():
            return ""
        try:
            lines = self.log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-max_lines:])
        except Exception:
            return ""
