"""Cloud + Local storage engine for YouTube Karaoke cache.

Supports:
- 'local': stores stems in local directory with automatic LRU size cap.
- 'gdrive': uploads separated stems directly to Google Drive via rclone, deletes local MP3s
  immediately (0 MB local disk usage!), and streams audio straight from Google Drive on demand.
"""

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

RCLONE_REMOTE = "ytkaraoke_gdrive"
RCLONE_FOLDER = "YT-Karaoke-Cache"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def find_rclone_exe() -> Optional[str]:
    found = shutil.which("rclone")
    if found:
        return found
    candidates = [
        Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages",
        Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links",
        Path("C:/Program Files/rclone"),
    ]
    for root in candidates:
        if not root.exists():
            continue
        if (root / "rclone.exe").exists():
            return str(root / "rclone.exe")
        for match in root.rglob("rclone.exe"):
            return str(match)
    return None


class GoogleDriveStorage:
    """Manages Google Drive cloud cache via rclone with a lightweight local JSON index."""

    def __init__(self, index_file: Path) -> None:
        self.index_file = index_file
        self._lock = threading.Lock()
        self.rclone_exe = find_rclone_exe()
        self._index: dict[str, dict] = {}
        self._load_index()

    def refresh_binary(self) -> Optional[str]:
        self.rclone_exe = find_rclone_exe()
        return self.rclone_exe

    def _load_index(self) -> None:
        if self.index_file.exists():
            try:
                self._index = json.loads(self.index_file.read_text(encoding="utf-8"))
            except Exception:
                self._index = {}

    def _save_index(self) -> None:
        try:
            self.index_file.write_text(json.dumps(self._index, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def is_configured(self) -> bool:
        exe = self.refresh_binary()
        if not exe:
            return False
        try:
            out = subprocess.check_output(
                [exe, "listremotes"],
                text=True,
                timeout=4,
                creationflags=CREATE_NO_WINDOW,
            )
            return f"{RCLONE_REMOTE}:" in out.splitlines()
        except Exception:
            return False

    def _remote_base(self, video_id: str, mode: str) -> str:
        return f"{RCLONE_REMOTE}:{RCLONE_FOLDER}/{video_id}/{mode}"

    def _key(self, video_id: str, mode: str) -> str:
        return f"{video_id}__{mode}"

    def has_song(self, video_id: str, mode: str) -> bool:
        with self._lock:
            return self._key(video_id, mode) in self._index

    def get_meta(self, video_id: str, mode: str) -> dict:
        with self._lock:
            return dict(self._index.get(self._key(video_id, mode), {}))

    def upload_song(self, video_id: str, mode: str, local_dir: Path, meta: dict) -> None:
        exe = self.refresh_binary()
        if not exe:
            raise RuntimeError("搵唔到 rclone，無法上傳到 Google Drive")

        remote_dir = self._remote_base(video_id, mode)
        # Ensure remote directory exists and upload instrumental.mp3, vocals.mp3, meta.json
        res = subprocess.run(
            [exe, "copy", str(local_dir), remote_dir, "--transfers", "3", "--quiet"],
            capture_output=True,
            text=True,
            timeout=180,
            creationflags=CREATE_NO_WINDOW,
        )
        if res.returncode != 0:
            raise RuntimeError(f"Google Drive 上傳失敗：{res.stderr.strip()}")

        size_bytes = sum(f.stat().st_size for f in local_dir.glob("*") if f.is_file())
        entry = {
            "video_id": video_id,
            "mode": mode,
            "title": meta.get("title", video_id),
            "duration": meta.get("duration", 0),
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "created_at": meta.get("created_at", time.time()),
            "storage": "gdrive",
            "url": f"https://www.youtube.com/watch?v={video_id}",
        }
        with self._lock:
            self._index[self._key(video_id, mode)] = entry
            self._save_index()

    def download_file(self, video_id: str, mode: str, filename: str, dest: Path) -> bool:
        exe = self.refresh_binary()
        if not exe:
            return False
        remote_file = f"{self._remote_base(video_id, mode)}/{filename}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        res = subprocess.run(
            [exe, "copyto", remote_file, str(dest), "--quiet"],
            capture_output=True,
            timeout=60,
            creationflags=CREATE_NO_WINDOW,
        )
        return res.returncode == 0 and dest.exists()

    def upload_single_file(self, video_id: str, mode: str, local_file: Path) -> bool:
        exe = self.refresh_binary()
        if not exe or not local_file.exists():
            return False
        remote_file = f"{self._remote_base(video_id, mode)}/{local_file.name}"
        res = subprocess.run(
            [exe, "copyto", str(local_file), remote_file, "--quiet"],
            capture_output=True,
            timeout=30,
            creationflags=CREATE_NO_WINDOW,
        )
        return res.returncode == 0

    def read_json(self, video_id: str, mode: str, filename: str) -> Optional[dict]:
        exe = self.refresh_binary()
        if not exe:
            return None
        remote_file = f"{self._remote_base(video_id, mode)}/{filename}"
        try:
            out = subprocess.check_output(
                [exe, "cat", remote_file],
                text=True,
                encoding="utf-8",
                timeout=12,
                creationflags=CREATE_NO_WINDOW,
            )
            return json.loads(out)
        except Exception:
            return None

    def stream_audio(self, video_id: str, mode: str, stem: str) -> Iterator[bytes]:
        exe = self.refresh_binary()
        if not exe:
            raise RuntimeError("rclone not found")
        remote_file = f"{self._remote_base(video_id, mode)}/{stem}.mp3"
        proc = subprocess.Popen(
            [exe, "cat", remote_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW,
        )
        try:
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                yield chunk
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                pass

    def sync_index_from_remote(self) -> int:
        """Scan Google Drive folder to rebuild index (e.g., on fresh install or restart)."""
        exe = self.refresh_binary()
        if not exe or not self.is_configured():
            return 0
        try:
            out = subprocess.check_output(
                [exe, "lsjson", f"{RCLONE_REMOTE}:{RCLONE_FOLDER}", "-R", "--files-only"],
                text=True,
                timeout=30,
                creationflags=CREATE_NO_WINDOW,
            )
            files = json.loads(out)
        except Exception:
            return len(self._index)

        grouped: dict[tuple[str, str], dict] = {}
        for f in files:
            path_parts = f.get("Path", "").replace("\\", "/").split("/")
            if len(path_parts) != 3:
                continue
            vid, mode, fname = path_parts
            g = grouped.setdefault((vid, mode), {"files": set(), "bytes": 0})
            g["files"].add(fname)
            g["bytes"] += int(f.get("Size", 0))

        with self._lock:
            new_index: dict[str, dict] = {}
            for (vid, mode), info in grouped.items():
                if "instrumental.mp3" in info["files"] and "vocals.mp3" in info["files"]:
                    key = self._key(vid, mode)
                    existing = self._index.get(key, {})
                    title = existing.get("title") or vid
                    duration = existing.get("duration") or 0
                    created_at = existing.get("created_at") or time.time()
                    # Fetch meta.json if title not cached yet
                    if title == vid and "meta.json" in info["files"]:
                        try:
                            meta_raw = subprocess.check_output(
                                [exe, "cat", f"{self._remote_base(vid, mode)}/meta.json"],
                                text=True,
                                timeout=8,
                                creationflags=CREATE_NO_WINDOW,
                            )
                            m = json.loads(meta_raw)
                            title = m.get("title", vid)
                            duration = m.get("duration", 0)
                            created_at = m.get("created_at", created_at)
                        except Exception:
                            pass
                    new_index[key] = {
                        "video_id": vid,
                        "mode": mode,
                        "title": title,
                        "duration": duration,
                        "size_mb": round(info["bytes"] / (1024 * 1024), 2),
                        "created_at": created_at,
                        "storage": "gdrive",
                        "url": f"https://www.youtube.com/watch?v={vid}",
                    }
            self._index = new_index
            self._save_index()
            return len(self._index)

    def list_items(self) -> list[dict]:
        with self._lock:
            items = list(self._index.values())
        items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return items

    def delete_song(self, video_id: str, mode: Optional[str] = None) -> int:
        exe = self.refresh_binary()
        if not exe:
            return 0
        target = f"{RCLONE_REMOTE}:{RCLONE_FOLDER}/{video_id}" + (f"/{mode}" if mode else "")
        subprocess.run(
            [exe, "purge", target],
            capture_output=True,
            timeout=30,
            creationflags=CREATE_NO_WINDOW,
        )
        removed = 0
        with self._lock:
            keys = [
                k for k, v in self._index.items()
                if v["video_id"] == video_id and (mode is None or v["mode"] == mode)
            ]
            for k in keys:
                self._index.pop(k, None)
                removed += 1
            self._save_index()
        return removed

    def purge_all(self) -> int:
        exe = self.refresh_binary()
        if not exe:
            return 0
        subprocess.run(
            [exe, "purge", f"{RCLONE_REMOTE}:{RCLONE_FOLDER}"],
            capture_output=True,
            timeout=60,
            creationflags=CREATE_NO_WINDOW,
        )
        with self._lock:
            n = len(self._index)
            self._index = {}
            self._save_index()
        return n
