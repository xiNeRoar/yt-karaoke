"""Windows-safe atomic JSON writer shared by separator.py, worker.py, lyrics_worker.py, and storage.py.

On Windows NTFS, os.replace(tmp, target) raises PermissionError if a concurrent reader
or antivirus scanner briefly holds target open. Using a thread-unique temp filename and
retrying os.replace guarantees zero temp-file collisions and zero torn/0-byte reads.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path


def write_json_atomic(path: Path, payload: dict, stamp_updated_at: bool = True) -> None:
    if stamp_updated_at:
        payload["updated_at"] = time.time()
    text = json.dumps(payload, ensure_ascii=False)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.monotonic_ns()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        for attempt in range(25):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt < 24:
                    time.sleep(min(0.03, 0.004 * (attempt + 1)))
                else:
                    raise
    finally:
        if tmp.exists():
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
