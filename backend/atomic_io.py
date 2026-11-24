"""Windows-safe atomic JSON writer shared by separator.py, worker.py, and lyrics_worker.py.

On Windows, os.replace(tmp, target) raises PermissionError for ~0.5ms if a concurrent
reader holds target open. Retrying os.replace with a tiny backoff guarantees readers
NEVER see a truncated 0-byte or half-written JSON file.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def write_json_atomic(path: Path, payload: dict, stamp_updated_at: bool = True) -> None:
    if stamp_updated_at:
        payload["updated_at"] = time.time()
    text = json.dumps(payload, ensure_ascii=False)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt < 9:
                time.sleep(0.008 * (attempt + 1))
            else:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
                raise
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise
