"""Invite Code & Device-Binding Manager for YouTube AI Karaoke.

Two-file single-writer architecture (zero split-brain / zero merge conflicts):
- `_access_keys.json`: written ONLY by PC Backend (via Desktop Tray App), synced to Google Drive & NAS Gateway.
- `_device_bindings.json`: written ONLY by 24/7 NAS Gateway on friend device activations, read by PC Backend
  so the Tray App shows live `used_devices / max_devices` counts.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import httpx

from atomic_io import write_json_atomic
from storage import CREATE_NO_WINDOW, RCLONE_FOLDER, RCLONE_REMOTE

# Typo-resistant Crockford-style alphabet (no 0/O/1/I confusion for friends typing codes)
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
ACCESS_KEYS_REMOTE = f"{RCLONE_REMOTE}:{RCLONE_FOLDER}/_access_keys.json"
DEVICE_BINDINGS_REMOTE = f"{RCLONE_REMOTE}:{RCLONE_FOLDER}/_device_bindings.json"


def normalize_code(raw: str) -> str:
    s = "".join(ch for ch in (raw or "").upper().strip() if ch.isalnum() or ch == "-")
    return s


def generate_invite_code(prefix: str = "KTV") -> str:
    p1 = "".join(secrets.choice(CODE_ALPHABET) for _ in range(4))
    p2 = "".join(secrets.choice(CODE_ALPHABET) for _ in range(4))
    return f"{prefix}-{p1}-{p2}"


class InviteManager:
    def __init__(
        self,
        base_dir: Path,
        gdrive,
        logger,
        worker_secret: str = "",
    ) -> None:
        self.base_dir = base_dir
        self.gdrive = gdrive
        self.log = logger
        self.worker_secret = worker_secret
        self.keys_file = base_dir / "access_keys.json"
        self.bindings_file = base_dir / "device_bindings.json"
        self.gateway_url = os.environ.get("KARAOKE_GATEWAY_URL", "http://192.168.1.112:18766").rstrip("/")

        self._lock = threading.RLock()
        self._push_lock = threading.Lock()
        self._push_pending = False
        self._push_running = False

        self._codes: dict[str, dict] = {}
        self._bindings: dict[str, dict] = {}
        self._last_bindings_pull: float = 0.0
        self._load_local()
        if not self._codes and self.gdrive.is_configured():
            threading.Thread(target=self.pull_keys_from_gdrive, daemon=True, name="invites-boot-pull").start()

    def pull_keys_from_gdrive(self) -> bool:
        """Disaster-recovery pull: restore _access_keys.json from Google Drive if local file is missing."""
        exe = self.gdrive.refresh_binary()
        if not exe or not self.gdrive.is_configured():
            return False
        try:
            res = subprocess.run(
                [exe, "cat", ACCESS_KEYS_REMOTE, "--drive-pacer-min-sleep", "200ms"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=20,
                creationflags=CREATE_NO_WINDOW,
            )
            if res.returncode == 0 and res.stdout.strip():
                doc = json.loads(res.stdout)
                codes = doc.get("codes") if isinstance(doc, dict) else None
                if isinstance(codes, dict) and codes:
                    with self._lock:
                        if not self._codes:
                            self._codes = codes
                            self._save_keys_local()
                    return True
        except Exception:
            pass
        return False

    def _load_local(self) -> None:
        if self.keys_file.exists():
            try:
                doc = json.loads(self.keys_file.read_text(encoding="utf-8"))
                codes = doc.get("codes") if isinstance(doc, dict) else None
                if isinstance(codes, dict):
                    self._codes = codes
            except Exception:
                self._codes = {}
        if self.bindings_file.exists():
            try:
                doc = json.loads(self.bindings_file.read_text(encoding="utf-8"))
                toks = doc.get("tokens") if isinstance(doc, dict) else None
                if isinstance(toks, dict):
                    self._bindings = toks
            except Exception:
                self._bindings = {}

    def _save_keys_local(self) -> dict:
        payload = {
            "version": 1,
            "updated_at": time.time(),
            "codes": dict(self._codes),
        }
        write_json_atomic(self.keys_file, payload, stamp_updated_at=False)
        return payload

    def _count_bindings_by_code(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self._lock:
            for info in self._bindings.values():
                if not isinstance(info, dict):
                    continue
                c = info.get("code")
                if c:
                    counts[c] = counts.get(c, 0) + 1
        return counts

    def pull_bindings_from_gdrive(self) -> bool:
        """Read NAS Gateway's _device_bindings.json so Tray App shows live bound device counts."""
        exe = self.gdrive.refresh_binary()
        if not exe or not self.gdrive.is_configured():
            return False
        try:
            res = subprocess.run(
                [exe, "cat", DEVICE_BINDINGS_REMOTE, "--drive-pacer-min-sleep", "200ms"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=20,
                creationflags=CREATE_NO_WINDOW,
            )
            if res.returncode == 0 and res.stdout.strip():
                doc = json.loads(res.stdout)
                toks = doc.get("tokens") if isinstance(doc, dict) else None
                if isinstance(toks, dict):
                    with self._lock:
                        self._bindings = toks
                        self._last_bindings_pull = time.time()
                    write_json_atomic(self.bindings_file, doc, stamp_updated_at=False)
                    return True
        except Exception:
            pass
        return False

    def push_keys_to_gdrive(self) -> bool:
        """Upload _access_keys.json to Google Drive and immediately ping NAS Gateway on LAN."""
        exe = self.gdrive.refresh_binary()
        if not exe or not self.gdrive.is_configured():
            return False
        with self._lock:
            payload = {
                "version": 1,
                "updated_at": time.time(),
                "codes": dict(self._codes),
            }
        tmp = self.base_dir / f"._access_keys.{os.getpid()}.{threading.get_ident()}.{time.monotonic_ns()}.tmp"
        ok = False
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            res = subprocess.run(
                [
                    exe, "copyto", str(tmp), ACCESS_KEYS_REMOTE,
                    "--retries", "4",
                    "--retries-sleep", "4s",
                    "--drive-pacer-min-sleep", "200ms",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                timeout=45,
                creationflags=CREATE_NO_WINDOW,
            )
            ok = res.returncode == 0
        except Exception:
            ok = False
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

        if ok:
            self._ping_gateway_sync()
        return ok

    def _ping_gateway_sync(self) -> bool:
        """Best-effort LAN ping so Synology NAS picks up new/revoked codes in <0.1s."""
        if not self.gateway_url or not self.worker_secret:
            return False
        try:
            r = httpx.post(
                f"{self.gateway_url}/api/internal/sync-auth",
                headers={"X-Worker-Secret": self.worker_secret},
                timeout=5.0,
            )
            return r.status_code == 200
        except Exception:
            return False

    def push_keys_async(self) -> None:
        with self._push_lock:
            self._push_pending = True
            if self._push_running:
                return
            self._push_running = True

        def _worker() -> None:
            try:
                while True:
                    time.sleep(0.5)
                    with self._push_lock:
                        if not self._push_pending:
                            self._push_running = False
                            return
                        self._push_pending = False
                    self.push_keys_to_gdrive()
            except Exception:
                with self._push_lock:
                    self._push_running = False

        threading.Thread(target=_worker, daemon=True, name="invites-sync-push").start()

    def sync_now(self) -> dict:
        pushed = self.push_keys_to_gdrive()
        pulled = self.pull_bindings_from_gdrive()
        return {"pushed_to_gdrive": pushed, "pulled_bindings": pulled, **self.list_codes(refresh_bindings=False)}

    def create_codes(self, count: int = 1, label: str = "", max_devices: int = 2) -> list[dict]:
        safe_count = max(1, min(500, int(count)))
        safe_max_dev = max(1, min(500, int(max_devices)))
        clean_label = (label or "").strip()[:48]
        now = time.time()
        created: list[dict] = []

        with self._lock:
            for _ in range(safe_count):
                code = generate_invite_code()
                while code in self._codes:
                    code = generate_invite_code()
                entry = {
                    "code": code,
                    "label": clean_label,
                    "max_devices": safe_max_dev,
                    "enabled": True,
                    "created_at": now,
                    "revoked_at": None,
                }
                self._codes[code] = entry
                created.append(dict(entry, used_devices=0))
            self._save_keys_local()

        self.log.info("Invites", f"Created {len(created)} invite code(s) | label='{clean_label}' max_devices={safe_max_dev}")
        self.push_keys_async()
        return created

    def set_enabled(self, code: str, enabled: bool) -> Optional[dict]:
        key = normalize_code(code)
        with self._lock:
            entry = self._codes.get(key)
            if not entry:
                return None
            entry["enabled"] = bool(enabled)
            entry["revoked_at"] = None if enabled else time.time()
            self._save_keys_local()
            out = dict(entry)
        self.log.info("Invites", f"{'Re-enabled' if enabled else 'Revoked'} invite code {key}")
        self.push_keys_async()
        return out

    def delete_code(self, code: str) -> bool:
        key = normalize_code(code)
        with self._lock:
            removed = self._codes.pop(key, None) is not None
            if removed:
                self._save_keys_local()
        if removed:
            self.log.info("Invites", f"Deleted invite code {key}")
            self.push_keys_async()
        return removed

    def list_codes(self, refresh_bindings: bool = False) -> dict:
        if refresh_bindings or (time.time() - self._last_bindings_pull > 30.0):
            self.pull_bindings_from_gdrive()
        counts = self._count_bindings_by_code()
        with self._lock:
            items = []
            for entry in self._codes.values():
                c = entry["code"]
                items.append({
                    **entry,
                    "used_devices": counts.get(c, 0),
                })
        items.sort(key=lambda x: (not x.get("enabled", True), -(x.get("created_at") or 0)))
        active_codes = sum(1 for x in items if x.get("enabled"))
        total_bound = sum(x.get("used_devices", 0) for x in items)
        return {
            "total_codes": len(items),
            "active_codes": active_codes,
            "total_bound_devices": total_bound,
            "items": items,
        }
