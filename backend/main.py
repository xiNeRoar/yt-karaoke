import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from invites import InviteManager
from separator import BASE_DIR, MODELS, manager

# Strict canonical YouTube 11-char video ID (prevents SSRF, injection, and path traversal)
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

# Per-session local admin token used ONLY by the Desktop Tray Controller EXE
TOKEN_FILE = BASE_DIR / ".admin_token"
ADMIN_TOKEN = secrets.token_hex(32)
TOKEN_FILE.write_text(ADMIN_TOKEN, encoding="utf-8")

# Optional shared secret between NAS Gateway and PC GPU Node when bound to LAN/Tailscale
WORKER_SECRET_FILE = BASE_DIR / ".worker_secret"
WORKER_SECRET = os.environ.get("KARAOKE_WORKER_SECRET", "").strip()
if not WORKER_SECRET and WORKER_SECRET_FILE.exists():
    WORKER_SECRET = WORKER_SECRET_FILE.read_text(encoding="utf-8").strip()

ALLOWED_ORIGINS = {"https://www.youtube.com"}
EXTENSION_RELOAD_NONCE = 1


def _is_allowed_origin(origin: Optional[str]) -> bool:
    if not origin:
        return True
    if origin in ALLOWED_ORIGINS:
        return True
    if origin.startswith("chrome-extension://"):
        return True
    return False


class ClientTracker:
    def __init__(self, ttl_sec: float = 40.0) -> None:
        self._lock = threading.Lock()
        self._ttl = ttl_sec
        self._clients: dict[str, dict] = {}

    def touch(self, client_id: Optional[str], ip: str, video_id: Optional[str] = None, singing: Optional[bool] = None) -> bool:
        if not client_id or not CLIENT_ID_RE.match(client_id):
            return True
        now = time.time()
        with self._lock:
            expired = [cid for cid, info in self._clients.items() if now - info.get("last_seen", 0) > self._ttl]
            for cid in expired:
                self._clients.pop(cid, None)
            if client_id not in self._clients and len(self._clients) >= manager.max_clients:
                return False
            entry = self._clients.get(client_id, {"ip": ip, "video_id": "", "singing": False})
            entry["ip"] = ip
            entry["last_seen"] = now
            if video_id is not None:
                entry["video_id"] = video_id
            if singing is not None:
                entry["singing"] = bool(singing)
            self._clients[client_id] = entry
            return True

    def remove(self, client_id: Optional[str]) -> None:
        if not client_id:
            return
        with self._lock:
            self._clients.pop(client_id, None)

    def snapshot(self) -> dict:
        now = time.time()
        with self._lock:
            expired = [cid for cid, info in self._clients.items() if now - info.get("last_seen", 0) > self._ttl]
            for cid in expired:
                self._clients.pop(cid, None)
            total = len(self._clients)
            singing = sum(1 for info in self._clients.values() if info.get("singing"))
            return {"active_clients": total, "active_singing": singing}


clients = ClientTracker()
invites = InviteManager(BASE_DIR, manager.gdrive, manager.log, worker_secret=WORKER_SECRET)
app = FastAPI(title="YouTube Karaoke AI Backend", version="2.2.0", docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def security_and_cors_middleware(request: Request, call_next):
    origin = request.headers.get("origin")
    if origin and not _is_allowed_origin(origin):
        manager.log.warning("Security", f"Blocked untrusted Origin={origin} on {request.method} {request.url.path}")
        return Response(content='{"detail":"Origin not allowed"}', status_code=403, media_type="application/json")

    # When KARAOKE_WORKER_SECRET is configured, all non-localhost requests (i.e. from NAS Gateway)
    # must present the matching X-Worker-Secret header.
    client_host = request.client.host if request.client else ""
    if WORKER_SECRET and client_host not in ("127.0.0.1", "::1") and request.method != "OPTIONS":
        provided = request.headers.get("x-worker-secret") or ""
        if not secrets.compare_digest(provided, WORKER_SECRET):
            manager.log.warning("Security", f"Rejected unauthorized remote worker call from {client_host}")
            return Response(content='{"detail":"Unauthorized worker request"}', status_code=403, media_type="application/json")

    if request.method == "OPTIONS":
        resp = Response(status_code=204)
    else:
        resp = await call_next(request)

    if origin and _is_allowed_origin(origin):
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Karaoke-Client"
        resp.headers["Access-Control-Allow-Private-Network"] = "true"

    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    return resp


class SeparateRequest(BaseModel):
    video_id: str
    mode: Literal["pure"] = "pure"


class LyricsRequest(BaseModel):
    video_id: str
    mode: Literal["pure"] = "pure"
    force_ai: bool = False


class HeartbeatRequest(BaseModel):
    client_id: str
    video_id: Optional[str] = None
    singing: bool = False
    leaving: bool = False


class SettingsRequest(BaseModel):
    device_mode: Optional[Literal["auto", "gpu", "cpu"]] = None
    storage_mode: Optional[Literal["local", "gdrive"]] = None
    cache_dir: Optional[str] = None
    max_cache_mb: Optional[int] = None
    pause_new_jobs: Optional[bool] = None
    max_queue_size: Optional[int] = None
    rate_limit_per_10min: Optional[int] = None
    max_clients: Optional[int] = None


class CreateInvitesRequest(BaseModel):
    count: int = 1
    label: str = ""
    max_devices: int = 2


class ToggleInviteRequest(BaseModel):
    code: str
    enabled: bool


def _validate_video_id(video_id: str) -> str:
    if not VIDEO_ID_RE.match(video_id):
        raise HTTPException(status_code=400, detail="Invalid YouTube video_id")
    return video_id


def _require_admin(x_admin_token: Optional[str]) -> None:
    if not x_admin_token or not secrets.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Admin token required")


def _catalog_version(items: list[dict]) -> str:
    if not items:
        return "0:0:0"
    total_mb = round(sum(float(i.get("size_mb") or 0) for i in items), 2)
    latest_ts = max(float(i.get("created_at") or 0) for i in items)
    return f"{len(items)}:{total_mb:.2f}:{latest_ts:.2f}"


@app.get("/api/health")
def health(request: Request, x_karaoke_client: Optional[str] = Header(default=None)):
    if x_karaoke_client:
        ip = request.client.host if request.client else "127.0.0.1"
        clients.touch(x_karaoke_client, ip)

    cached = manager.list_cache()
    local_mb = round(sum(item["size_mb"] for item in cached if item.get("storage") == "local"), 2)
    gdrive_mb = round(sum(item["size_mb"] for item in cached if item.get("storage") == "gdrive"), 2)
    free_vram_mb, total_vram_mb = manager.get_vram_free_total_mb()
    csnap = clients.snapshot()

    mig = manager.get_migration_state()
    return {
        "status": "ok",
        "gpu_online": not manager.pause_new_jobs,
        "gpu_offline_message": "AI 伺服器休息中，現有歌庫仍可免費唱" if manager.pause_new_jobs else "",
        "cuda_available": manager.cuda_available,
        "gpu_name": manager.gpu_name,
        "device_mode": manager.device_mode,
        "vram_allocated_mb": manager.get_vram_mb(),
        "vram_free_mb": free_vram_mb,
        "vram_total_mb": total_vram_mb,
        "ffmpeg_found": bool(manager.ffmpeg_dir),
        "storage_mode": manager.storage_mode,
        "gdrive_connected": manager.gdrive.is_configured(),
        "rclone_found": bool(manager.gdrive.refresh_binary()),
        "cache_dir": str(manager.cache_dir),
        "cache_count": len(cached),
        "catalog_version": _catalog_version(cached),
        "cache_size_mb": round(local_mb + gdrive_mb, 2),
        "local_cache_mb": local_mb,
        "gdrive_cache_mb": gdrive_mb,
        "max_cache_mb": manager.max_cache_mb,
        "active_clients": csnap["active_clients"],
        "active_singing": csnap["active_singing"],
        "queue_count": manager.get_queue_count(),
        "pending_hq_upgrades": manager.count_pending_hq_upgrades(),
        "pause_new_jobs": manager.pause_new_jobs,
        "max_queue_size": manager.max_queue_size,
        "rate_limit_per_10min": manager.rate_limit_per_10min,
        "max_clients": manager.max_clients,
        "extension_reload_nonce": EXTENSION_RELOAD_NONCE,
        "migration_running": mig["running"],
        "migration_done": mig["done"],
        "migration_total": mig["total"],
        "migration_message": mig["message"],
        "migration_error": mig["error"],
    }


@app.post("/api/heartbeat")
def heartbeat(req: HeartbeatRequest, request: Request):
    ip = request.client.host if request.client else "127.0.0.1"
    if req.leaving:
        clients.remove(req.client_id)
    else:
        vid = req.video_id if (req.video_id and VIDEO_ID_RE.match(req.video_id)) else None
        allowed = clients.touch(req.client_id, ip, video_id=vid, singing=req.singing)
        if not allowed:
            raise HTTPException(status_code=503, detail="目前已達最大連線人數上限，請稍後再試。")
    return {"status": "ok", **clients.snapshot(), "extension_reload_nonce": EXTENSION_RELOAD_NONCE}


@app.post("/api/settings")
def update_settings(req: SettingsRequest, x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    if req.device_mode is not None:
        if req.device_mode in ("gpu", "auto") and not manager.cuda_available:
            raise HTTPException(status_code=400, detail="CUDA GPU is not available on this machine")
        manager.device_mode = req.device_mode
    if req.storage_mode is not None:
        if req.storage_mode == "gdrive" and not manager.gdrive.is_configured():
            raise HTTPException(status_code=400, detail="請先登入連接 Google Drive")
        manager.storage_mode = req.storage_mode
    if req.max_cache_mb is not None:
        manager.max_cache_mb = max(100, min(50000, int(req.max_cache_mb)))
        manager.enforce_cache_limit()
    if req.cache_dir is not None and req.cache_dir.strip():
        target = Path(req.cache_dir.strip()).resolve()
        target.mkdir(parents=True, exist_ok=True)
        manager.set_cache_dir(target)
    if req.pause_new_jobs is not None:
        manager.pause_new_jobs = bool(req.pause_new_jobs)
    if req.max_queue_size is not None:
        manager.max_queue_size = max(1, min(20, int(req.max_queue_size)))
    if req.rate_limit_per_10min is not None:
        manager.rate_limit_per_10min = max(1, min(50, int(req.rate_limit_per_10min)))
    if req.max_clients is not None:
        manager.max_clients = max(5, min(500, int(req.max_clients)))
    manager.save_config()
    return {"status": "ok", "device_mode": manager.device_mode, "storage_mode": manager.storage_mode}


@app.post("/api/admin/reload-extension")
def trigger_extension_reload(x_admin_token: Optional[str] = Header(default=None)):
    global EXTENSION_RELOAD_NONCE
    _require_admin(x_admin_token)
    EXTENSION_RELOAD_NONCE += 1
    return {"status": "ok", "extension_reload_nonce": EXTENSION_RELOAD_NONCE}


@app.post("/api/admin/shutdown")
def admin_shutdown(x_admin_token: Optional[str] = Header(default=None)):
    """Cleanly stop the backend (and any active worker subprocesses) even if launched externally."""
    _require_admin(x_admin_token)

    def _delayed_exit() -> None:
        time.sleep(0.15)
        try:
            manager.terminate_active_workers()
        except Exception:
            pass
        os._exit(0)

    threading.Thread(target=_delayed_exit, daemon=True, name="admin-shutdown").start()
    return {"status": "stopping"}


@app.post("/api/admin/gdrive/migrate")
def gdrive_migrate(x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    try:
        res = manager.start_migrate_local_to_gdrive()
        return {"status": "ok", **res}
    except Exception as exc:
        manager.log.error("Migrate", f"Failed to start migration: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/admin/logs/sync")
def sync_logs_now(x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    ok = manager.log.sync_to_gdrive()
    return {"status": "ok", "synced": ok}


@app.get("/api/admin/logs")
def get_recent_logs(lines: int = 120, x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    return {"status": "ok", "log": manager.log.tail(max(10, min(500, lines)))}


@app.post("/api/admin/gdrive/sync")
def gdrive_sync(x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    count = manager.gdrive.sync_index_from_remote(force_full_scan=True)
    if manager.gdrive.is_configured():
        manager.storage_mode = "gdrive"
        manager.save_config()
    return {"status": "ok", "gdrive_count": count, "storage_mode": manager.storage_mode}


# -------------------- Invite Codes & Device-Binding Admin API (Tray App Only) --------------------
@app.get("/api/admin/invites")
def list_invites(refresh: int = 0, x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    return {"status": "ok", **invites.list_codes(refresh_bindings=bool(refresh))}


@app.post("/api/admin/invites/create")
def create_invites(req: CreateInvitesRequest, x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    created = invites.create_codes(count=req.count, label=req.label, max_devices=req.max_devices)
    return {"status": "ok", "created": created, **invites.list_codes(refresh_bindings=False)}


@app.post("/api/admin/invites/toggle")
def toggle_invite(req: ToggleInviteRequest, x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    updated = invites.set_enabled(req.code, req.enabled)
    if not updated:
        raise HTTPException(status_code=404, detail="邀請碼不存在")
    return {"status": "ok", "item": updated}


@app.delete("/api/admin/invites/{code}")
def delete_invite(code: str, x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    if not invites.delete_code(code):
        raise HTTPException(status_code=404, detail="邀請碼不存在")
    return {"status": "ok"}


@app.post("/api/admin/invites/sync")
def sync_invites(x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    return {"status": "ok", **invites.sync_now()}


@app.get("/api/status/{video_id}")
def get_status(
    video_id: str,
    request: Request,
    mode: Literal["pure"] = "pure",
    x_karaoke_client: Optional[str] = Header(default=None),
):
    _validate_video_id(video_id)
    if x_karaoke_client:
        ip = request.client.host if request.client else "127.0.0.1"
        clients.touch(x_karaoke_client, ip, video_id=video_id)
    return manager.get_status(video_id, mode)


@app.post("/api/separate")
def separate(
    req: SeparateRequest,
    request: Request,
    x_karaoke_client: Optional[str] = Header(default=None),
):
    _validate_video_id(req.video_id)
    ip = request.client.host if request.client else "127.0.0.1"
    if x_karaoke_client:
        clients.touch(x_karaoke_client, ip, video_id=req.video_id)
    try:
        return manager.start_separation(req.video_id, req.mode, client_ip=ip)
    except PermissionError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except OverflowError as exc:
        raise HTTPException(status_code=429, detail=str(exc))


@app.get("/api/audio/{video_id}/{mode}/{stem}")
def get_audio(
    video_id: str,
    mode: Literal["pure"],
    stem: Literal["instrumental", "vocals"],
):
    _validate_video_id(video_id)
    # 1) Serve from local cache if present
    audio_path = (manager.get_cache_dir(video_id, mode) / f"{stem}.mp3").resolve()
    if str(audio_path).startswith(str(manager.cache_dir.resolve())) and audio_path.exists():
        return FileResponse(
            path=str(audio_path),
            media_type="audio/mpeg",
            filename=f"{video_id}_{mode}_{stem}.mp3",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    # 2) Stream directly from Google Drive cloud cache (0 MB local disk!)
    if manager.gdrive.has_song(video_id, mode):
        try:
            stream = manager.gdrive.open_audio_stream(video_id, mode, stem)
        except FileNotFoundError:
            raise HTTPException(status_code=502, detail="Cloud audio temporarily unavailable")
        except Exception:
            raise HTTPException(status_code=502, detail="Cloud audio stream error")
        return StreamingResponse(
            stream,
            media_type="audio/mpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    raise HTTPException(status_code=404, detail="Audio stem not found in cache")


@app.get("/api/lyrics/{video_id}")
def get_lyrics(video_id: str, mode: Literal["pure"] = "pure"):
    _validate_video_id(video_id)
    return manager.get_lyrics(video_id, mode)


@app.post("/api/lyrics")
def generate_lyrics(req: LyricsRequest, request: Request):
    _validate_video_id(req.video_id)
    if not req.force_ai:
        existing = manager.get_lyrics(req.video_id, req.mode)
        if existing.get("status") == "ready":
            return existing
    if manager.pause_new_jobs:
        raise HTTPException(status_code=503, detail="AI 伺服器休息中，現有歌庫仍可免費唱")
    ip = request.client.host if request.client else "127.0.0.1"
    if ip not in ("127.0.0.1", "::1") and not manager.check_rate_limit(ip):
        raise HTTPException(
            status_code=429,
            detail=f"請求過於頻繁（每 10 分鐘上限 {manager.rate_limit_per_10min} 次），請稍後再試。",
        )
    return manager.start_lyrics_generation(req.video_id, req.mode, force_ai=req.force_ai)


@app.get("/api/cache")
def get_cache(limit: int = 200, offset: int = 0, q: str = ""):
    all_items = manager.list_cache()
    total_size_mb = round(sum(i["size_mb"] for i in all_items), 2)
    cat_ver = _catalog_version(all_items)

    needle = q.strip().lower()
    if needle:
        filtered = [
            i for i in all_items
            if needle in (i.get("title") or "").lower() or needle in (i.get("video_id") or "").lower()
        ]
    else:
        filtered = all_items

    safe_offset = max(0, offset)
    if limit <= 0:
        page = filtered[safe_offset : safe_offset + 5000]
    else:
        safe_limit = max(1, min(1000, limit))
        page = filtered[safe_offset : safe_offset + safe_limit]

    return {
        "count": len(filtered),
        "total_songs": len(all_items),
        "returned": len(page),
        "has_more": (safe_offset + len(page)) < len(filtered),
        "catalog_version": cat_ver,
        "total_size_mb": total_size_mb,
        "max_cache_mb": manager.max_cache_mb,
        "storage_mode": manager.storage_mode,
        "cache_dir": str(manager.cache_dir),
        "items": page,
    }


@app.delete("/api/cache")
def clear_all_cache(x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    removed = manager.delete_cache()
    return {"status": "ok", "removed": removed}


@app.delete("/api/cache/{video_id}")
def clear_video_cache(
    video_id: str,
    mode: Optional[Literal["pure"]] = None,
    x_admin_token: Optional[str] = Header(default=None),
):
    _require_admin(x_admin_token)
    _validate_video_id(video_id)
    removed = manager.delete_cache(video_id, mode)
    return {"status": "ok", "removed": removed}


if __name__ == "__main__":
    import os
    import uvicorn

    default_host = "0.0.0.0" if WORKER_SECRET else "127.0.0.1"
    bind_host = os.environ.get("KARAOKE_HOST", default_host)
    bind_port = int(os.environ.get("KARAOKE_PORT", "18765"))
    uvicorn.run("main:app", host=bind_host, port=bind_port, reload=False)
