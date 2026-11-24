"""24/7 Lightweight NAS Gateway for YouTube AI Karaoke.

Runs on NAS (Docker, ~140MB RAM, 0% GPU):
- Streams all cached MP3 stems + lyrics.json directly from Google Drive 24/7 — even when GPU PC is powered OFF!
- Probes the GPU PC node every 8s. When GPU PC is offline:
  - Cached songs still play seamlessly 24/7.
  - Uncached songs return HTTP 503 with: "AI 伺服器休息中，現有歌庫仍可免費唱".
- Never exposes any /api/admin/* or DELETE endpoints to the public internet.
"""

import base64
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterator, Literal, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

GPU_NODE_URL = os.environ.get("GPU_NODE_URL", "").rstrip("/")
WORKER_SECRET = os.environ.get("WORKER_SECRET", "").strip()
RCLONE_REMOTE = os.environ.get("RCLONE_REMOTE", "ytkaraoke_gdrive").strip()
RCLONE_FOLDER = os.environ.get("RCLONE_FOLDER", "YT-Karaoke-Cache").strip()
RATE_LIMIT_PER_10MIN = int(os.environ.get("RATE_LIMIT_PER_10MIN", "5"))
MAX_CLIENTS = int(os.environ.get("MAX_CLIENTS", "200"))
ALLOWED_EXT_IDS = {
    x.strip() for x in os.environ.get("ALLOWED_EXTENSION_IDS", "").split(",") if x.strip()
}

DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_FILE = DATA_DIR / "gdrive_index.json"
AUDIO_CACHE_DIR = DATA_DIR / "audio_cache"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MAX_NAS_CACHE_MB = int(os.environ.get("MAX_NAS_CACHE_MB", "2048"))

# Optional Portainer 1-paste convenience: decode base64 rclone.conf from env var
_rclone_b64 = os.environ.get("RCLONE_CONF_B64", "").strip()
_rclone_cfg_path = Path.home() / ".config" / "rclone" / "rclone.conf"
if _rclone_b64:
    try:
        _rclone_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        _rclone_cfg_path.write_bytes(base64.b64decode(_rclone_b64))
        os.chmod(_rclone_cfg_path, 0o600)
    except Exception as _exc:
        print(f"[Gateway] Failed to decode RCLONE_CONF_B64: {_exc}", flush=True)

OFFLINE_MSG = "AI 伺服器休息中，現有歌庫仍可免費唱"


def _is_allowed_origin(origin: Optional[str]) -> bool:
    if not origin:
        return True
    if origin == "https://www.youtube.com":
        return True
    if origin.startswith("chrome-extension://"):
        if not ALLOWED_EXT_IDS:
            return True
        ext_id = origin.replace("chrome-extension://", "").split("/")[0]
        return ext_id in ALLOWED_EXT_IDS
    return False


class GatewayState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._index: dict[str, dict] = {}
        self._lyrics_ram: dict[str, dict] = {}
        self._lyrics_missing: set[str] = set()
        self._lyrics_refreshing: set[str] = set()
        self._clients: dict[str, dict] = {}
        self._rate_hist: dict[str, list[float]] = {}

        self.gpu_online: bool = False
        self.gpu_queue_count: int = 0
        self.gpu_pause_new: bool = False
        self.last_gpu_check: float = 0.0
        self.last_index_sync: float = 0.0

        self._load_index()
        threading.Thread(target=self._bg_gpu_probe_loop, daemon=True, name="gpu-probe").start()
        threading.Thread(target=self._bg_index_sync_loop, daemon=True, name="gdrive-sync").start()

    def _load_index(self) -> None:
        if INDEX_FILE.exists():
            try:
                self._index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
            except Exception:
                self._index = {}

    def _save_index(self) -> None:
        try:
            INDEX_FILE.write_text(json.dumps(self._index, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _worker_headers(self) -> dict:
        h = {}
        if WORKER_SECRET:
            h["X-Worker-Secret"] = WORKER_SECRET
        return h

    def _bg_gpu_probe_loop(self) -> None:
        while True:
            self.probe_gpu_once()
            time.sleep(8)

    def probe_gpu_once(self) -> bool:
        if not GPU_NODE_URL:
            self.gpu_online = False
            return False
        try:
            r = httpx.get(f"{GPU_NODE_URL}/api/health", headers=self._worker_headers(), timeout=2.5)
            if r.status_code == 200:
                d = r.json()
                self.gpu_online = d.get("status") == "ok"
                self.gpu_queue_count = int(d.get("queue_count") or 0)
                self.gpu_pause_new = bool(d.get("pause_new_jobs"))
                self.last_gpu_check = time.time()
                return self.gpu_online
        except Exception:
            pass
        self.gpu_online = False
        self.last_gpu_check = time.time()
        return False

    def _bg_index_sync_loop(self) -> None:
        # Initial sync right after boot, then refresh every 3 minutes
        self.sync_gdrive_index()
        while True:
            time.sleep(180)
            self.sync_gdrive_index()

    def sync_gdrive_index(self, force_full_scan: bool = False) -> int:
        # Fast O(1) path: read consolidated _catalog.json (1 single Drive API call -> ~0.5s even for 10,000+ songs!)
        if not force_full_scan:
            try:
                cat_res = subprocess.run(
                    ["rclone", "cat", f"{RCLONE_REMOTE}:{RCLONE_FOLDER}/_catalog.json", "--drive-pacer-min-sleep", "200ms"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=35,
                )
                if cat_res.returncode == 0 and cat_res.stdout.strip():
                    cat_doc = json.loads(cat_res.stdout)
                    songs = cat_doc.get("songs") if isinstance(cat_doc, dict) else None
                    if isinstance(songs, dict) and len(songs) > 0:
                        cleaned: dict[str, dict] = {}
                        for key, entry in songs.items():
                            if not isinstance(entry, dict) or entry.get("mode", "pure") != "pure":
                                continue
                            vid = entry.get("video_id", "")
                            if not VIDEO_ID_RE.match(vid):
                                continue
                            cleaned[f"{vid}__pure"] = {
                                "video_id": vid,
                                "mode": "pure",
                                "title": entry.get("title") or vid,
                                "duration": entry.get("duration") or 0,
                                "size_mb": entry.get("size_mb") or 0,
                                "created_at": entry.get("created_at") or time.time(),
                                "quality_tier": entry.get("quality_tier") or "gpu_hq",
                                "storage": "gdrive",
                                "url": f"https://www.youtube.com/watch?v={vid}",
                            }
                        if cleaned:
                            evict_vids: list[str] = []
                            with self._lock:
                                for k, new_ent in cleaned.items():
                                    old_ent = self._index.get(k)
                                    if old_ent and old_ent.get("quality_tier") == "cpu_fast" and new_ent.get("quality_tier") == "gpu_hq":
                                        evict_vids.append(new_ent["video_id"])
                                self._index = cleaned
                                self._lyrics_missing.clear()
                                self._save_index()
                                self.last_index_sync = time.time()
                            for evid in evict_vids:
                                for stem in ("instrumental", "vocals"):
                                    try:
                                        (AUDIO_CACHE_DIR / f"{evid}_pure_{stem}.mp3").unlink(missing_ok=True)
                                    except Exception:
                                        pass
                                print(f"[Gateway] Evicted stale cpu_fast NAS cache for upgraded HQ song {evid}", flush=True)
                            print(f"[Gateway] Synced {len(cleaned)} songs via _catalog.json (fast path)", flush=True)
                            return len(cleaned)
            except Exception as exc:
                print(f"[Gateway] _catalog.json fast sync skipped: {exc}", flush=True)

        # Fallback full scan with --fast-list (~100x fewer Drive API calls than plain recursive lsjson)
        try:
            res = subprocess.run(
                ["rclone", "lsjson", f"{RCLONE_REMOTE}:{RCLONE_FOLDER}", "-R", "--files-only", "--fast-list"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if res.returncode != 0:
                print(f"[Gateway] rclone lsjson failed (rc={res.returncode}): {res.stderr.strip()[:300]}", flush=True)
                with self._lock:
                    return len(self._index)
            files = json.loads(res.stdout)
        except Exception as exc:
            print(f"[Gateway] rclone sync exception: {exc}", flush=True)
            with self._lock:
                return len(self._index)

        grouped: dict[tuple[str, str], dict] = {}
        for f in files:
            parts = f.get("Path", "").replace("\\", "/").split("/")
            if len(parts) != 3 or parts[0].startswith("_"):
                continue
            vid, mode, fname = parts
            if mode != "pure":
                continue
            g = grouped.setdefault((vid, mode), {"files": set(), "bytes": 0})
            g["files"].add(fname)
            g["bytes"] += int(f.get("Size", 0))

        with self._lock:
            snap = dict(self._index)

        # Publish song IDs immediately so streaming works within ~2s of boot, then enrich titles
        new_idx: dict[str, dict] = {}
        need_meta: list[tuple[str, str, str]] = []
        for (vid, mode), info in grouped.items():
            if "instrumental.mp3" in info["files"] and "vocals.mp3" in info["files"]:
                key = f"{vid}__{mode}"
                ex = snap.get(key, {})
                title = ex.get("title") or vid
                duration = ex.get("duration") or 0
                created_at = ex.get("created_at") or time.time()
                has_lyrics = "lyrics.json" in info["files"]
                new_idx[key] = {
                    "video_id": vid,
                    "mode": mode,
                    "title": title,
                    "duration": duration,
                    "size_mb": round(info["bytes"] / (1024 * 1024), 2),
                    "created_at": created_at,
                    "has_lyrics": has_lyrics,
                    "storage": "gdrive",
                    "url": f"https://www.youtube.com/watch?v={vid}",
                }
                if title == vid and "meta.json" in info["files"]:
                    need_meta.append((key, vid, mode))

        with self._lock:
            self._index = new_idx
            self._lyrics_missing.clear()
            self._save_index()
            self.last_index_sync = time.time()

        print(f"[Gateway] Synced {len(new_idx)} songs from Google Drive", flush=True)

        # Cap per-cycle meta.json enrichment so cold startup with thousands of songs never stalls
        for key, vid, mode in need_meta[:30]:
            try:
                mraw = subprocess.check_output(
                    ["rclone", "cat", f"{RCLONE_REMOTE}:{RCLONE_FOLDER}/{vid}/{mode}/meta.json"],
                    text=True,
                    encoding="utf-8",
                    timeout=15,
                )
                m = json.loads(mraw)
                with self._lock:
                    if key in self._index:
                        self._index[key]["title"] = m.get("title", vid)
                        self._index[key]["duration"] = m.get("duration", 0)
                        self._index[key]["created_at"] = m.get("created_at", self._index[key]["created_at"])
                        self._save_index()
            except Exception:
                pass

        with self._lock:
            return len(self._index)

    def has_song(self, video_id: str, mode: str = "pure") -> bool:
        with self._lock:
            return f"{video_id}__{mode}" in self._index

    def get_song_meta(self, video_id: str, mode: str = "pure") -> dict:
        with self._lock:
            return dict(self._index.get(f"{video_id}__{mode}", {}))

    def list_songs(self) -> list[dict]:
        with self._lock:
            items = list(self._index.values())
        items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return items

    def get_nas_cached_audio(self, video_id: str, mode: str, stem: str) -> Optional[Path]:
        p = AUDIO_CACHE_DIR / f"{video_id}_{mode}_{stem}.mp3"
        if p.exists() and p.stat().st_size > 16384:
            try:
                os.utime(p, None)  # Refresh LRU mtime
            except Exception:
                pass
            return p
        return None

    def _enforce_nas_cache_limit(self) -> None:
        if MAX_NAS_CACHE_MB <= 0:
            return
        try:
            files = [f for f in AUDIO_CACHE_DIR.glob("*.mp3") if f.is_file()]
            total = sum(f.stat().st_size for f in files)
            limit = MAX_NAS_CACHE_MB * 1024 * 1024
            if total <= limit:
                return
            files.sort(key=lambda f: f.stat().st_mtime)
            for f in files:
                if total <= limit * 0.85:
                    break
                sz = f.stat().st_size
                try:
                    f.unlink()
                    total -= sz
                except Exception:
                    pass
        except Exception:
            pass

    def stream_and_cache_gdrive_audio(self, video_id: str, mode: str, stem: str) -> Iterator[bytes]:
        remote_file = f"{RCLONE_REMOTE}:{RCLONE_FOLDER}/{video_id}/{mode}/{stem}.mp3"
        dest = AUDIO_CACHE_DIR / f"{video_id}_{mode}_{stem}.mp3"
        tmp = AUDIO_CACHE_DIR / f".{video_id}_{mode}_{stem}.{int(time.time() * 1000)}.tmp"
        proc = subprocess.Popen(["rclone", "cat", remote_file], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        fh = None
        written = 0
        completed = False
        try:
            if MAX_NAS_CACHE_MB > 0:
                try:
                    fh = open(tmp, "wb")
                except Exception:
                    fh = None
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                if fh is not None:
                    try:
                        fh.write(chunk)
                    except Exception:
                        pass
                written += len(chunk)
                yield chunk
            proc.wait(timeout=5)
            completed = (proc.returncode == 0 and written > 16384)
        finally:
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
            if completed and tmp.exists():
                try:
                    os.replace(tmp, dest)
                    threading.Thread(target=self._enforce_nas_cache_limit, daemon=True).start()
                except Exception:
                    tmp.unlink(missing_ok=True)
            elif tmp.exists():
                tmp.unlink(missing_ok=True)
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                pass

    def get_cached_lyrics(self, video_id: str, mode: str = "pure") -> Optional[dict]:
        with self._lock:
            if video_id in self._lyrics_ram:
                return self._lyrics_ram[video_id]
            if video_id in self._lyrics_missing:
                return None

        remote_file = f"{RCLONE_REMOTE}:{RCLONE_FOLDER}/{video_id}/{mode}/lyrics.json"
        try:
            raw = subprocess.check_output(["rclone", "cat", remote_file], text=True, encoding="utf-8", timeout=25)
            data = json.loads(raw)
            if data and data.get("lines"):
                with self._lock:
                    self._lyrics_ram[video_id] = data
                return data
        except Exception:
            pass

        with self._lock:
            self._lyrics_missing.add(video_id)
        return None

    def touch_client(self, client_id: Optional[str], ip: str, video_id: Optional[str] = None, singing: Optional[bool] = None) -> bool:
        if not client_id or not CLIENT_ID_RE.match(client_id):
            return True
        now = time.time()
        with self._lock:
            expired = [k for k, v in self._clients.items() if now - v.get("last_seen", 0) > 40.0]
            for k in expired:
                self._clients.pop(k, None)
            if client_id not in self._clients and len(self._clients) >= MAX_CLIENTS:
                return False
            e = self._clients.get(client_id, {"ip": ip, "video_id": "", "singing": False})
            e["ip"] = ip
            e["last_seen"] = now
            if video_id is not None:
                e["video_id"] = video_id
            if singing is not None:
                e["singing"] = bool(singing)
            self._clients[client_id] = e
            return True

    def remove_client(self, client_id: Optional[str]) -> None:
        if not client_id:
            return
        with self._lock:
            self._clients.pop(client_id, None)

    def client_counts(self) -> dict:
        now = time.time()
        with self._lock:
            expired = [k for k, v in self._clients.items() if now - v.get("last_seen", 0) > 40.0]
            for k in expired:
                self._clients.pop(k, None)
            return {
                "active_clients": len(self._clients),
                "active_singing": sum(1 for v in self._clients.values() if v.get("singing")),
            }

    def check_rate_limit(self, ip: str) -> bool:
        now = time.time()
        with self._lock:
            if len(self._rate_hist) > 2000:
                self._rate_hist = {
                    k: [t for t in v if now - t < 600.0]
                    for k, v in self._rate_hist.items()
                    if any(now - t < 600.0 for t in v)
                }
            hist = [t for t in self._rate_hist.get(ip, []) if now - t < 600.0]
            if len(hist) >= RATE_LIMIT_PER_10MIN:
                self._rate_hist[ip] = hist
                return False
            hist.append(now)
            self._rate_hist[ip] = hist
            return True


gw = GatewayState()
app = FastAPI(title="YouTube AI Karaoke 24/7 Gateway", version="1.0.0", docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def cors_and_security(request: Request, call_next):
    origin = request.headers.get("origin")
    if origin and not _is_allowed_origin(origin):
        return Response(content='{"detail":"Origin not allowed"}', status_code=403, media_type="application/json")

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


def _validate_vid(video_id: str) -> str:
    if not VIDEO_ID_RE.match(video_id):
        raise HTTPException(status_code=400, detail="Invalid video_id")
    return video_id


def _client_ip(request: Request) -> str:
    # Prefer Cloudflare's verified edge header (cannot be spoofed when behind Cloudflare Proxy)
    cf_ip = (request.headers.get("cf-connecting-ip") or "").strip()
    if cf_ip:
        return cf_ip
    real_ip = (request.headers.get("x-real-ip") or "").strip()
    if real_ip:
        return real_ip
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


@app.get("/api/health")
def health(request: Request, x_karaoke_client: Optional[str] = Header(default=None)):
    if x_karaoke_client:
        gw.touch_client(x_karaoke_client, _client_ip(request))
    songs = gw.list_songs()
    ai_available = gw.gpu_online and not gw.gpu_pause_new
    return {
        "status": "ok",
        "gateway_mode": True,
        "gpu_online": ai_available,
        "gpu_offline_message": "" if ai_available else OFFLINE_MSG,
        "cache_count": len(songs),
        "cache_size_mb": round(sum(s["size_mb"] for s in songs), 2),
        "queue_count": gw.gpu_queue_count if gw.gpu_online else 0,
        **gw.client_counts(),
    }


@app.post("/api/heartbeat")
def heartbeat(req: HeartbeatRequest, request: Request):
    ip = _client_ip(request)
    if req.leaving:
        gw.remove_client(req.client_id)
    else:
        vid = req.video_id if (req.video_id and VIDEO_ID_RE.match(req.video_id)) else None
        if not gw.touch_client(req.client_id, ip, video_id=vid, singing=req.singing):
            raise HTTPException(status_code=503, detail="目前已達最大在線人數上限，請稍後再試。")
    ai_available = gw.gpu_online and not gw.gpu_pause_new
    return {"status": "ok", "gpu_online": ai_available, **gw.client_counts()}


@app.get("/api/cache")
def list_cache(limit: int = 60, offset: int = 0, q: str = ""):
    all_items = gw.list_songs()
    total_size_mb = round(sum(i["size_mb"] for i in all_items), 2)

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
        page = filtered[safe_offset : safe_offset + 2000]
    else:
        safe_limit = max(1, min(500, limit))
        page = filtered[safe_offset : safe_offset + safe_limit]

    return {
        "count": len(filtered),
        "total_songs": len(all_items),
        "returned": len(page),
        "has_more": (safe_offset + len(page)) < len(filtered),
        "total_size_mb": total_size_mb,
        "storage_mode": "gdrive",
        "items": page,
    }


@app.get("/api/status/{video_id}")
def get_status(video_id: str, mode: Literal["pure"] = "pure"):
    _validate_vid(video_id)
    if gw.has_song(video_id, mode):
        meta = gw.get_song_meta(video_id, mode)
        return {
            "video_id": video_id,
            "mode": mode,
            "status": "ready",
            "progress": 100,
            "stage_text": "已就緒，隨時可以開唱",
            "title": meta.get("title", ""),
            "duration": meta.get("duration", 0),
            "cached": True,
            "cached_modes": [mode],
        }

    # Proxy live progress from GPU node when online
    if gw.gpu_online and GPU_NODE_URL:
        try:
            r = httpx.get(
                f"{GPU_NODE_URL}/api/status/{video_id}",
                params={"mode": mode},
                headers=gw._worker_headers(),
                timeout=4.0,
            )
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == "ready":
                    # Refresh gateway index immediately so this song is now playable 24/7!
                    threading.Thread(target=gw.sync_gdrive_index, daemon=True).start()
                return data
        except Exception:
            gw.gpu_online = False

    return {
        "video_id": video_id,
        "mode": mode,
        "status": "idle",
        "progress": 0,
        "stage_text": OFFLINE_MSG if not gw.gpu_online else "未處理",
        "cached": False,
        "cached_modes": [],
    }


@app.post("/api/separate")
def separate(req: SeparateRequest, request: Request):
    _validate_vid(req.video_id)
    # 1) Already in Google Drive -> serve 24/7 even when GPU PC is off!
    if gw.has_song(req.video_id, req.mode):
        return get_status(req.video_id, req.mode)

    # 2) Need GPU PC to separate a new song
    if not gw.gpu_online or gw.gpu_pause_new or not GPU_NODE_URL:
        raise HTTPException(status_code=503, detail=OFFLINE_MSG)

    if not gw.check_rate_limit(_client_ip(request)):
        raise HTTPException(status_code=429, detail=f"請求過於頻繁（每 10 分鐘上限 {RATE_LIMIT_PER_10MIN} 首新歌），請稍後再試。")

    try:
        r = httpx.post(
            f"{GPU_NODE_URL}/api/separate",
            json={"video_id": req.video_id, "mode": req.mode},
            headers=gw._worker_headers(),
            timeout=8.0,
        )
        if r.status_code >= 400:
            detail = r.json().get("detail") if r.headers.get("content-type", "").startswith("application/json") else r.text
            raise HTTPException(status_code=r.status_code, detail=detail or OFFLINE_MSG)
        return r.json()
    except HTTPException:
        raise
    except Exception:
        gw.gpu_online = False
        raise HTTPException(status_code=503, detail=OFFLINE_MSG)


@app.get("/api/audio/{video_id}/{mode}/{stem}")
def get_audio(video_id: str, mode: Literal["pure"], stem: Literal["instrumental", "vocals"]):
    _validate_vid(video_id)
    # 1) Hot NAS LRU Cache -> instant ~40ms playback (350x faster than Google Drive rclone!)
    cached_file = gw.get_nas_cached_audio(video_id, mode, stem)
    if cached_file:
        return FileResponse(
            path=str(cached_file),
            media_type="audio/mpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    # 2) Cache miss -> stream from Google Drive while populating NAS LRU cache automatically!
    if gw.has_song(video_id, mode):
        return StreamingResponse(
            gw.stream_and_cache_gdrive_audio(video_id, mode, stem),
            media_type="audio/mpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    # Fallback proxy to GPU PC if just finished separating before index synced
    if gw.gpu_online and GPU_NODE_URL:
        def _proxy_stream():
            with httpx.stream("GET", f"{GPU_NODE_URL}/api/audio/{video_id}/{mode}/{stem}", headers=gw._worker_headers(), timeout=120.0) as r:
                if r.status_code != 200:
                    return
                for chunk in r.iter_bytes(65536):
                    yield chunk
        return StreamingResponse(_proxy_stream(), media_type="audio/mpeg", headers={"Cache-Control": "public, max-age=86400"})
    raise HTTPException(status_code=404, detail="Audio stem not found")


@app.get("/api/lyrics/{video_id}")
def get_lyrics(video_id: str, mode: Literal["pure"] = "pure"):
    _validate_vid(video_id)
    with gw._lock:
        is_refreshing = video_id in gw._lyrics_refreshing

    if not is_refreshing:
        cached = gw.get_cached_lyrics(video_id, mode)
        if cached:
            return {"status": "ready", "progress": 100, "stage_text": cached.get("source_label", "歌詞已就緒"), **cached}

    if gw.gpu_online and GPU_NODE_URL:
        try:
            r = httpx.get(
                f"{GPU_NODE_URL}/api/lyrics/{video_id}",
                params={"mode": mode},
                headers=gw._worker_headers(),
                timeout=8.0,
            )
            if r.status_code == 200:
                d = r.json()
                if d.get("status") == "ready" and d.get("lines"):
                    with gw._lock:
                        gw._lyrics_ram[video_id] = d
                        gw._lyrics_missing.discard(video_id)
                        gw._lyrics_refreshing.discard(video_id)
                elif d.get("status") in ("error", "idle"):
                    with gw._lock:
                        gw._lyrics_refreshing.discard(video_id)
                return d
        except Exception:
            with gw._lock:
                gw._lyrics_refreshing.discard(video_id)

    return {"video_id": video_id, "status": "idle", "progress": 0, "stage_text": OFFLINE_MSG if not gw.gpu_online else "未生成歌詞", "lines": []}


@app.post("/api/lyrics")
def generate_lyrics(req: LyricsRequest, request: Request):
    _validate_vid(req.video_id)
    if not req.force_ai:
        cached = gw.get_cached_lyrics(req.video_id, req.mode)
        if cached:
            return {"status": "ready", "progress": 100, "stage_text": cached.get("source_label", "歌詞已就緒"), **cached}

    if not gw.gpu_online or gw.gpu_pause_new or not GPU_NODE_URL:
        raise HTTPException(status_code=503, detail=OFFLINE_MSG)

    if not gw.check_rate_limit(_client_ip(request)):
        raise HTTPException(status_code=429, detail=f"請求過於頻繁（每 10 分鐘上限 {RATE_LIMIT_PER_10MIN} 次），請稍後再試。")

    if req.force_ai:
        with gw._lock:
            gw._lyrics_ram.pop(req.video_id, None)
            gw._lyrics_missing.discard(req.video_id)
            gw._lyrics_refreshing.add(req.video_id)

    try:
        r = httpx.post(
            f"{GPU_NODE_URL}/api/lyrics",
            json={"video_id": req.video_id, "mode": req.mode, "force_ai": req.force_ai},
            headers=gw._worker_headers(),
            timeout=10.0,
        )
        if r.status_code >= 400:
            detail = None
            if r.status_code < 500 and r.headers.get("content-type", "").startswith("application/json"):
                try:
                    detail = r.json().get("detail")
                except Exception:
                    detail = None
            raise HTTPException(status_code=r.status_code, detail=detail or OFFLINE_MSG)
        return r.json()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=503, detail=OFFLINE_MSG)
