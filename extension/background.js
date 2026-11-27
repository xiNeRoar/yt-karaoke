const API_BASE = "http://127.0.0.1:18765";
const API_ORIGIN = new URL(API_BASE).origin;
const IS_LOCAL_DEV = API_BASE.startsWith("http://127.0.0.1") || API_BASE.startsWith("http://localhost");
const VIDEO_ID_RE = /^[A-Za-z0-9_-]{11}$/;
let lastReloadNonce = null;

// Shared zero-spam catalog ID cache across all YouTube tabs (powers instant visual DB indicators)
let _catalogCache = { ids: [], version: "", fetchedAt: 0, inFlight: null };

function isSafeApiPath(path) {
  if (typeof path !== "string" || !path.startsWith("/api/")) return false;
  if (/[\s\\]|\/\/|\.\./.test(path)) return false;
  if (path.startsWith("/api/admin/") || path.startsWith("/api/internal/")) return false;
  try {
    const resolved = new URL(path, API_BASE);
    return resolved.origin === API_ORIGIN && resolved.pathname.startsWith("/api/");
  } catch (_e) {
    return false;
  }
}

async function fetchCatalogVideoIds(force = false) {
  const now = Date.now();
  if (!force && _catalogCache.fetchedAt > 0 && now - _catalogCache.fetchedAt < 60000) {
    return _catalogCache.ids;
  }
  if (_catalogCache.inFlight) {
    return _catalogCache.inFlight;
  }
  _catalogCache.inFlight = (async () => {
    const st = await getAuthState();
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 12000);
    try {
      const headers = {};
      if (st.token) headers["X-Karaoke-Token"] = st.token;
      const res = await fetch(`${API_BASE}/api/cache?limit=0`, { headers, signal: ctrl.signal });
      if (!res.ok) {
        if (res.status === 401) {
          await saveAuthToken("");
          _catalogCache.ids = [];
          _catalogCache.version = "";
          _catalogCache.fetchedAt = 0;
        }
        return _catalogCache.ids;
      }
      const data = await res.json().catch(() => null);
      if (data && Array.isArray(data.items)) {
        const seen = new Set();
        for (const it of data.items) {
          if (it && (it.mode === "pure" || !it.mode) && VIDEO_ID_RE.test(String(it.video_id || ""))) {
            seen.add(it.video_id);
          }
        }
        _catalogCache.ids = Array.from(seen);
        if (typeof data.catalog_version === "string" && data.catalog_version) {
          _catalogCache.version = data.catalog_version;
        }
        _catalogCache.fetchedAt = Date.now();
      }
      return _catalogCache.ids;
    } catch (_e) {
      return _catalogCache.ids;
    } finally {
      clearTimeout(timer);
      _catalogCache.inFlight = null;
    }
  })();
  return _catalogCache.inFlight;
}

// Persist & cache the user's bound invite device token + stable device ID in chrome.storage.local
let _cachedAuth = { loaded: false, token: "", deviceId: "" };

function _genDeviceId() {
  const bytes = new Uint8Array(12);
  crypto.getRandomValues(bytes);
  return "dev_" + Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

async function getAuthState() {
  if (_cachedAuth.loaded && _cachedAuth.deviceId) {
    return { token: _cachedAuth.token, deviceId: _cachedAuth.deviceId };
  }
  try {
    const stored = await chrome.storage.local.get(["ktv_access_token", "ktv_device_id"]);
    let deviceId = stored.ktv_device_id || "";
    if (!deviceId) {
      deviceId = _genDeviceId();
      await chrome.storage.local.set({ ktv_device_id: deviceId });
    }
    _cachedAuth = {
      loaded: true,
      token: typeof stored.ktv_access_token === "string" ? stored.ktv_access_token : "",
      deviceId,
    };
  } catch (_e) {
    if (!_cachedAuth.deviceId) _cachedAuth.deviceId = _genDeviceId();
    _cachedAuth.loaded = true;
  }
  return { token: _cachedAuth.token, deviceId: _cachedAuth.deviceId };
}

async function saveAuthToken(token) {
  const st = await getAuthState();
  _cachedAuth.token = token || "";
  try {
    if (token) {
      await chrome.storage.local.set({ ktv_access_token: token, ktv_device_id: st.deviceId });
    } else {
      await chrome.storage.local.remove(["ktv_access_token"]);
    }
  } catch (_e) {}
}

if (chrome.storage && chrome.storage.onChanged) {
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === "local" && changes.ktv_access_token) {
      _cachedAuth.token = typeof changes.ktv_access_token.newValue === "string" ? changes.ktv_access_token.newValue : "";
      if (!_cachedAuth.token) {
        _catalogCache.ids = [];
        _catalogCache.version = "";
        _catalogCache.fetchedAt = 0;
      }
    }
  });
}

// Right after local dev auto-reload finishes, gently refresh open YouTube watch tabs so new content.js takes effect
chrome.runtime.onInstalled.addListener((details) => {
  if (IS_LOCAL_DEV && (details.reason === "update" || details.reason === "install")) {
    chrome.tabs.query({ url: "https://www.youtube.com/watch*" }, (tabs) => {
      for (const tab of tabs || []) {
        if (tab.id) {
          try { chrome.tabs.reload(tab.id); } catch (_e) {}
        }
      }
    });
  }
});

function checkReloadNonce(nonce) {
  if (!IS_LOCAL_DEV || typeof nonce !== "number") return;
  if (lastReloadNonce === null) {
    lastReloadNonce = nonce;
    return;
  }
  if (nonce > lastReloadNonce) {
    lastReloadNonce = nonce;
    chrome.runtime.reload();
  }
}

// Local dev auto-reload watcher (skipped in production Chrome Web Store builds)
if (IS_LOCAL_DEV) {
  setInterval(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (res.ok) {
        const data = await res.json();
        checkReloadNonce(data.extension_reload_nonce);
      }
    } catch (_e) {}
  }, 30000);
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!sender || sender.id !== chrome.runtime.id) return false;
  if (!message || !message.type) return false;

  (async () => {
    try {
      if (message.type === "AUTH_GET_STATE") {
        const st = await getAuthState();
        sendResponse({ ok: true, token: st.token, deviceId: st.deviceId });
        return;
      }

      if (message.type === "AUTH_CLEAR") {
        await saveAuthToken("");
        _catalogCache = { ids: [], version: "", fetchedAt: 0, inFlight: null };
        sendResponse({ ok: true });
        return;
      }

      if (message.type === "CATALOG_GET_IDS") {
        const ids = await fetchCatalogVideoIds(!!message.force);
        sendResponse({ ok: true, ids, version: _catalogCache.version });
        return;
      }

      if (message.type === "CATALOG_ADD_ID") {
        const vid = typeof message.videoId === "string" ? message.videoId : "";
        if (VIDEO_ID_RE.test(vid) && !_catalogCache.ids.includes(vid)) {
          _catalogCache.ids.push(vid);
        }
        sendResponse({ ok: true, ids: _catalogCache.ids });
        return;
      }

      if (message.type === "AUTH_ACTIVATE") {
        const code = typeof message.code === "string" ? message.code.trim() : "";
        if (!code) {
          sendResponse({ ok: false, error: "請輸入邀請碼" });
          return;
        }
        const st = await getAuthState();
        const ctrl = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), 15000);
        try {
          const res = await fetch(`${API_BASE}/api/auth/activate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code, device_id: st.deviceId }),
            signal: ctrl.signal,
          });
          const data = await res.json().catch(() => ({}));
          if (res.ok && data && data.token) {
            await saveAuthToken(data.token);
            _catalogCache.fetchedAt = 0;
            sendResponse({ ok: true, status: res.status, data });
          } else {
            sendResponse({ ok: false, status: res.status, error: (data && data.detail) || "邀請碼啟用失敗", data });
          }
        } finally {
          clearTimeout(timer);
        }
        return;
      }

      if (message.type === "API_REQUEST") {
        const { path, method = "GET", body = null, clientId = null } = message;
        const safeMethod = method === "POST" ? "POST" : "GET";
        if (!isSafeApiPath(path)) {
          sendResponse({ ok: false, error: "Invalid API path" });
          return;
        }
        const st = await getAuthState();
        const ctrl = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), 20000);
        const init = {
          method: safeMethod,
          headers: {},
          signal: ctrl.signal,
        };
        if (clientId && /^[A-Za-z0-9_-]{8,64}$/.test(String(clientId))) {
          init.headers["X-Karaoke-Client"] = String(clientId);
        }
        if (st.token) {
          init.headers["X-Karaoke-Token"] = st.token;
        }
        if (body !== null && safeMethod === "POST") {
          init.headers["Content-Type"] = "application/json";
          init.body = JSON.stringify(body);
        }
        try {
          const res = await fetch(`${API_BASE}${path}`, init);
          const data = await res.json().catch(() => ({}));
          if (res.status === 401 && data && data.auth_required) {
            await saveAuthToken("");
          }
          if (data && typeof data.catalog_version === "string" && data.catalog_version && data.catalog_version !== _catalogCache.version) {
            _catalogCache.version = data.catalog_version;
            _catalogCache.fetchedAt = 0;
          }
          if (data && typeof data.extension_reload_nonce === "number") {
            checkReloadNonce(data.extension_reload_nonce);
          }
          sendResponse({ ok: res.ok, status: res.status, data });
        } finally {
          clearTimeout(timer);
        }
        return;
      }

      sendResponse({ ok: false, error: "Unknown message type" });
    } catch (err) {
      sendResponse({ ok: false, error: String(err && err.message ? err.message : err) });
    }
  })();

  return true;
});

// Stream audio binary over a long-lived Port in 256KB chunks (immune to sendMessage size limits)
chrome.runtime.onConnect.addListener((port) => {
  if (!port.sender || port.sender.id !== chrome.runtime.id) {
    try { port.disconnect(); } catch (_e) {}
    return;
  }
  if (port.name !== "AUDIO_STREAM") return;
  let activeCtrl = null;
  let disconnected = false;

  port.onDisconnect.addListener(() => {
    disconnected = true;
    if (activeCtrl) {
      try { activeCtrl.abort(); } catch (_e) {}
    }
  });

  port.onMessage.addListener(async (msg) => {
    if (!msg || !isSafeApiPath(msg.path) || !/^\/api\/audio\/[A-Za-z0-9_-]{11}\/pure\/(instrumental|vocals)$/.test(msg.path)) return;
    const st = await getAuthState();
    const ctrl = new AbortController();
    activeCtrl = ctrl;
    const timer = setTimeout(() => ctrl.abort(), 60000);
    try {
      const headers = {};
      if (st.token) headers["X-Karaoke-Token"] = st.token;
      const res = await fetch(`${API_BASE}${msg.path}`, { headers, signal: ctrl.signal });
      if (disconnected) return;
      if (!res.ok) {
        if (res.status === 401) {
          await saveAuthToken("");
        }
        port.postMessage({ type: "error", status: res.status, error: `HTTP ${res.status}` });
        return;
      }
      const buf = await res.arrayBuffer();
      if (disconnected) return;
      const bytes = new Uint8Array(buf);
      const CHUNK = 524288;
      port.postMessage({ type: "meta", totalBytes: bytes.length });

      for (let off = 0; off < bytes.length; off += CHUNK) {
        if (disconnected) return;
        const slice = bytes.subarray(off, Math.min(off + CHUNK, bytes.length));
        let bin = "";
        const sub = 0x8000;
        for (let i = 0; i < slice.length; i += sub) {
          bin += String.fromCharCode.apply(null, slice.subarray(i, i + sub));
        }
        port.postMessage({ type: "chunk", b64: btoa(bin) });
      }
      if (!disconnected) port.postMessage({ type: "done" });
    } catch (err) {
      if (!disconnected) {
        try {
          port.postMessage({ type: "error", error: String(err && err.message ? err.message : err) });
        } catch (_e) {}
      }
    } finally {
      clearTimeout(timer);
      if (activeCtrl === ctrl) activeCtrl = null;
    }
  });
});
