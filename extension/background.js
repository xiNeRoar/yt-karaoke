const API_BASE = "http://127.0.0.1:18765";
let lastReloadNonce = null;

// Right after auto-reload finishes, gently refresh open YouTube watch tabs so new content.js takes effect
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === "update" || details.reason === "install") {
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
  if (typeof nonce !== "number") return;
  if (lastReloadNonce === null) {
    lastReloadNonce = nonce;
    return;
  }
  if (nonce > lastReloadNonce) {
    lastReloadNonce = nonce;
    chrome.runtime.reload();
  }
}

// Gentle background health check (no-op on public gateway unless reload nonce increments)
setInterval(async () => {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (res.ok) {
      const data = await res.json();
      checkReloadNonce(data.extension_reload_nonce);
    }
  } catch (_e) {}
}, 30000);

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || !message.type) return false;

  (async () => {
    try {
      if (message.type === "API_REQUEST") {
        const { path, method = "GET", body = null, clientId = null } = message;
        const init = {
          method,
          headers: {},
        };
        if (clientId) {
          init.headers["X-Karaoke-Client"] = clientId;
        }
        if (body !== null) {
          init.headers["Content-Type"] = "application/json";
          init.body = JSON.stringify(body);
        }
        const res = await fetch(`${API_BASE}${path}`, init);
        const data = await res.json().catch(() => ({}));
        if (data && typeof data.extension_reload_nonce === "number") {
          checkReloadNonce(data.extension_reload_nonce);
        }
        sendResponse({ ok: res.ok, status: res.status, data });
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
  if (port.name !== "AUDIO_STREAM") return;

  port.onMessage.addListener(async (msg) => {
    if (!msg || !msg.path) return;
    try {
      const res = await fetch(`${API_BASE}${msg.path}`);
      if (!res.ok) {
        port.postMessage({ type: "error", error: `HTTP ${res.status}` });
        return;
      }
      const buf = await res.arrayBuffer();
      const bytes = new Uint8Array(buf);
      const CHUNK = 262144;
      port.postMessage({ type: "meta", totalBytes: bytes.length });

      for (let off = 0; off < bytes.length; off += CHUNK) {
        const slice = bytes.subarray(off, Math.min(off + CHUNK, bytes.length));
        let bin = "";
        const sub = 0x8000;
        for (let i = 0; i < slice.length; i += sub) {
          bin += String.fromCharCode.apply(null, slice.subarray(i, i + sub));
        }
        port.postMessage({ type: "chunk", b64: btoa(bin) });
      }
      port.postMessage({ type: "done" });
    } catch (err) {
      try {
        port.postMessage({ type: "error", error: String(err && err.message ? err.message : err) });
      } catch (_e) {}
    }
  });
});
