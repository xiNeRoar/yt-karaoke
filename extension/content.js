(() => {
  "use strict";

  const API_BASE = "http://127.0.0.1:18765";
  const VIDEO_ID_RE = /^[A-Za-z0-9_-]{11}$/;
  const AUDIO_PATH_RE = /^\/api\/audio\/[A-Za-z0-9_-]{11}\/pure\/(instrumental|vocals)$/;
  const CLIENT_ID = (() => {
    const b = new Uint8Array(6);
    crypto.getRandomValues(b);
    return `yt_${Date.now().toString(36)}_${Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("")}`;
  })();

  // Trusted-Types-safe inline SVG icon factory (minimalist Lucide stroke icons)
  function svgIcon(paths, size = 14) {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("width", String(size));
    svg.setAttribute("height", String(size));
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", "2");
    svg.setAttribute("stroke-linecap", "round");
    svg.setAttribute("stroke-linejoin", "round");
    svg.setAttribute("class", "ktv-icon-svg");
    for (const d of paths) {
      const p = document.createElementNS(ns, "path");
      p.setAttribute("d", d);
      svg.appendChild(p);
    }
    return svg;
  }

  const ICONS = {
    mic: () => svgIcon(["M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z", "M19 10v2a7 7 0 0 1-14 0v-2", "M12 19v3", "M8 22h8"], 15),
    sliders: () => svgIcon(["M4 21v-7", "M4 10V3", "M12 21v-9", "M12 8V3", "M20 21v-5", "M20 12V3", "M1 14h6", "M9 8h6", "M17 16h6"], 13),
    music: () => svgIcon(["M9 18V5l12-2v13", "M6 18a3 3 0 1 0 0 .01", "M18 16a3 3 0 1 0 0 .01"], 13),
    captions: () => svgIcon(["M3 5h18a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Z", "M7 15h4", "M15 15h2", "M7 11h2", "M13 11h4"], 13),
    zap: (size = 13) => svgIcon(["M13 2L3 14h9l-1 8 10-12h-9l1-8z"], size),
    sparkles: (size = 13) => svgIcon(["M12 3l1.8 4.2L18 9l-4.2 1.8L12 15l-1.8-4.2L6 9l4.2-1.8L12 3z", "M19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9L19 15z"], size),
  };

  // Helper to create DOM elements safely (100% Trusted Types compliant on YouTube)
  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "className") node.className = v;
      else if (k === "textContent") node.textContent = v;
      else if (k.startsWith("on") && typeof v === "function") {
        node.addEventListener(k.slice(2).toLowerCase(), v);
      } else if (v !== undefined && v !== null) {
        node.setAttribute(k, String(v));
      }
    }
    for (const child of children) {
      if (typeof child === "string") node.appendChild(document.createTextNode(child));
      else if (child) node.appendChild(child);
    }
    return node;
  }

  function apiRequest(path, method = "GET", body = null) {
    return new Promise((resolve) => {
      try {
        chrome.runtime.sendMessage({ type: "API_REQUEST", path, method, body, clientId: CLIENT_ID }, (resp) => {
          if (chrome.runtime.lastError) {
            resolve({ ok: false, error: chrome.runtime.lastError.message });
            return;
          }
          const out = resp || { ok: false, error: "No response" };
          if (out.status === 401 && out.data && out.data.auth_required) {
            state.authRequired = true;
            state.authenticated = false;
            if (ui && ui.authMsg && out.data.detail) {
              ui.authMsg.textContent = out.data.detail;
              ui.authMsg.classList.add("error");
            }
            updateUI();
          }
          resolve(out);
        });
      } catch (err) {
        resolve({ ok: false, error: String(err) });
      }
    });
  }

  function getAuthToken() {
    return new Promise((resolve) => {
      try {
        chrome.runtime.sendMessage({ type: "AUTH_GET_STATE" }, (resp) => {
          if (chrome.runtime.lastError || !resp || !resp.ok) resolve("");
          else resolve(resp.token || "");
        });
      } catch (_e) {
        resolve("");
      }
    });
  }

  function activateInviteCode(code) {
    return new Promise((resolve) => {
      try {
        chrome.runtime.sendMessage({ type: "AUTH_ACTIVATE", code }, (resp) => {
          if (chrome.runtime.lastError) resolve({ ok: false, error: chrome.runtime.lastError.message });
          else resolve(resp || { ok: false, error: "無回應" });
        });
      } catch (err) {
        resolve({ ok: false, error: String(err) });
      }
    });
  }

  // Bounded browser Cache API store (max 12 recent songs = 24 stems) + hover prefetch map
  // Strictly gated on authentication and wiped on AUTH_CLEAR/401 so revoked tokens never replay audio!
  const AUDIO_CACHE_NAME = "yt-ai-ktv-audio-v1";
  const AUDIO_CACHE_MAX_ENTRIES = 24;
  const _audioPrefetchMap = new Map(); // path -> Promise<ArrayBuffer> (max 2 entries)

  async function ktvAudioCacheClear() {
    _audioPrefetchMap.clear();
    if (typeof caches === "undefined") return;
    try {
      await caches.delete(AUDIO_CACHE_NAME);
    } catch (_e) {}
  }

  async function ktvAudioCacheGet(path) {
    if (typeof caches === "undefined") return null;
    if (state.authRequired && !state.authenticated) return null;
    try {
      const cache = await caches.open(AUDIO_CACHE_NAME);
      const match = await cache.match(`${API_BASE}${path}`);
      if (!match) return null;
      const buf = await match.arrayBuffer();
      return buf && buf.byteLength > 8192 ? buf : null;
    } catch (_e) {
      return null;
    }
  }

  async function ktvAudioCachePut(path, bufCopy) {
    if (typeof caches === "undefined" || !bufCopy || bufCopy.byteLength <= 8192) return;
    if (state.authRequired && !state.authenticated) return;
    try {
      const cache = await caches.open(AUDIO_CACHE_NAME);
      const reqUrl = `${API_BASE}${path}`;
      await cache.put(
        reqUrl,
        new Response(bufCopy, {
          status: 200,
          headers: { "Content-Type": "audio/mpeg", "X-KTV-Cached-At": String(Date.now()) },
        })
      );
      const keys = await cache.keys();
      if (keys.length > AUDIO_CACHE_MAX_ENTRIES) {
        const dropCount = keys.length - AUDIO_CACHE_MAX_ENTRIES;
        for (let i = 0; i < dropCount; i++) {
          await cache.delete(keys[i]);
        }
      }
    } catch (_e) {}
  }

  async function _downloadAudioRaw(path) {
    const token = await getAuthToken();
    const headers = token ? { "X-Karaoke-Token": token } : {};
    // 1) Direct HTTPS fetch with full-body timeout + 1 auto-retry (zero Base64 overhead!)
    for (let attempt = 0; attempt < 2; attempt++) {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 45000);
      try {
        const res = await fetch(`${API_BASE}${path}`, { headers, signal: ctrl.signal });
        if (res.status === 401) {
          chrome.runtime.sendMessage({ type: "AUTH_CLEAR" });
          ktvAudioCacheClear();
          state.authRequired = true;
          state.authenticated = false;
          updateUI();
          throw new Error("邀請碼已失效，請重新輸入邀請碼");
        }
        if (res.ok) {
          const buf = await res.arrayBuffer();
          if (buf && buf.byteLength > 8192) {
            return buf;
          }
        }
      } catch (err) {
        if (err && String(err.message || "").includes("邀請碼")) throw err;
        // Retry once, then fall through to chunked Port stream via Service Worker
      } finally {
        clearTimeout(timer);
      }
    }

    // 2) Chunked Port streaming via background service worker (immune to PNA & sendMessage size limits)
    return await new Promise((resolve, reject) => {
      let port;
      try {
        port = chrome.runtime.connect({ name: "AUDIO_STREAM" });
      } catch (err) {
        reject(new Error(`無法連線插件背景服務：${err.message}`));
        return;
      }

      const chunks = [];
      let totalLen = 0;
      let settled = false;

      const timeout = setTimeout(() => {
        if (!settled) {
          settled = true;
          try { port.disconnect(); } catch (_e) {}
          reject(new Error("載入音軌逾時，請重試"));
        }
      }, 45000);

      port.onMessage.addListener((msg) => {
        if (settled) return;
        if (msg.type === "chunk" && msg.b64) {
          const bin = atob(msg.b64);
          const arr = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
          chunks.push(arr);
          totalLen += arr.length;
        } else if (msg.type === "done") {
          settled = true;
          clearTimeout(timeout);
          try { port.disconnect(); } catch (_e) {}
          const merged = new Uint8Array(totalLen);
          let pos = 0;
          for (const c of chunks) {
            merged.set(c, pos);
            pos += c.length;
          }
          resolve(merged.buffer);
        } else if (msg.type === "error") {
          settled = true;
          clearTimeout(timeout);
          try { port.disconnect(); } catch (_e) {}
          if (msg.status === 401) {
            ktvAudioCacheClear();
            state.authRequired = true;
            state.authenticated = false;
            updateUI();
            reject(new Error("邀請碼已失效，請重新輸入邀請碼"));
            return;
          }
          reject(new Error(msg.error || "無法載入分離後嘅音訊"));
        }
      });

      port.onDisconnect.addListener(() => {
        if (!settled) {
          settled = true;
          clearTimeout(timeout);
          reject(new Error("音訊傳送中斷，請重新載入插件再試"));
        }
      });

      port.postMessage({ path });
    });
  }

  function prefetchAudioPath(path) {
    if (typeof path !== "string" || !AUDIO_PATH_RE.test(path)) return;
    if (state.authRequired && !state.authenticated) return;
    if (_audioPrefetchMap.has(path)) return;
    while (_audioPrefetchMap.size >= 2) {
      const oldest = _audioPrefetchMap.keys().next().value;
      _audioPrefetchMap.delete(oldest);
    }
    const p = (async () => {
      const cached = await ktvAudioCacheGet(path);
      if (cached) return cached;
      const fresh = await _downloadAudioRaw(path);
      ktvAudioCachePut(path, fresh.slice(0));
      return fresh;
    })().catch(() => {
      _audioPrefetchMap.delete(path);
      return null;
    });
    _audioPrefetchMap.set(path, p);
  }

  async function fetchAudioArrayBuffer(path) {
    if (typeof path !== "string" || !AUDIO_PATH_RE.test(path)) {
      throw new Error("Invalid audio path");
    }
    // 0A) Consume hover-prefetched buffer if ready or in-flight (returns fresh clone so decodeAudioData never detaches shared copy!)
    if (_audioPrefetchMap.has(path)) {
      const p = _audioPrefetchMap.get(path);
      _audioPrefetchMap.delete(path);
      const preBuf = await p;
      if (preBuf && preBuf.byteLength > 8192) {
        return preBuf.slice(0);
      }
    }
    // 0B) Instant local browser Cache API hit (< 40ms replay for last 12 songs!)
    const cached = await ktvAudioCacheGet(path);
    if (cached) return cached;

    const fresh = await _downloadAudioRaw(path);
    ktvAudioCachePut(path, fresh.slice(0));
    return fresh;
  }

  // Fallback ScriptProcessor pitch shifter (identical algorithm) if AudioWorklet is blocked
  function createFallbackPitchNode(audioCtx) {
    const node = audioCtx.createScriptProcessor(2048, 2, 2);
    const bufSize = 16384;
    const bufMask = bufSize - 1;
    const bufL = new Float32Array(bufSize);
    const bufR = new Float32Array(bufSize);
    let writePos = 0;
    const W = 2048;
    const minD = 64;
    let phase = 0;
    let currentRatio = 1.0;
    let targetSemitones = 0;

    function readHermite(buf, pos) {
      const i1 = Math.floor(pos);
      const frac = pos - i1;
      const y0 = buf[(i1 - 1) & bufMask];
      const y1 = buf[i1 & bufMask];
      const y2 = buf[(i1 + 1) & bufMask];
      const y3 = buf[(i1 + 2) & bufMask];
      const c0 = y1;
      const c1 = 0.5 * (y2 - y0);
      const c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
      const c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
      return ((c3 * frac + c2) * frac + c1) * frac + c0;
    }

    node.onaudioprocess = (ev) => {
      const inL = ev.inputBuffer.getChannelData(0);
      const inR = ev.inputBuffer.numberOfChannels > 1 ? ev.inputBuffer.getChannelData(1) : inL;
      const outL = ev.outputBuffer.getChannelData(0);
      const outR = ev.outputBuffer.getChannelData(1);
      const targetRatio = Math.pow(2, targetSemitones / 12);

      for (let i = 0; i < outL.length; i++) {
        currentRatio += (targetRatio - currentRatio) * 0.003;
        bufL[writePos] = inL[i];
        bufR[writePos] = inR[i];

        const p0 = phase;
        let p1 = p0 + 0.5;
        if (p1 >= 1.0) p1 -= 1.0;

        const w0 = Math.sin(Math.PI * p0);
        const w1 = Math.sin(Math.PI * p1);
        const r0 = writePos - (minD + p0 * W);
        const r1 = writePos - (minD + p1 * W);

        outL[i] = readHermite(bufL, r0) * w0 + readHermite(bufL, r1) * w1;
        outR[i] = readHermite(bufR, r0) * w0 + readHermite(bufR, r1) * w1;

        phase += (1.0 - currentRatio) / W;
        if (phase < 0) phase += 1;
        else if (phase >= 1) phase -= 1;

        writePos = (writePos + 1) & bufMask;
      }
    };

    return {
      node,
      setSemitones(val) {
        targetSemitones = val;
      },
      reset(semi = 0) {
        bufL.fill(0);
        bufR.fill(0);
        writePos = 0;
        phase = 0;
        targetSemitones = semi;
        currentRatio = Math.pow(2, semi / 12);
      },
    };
  }

  class KaraokeAudioEngine {
    constructor() {
      this.ctx = null;
      this.instBuffer = null;
      this.vocBuffer = null;
      this.instSource = null;
      this.vocSource = null;
      this.instGain = null;
      this.vocGain = null;
      this.mixBus = null;
      this.pitchWrapper = null; // { node, setSemitones }
      this.masterGain = null;
      this.pitchRouted = false;

      this.isPlaying = false;
      this.startCtxTime = 0;
      this.startOffset = 0;
      this.playbackRate = 1.0;
      this.vocalVolume = 0.0;
      this.semitones = 0;
      this.masterVolume = 1.0;
      this.loadedKey = null; // `${videoId}_${mode}`
      this._loadedVideoId = null;
      this._loadedMode = null;
      this._vocLoadingKey = null;
      this._loadSeq = 0;
    }

    async ensureContext() {
      if (!this.ctx) {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        this.ctx = new AudioCtx();
        this.instGain = this.ctx.createGain();
        this.vocGain = this.ctx.createGain();
        this.mixBus = this.ctx.createGain();
        this.masterGain = this.ctx.createGain();

        this.instGain.gain.value = 1.0;
        this.vocGain.gain.value = this.vocalVolume;
        this.masterGain.gain.value = this.masterVolume;

        this.instGain.connect(this.mixBus);
        this.vocGain.connect(this.mixBus);

        // Try loading AudioWorklet pitch processor with a 1.5s safety timeout (YouTube CSP/TrustedTypes can stall addModule)
        try {
          const workletUrl = chrome.runtime.getURL("pitch-worklet.js");
          await Promise.race([
            this.ctx.audioWorklet.addModule(workletUrl),
            new Promise((_, rej) => setTimeout(() => rej(new Error("AudioWorklet timeout")), 1500)),
          ]);
          const workletNode = new AudioWorkletNode(this.ctx, "karaoke-pitch-processor", {
            numberOfInputs: 1,
            numberOfOutputs: 1,
            outputChannelCount: [2],
          });
          this.pitchWrapper = {
            node: workletNode,
            setSemitones: (val) => {
              const p = workletNode.parameters.get("semitones");
              if (p) p.setValueAtTime(val, this.ctx.currentTime);
            },
            reset: (semi = 0) => {
              try {
                workletNode.port.postMessage({ type: "reset", semitones: semi });
              } catch (_e) {}
            },
          };
        } catch (err) {
          console.warn("[YT AI KTV] AudioWorklet fallback engaged:", err);
          this.pitchWrapper = createFallbackPitchNode(this.ctx);
        }

        // Initial routing: direct bypass for original key
        this.mixBus.connect(this.masterGain);
        this.masterGain.connect(this.ctx.destination);
        this.pitchRouted = false;
      }

      // Never hang indefinitely if AudioContext.resume() waits for a user gesture
      if (this.ctx.state === "suspended") {
        await Promise.race([
          this.ctx.resume().catch(() => {}),
          new Promise((res) => setTimeout(res, 400)),
        ]);
      }
    }

    _effectiveSemitones() {
      // Automatically preserve musical pitch when user slows/speeds YouTube playbackRate (e.g. 0.75x practice)!
      const rateComp = this.playbackRate && Math.abs(this.playbackRate - 1.0) > 0.005
        ? -12.0 * Math.log2(this.playbackRate)
        : 0.0;
      return Math.max(-12, Math.min(12, this.semitones + rateComp));
    }

    _updateRouting() {
      if (!this.ctx || !this.mixBus || !this.masterGain) return;
      const effSemi = this._effectiveSemitones();
      const needPitch = Math.abs(effSemi) > 0.01 && !!this.pitchWrapper;
      if (this.pitchWrapper) {
        this.pitchWrapper.setSemitones(effSemi);
      }
      if (needPitch === this.pitchRouted) return;

      try {
        this.mixBus.disconnect();
      } catch (_e) {}
      if (this.pitchWrapper) {
        try {
          this.pitchWrapper.node.disconnect();
        } catch (_e) {}
      }

      if (needPitch) {
        if (this.pitchWrapper.reset) this.pitchWrapper.reset(effSemi);
        this.mixBus.connect(this.pitchWrapper.node);
        this.pitchWrapper.node.connect(this.masterGain);
        this.pitchRouted = true;
      } else {
        this.mixBus.connect(this.masterGain);
        this.pitchRouted = false;
      }
    }

    _attachVocalsBuffer(vocBuf, key, seq) {
      if (seq !== this._loadSeq || this.loadedKey !== key) return;
      this.vocBuffer = vocBuf;
      if (this.isPlaying && this.ctx && !this.vocSource) {
        const tStart = this.ctx.currentTime + 0.015;
        const targetOff = this.startOffset + (tStart - this.startCtxTime) * this.playbackRate;
        if (targetOff >= 0 && targetOff < vocBuf.duration - 0.05) {
          const vocSrc = this.ctx.createBufferSource();
          vocSrc.buffer = vocBuf;
          vocSrc.playbackRate.value = this.playbackRate;
          vocSrc.connect(this.vocGain);
          vocSrc.start(tStart, targetOff);
          this.vocSource = vocSrc;
        }
      }
    }

    ensureVocalsLoaded() {
      if (!this.ctx || !this.loadedKey || this.vocBuffer) return;
      if (!this._loadedVideoId || !this._loadedMode) return;
      if (this._vocLoadingKey === this.loadedKey) return;
      const key = this.loadedKey;
      const seq = this._loadSeq;
      const path = `/api/audio/${this._loadedVideoId}/${this._loadedMode}/vocals`;
      this._vocLoadingKey = key;
      fetchAudioArrayBuffer(path)
        .then((raw) => this.ctx.decodeAudioData(raw))
        .then((vocBuf) => {
          if (this._vocLoadingKey === key) this._vocLoadingKey = null;
          this._attachVocalsBuffer(vocBuf, key, seq);
        })
        .catch(() => {
          if (this._vocLoadingKey === key) this._vocLoadingKey = null;
        });
    }

    async loadTracks(videoId, mode, onProgress) {
      const seq = ++this._loadSeq;
      if (onProgress) onProgress("載入伴奏中...", 94);
      await this.ensureContext();
      if (seq !== this._loadSeq) return;
      const key = `${videoId}_${mode}`;
      if (this.loadedKey === key && this.instBuffer && (this.vocBuffer || this.vocalVolume <= 0.001)) {
        if (!this.vocBuffer) this.ensureVocalsLoaded();
        return;
      }

      if (onProgress) onProgress("載入伴奏中...", 95);
      // Pipeline download -> decode per stem independently (start decoding instrumental the instant IT finishes!)
      const instBufP = fetchAudioArrayBuffer(`/api/audio/${videoId}/${mode}/instrumental`).then((raw) =>
        this.ctx.decodeAudioData(raw)
      );
      this._vocLoadingKey = key;
      const vocBufP = fetchAudioArrayBuffer(`/api/audio/${videoId}/${mode}/vocals`).then((raw) =>
        this.ctx.decodeAudioData(raw)
      );

      if (this.vocalVolume > 0.001) {
        try {
          const [instBuf, vocBuf] = await Promise.all([instBufP, vocBufP]);
          if (seq !== this._loadSeq) return;
          this.instBuffer = instBuf;
          this.vocBuffer = vocBuf;
          this.loadedKey = key;
          this._loadedVideoId = videoId;
          this._loadedMode = mode;
        } finally {
          if (this._vocLoadingKey === key) this._vocLoadingKey = null;
        }
        return;
      }

      // Default KTV mode (guide vocal = 0%): start playing IMMEDIATELY as soon as instrumental is decoded!
      // Attach rejection guard BEFORE awaiting instBufP so fast vocals errors (or instBufP throw) never trigger unhandledrejection!
      vocBufP.catch(() => {
        if (this._vocLoadingKey === key) this._vocLoadingKey = null;
      });
      let instBuf;
      try {
        instBuf = await instBufP;
      } catch (err) {
        if (this._vocLoadingKey === key) this._vocLoadingKey = null;
        throw err;
      }
      if (seq !== this._loadSeq) return;
      this.instBuffer = instBuf;
      this.vocBuffer = null;
      this.loadedKey = key;
      this._loadedVideoId = videoId;
      this._loadedMode = mode;

      vocBufP
        .then((vocBuf) => {
          if (this._vocLoadingKey === key) this._vocLoadingKey = null;
          this._attachVocalsBuffer(vocBuf, key, seq);
        })
        .catch(() => {
          if (this._vocLoadingKey === key) this._vocLoadingKey = null;
        });
    }

    stopSources() {
      if (this.instSource) {
        try {
          this.instSource.onended = null;
          this.instSource.stop();
          this.instSource.disconnect();
        } catch (_e) {}
        this.instSource = null;
      }
      if (this.vocSource) {
        try {
          this.vocSource.onended = null;
          this.vocSource.stop();
          this.vocSource.disconnect();
        } catch (_e) {}
        this.vocSource = null;
      }
      if (this.pitchWrapper && this.pitchWrapper.reset) {
        this.pitchWrapper.reset(this._effectiveSemitones());
      }
      this.isPlaying = false;
    }

    startAt(offsetSec, playbackRate = 1.0) {
      if (!this.ctx || !this.instBuffer) return;
      this.stopSources();

      const maxDur = this.vocBuffer ? Math.min(this.instBuffer.duration, this.vocBuffer.duration) : this.instBuffer.duration;
      const safeOffset = Math.max(0, offsetSec);
      if (safeOffset >= maxDur - 0.05) {
        return;
      }

      if (this.ctx.state === "suspended") {
        this.ctx.resume().catch(() => {});
        // Do NOT mark isPlaying=true while AudioContext is still suspended (prevents 4x/sec restart leak!)
        return;
      }

      this.playbackRate = playbackRate;
      this._updateRouting();

      const instSrc = this.ctx.createBufferSource();
      instSrc.buffer = this.instBuffer;
      instSrc.playbackRate.value = playbackRate;
      instSrc.connect(this.instGain);

      let vocSrc = null;
      if (this.vocBuffer) {
        vocSrc = this.ctx.createBufferSource();
        vocSrc.buffer = this.vocBuffer;
        vocSrc.playbackRate.value = playbackRate;
        vocSrc.connect(this.vocGain);
      }

      const now = this.ctx.currentTime + 0.01;
      instSrc.onended = () => {
        if (this.instSource === instSrc) {
          this.isPlaying = false;
        }
      };
      instSrc.start(now, safeOffset);
      if (vocSrc) vocSrc.start(now, safeOffset);

      this.instSource = instSrc;
      this.vocSource = vocSrc;
      this.startCtxTime = now;
      this.startOffset = safeOffset;
      this.isPlaying = true;
    }

    getCurrentPosition() {
      if (!this.isPlaying || !this.ctx) return this.startOffset;
      return this.startOffset + (this.ctx.currentTime - this.startCtxTime) * this.playbackRate;
    }

    setVocalVolume(level) {
      this.vocalVolume = Math.max(0, Math.min(1, level));
      if (this.vocGain && this.ctx) {
        this.vocGain.gain.setTargetAtTime(this.vocalVolume, this.ctx.currentTime, 0.02);
      }
      if (this.vocalVolume > 0.001 && this.loadedKey && !this.vocBuffer) {
        this.ensureVocalsLoaded();
      }
    }

    setMasterVolume(vol) {
      this.masterVolume = Math.max(0, Math.min(1.5, vol));
      if (this.masterGain && this.ctx) {
        this.masterGain.gain.setTargetAtTime(this.masterVolume, this.ctx.currentTime, 0.02);
      }
    }

    setSemitones(semi) {
      this.semitones = Math.max(-6, Math.min(6, Math.round(semi)));
      if (this.pitchWrapper) {
        this.pitchWrapper.setSemitones(this.semitones);
      }
      this._updateRouting();
    }

    clearBuffers() {
      this._loadSeq++;
      this.stopSources();
      this.instBuffer = null;
      this.vocBuffer = null;
      this.loadedKey = null;
      this._loadedVideoId = null;
      this._loadedMode = null;
      this._vocLoadingKey = null;
    }
  }

  // Global controller state
  const engine = new KaraokeAudioEngine();
  const state = {
    videoId: null,
    mode: "pure", // "pure" | "karaoke"
    active: false,
    busy: false,
    backendOnline: false,
    authRequired: false,
    authenticated: true,
    gpuOnline: true,
    gpuOfflineMsg: "AI 伺服器休息中，現有歌庫仍可免費唱",
    deviceMode: "gpu",
    gpuName: "",
    cachedModes: [],
    cachedVideoIds: new Set(),
    catalogLoaded: false,
    pollTimer: null,
    syncTimer: null,
    attachedVideo: null,
    videoListeners: [],
    prevVideoMuted: false,
    // On-demand KTV Lyrics Overlay (default OFF per song)
    lyricsEnabled: false,
    lyricsLoading: false,
    lyricsLines: [],
    lyricsSourceLabel: "",
    lyricsOffsetSec: 0,
    lyricsPollTimer: null,
    lyricsRenderTimer: null,
    lastLyricLineIdx: -999,
  };

  let ui = null;
  let lyricsOverlay = null;

  function getYouTubeVideoId() {
    if (!location.pathname.startsWith("/watch")) return null;
    const params = new URLSearchParams(location.search);
    const v = params.get("v");
    return v && VIDEO_ID_RE.test(v) ? v : null;
  }

  function extractVideoIdFromHref(href) {
    if (!href || typeof href !== "string") return null;
    try {
      const u = new URL(href, location.origin);
      if (u.pathname === "/watch") {
        const v = u.searchParams.get("v");
        return v && VIDEO_ID_RE.test(v) ? v : null;
      }
      if (u.pathname.startsWith("/shorts/")) {
        const seg = u.pathname.slice(8).split("/")[0];
        return seg && VIDEO_ID_RE.test(seg) ? seg : null;
      }
    } catch (_e) {}
    return null;
  }

  function getVideoElement() {
    return document.querySelector("video.html5-main-video") || document.querySelector("video");
  }

  function isAdPlaying() {
    return !!document.querySelector(".ad-showing, .ad-interrupting");
  }

  function setStatus(text, isError = false, progress = null) {
    if (!ui) return;
    if (progress !== null) {
      ui.progressWrap.classList.add("visible");
      ui.progressFill.style.width = `${Math.max(0, Math.min(100, progress))}%`;
    } else if (!text) {
      ui.progressWrap.classList.remove("visible");
    } else {
      ui.progressWrap.classList.add("visible");
    }
    ui.statusText.textContent = text || "";
    ui.statusText.classList.toggle("error", !!isError);
  }

  function updateUI() {
    if (!ui) return;

    ui.panel.classList.toggle("ktv-active", state.active);
    ui.dot.classList.toggle("online", state.backendOnline);

    const isCurrentModeCached =
      state.cachedModes.includes("pure") || (state.videoId ? state.cachedVideoIds.has(state.videoId) : false);

    const needInvite = state.backendOnline && state.authRequired && !state.authenticated;
    const gpuRestingForNewSong = state.backendOnline && !needInvite && !state.gpuOnline && !isCurrentModeCached && !state.active;

    // Pure visual DB-status indicator (zero text): emerald bolt = already in DB (instant), dashed sparkles = needs AI gen
    if (ui.dbVisual) {
      if (!state.backendOnline || needInvite) {
        ui.dbVisual.className = "ktv-db-visual";
        ui.dbVisual.title = "";
      } else if (isCurrentModeCached) {
        ui.dbVisual.className = "ktv-db-visual in-db";
        ui.dbVisual.title = "歌庫已有伴奏 · 免生成即唱";
        ui.dbVisualZap.style.display = "inline-flex";
        ui.dbVisualSparkles.style.display = "none";
      } else {
        ui.dbVisual.className = "ktv-db-visual needs-gen";
        ui.dbVisual.title = "新歌 · 需 AI 生成伴奏";
        ui.dbVisualZap.style.display = "none";
        ui.dbVisualSparkles.style.display = "inline-flex";
      }
    }

    if (!state.backendOnline) {
      ui.deviceBadge.textContent = "服務未連線";
      ui.deviceBadge.style.display = "inline-block";
    } else if (needInvite) {
      ui.deviceBadge.textContent = "邀請制";
      ui.deviceBadge.style.display = "inline-block";
    } else if (!state.gpuOnline) {
      ui.deviceBadge.textContent = "歌庫在線 · AI 休息中";
      ui.deviceBadge.style.display = "inline-block";
    } else {
      ui.deviceBadge.style.display = "none";
    }

    if (ui.authRow) {
      ui.authRow.classList.toggle("visible", needInvite);
    }

    ui.toggleBtn.style.display = needInvite ? "none" : "inline-flex";
    ui.toggleBtn.disabled = state.busy || !state.backendOnline || gpuRestingForNewSong || needInvite;
    ui.toggleBtn.classList.toggle("active", state.active);
    const showBtnVisual = !state.busy && !state.active && !gpuRestingForNewSong && state.backendOnline && !needInvite;
    ui.toggleBtnZap.style.display = showBtnVisual && isCurrentModeCached ? "inline-flex" : "none";
    ui.toggleBtnSparkles.style.display = showBtnVisual && !isCurrentModeCached ? "inline-flex" : "none";
    if (state.busy) {
      ui.toggleBtnLabel.textContent = "準備伴奏中...";
    } else if (state.active) {
      ui.toggleBtnLabel.textContent = "關閉唱K";
    } else if (gpuRestingForNewSong) {
      ui.toggleBtnLabel.textContent = "AI 休息中";
    } else {
      ui.toggleBtnLabel.textContent = "開始唱K";
    }

    ui.mixerRow.classList.toggle("visible", state.active);

    // Pitch display
    const s = engine.semitones;
    ui.pitchVal.textContent = s === 0 ? "原調" : s > 0 ? `+${s}` : `${s}`;

    // Vocal volume display & presets
    const pct = Math.round(engine.vocalVolume * 100);
    ui.vocalVal.textContent = `${pct}%`;
    ui.vocalSlider.value = String(pct);
    ui.preset0.classList.toggle("active-preset", pct === 0);
    ui.preset25.classList.toggle("active-preset", pct === 25);
    ui.preset100.classList.toggle("active-preset", pct === 100);

    // Lyrics toggle & fine-tune controls
    ui.lyricsToggleBtn.classList.toggle("active-preset", state.lyricsEnabled);
    if (state.lyricsLoading) {
      ui.lyricsToggleBtn.textContent = "對齊中";
    } else {
      ui.lyricsToggleBtn.textContent = state.lyricsEnabled ? "開" : "關";
    }
    ui.lyricsExtraWrap.style.display = state.lyricsEnabled ? "inline-flex" : "none";
    const lo = state.lyricsOffsetSec;
    ui.lyricsOffsetVal.textContent = lo === 0 ? "0.0s" : lo > 0 ? `+${lo.toFixed(1)}s` : `${lo.toFixed(1)}s`;
    ui.lyricsAiBtn.disabled = state.lyricsLoading;
  }

  function detachVideoListeners() {
    if (state.attachedVideo && state.videoListeners.length) {
      for (const [ev, fn] of state.videoListeners) {
        state.attachedVideo.removeEventListener(ev, fn);
      }
    }
    state.videoListeners = [];
    state.attachedVideo = null;
  }

  function syncAudioNow(forceRestart = false) {
    if (!state.active) return;
    const video = getVideoElement();
    if (!video) return;
    if (video !== state.attachedVideo) {
      attachVideoSync();
      return;
    }

    if (isAdPlaying()) {
      if (engine.isPlaying) engine.stopSources();
      return;
    }

    if (!video.muted) {
      video.muted = true;
    }
    engine.setMasterVolume(video.volume);

    if (video.paused || video.ended || video.readyState < 2) {
      if (engine.isPlaying) engine.stopSources();
      return;
    }

    const targetTime = Math.max(0, video.currentTime);
    const rate = video.playbackRate || 1.0;

    if (!engine.isPlaying || forceRestart || engine.playbackRate !== rate) {
      engine.startAt(targetTime, rate);
      return;
    }

    const drift = Math.abs(engine.getCurrentPosition() - targetTime);
    if (drift > 0.14) {
      engine.startAt(targetTime, rate);
    }
  }

  function attachVideoSync() {
    detachVideoListeners();
    const video = getVideoElement();
    if (!video) return;

    state.prevVideoMuted = video.muted;
    video.muted = true;
    engine.setMasterVolume(video.volume);

    const onPlay = () => syncAudioNow(true);
    const onPause = () => engine.stopSources();
    const onSeeking = () => engine.stopSources();
    const onSeeked = () => syncAudioNow(true);
    const onWaiting = () => engine.stopSources();
    const onPlaying = () => syncAudioNow(true);
    const onRate = () => syncAudioNow(true);
    const onVolume = () => engine.setMasterVolume(video.volume);

    const pairs = [
      ["play", onPlay],
      ["pause", onPause],
      ["seeking", onSeeking],
      ["seeked", onSeeked],
      ["waiting", onWaiting],
      ["playing", onPlaying],
      ["ratechange", onRate],
      ["volumechange", onVolume],
      ["ended", onPause],
    ];

    for (const [ev, fn] of pairs) {
      video.addEventListener(ev, fn);
    }
    state.attachedVideo = video;
    state.videoListeners = pairs;

    if (state.syncTimer) clearInterval(state.syncTimer);
    state.syncTimer = setInterval(() => syncAudioNow(false), 250);

    syncAudioNow(true);
  }

  function deactivateKaraoke(restoreMute = true) {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
    if (state.syncTimer) {
      clearInterval(state.syncTimer);
      state.syncTimer = null;
    }

    engine.stopSources();
    const video = state.attachedVideo || getVideoElement();
    detachVideoListeners();

    if (restoreMute && video && state.active) {
      video.muted = state.prevVideoMuted;
    }

    state.active = false;
    state.busy = false;
    state.lyricsLines = [];
    disableLyricsOverlay();
    updateUI();
  }

  // -------------------- Authentic HK KTV Staggered Dual-Line Sweeping Overlay --------------------
  function makeSweepSlot() {
    const baseEl = el("span", { className: "ktv-layer-base" });
    const fillEl = el("span", { className: "ktv-layer-fill" });
    const wrap = el("div", { className: "ktv-sweep-wrap" }, [baseEl, fillEl]);
    return { wrap, baseEl, fillEl, lastText: "", lastSpeaker: "" };
  }

  function ensureLyricsOverlay() {
    if (lyricsOverlay && lyricsOverlay.root.isConnected) return lyricsOverlay;
    const player = document.querySelector("#movie_player");
    if (!player) return null;

    const statusEl = el("div", { className: "ktv-lyric-status" });
    const cdDots = [
      el("span", { className: "ktv-cd-dot" }),
      el("span", { className: "ktv-cd-dot" }),
      el("span", { className: "ktv-cd-dot" }),
    ];
    const countdownEl = el("div", { className: "ktv-countdown" }, cdDots);

    const topSlot = makeSweepSlot();
    const bottomSlot = makeSweepSlot();

    const topRow = el("div", { className: "ktv-stage-row row-top" }, [countdownEl, topSlot.wrap]);
    const bottomRow = el("div", { className: "ktv-stage-row row-bottom" }, [bottomSlot.wrap]);
    const root = el("div", { id: "yt-ai-ktv-lyrics-overlay" }, [statusEl, topRow, bottomRow]);
    player.appendChild(root);

    lyricsOverlay = { root, statusEl, countdownEl, cdDots, topSlot, bottomSlot };
    return lyricsOverlay;
  }

  function stopLyricsPolling() {
    if (state.lyricsPollTimer) {
      clearInterval(state.lyricsPollTimer);
      state.lyricsPollTimer = null;
    }
  }

  function stopLyricsRenderLoop() {
    if (state.lyricsRenderTimer) {
      cancelAnimationFrame(state.lyricsRenderTimer);
      state.lyricsRenderTimer = null;
    }
  }

  function startLyricsRenderLoop() {
    stopLyricsRenderLoop();
    const step = () => {
      if (!state.lyricsEnabled) {
        state.lyricsRenderTimer = null;
        return;
      }
      renderLyricsFrame();
      state.lyricsRenderTimer = requestAnimationFrame(step);
    };
    state.lyricsRenderTimer = requestAnimationFrame(step);
  }

  function disableLyricsOverlay() {
    stopLyricsPolling();
    stopLyricsRenderLoop();
    state.lyricsEnabled = false;
    state.lyricsLoading = false;
    state.lastLyricLineIdx = -999;
    if (lyricsOverlay) {
      lyricsOverlay.root.classList.remove("visible");
    }
  }

  // Real HK KTV line-breaking standard (NO font shrinking ever!):
  // Uses Chromium's native Intl.Segmenter (zh-Hant / ja / ko) + singer breath pauses
  // so CJK/Japanese/Korean words are NEVER torn in half and every line keeps 100% uniform KTV font size!
  const _wordSegmenters =
    typeof Intl !== "undefined" && typeof Intl.Segmenter === "function"
      ? {
          zh: new Intl.Segmenter("zh-Hant", { granularity: "word" }),
          ja: new Intl.Segmenter("ja", { granularity: "word" }),
          ko: new Intl.Segmenter("ko", { granularity: "word" }),
        }
      : null;

  function normalizeKtvLines(rawLines, maxCjkUnits = 10.5) {
    if (!Array.isArray(rawLines) || rawLines.length === 0) return [];
    const out = [];

    const nonSpaceLen = (s) => (s || "").replace(/\s+/g, "").length;
    const visualUnits = (s) => {
      let u = 0;
      for (const ch of s || "") {
        if (/\s/.test(ch)) u += 0.35;
        else if (/[\u3000-\u9fff\uac00-\ud7af\uff00-\uffef]/.test(ch)) u += 1.0;
        else u += 0.56;
      }
      return u;
    };
    const isCjkText = (s) => {
      const cjk = (s.match(/[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]/g) || []).length;
      return cjk >= 3 || (nonSpaceLen(s) > 0 && cjk / nonSpaceLen(s) >= 0.35);
    };

    function buildCharWordMap(words) {
      const map = [];
      if (!Array.isArray(words)) return map;
      for (let wi = 0; wi < words.length; wi++) {
        const w = words[wi];
        const clean = (w.word || "").replace(/\s+/g, "");
        for (let k = 0; k < clean.length; k++) {
          map.push({ wi, w });
        }
      }
      return map;
    }

    function getValidCjkWordCuts(piece) {
      const cuts = new Set();
      if (_wordSegmenters) {
        let segInst = _wordSegmenters.zh;
        if (/[\u3040-\u30ff]/.test(piece)) segInst = _wordSegmenters.ja;
        else if (/[\uac00-\ud7af]/.test(piece)) segInst = _wordSegmenters.ko;
        for (const seg of segInst.segment(piece)) {
          const endIdx = seg.index + seg.segment.length;
          if (endIdx > 0 && endIdx < piece.length) cuts.add(endIdx);
        }
      }
      return cuts;
    }

    function splitLongCjkPiece(piece, charOffset, charMap, maxUnits) {
      if (visualUnits(piece) <= maxUnits || piece.length <= 4) return [piece];
      const len = piece.length;
      const partsNeeded = Math.max(2, Math.ceil(visualUnits(piece) / (maxUnits - 0.5)));
      const target = Math.round(len / partsNeeded);
      const minCut = Math.max(3, Math.floor(target * 0.58));
      const maxCut = Math.min(len - 3, Math.ceil(target * 1.42));
      const wordCuts = getValidCjkWordCuts(piece);

      let bestCut = Math.max(minCut, Math.min(maxCut, target));
      let bestScore = -1e9;

      for (let c = minCut; c <= maxCut; c++) {
        const headText = piece.slice(0, c);
        // Never allow a head chunk that still exceeds maxUnits if a shorter valid cut exists
        const overPenalty = visualUnits(headText) > maxUnits ? -3.0 : 0.0;
        // Strong preference for real Chinese word/idiom boundaries from Intl.Segmenter
        const isGrammaticalCut = wordCuts.size === 0 || wordCuts.has(c);
        const grammarBonus = isGrammaticalCut ? 2.5 : -2.5;

        // Breath pause bonus from Whisper word timestamps
        let gapBonus = 0;
        const idxA = charOffset + c - 1;
        const idxB = charOffset + c;
        if (idxA >= 0 && idxB < charMap.length && charMap[idxA].wi !== charMap[idxB].wi) {
          const gap = Math.max(0, (charMap[idxB].w.start || 0) - (charMap[idxA].w.end || 0));
          gapBonus = Math.min(2.0, gap * 2.2);
        }

        // Balance bonus: stay close to balanced split and avoid tiny 2-char tails
        const tailLen = len - c;
        const orphanPenalty = (c <= 2 || tailLen <= 2) ? -2.0 : 0.0;
        const balancePenalty = -Math.abs(c - target) * 0.28;

        const score = overPenalty + grammarBonus + gapBonus + orphanPenalty + balancePenalty;
        if (score > bestScore) {
          bestScore = score;
          bestCut = c;
        }
      }

      const head = piece.slice(0, bestCut).trim();
      const tail = piece.slice(bestCut).trim();
      return [head, ...splitLongCjkPiece(tail, charOffset + bestCut, charMap, maxUnits)];
    }

    const stripQuestionMarks = (s) =>
      String(s || "")
        .replace(/(?<=[^\s])[?？]+(?=[^\s])/g, " ")
        .replace(/[?？]+/g, "")
        .replace(/\s+/g, " ")
        .trim();

    // Backward-compatibility fallback for older cached duet lyrics that still have raw '男：'/'女：'/'合：' or '（男）'/'（女）'/'（合）' prefixes
    const hasExplicitSpeakerMeta = rawLines.some((l) => l && typeof l.speaker === "string" && l.speaker);
    const LEGACY_SPK_RE = /^\s*(?:[\(（]?\s*(男聲?|女聲?|合唱?|齊|男女合?|Male|Female|All|Both|Chorus)\s*[:：]|[\(（\[【]\s*(男聲?|女聲?|合唱?|齊|男女合?|Male|Female|All|Both|Chorus)\s*[\)）\]】])\s*/i;
    const legacySpkRole = (lbl) => {
      const k = String(lbl || "").trim().toLowerCase();
      if (k.startsWith("男") && !k.includes("女")) return "male";
      if (k === "male") return "male";
      if (k.startsWith("女") && !k.includes("男")) return "female";
      if (k === "female") return "female";
      return "all";
    };
    const enableLegacySpkClean =
      !hasExplicitSpeakerMeta &&
      rawLines.filter((l) => l && l.text && LEGACY_SPK_RE.test(String(l.text))).length >= 2;
    let legacyCurSpeaker = "";

    for (const line of rawLines) {
      let text = stripQuestionMarks(line && line.text ? String(line.text) : "");
      if (!text) continue;
      const start = Number(line.start) || 0;
      const end = Math.max(start + 0.3, Number(line.end) || start + 3);
      const duetTrack = line && typeof line.duet_track === "string" ? line.duet_track : "";
      let speaker = line && typeof line.speaker === "string" ? line.speaker : "";
      const rawWords = Array.isArray(line.words) ? line.words : [];
      const words = [];
      for (const w of rawWords) {
        const cleanW = stripQuestionMarks(w && w.word ? String(w.word) : "");
        if (!cleanW) continue;
        words.push({ ...w, word: cleanW });
      }
      if (enableLegacySpkClean) {
        const mSp = text.match(LEGACY_SPK_RE);
        if (mSp) {
          const rSp = legacySpkRole(mSp[1] || mSp[2]);
          speaker = rSp;
          if (duetTrack !== "counter") legacyCurSpeaker = rSp;
          const unwrapped = text.replace(/^\s*[\(（]\s*/, "").replace(/\s*[\)）]\s*$/, "");
          text = text.replace(LEGACY_SPK_RE, "").trim() || unwrapped.replace(LEGACY_SPK_RE, "").trim() || text;
          while (words.length > 0 && (/[:：]$/.test(words[0].word || "") || LEGACY_SPK_RE.test(words[0].word || ""))) {
            words.shift();
          }
        } else if (legacyCurSpeaker) {
          speaker = legacyCurSpeaker;
        }
      }
      const cjk = isCjkText(text);
      // Keep simultaneous duet pairs intact on a single KTV row up to 12 CJK units so top/bottom pairing never splits
      const effMaxCjk = duetTrack === "lead" || duetTrack === "counter" ? Math.max(maxCjkUnits, 12.0) : maxCjkUnits;

      let subTexts = [text];
      if (cjk && visualUnits(text) > effMaxCjk) {
        const charMap = buildCharWordMap(words);
        const rawPieces = text
          .split(/[\s,，、；;]+/)
          .map((s) => s.trim())
          .filter(Boolean);

        // Break any overly long spaceless chunk at real Chinese word boundaries + breath pauses
        const atomPieces = [];
        let cursor = 0;
        for (const p of rawPieces) {
          if (visualUnits(p) > effMaxCjk) {
            const broken = splitLongCjkPiece(p, cursor, charMap, effMaxCjk);
            for (const b of broken) atomPieces.push(b);
          } else {
            atomPieces.push(p);
          }
          cursor += p.length;
        }

        // Greedily combine short adjacent phrases ONLY when combined fits cleanly within effMaxCjk
        subTexts = [];
        for (const p of atomPieces) {
          if (subTexts.length > 0 && visualUnits(`${subTexts[subTexts.length - 1]} ${p}`) <= effMaxCjk) {
            subTexts[subTexts.length - 1] = `${subTexts[subTexts.length - 1]} ${p}`;
          } else {
            subTexts.push(p);
          }
        }
      } else if (!cjk && visualUnits(text) > effMaxCjk * 1.45) {
        const toks = text.split(/\s+/).filter(Boolean);
        const enBudget = effMaxCjk * 1.4;
        subTexts = [];
        let cur = "";
        for (const tk of toks) {
          const cand = cur ? `${cur} ${tk}` : tk;
          if (cur && visualUnits(cand) > enBudget) {
            subTexts.push(cur);
            cur = tk;
          } else {
            cur = cand;
          }
        }
        if (cur) subTexts.push(cur);
      }

      if (subTexts.length <= 1) {
        out.push({ start, end, text, words, speaker, duetTrack });
        continue;
      }

      // Slice word timestamps (or proportional timing) cleanly across every split KTV sub-line
      // Compare alphanumeric/CJK content chars only so stripped punctuation in aligner tokens never skews split boundaries!
      const contentLen = (s) => (s || "").replace(/[\s\p{P}\p{S}]+/gu, "").length;
      const totalChars = Math.max(1, subTexts.reduce((acc, s) => acc + Math.max(1, contentLen(s) || nonSpaceLen(s)), 0));
      let wordIdx = 0;
      let elapsedChars = 0;
      let prevSubEnd = start;

      for (let si = 0; si < subTexts.length; si++) {
        const stxt = subTexts[si];
        const need = Math.max(1, contentLen(stxt) || nonSpaceLen(stxt));
        const subWords = [];

        if (words.length > 0 && wordIdx < words.length) {
          let collected = 0;
          while (wordIdx < words.length && (si === subTexts.length - 1 || collected < need)) {
            const w = words[wordIdx];
            const wlen = Math.max(1, contentLen(w.word || "") || nonSpaceLen(w.word || ""));
            if (si < subTexts.length - 1 && subWords.length > 0 && collected + wlen > need) {
              break;
            }
            subWords.push(w);
            collected += wlen;
            wordIdx++;
          }
        }

        let subStart;
        let subEnd;
        if (subWords.length > 0) {
          subStart = Number(subWords[0].start);
          subEnd = Math.max(subStart + 0.25, Number(subWords[subWords.length - 1].end));
        } else {
          subStart = start + ((end - start) * elapsedChars) / totalChars;
          subEnd = start + ((end - start) * (elapsedChars + need)) / totalChars;
        }
        if (si > 0 && subStart < prevSubEnd) {
          subStart = prevSubEnd;
        }
        subEnd = Math.max(subStart + 0.25, subEnd);
        prevSubEnd = subEnd;
        elapsedChars += need;
        out.push({
          start: Math.round(subStart * 100) / 100,
          end: Math.round(subEnd * 100) / 100,
          text: stxt,
          words: subWords,
          speaker,
          duetTrack: "",
        });
      }
    }

    // Annotate authentic HK KTV section boundaries & slot parity:
    // - After any instrumental interlude (>= 4.5s gap), reset to TOP row (slot 0)
    // - Mark the line right before an interlude as isSectionEnd so the other row stays blank during the break
    const KTV_INTERLUDE_GAP_SEC = 4.5;
    for (let i = 0; i < out.length; i++) {
      if (i === 0) {
        out[i].ktvSlot = 0;
        out[i].isSectionStart = true;
      } else {
        const prevEnd =
          i >= 2 && out[i - 1].duetTrack === "counter" && out[i - 2].duetTrack === "lead"
            ? Math.max(out[i - 1].end, out[i - 2].end)
            : out[i - 1].end;
        const gap = out[i].start - prevEnd;
        if (gap >= KTV_INTERLUDE_GAP_SEC) {
          out[i - 1].isSectionEnd = true;
          out[i].isSectionStart = true;
          out[i].ktvSlot = 0;
        } else {
          out[i - 1].isSectionEnd = false;
          out[i].isSectionStart = false;
          out[i].ktvSlot = 1 - out[i - 1].ktvSlot;
        }
      }
      if (i === out.length - 1) {
        out[i].isSectionEnd = true;
      }
    }

    return out;
  }

  function computeLineSweepProgress(line, now) {
    if (!line) return 0;
    if (now <= line.start) return 0;
    if (now >= line.end) return 1;

    // Smooth visual-width-weighted sweep when word-level timestamps exist (space-aware for EN/KO & mixed CJK+EN!)
    if (line.words && line.words.length > 0) {
      const wordVisualWidth = (s) => {
        let u = 0;
        for (const ch of s || "") {
          if (/\s/.test(ch)) u += 0.35;
          else if (/[\u3000-\u9fff\uac00-\ud7af\uff00-\uffef]/.test(ch)) u += 1.0;
          else u += 0.56;
        }
        return Math.max(0.4, u);
      };
      const hasSpacesInText = /\s/.test(line.text || "");
      const needsInterWordSpace = (prevW, nextW) => {
        if (!hasSpacesInText) return false;
        const a = (prevW || "").trim();
        const b = (nextW || "").trim();
        if (!a || !b) return false;
        const aTail = a[a.length - 1];
        const bHead = b[0];
        // English/Korean/mixed-script tokens separated by space in rendered line.text occupy ~0.35 visual units
        const aCjk = /[\u4e00-\u9fff\u3040-\u30ff]/.test(aTail);
        const bCjk = /[\u4e00-\u9fff\u3040-\u30ff]/.test(bHead);
        return !(aCjk && bCjk);
      };
      let totalWeight = 0;
      const weights = [];
      const gapsAfter = [];
      for (let i = 0; i < line.words.length; i++) {
        const wt = wordVisualWidth(line.words[i].word || "");
        const gapW = i + 1 < line.words.length && needsInterWordSpace(line.words[i].word, line.words[i + 1].word) ? 0.35 : 0;
        weights.push(wt);
        gapsAfter.push(gapW);
        totalWeight += wt + gapW;
      }
      let acc = 0;
      for (let i = 0; i < line.words.length; i++) {
        const w = line.words[i];
        const wt = weights[i];
        if (now < w.start) {
          return acc / totalWeight;
        }
        if (now <= w.end) {
          const wDur = Math.max(0.05, w.end - w.start);
          const frac = Math.max(0, Math.min(1, (now - w.start) / wDur));
          return (acc + wt * frac) / totalWeight;
        }
        acc += wt + gapsAfter[i];
      }
      return 1;
    }

    // Smooth linear sweep for line-timed lyrics (YouTube CC / LRCLIB)
    const dur = Math.max(0.2, line.end - line.start);
    return Math.max(0, Math.min(1, (now - line.start) / dur));
  }

  function setSlotContent(slot, text, progress, speaker = "") {
    const spk = text ? String(speaker || "") : "";
    if (slot.lastSpeaker !== spk) {
      if (spk) slot.wrap.dataset.speaker = spk;
      else delete slot.wrap.dataset.speaker;
      slot.lastSpeaker = spk;
    }
    if (!text) {
      if (slot.lastText !== "") {
        slot.baseEl.textContent = "";
        slot.fillEl.textContent = "";
        slot.lastText = "";
      }
      slot.fillEl.style.clipPath = "inset(0 100% 0 0)";
      return;
    }
    if (slot.lastText !== text) {
      slot.baseEl.textContent = text;
      slot.fillEl.textContent = text;
      slot.lastText = text;
    }
    const rightPct = Math.max(0, Math.min(100, (1 - progress) * 100));
    slot.fillEl.style.clipPath = `inset(0 ${rightPct.toFixed(2)}% 0 0)`;
  }

  function renderLyricsFrame() {
    if (!state.lyricsEnabled) return;
    const ov = ensureLyricsOverlay();
    if (!ov) return;

    if (isAdPlaying()) {
      ov.root.classList.remove("visible");
      return;
    }

    if (state.lyricsLoading) {
      ov.root.classList.add("visible");
      setSlotContent(ov.topSlot, "", 0);
      setSlotContent(ov.bottomSlot, "", 0);
      ov.countdownEl.classList.remove("active");
      return;
    }

    const lines = state.lyricsLines;
    if (!lines || lines.length === 0) {
      if (ov.statusEl.textContent) {
        ov.root.classList.add("visible");
      } else {
        ov.root.classList.remove("visible");
      }
      return;
    }

    const video = getVideoElement();
    if (!video) return;
    // Lock lyrics clock to actual WebAudio playback position when karaoke audio is running
    // (eliminates up to 140ms HTML5 video.currentTime quantization jitter!)
    const baseTime = state.active && engine.isPlaying ? engine.getCurrentPosition() : video.currentTime;
    const now = baseTime + state.lyricsOffsetSec;

    // Find current active line OR the upcoming line right before singing begins
    let activeIdx = -1;
    let nextUpcomingIdx = -1;

    for (let i = 0; i < lines.length; i++) {
      // Smooth KTV hold: keep finished line comfortably lit for 0.45s (never overlapping next line's start)
      const nextStart = i + 1 < lines.length ? lines[i + 1].start : lines[i].end + 1.0;
      const holdUntil = Math.min(nextStart - 0.05, lines[i].end + 0.45);
      if (now >= lines[i].start && now <= Math.max(lines[i].end, holdUntil)) {
        activeIdx = i;
        break;
      }
      if (lines[i].start > now) {
        nextUpcomingIdx = i;
        break;
      }
    }

    let topIdx = -1;
    let bottomIdx = -1;
    let countdownSec = -1;

    if (activeIdx !== -1) {
      // Classic HK KTV alternating slots: only pre-load next line when NOT entering an interlude
      const cur = lines[activeIdx];
      const hasImmediateNext = activeIdx + 1 < lines.length && !cur.isSectionEnd;
      if (cur.ktvSlot === 0) {
        topIdx = activeIdx;
        if (hasImmediateNext) bottomIdx = activeIdx + 1;
      } else {
        bottomIdx = activeIdx;
        if (hasImmediateNext) topIdx = activeIdx + 1;
      }
    } else if (nextUpcomingIdx !== -1) {
      const nextLine = lines[nextUpcomingIdx];
      const timeUntil = nextLine.start - now;
      const isAfterBreak = nextUpcomingIdx === 0 || !!nextLine.isSectionStart;
      // After an interlude, stay cleanly hidden until 3.5s before singing resumes (always on TOP row)
      // Mid-verse pauses (<4.5s) keep the upcoming line ready seamlessly with no flicker
      const previewLead = isAfterBreak ? 3.5 : 4.5;

      if (timeUntil <= previewLead) {
        const hasImmediateFollowing = nextUpcomingIdx + 1 < lines.length && !nextLine.isSectionEnd;
        if (nextLine.ktvSlot === 0) {
          topIdx = nextUpcomingIdx;
          if (hasImmediateFollowing) bottomIdx = nextUpcomingIdx + 1;
        } else {
          bottomIdx = nextUpcomingIdx;
          if (hasImmediateFollowing) topIdx = nextUpcomingIdx + 1;
        }
        // Classic 3-2-1 KTV countdown dots before intro (>=3s) or after an instrumental break
        const allowCountdown =
          (nextUpcomingIdx === 0 && nextLine.start >= 3.0) ||
          (nextUpcomingIdx > 0 && !!nextLine.isSectionStart);
        if (allowCountdown && timeUntil <= 3.0) {
          countdownSec = timeUntil;
        }
      } else {
        // Long instrumental interlude — hide overlay cleanly so MV shines
        setSlotContent(ov.topSlot, "", 0);
        setSlotContent(ov.bottomSlot, "", 0);
        ov.root.classList.remove("visible");
        ov.countdownEl.classList.remove("active");
        return;
      }
    } else {
      // Past final line
      setSlotContent(ov.topSlot, "", 0);
      setSlotContent(ov.bottomSlot, "", 0);
      ov.root.classList.remove("visible");
      ov.countdownEl.classList.remove("active");
      return;
    }

    ov.root.classList.add("visible");
    ov.statusEl.textContent = "";

    // Update 3-2-1 KTV intro countdown dots (matches upcoming singer color in HK KTV duets!)
    if (countdownSec > 0) {
      const cdSpk = nextUpcomingIdx >= 0 && lines[nextUpcomingIdx] ? String(lines[nextUpcomingIdx].speaker || "") : "";
      if (cdSpk) ov.countdownEl.dataset.speaker = cdSpk;
      else delete ov.countdownEl.dataset.speaker;
      ov.countdownEl.classList.add("active");
      ov.cdDots[0].classList.toggle("lit", countdownSec <= 3.0);
      ov.cdDots[1].classList.toggle("lit", countdownSec <= 2.0);
      ov.cdDots[2].classList.toggle("lit", countdownSec <= 1.0);
    } else {
      ov.countdownEl.classList.remove("active");
      delete ov.countdownEl.dataset.speaker;
    }

    const topLine = topIdx >= 0 ? lines[topIdx] : null;
    const bottomLine = bottomIdx >= 0 ? lines[bottomIdx] : null;

    setSlotContent(ov.topSlot, topLine ? topLine.text : "", topLine ? computeLineSweepProgress(topLine, now) : 0, topLine ? topLine.speaker : "");
    setSlotContent(ov.bottomSlot, bottomLine ? bottomLine.text : "", bottomLine ? computeLineSweepProgress(bottomLine, now) : 0, bottomLine ? bottomLine.speaker : "");
  }

  let lyricsReqSeq = 0;

  async function requestLyrics(forceAi = false) {
    if (!state.videoId) return;
    const reqSeq = ++lyricsReqSeq;
    const targetVideoId = state.videoId;
    const targetMode = state.mode;

    stopLyricsRenderLoop();
    state.lyricsEnabled = true;
    state.lyricsLoading = true;
    state.lastLyricLineIdx = -999;
    if (forceAi) state.lyricsLines = [];
    updateUI();

    const ov = ensureLyricsOverlay();
    if (ov) {
      ov.root.classList.add("visible");
      ov.statusEl.textContent = forceAi ? "對齊歌詞中..." : "載入歌詞中...";
      setSlotContent(ov.topSlot, "", 0);
      setSlotContent(ov.bottomSlot, "", 0);
      ov.countdownEl.classList.remove("active");
    }

    const res = await apiRequest("/api/lyrics", "POST", {
      video_id: targetVideoId,
      mode: targetMode,
      force_ai: forceAi,
    });
    if (reqSeq !== lyricsReqSeq || state.videoId !== targetVideoId || !state.lyricsEnabled) return;

    if (!res.ok) {
      state.lyricsLoading = false;
      const friendlyErr =
        (res.status === 503 || res.status === 429) && res.data && res.data.detail
          ? res.data.detail
          : "暫未能載入歌詞";
      if (ov) ov.statusEl.textContent = friendlyErr;
      updateUI();
      return;
    }

    if (res.data && res.data.status === "ready" && res.data.lines) {
      state.lyricsLoading = false;
      state.lyricsLines = normalizeKtvLines(res.data.lines);
      state.lyricsSourceLabel = res.data.source_label || "";
      startLyricsRenderLoop();
      updateUI();
      return;
    }

    stopLyricsPolling();
    let inFlight = false;
    state.lyricsPollTimer = setInterval(async () => {
      if (inFlight || reqSeq !== lyricsReqSeq || !state.lyricsEnabled || state.videoId !== targetVideoId) return;
      inFlight = true;
      try {
        const stRes = await apiRequest(`/api/lyrics/${targetVideoId}?mode=${targetMode}`);
        if (reqSeq !== lyricsReqSeq || state.videoId !== targetVideoId || !state.lyricsEnabled) return;
        if (!stRes.ok || !stRes.data) return;
        const d = stRes.data;
        if (d.status === "ready" && d.lines) {
          stopLyricsPolling();
          state.lyricsLoading = false;
          state.lyricsLines = normalizeKtvLines(d.lines);
          state.lyricsSourceLabel = d.source_label || "";
          startLyricsRenderLoop();
          updateUI();
        } else if (d.status === "error" || d.status === "idle") {
          stopLyricsPolling();
          state.lyricsLoading = false;
          if (ov) ov.statusEl.textContent = d.status === "idle" ? "請再按一次載入歌詞" : (d.stage_text || d.error || "無法生成歌詞");
          updateUI();
        } else if (ov) {
          ov.statusEl.textContent = d.stage_text || "載入歌詞中...";
        }
      } finally {
        inFlight = false;
      }
    }, 1000);
  }

  function toggleLyrics() {
    if (state.lyricsEnabled) {
      disableLyricsOverlay();
      updateUI();
      return;
    }
    if (state.lyricsLines && state.lyricsLines.length > 0) {
      state.lyricsEnabled = true;
      state.lastLyricLineIdx = -999;
      startLyricsRenderLoop();
      updateUI();
    } else {
      requestLyrics(false);
    }
  }

  async function activateLoadedTracks(videoId, mode) {
    await engine.loadTracks(videoId, mode, (msg, pct) => {
      if (state.videoId === videoId) setStatus(msg, false, pct);
    });
    if (state.videoId !== videoId || state.mode !== mode) return;
    const vocRequired = engine.vocalVolume > 0.001;
    if (engine.loadedKey !== `${videoId}_${mode}` || !engine.instBuffer || (vocRequired && !engine.vocBuffer)) {
      state.busy = false;
      setStatus("音訊載入未完成，請重試", true, 0);
      updateUI();
      return;
    }
    state.active = true;
    state.busy = false;
    attachVideoSync();
    setStatus("", false, null);
    updateUI();
    sendHeartbeat(false);
  }

  function stopPolling() {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }

  function startPollingJob(targetVideoId, targetMode, autoActivate) {
    stopPolling();
    let inFlight = false;

    const tick = async () => {
      if (inFlight) return;
      if (state.videoId !== targetVideoId || state.mode !== targetMode) {
        stopPolling();
        return;
      }
      inFlight = true;
      try {
        const stRes = await apiRequest(`/api/status/${targetVideoId}?mode=${targetMode}`);
        if (state.videoId !== targetVideoId || state.mode !== targetMode) return;
        if (!stRes.ok || !stRes.data) return;
        const st = stRes.data;
        state.cachedModes = st.cached_modes || state.cachedModes;

        if (st.status === "ready") {
          stopPolling();
          markVideoIdCached(targetVideoId);
          if (autoActivate) {
            try {
              await activateLoadedTracks(targetVideoId, targetMode);
            } catch (e) {
              state.busy = false;
              setStatus(`音訊載入失敗：${e.message}`, true, 0);
              updateUI();
            }
          } else {
            state.busy = false;
            setStatus("", false, null);
            updateUI();
          }
        } else if (st.status === "error") {
          stopPolling();
          state.busy = false;
          setStatus(st.stage_text || st.error || "處理失敗", true, 0);
          updateUI();
        } else if (st.status === "idle") {
          stopPolling();
          state.busy = false;
          setStatus("", false, null);
          updateUI();
        } else {
          setStatus(st.stage_text || "正在處理中...", false, st.progress || 15);
        }
      } finally {
        inFlight = false;
      }
    };

    // Fire immediately, then smoothly every 800ms
    tick();
    state.pollTimer = setInterval(tick, 800);
  }

  async function startKaraokeFlow() {
    if (!state.videoId || state.busy) return;
    const targetVideoId = state.videoId;
    const targetMode = state.mode;

    state.busy = true;
    updateUI();

    try {
      // Ensure AudioContext unlocks inside user click gesture
      await engine.ensureContext();

      setStatus("正在準備歌曲...", false, 5);
      const res = await apiRequest("/api/separate", "POST", {
        video_id: targetVideoId,
        mode: targetMode,
      });
      if (state.videoId !== targetVideoId || state.mode !== targetMode) return;

      if (!res.ok) {
        throw new Error(res.error || (res.data && res.data.detail) || "未能連線伺服器");
      }

      const job = res.data;
      if (job.status === "ready") {
        if (!state.cachedModes.includes(targetMode)) state.cachedModes.push(targetMode);
        markVideoIdCached(targetVideoId);
        await activateLoadedTracks(targetVideoId, targetMode);
        return;
      }

      setStatus(job.stage_text || "正在處理中...", false, job.progress || 10);
      startPollingJob(targetVideoId, targetMode, true);
    } catch (err) {
      if (state.videoId !== targetVideoId || state.mode !== targetMode) return;
      state.busy = false;
      setStatus(`啟動失敗：${err.message}`, true, 0);
      updateUI();
    }
  }

  async function refreshBackendAndVideoStatus() {
    const targetVideoId = state.videoId;
    const targetMode = state.mode;
    const health = await apiRequest("/api/health");
    if (state.videoId !== targetVideoId) return;
    if (health.ok && health.data && health.data.status === "ok") {
      state.backendOnline = true;
      state.authRequired = !!health.data.auth_required;
      state.authenticated = health.data.authenticated !== false;
      if (state.authRequired && !state.authenticated) {
        ktvAudioCacheClear();
        state.cachedModes = [];
        state.cachedVideoIds.clear();
        annotateYouTubeThumbnails();
        setStatus("", false, null);
        updateUI();
        return;
      }
      state.gpuOnline = health.data.gpu_online !== false;
      state.gpuOfflineMsg = health.data.gpu_offline_message || "AI 伺服器休息中，現有歌庫仍可免費唱";
      state.deviceMode = health.data.device_mode || "gpu";
      state.gpuName = (health.data.gpu_name || "").replace("NVIDIA GeForce ", "");
    } else {
      state.backendOnline = false;
      setStatus("唱K 服務未啟動，請稍後再試。", true, null);
      updateUI();
      return;
    }

    if (targetVideoId) {
      const stRes = await apiRequest(`/api/status/${targetVideoId}?mode=${targetMode}`);
      if (state.videoId !== targetVideoId || state.mode !== targetMode) return;
      if (stRes.ok && stRes.data) {
        const st = stRes.data;
        state.cachedModes = st.cached_modes || [];
        if (state.cachedModes.includes("pure")) {
          markVideoIdCached(targetVideoId);
        }
        if (
          st.status === "queued" ||
          st.status === "downloading" ||
          st.status === "separating" ||
          st.status === "finalizing"
        ) {
          state.busy = true;
          setStatus(st.stage_text || "正在處理中...", false, st.progress || 10);
          // Automatically resume live progress polling even after F5 refresh!
          startPollingJob(state.videoId, state.mode, false);
        } else if (!state.busy && !state.gpuOnline && !state.cachedModes.includes("pure")) {
          setStatus(state.gpuOfflineMsg, false, null);
        } else if (!state.busy) {
          setStatus("", false, null);
        }
      }
    }
    updateUI();
    sendHeartbeat(false);
    refreshCatalogAndThumbnails(false);
  }

  function handleModeSelect(newMode) {
    if (state.mode === newMode || state.busy) return;
    const wasActive = state.active;
    state.mode = newMode;
    if (wasActive) {
      deactivateKaraoke(true);
      updateUI();
      startKaraokeFlow();
    } else {
      refreshBackendAndVideoStatus();
    }
  }

  function buildPanel() {
    const dot = el("span", { className: "ktv-dot" });
    const deviceBadge = el("span", { className: "ktv-badge", textContent: "連線中..." });

    // Pure visual DB-status indicator (zero text): emerald bolt = already in DB, dashed sparkles = needs AI gen
    const dbVisualZap = el("span", { className: "ktv-btn-icon", style: "display:none" }, [ICONS.zap(12)]);
    const dbVisualSparkles = el("span", { className: "ktv-btn-icon", style: "display:none" }, [ICONS.sparkles(12)]);
    const dbVisual = el("span", { className: "ktv-db-visual", "aria-label": "歌庫狀態" }, [
      dbVisualZap,
      dbVisualSparkles,
    ]);

    const toggleBtnZap = el("span", { className: "ktv-btn-icon in-db", style: "display:none" }, [ICONS.zap(13)]);
    const toggleBtnSparkles = el("span", { className: "ktv-btn-icon", style: "display:none" }, [ICONS.sparkles(13)]);
    const toggleBtnLabel = el("span", { textContent: "開始唱K" });
    const toggleBtn = el(
      "button",
      {
        className: "ktv-primary-btn",
        type: "button",
        onPointerEnter: () => {
          if (!state.active && !state.busy && state.videoId && state.cachedModes.includes(state.mode)) {
            prefetchAudioPath(`/api/audio/${state.videoId}/${state.mode}/instrumental`);
          }
        },
        onClick: () => {
          if (state.active) {
            deactivateKaraoke(true);
            sendHeartbeat();
          } else {
            startKaraokeFlow();
          }
        },
      },
      [toggleBtnZap, toggleBtnSparkles, toggleBtnLabel]
    );

    const progressFill = el("div", { className: "ktv-progress-bar-fill" });
    const statusText = el("div", { className: "ktv-status-text" });
    const progressWrap = el("div", { className: "ktv-progress-wrap" }, [
      el("div", { className: "ktv-progress-bar-bg" }, [progressFill]),
      statusText,
    ]);

    // Guide vocal slider
    const vocalVal = el("span", { className: "ktv-val", textContent: "0%" });
    const vocalSlider = el("input", {
      className: "ktv-slider",
      type: "range",
      min: "0",
      max: "100",
      step: "1",
      value: "0",
      onInput: (e) => {
        engine.setVocalVolume(Number(e.target.value) / 100);
        updateUI();
      },
    });

    const setVocalPreset = (pct) => {
      engine.setVocalVolume(pct / 100);
      updateUI();
    };
    const preset0 = el("button", {
      className: "ktv-mini-btn active-preset",
      type: "button",
      textContent: "伴奏",
      onClick: () => setVocalPreset(0),
    });
    const preset25 = el("button", {
      className: "ktv-mini-btn",
      type: "button",
      textContent: "導唱",
      onClick: () => setVocalPreset(25),
    });
    const preset100 = el("button", {
      className: "ktv-mini-btn",
      type: "button",
      textContent: "原唱",
      onClick: () => setVocalPreset(100),
    });

    // Pitch shift controls (±6 semitones — click middle value to reset to original key)
    const pitchVal = el("button", {
      className: "ktv-mini-btn val-reset-btn",
      type: "button",
      title: "點擊重設回原調",
      textContent: "原調",
      onClick: () => {
        engine.setSemitones(0);
        updateUI();
      },
    });
    const pitchDown = el("button", {
      className: "ktv-mini-btn",
      type: "button",
      title: "降 1 個半音",
      textContent: "−",
      onClick: () => {
        engine.setSemitones(engine.semitones - 1);
        updateUI();
      },
    });
    const pitchUp = el("button", {
      className: "ktv-mini-btn",
      type: "button",
      title: "升 1 個半音",
      textContent: "+",
      onClick: () => {
        engine.setSemitones(engine.semitones + 1);
        updateUI();
      },
    });

    // On-demand KTV Lyrics toggle + unified timing stepper + AI align
    const lyricsToggleBtn = el("button", {
      className: "ktv-mini-btn pill-standalone",
      type: "button",
      title: "開關同步歌詞",
      textContent: "關",
      onClick: () => toggleLyrics(),
    });
    const lyricsOffsetVal = el("button", {
      className: "ktv-mini-btn val-reset-btn",
      type: "button",
      title: "點擊重設回 0.0s",
      textContent: "0.0s",
      onClick: () => {
        state.lyricsOffsetSec = 0;
        renderLyricsFrame();
        updateUI();
      },
    });
    const adjustLyricsOffset = (delta) => {
      state.lyricsOffsetSec = Math.round((state.lyricsOffsetSec + delta) * 10) / 10;
      renderLyricsFrame();
      updateUI();
    };
    const lyricsAiBtn = el("button", {
      className: "ktv-mini-btn pill-standalone",
      type: "button",
      title: "重新對齊歌詞（適合 Live 或翻唱版）",
      textContent: "AI 對齊",
      onClick: () => requestLyrics(true),
    });
    const lyricsExtraWrap = el("span", { style: "display:none;align-items:center;gap:6px;" }, [
      el("div", { className: "ktv-seg" }, [
        el("button", {
          className: "ktv-mini-btn",
          type: "button",
          textContent: "−",
          onClick: () => adjustLyricsOffset(-0.2),
        }),
        lyricsOffsetVal,
        el("button", {
          className: "ktv-mini-btn",
          type: "button",
          textContent: "+",
          onClick: () => adjustLyricsOffset(0.2),
        }),
      ]),
      lyricsAiBtn,
    ]);

    const mixerRow = el("div", { className: "ktv-mixer-row" }, [
      el("div", { className: "ktv-control-card" }, [
        el("span", { className: "ktv-label" }, [ICONS.sliders(), "導唱"]),
        el("div", { className: "ktv-card-body" }, [
          vocalSlider,
          vocalVal,
          el("div", { className: "ktv-seg" }, [preset0, preset25, preset100]),
        ]),
      ]),
      el("div", { className: "ktv-control-card" }, [
        el("span", { className: "ktv-label" }, [ICONS.music(), "升降 Key"]),
        el("div", { className: "ktv-card-body" }, [
          el("div", { className: "ktv-seg" }, [pitchDown, pitchVal, pitchUp]),
        ]),
      ]),
      el("div", { className: "ktv-control-card" }, [
        el("span", { className: "ktv-label" }, [ICONS.captions(), "KTV 歌詞"]),
        el("div", { className: "ktv-card-body" }, [lyricsToggleBtn, lyricsExtraWrap]),
      ]),
    ]);

    const topRow = el("div", { className: "ktv-row" }, [
      el("div", { className: "ktv-brand" }, [
        dot,
        ICONS.mic(),
        el("span", { textContent: "唱K 去人聲" }),
        dbVisual,
        deviceBadge,
      ]),
      el("div", { className: "ktv-controls-group" }, [toggleBtn]),
    ]);

    // Friends-only invite code activation row (auto-hidden once activated or on localhost)
    const authInput = el("input", {
      className: "ktv-auth-input",
      type: "text",
      placeholder: "輸入邀請碼 (例如 KTV-XXXX-XXXX)",
      autocomplete: "off",
      spellcheck: "false",
    });
    const authMsg = el("span", { className: "ktv-auth-msg", textContent: "只需輸入一次，此電腦將永久記住" });
    const authSubmitBtn = el("button", {
      className: "ktv-primary-btn",
      type: "button",
      textContent: "啟用",
    });
    const submitInvite = async () => {
      const code = (authInput.value || "").trim();
      if (!code) {
        authMsg.textContent = "請先輸入邀請碼";
        authMsg.classList.add("error");
        return;
      }
      authSubmitBtn.disabled = true;
      authSubmitBtn.textContent = "驗證中...";
      authMsg.classList.remove("error");
      authMsg.textContent = "正在驗證邀請碼...";
      const res = await activateInviteCode(code);
      authSubmitBtn.disabled = false;
      authSubmitBtn.textContent = "啟用";
      if (res.ok) {
        authInput.value = "";
        state.authenticated = true;
        authMsg.textContent = "啟用成功！";
        await refreshBackendAndVideoStatus();
      } else {
        authMsg.textContent = res.error || "邀請碼無效或已達上限";
        authMsg.classList.add("error");
      }
    };
    authSubmitBtn.addEventListener("click", submitInvite);
    authInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") submitInvite();
    });
    const authRow = el("div", { className: "ktv-auth-row" }, [authInput, authSubmitBtn, authMsg]);

    const panel = el("div", { id: "yt-ai-ktv-panel" }, [topRow, authRow, progressWrap, mixerRow]);

    ui = {
      panel,
      dot,
      deviceBadge,
      dbVisual,
      dbVisualZap,
      dbVisualSparkles,
      toggleBtn,
      toggleBtnZap,
      toggleBtnSparkles,
      toggleBtnLabel,
      authRow,
      authInput,
      authMsg,
      progressWrap,
      progressFill,
      statusText,
      mixerRow,
      vocalSlider,
      vocalVal,
      preset0,
      preset25,
      preset100,
      pitchVal,
      lyricsToggleBtn,
      lyricsExtraWrap,
      lyricsOffsetVal,
      lyricsAiBtn,
    };

    return panel;
  }

  // -------------------- Pure Visual DB-Cached Thumbnail Badges (All YouTube Pages) --------------------
  let thumbScanTimer = null;

  function markVideoIdCached(vid) {
    if (!vid || !VIDEO_ID_RE.test(vid)) return;
    if (!state.cachedVideoIds.has(vid)) {
      state.cachedVideoIds.add(vid);
      try {
        chrome.runtime.sendMessage({ type: "CATALOG_ADD_ID", videoId: vid });
      } catch (_e) {
        // Ignore background disconnect
      }
    }
    scheduleThumbnailScan();
  }

  function fetchCatalogIdsFromBg(force = false) {
    return new Promise((resolve) => {
      try {
        chrome.runtime.sendMessage({ type: "CATALOG_GET_IDS", force: !!force }, (resp) => {
          if (chrome.runtime.lastError || !resp || !resp.ok || !Array.isArray(resp.ids)) {
            resolve(null);
          } else {
            resolve(resp.ids);
          }
        });
      } catch (_e) {
        resolve(null);
      }
    });
  }

  async function refreshCatalogAndThumbnails(force = false) {
    if (state.authRequired && !state.authenticated) {
      state.cachedVideoIds.clear();
      annotateYouTubeThumbnails();
      return;
    }
    const ids = await fetchCatalogIdsFromBg(force);
    if (ids === null) return;
    const next = new Set();
    for (const id of ids) {
      if (typeof id === "string" && VIDEO_ID_RE.test(id)) {
        next.add(id);
      }
    }
    if (state.videoId && state.cachedModes.includes("pure")) {
      next.add(state.videoId);
    }
    state.cachedVideoIds = next;
    state.catalogLoaded = true;
    updateUI();
    annotateYouTubeThumbnails();
  }

  function annotateYouTubeThumbnails() {
    const anchors = document.querySelectorAll(
      "ytd-thumbnail a#thumbnail[href], a.yt-lockup-view-model__content-image[href], a.yt-lockup-view-model-wiz__content-image[href], ytd-playlist-thumbnail a[href]"
    );
    const hasCache = state.cachedVideoIds.size > 0;
    for (const anchor of anchors) {
      const existing = anchor.querySelector(":scope > .ktv-thumb-db-dot");
      if (!hasCache) {
        if (existing) existing.remove();
        continue;
      }
      const vid = extractVideoIdFromHref(anchor.getAttribute("href"));
      const inDb = !!(vid && state.cachedVideoIds.has(vid));
      if (inDb) {
        if (!existing) {
          if (window.getComputedStyle(anchor).position === "static") {
            anchor.classList.add("ktv-thumb-host");
          }
          const badge = el(
            "span",
            {
              className: "ktv-thumb-db-dot",
              title: "歌庫已有伴奏 · 免生成即唱",
              "aria-hidden": "true",
            },
            [ICONS.zap(12)]
          );
          anchor.appendChild(badge);
        }
      } else if (existing) {
        existing.remove();
      }
    }
  }

  function scheduleThumbnailScan() {
    if (thumbScanTimer) return;
    thumbScanTimer = setTimeout(() => {
      thumbScanTimer = null;
      annotateYouTubeThumbnails();
    }, 260);
  }

  function startThumbnailObserver() {
    if (!document.body || typeof MutationObserver !== "function") return;
    const obs = new MutationObserver((mutations) => {
      if (state.cachedVideoIds.size === 0) return;
      for (const m of mutations) {
        if (m.type === "childList" && m.addedNodes.length > 0) {
          for (const node of m.addedNodes) {
            if (node.nodeType === 1 && !node.classList.contains("ktv-thumb-db-dot")) {
              scheduleThumbnailScan();
              return;
            }
          }
        } else if (m.type === "attributes" && m.attributeName === "href") {
          scheduleThumbnailScan();
          return;
        }
      }
    });
    obs.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["href"],
    });
  }

  function mountPanel() {
    if (!getYouTubeVideoId()) return;

    const existing = document.getElementById("yt-ai-ktv-panel");
    if (existing && existing.isConnected) return;

    const anchor =
      document.querySelector("#below #above-the-fold") ||
      document.querySelector("#below") ||
      document.querySelector("#primary-inner");

    if (!anchor) return;

    const panel = ui ? ui.panel : buildPanel();
    if (anchor.id === "above-the-fold" && anchor.parentElement) {
      anchor.parentElement.insertBefore(panel, anchor);
    } else {
      anchor.insertBefore(panel, anchor.firstChild);
    }
    updateUI();
  }

  function onNavigate() {
    const newVideoId = getYouTubeVideoId();
    if (newVideoId !== state.videoId) {
      deactivateKaraoke(true);
      engine.clearBuffers();
      engine.setSemitones(0);
      engine.setVocalVolume(0);
      state.lyricsOffsetSec = 0;
      state.lyricsLines = [];
      state.lyricsSourceLabel = "";
      disableLyricsOverlay();
      state.cachedModes = [];
      state.videoId = newVideoId;

      if (!newVideoId) {
        const existing = document.getElementById("yt-ai-ktv-panel");
        if (existing) existing.remove();
        refreshCatalogAndThumbnails(false);
        return;
      }
      mountPanel();
      refreshBackendAndVideoStatus();
    } else if (newVideoId) {
      mountPanel();
      scheduleThumbnailScan();
    } else {
      refreshCatalogAndThumbnails(false);
    }
  }

  function sendHeartbeat(leaving = false) {
    if (state.authRequired && !state.authenticated) return;
    if (!state.backendOnline && !leaving) return;
    apiRequest("/api/heartbeat", "POST", {
      client_id: CLIENT_ID,
      video_id: state.videoId,
      singing: state.active,
      leaving,
    });
  }

  // Instant cross-tab sync: if user enters/clears invite code in Popup, unlock/lock open YouTube tabs immediately!
  if (chrome.storage && chrome.storage.onChanged) {
    chrome.storage.onChanged.addListener((changes, area) => {
      if (area === "local" && changes.ktv_access_token) {
        if (!changes.ktv_access_token.newValue) {
          ktvAudioCacheClear();
          state.cachedVideoIds.clear();
          annotateYouTubeThumbnails();
        }
        if (state.videoId) {
          refreshBackendAndVideoStatus();
        } else {
          refreshCatalogAndThumbnails(true);
        }
      }
    });
  }

  // Listen to YouTube SPA navigation & DOM updates
  window.addEventListener("yt-navigate-finish", () => {
    setTimeout(onNavigate, 200);
  });

  window.addEventListener("beforeunload", () => {
    deactivateKaraoke(true);
    sendHeartbeat(true);
  });

  // Gentle 15s heartbeat so Desktop Tray App displays live active users / singing count
  setInterval(() => {
    if (getYouTubeVideoId()) sendHeartbeat(false);
  }, 15000);

  // Periodic guard for YouTube dynamic layout re-renders
  setInterval(() => {
    const vid = getYouTubeVideoId();
    if (vid !== state.videoId) {
      onNavigate();
    } else if (vid && !document.getElementById("yt-ai-ktv-panel")) {
      mountPanel();
    }
  }, 1500);

  // Initial kick-off (watch panel + visual DB thumbnail indicators across all YouTube pages)
  startThumbnailObserver();
  setTimeout(onNavigate, 600);
})();
