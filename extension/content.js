(() => {
  "use strict";

  const API_BASE = "http://127.0.0.1:18765";
  const CLIENT_ID = `yt_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;

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
    clock: () => svgIcon(["M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20Z", "M12 6v6l4 2"], 13),
    captions: () => svgIcon(["M3 5h18a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Z", "M7 15h4", "M15 15h2", "M7 11h2", "M13 11h4"], 13),
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
          resolve(resp || { ok: false, error: "No response" });
        });
      } catch (err) {
        resolve({ ok: false, error: String(err) });
      }
    });
  }

  async function fetchAudioArrayBuffer(path) {
    // 1) Try direct localhost fetch first (fastest, zero serialization)
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 6000);
      const res = await fetch(`${API_BASE}${path}`, { signal: ctrl.signal });
      clearTimeout(timer);
      if (res.ok) {
        return await res.arrayBuffer();
      }
    } catch (_err) {
      // Fall through to chunked Port stream via Service Worker
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
      }, 20000);

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

    _updateRouting() {
      if (!this.ctx || !this.mixBus || !this.masterGain) return;
      const needPitch = this.semitones !== 0 && !!this.pitchWrapper;
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
        this.mixBus.connect(this.pitchWrapper.node);
        this.pitchWrapper.node.connect(this.masterGain);
        this.pitchRouted = true;
      } else {
        this.mixBus.connect(this.masterGain);
        this.pitchRouted = false;
      }
    }

    async loadTracks(videoId, mode, onProgress) {
      if (onProgress) onProgress("正在初始化音訊引擎...", 94);
      await this.ensureContext();
      const key = `${videoId}_${mode}`;
      if (this.loadedKey === key && this.instBuffer && this.vocBuffer) {
        return;
      }

      if (onProgress) onProgress("正在讀取伴奏軌...", 95);
      const instRaw = await fetchAudioArrayBuffer(`/api/audio/${videoId}/${mode}/instrumental`);
      if (onProgress) onProgress("正在讀取導唱人聲軌...", 97);
      const vocRaw = await fetchAudioArrayBuffer(`/api/audio/${videoId}/${mode}/vocals`);

      if (onProgress) onProgress("正在解碼雙軌音訊...", 99);
      const [instBuf, vocBuf] = await Promise.all([
        this.ctx.decodeAudioData(instRaw.slice(0)),
        this.ctx.decodeAudioData(vocRaw.slice(0)),
      ]);

      this.instBuffer = instBuf;
      this.vocBuffer = vocBuf;
      this.loadedKey = key;
    }

    stopSources() {
      if (this.instSource) {
        try {
          this.instSource.stop();
          this.instSource.disconnect();
        } catch (_e) {}
        this.instSource = null;
      }
      if (this.vocSource) {
        try {
          this.vocSource.stop();
          this.vocSource.disconnect();
        } catch (_e) {}
        this.vocSource = null;
      }
      this.isPlaying = false;
    }

    startAt(offsetSec, playbackRate = 1.0) {
      if (!this.ctx || !this.instBuffer || !this.vocBuffer) return;
      this.stopSources();

      const maxDur = Math.min(this.instBuffer.duration, this.vocBuffer.duration);
      const safeOffset = Math.max(0, offsetSec);
      if (safeOffset >= maxDur - 0.05) {
        return;
      }

      if (this.ctx.state === "suspended") {
        this.ctx.resume().catch(() => {});
      }

      const instSrc = this.ctx.createBufferSource();
      const vocSrc = this.ctx.createBufferSource();
      instSrc.buffer = this.instBuffer;
      vocSrc.buffer = this.vocBuffer;
      instSrc.playbackRate.value = playbackRate;
      vocSrc.playbackRate.value = playbackRate;

      instSrc.connect(this.instGain);
      vocSrc.connect(this.vocGain);

      const now = this.ctx.currentTime + 0.01;
      instSrc.start(now, safeOffset);
      vocSrc.start(now, safeOffset);

      this.instSource = instSrc;
      this.vocSource = vocSrc;
      this.startCtxTime = now;
      this.startOffset = safeOffset;
      this.playbackRate = playbackRate;
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
      this.stopSources();
      this.instBuffer = null;
      this.vocBuffer = null;
      this.loadedKey = null;
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
    deviceMode: "gpu",
    gpuName: "",
    cachedModes: [],
    syncOffsetMs: 0,
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
    return params.get("v");
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

    if (!state.backendOnline) {
      ui.deviceBadge.textContent = "服務未連線";
      ui.deviceBadge.className = "ktv-badge";
      ui.deviceBadge.style.display = "inline-block";
    } else {
      ui.deviceBadge.style.display = "none";
    }

    const isCurrentModeCached = state.cachedModes.includes(state.mode);
    ui.cacheBadge.style.display = isCurrentModeCached ? "inline-block" : "none";

    ui.modePureBtn.classList.toggle("selected", state.mode === "pure");
    ui.modeKaraokeBtn.classList.toggle("selected", state.mode === "karaoke");

    ui.toggleBtn.disabled = state.busy || !state.backendOnline;
    ui.toggleBtn.classList.toggle("active", state.active);
    if (state.busy) {
      ui.toggleBtn.textContent = "準備伴奏中...";
    } else if (state.active) {
      ui.toggleBtn.textContent = "關閉唱K";
    } else if (isCurrentModeCached) {
      ui.toggleBtn.textContent = "立即開唱";
    } else {
      ui.toggleBtn.textContent = "開始唱K";
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

    // Sync offset display
    ui.syncVal.textContent =
      state.syncOffsetMs === 0 ? "0ms" : state.syncOffsetMs > 0 ? `+${state.syncOffsetMs}ms` : `${state.syncOffsetMs}ms`;

    // Lyrics toggle & fine-tune controls
    ui.lyricsToggleBtn.classList.toggle("active-preset", state.lyricsEnabled);
    if (state.lyricsLoading) {
      ui.lyricsToggleBtn.textContent = "對齊中...";
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

    const targetTime = Math.max(0, video.currentTime + state.syncOffsetMs / 1000);
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
    disableLyricsOverlay();
    updateUI();
  }

  // -------------------- Authentic HK KTV Staggered Dual-Line Sweeping Overlay --------------------
  function makeSweepSlot() {
    const baseEl = el("span", { className: "ktv-layer-base" });
    const fillEl = el("span", { className: "ktv-layer-fill" });
    const wrap = el("div", { className: "ktv-sweep-wrap" }, [baseEl, fillEl]);
    return { wrap, baseEl, fillEl, lastText: "" };
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

  function computeLineSweepProgress(line, now) {
    if (!line) return 0;
    if (now <= line.start) return 0;
    if (now >= line.end) return 1;

    // Smooth word-weighted sweep when word-level timestamps exist
    if (line.words && line.words.length > 0) {
      let totalWeight = 0;
      const weights = [];
      for (const w of line.words) {
        const len = Math.max(1, (w.word || "").length);
        weights.push(len);
        totalWeight += len;
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
        acc += wt;
      }
      return 1;
    }

    // Smooth linear sweep for line-timed lyrics (YouTube CC / LRCLIB)
    const dur = Math.max(0.2, line.end - line.start);
    return Math.max(0, Math.min(1, (now - line.start) / dur));
  }

  function setSlotContent(slot, text, progress) {
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
      ov.root.classList.remove("visible");
      return;
    }

    const video = getVideoElement();
    if (!video) return;
    const now = video.currentTime + state.lyricsOffsetSec;

    // Find current active line OR the upcoming line right before singing begins
    let activeIdx = -1;
    let nextUpcomingIdx = -1;

    for (let i = 0; i < lines.length; i++) {
      if (now >= lines[i].start && now <= lines[i].end + 0.15) {
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
      // Classic HK KTV alternating slots: even lines on Top-Left, odd lines on Bottom-Right!
      if (activeIdx % 2 === 0) {
        topIdx = activeIdx;
        if (activeIdx + 1 < lines.length) bottomIdx = activeIdx + 1;
      } else {
        bottomIdx = activeIdx;
        if (activeIdx + 1 < lines.length) topIdx = activeIdx + 1;
      }
    } else if (nextUpcomingIdx !== -1) {
      const timeUntil = lines[nextUpcomingIdx].start - now;
      if (timeUntil <= 4.5) {
        // Preview the upcoming pair cleanly right before the verse enters
        if (nextUpcomingIdx % 2 === 0) {
          topIdx = nextUpcomingIdx;
          if (nextUpcomingIdx + 1 < lines.length) bottomIdx = nextUpcomingIdx + 1;
        } else {
          bottomIdx = nextUpcomingIdx;
          if (nextUpcomingIdx + 1 < lines.length) topIdx = nextUpcomingIdx + 1;
        }
        // Classic 3-2-1 KTV countdown dots before intro or after a long >6s instrumental break
        const prevEnd = nextUpcomingIdx > 0 ? lines[nextUpcomingIdx - 1].end : -99;
        if (timeUntil <= 3.0 && lines[nextUpcomingIdx].start - prevEnd >= 5.0) {
          countdownSec = timeUntil;
        }
      } else {
        // Long instrumental interlude — hide overlay cleanly so MV shines
        ov.root.classList.remove("visible");
        ov.countdownEl.classList.remove("active");
        return;
      }
    } else {
      // Past final line
      ov.root.classList.remove("visible");
      ov.countdownEl.classList.remove("active");
      return;
    }

    ov.root.classList.add("visible");
    ov.statusEl.textContent = "";

    // Update 3-2-1 KTV intro countdown dots
    if (countdownSec > 0) {
      ov.countdownEl.classList.add("active");
      ov.cdDots[0].classList.toggle("lit", countdownSec <= 3.0);
      ov.cdDots[1].classList.toggle("lit", countdownSec <= 2.0);
      ov.cdDots[2].classList.toggle("lit", countdownSec <= 1.0);
    } else {
      ov.countdownEl.classList.remove("active");
    }

    const topLine = topIdx >= 0 ? lines[topIdx] : null;
    const bottomLine = bottomIdx >= 0 ? lines[bottomIdx] : null;

    setSlotContent(ov.topSlot, topLine ? topLine.text : "", topLine ? computeLineSweepProgress(topLine, now) : 0);
    setSlotContent(ov.bottomSlot, bottomLine ? bottomLine.text : "", bottomLine ? computeLineSweepProgress(bottomLine, now) : 0);
  }

  async function requestLyrics(forceAi = false) {
    if (!state.videoId) return;
    const targetVideoId = state.videoId;
    const targetMode = state.mode;

    state.lyricsEnabled = true;
    state.lyricsLoading = true;
    state.lastLyricLineIdx = -999;
    if (forceAi) state.lyricsLines = [];
    updateUI();

    const ov = ensureLyricsOverlay();
    if (ov) {
      ov.root.classList.add("visible");
      ov.statusEl.textContent = forceAi ? "AI 正在對準純人聲逐字對齊歌詞..." : "正在載入同步歌詞...";
      setSlotContent(ov.topSlot, "", 0);
      setSlotContent(ov.bottomSlot, "", 0);
      ov.countdownEl.classList.remove("active");
    }

    const res = await apiRequest("/api/lyrics", "POST", {
      video_id: targetVideoId,
      mode: targetMode,
      force_ai: forceAi,
    });

    if (!res.ok) {
      state.lyricsLoading = false;
      if (ov) ov.statusEl.textContent = `歌詞載入失敗：${res.error || "未知錯誤"}`;
      updateUI();
      return;
    }

    if (res.data && res.data.status === "ready" && res.data.lines) {
      state.lyricsLoading = false;
      state.lyricsLines = res.data.lines;
      state.lyricsSourceLabel = res.data.source_label || "";
      startLyricsRenderLoop();
      updateUI();
      return;
    }

    stopLyricsPolling();
    let inFlight = false;
    state.lyricsPollTimer = setInterval(async () => {
      if (inFlight || !state.lyricsEnabled || state.videoId !== targetVideoId) return;
      inFlight = true;
      try {
        const stRes = await apiRequest(`/api/lyrics/${targetVideoId}?mode=${targetMode}`);
        if (!stRes.ok || !stRes.data) return;
        const d = stRes.data;
        if (d.status === "ready" && d.lines) {
          stopLyricsPolling();
          state.lyricsLoading = false;
          state.lyricsLines = d.lines;
          state.lyricsSourceLabel = d.source_label || "";
          startLyricsRenderLoop();
          updateUI();
        } else if (d.status === "error") {
          stopLyricsPolling();
          state.lyricsLoading = false;
          if (ov) ov.statusEl.textContent = d.stage_text || d.error || "無法生成歌詞";
          updateUI();
        } else if (ov) {
          ov.statusEl.textContent = d.stage_text || "正在生成歌詞...";
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
    await engine.loadTracks(videoId, mode, (msg, pct) => setStatus(msg, false, pct));
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
        if (!stRes.ok || !stRes.data) return;
        const st = stRes.data;
        state.cachedModes = st.cached_modes || state.cachedModes;

        if (st.status === "ready") {
          stopPolling();
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

      if (!res.ok) {
        throw new Error(res.error || (res.data && res.data.detail) || "未能連接本機 Backend");
      }

      const job = res.data;
      if (job.status === "ready") {
        if (!state.cachedModes.includes(targetMode)) state.cachedModes.push(targetMode);
        await activateLoadedTracks(targetVideoId, targetMode);
        return;
      }

      setStatus(job.stage_text || "正在處理中...", false, job.progress || 10);
      startPollingJob(targetVideoId, targetMode, true);
    } catch (err) {
      state.busy = false;
      setStatus(`啟動失敗：${err.message}`, true, 0);
      updateUI();
    }
  }

  async function refreshBackendAndVideoStatus() {
    const health = await apiRequest("/api/health");
    if (health.ok && health.data && health.data.status === "ok") {
      state.backendOnline = true;
      state.deviceMode = health.data.device_mode || "gpu";
      state.gpuName = (health.data.gpu_name || "").replace("NVIDIA GeForce ", "");
    } else {
      state.backendOnline = false;
      setStatus("唱K 服務未啟動，請稍後再試。", true, null);
      updateUI();
      return;
    }

    if (state.videoId) {
      const stRes = await apiRequest(`/api/status/${state.videoId}?mode=${state.mode}`);
      if (stRes.ok && stRes.data) {
        const st = stRes.data;
        state.cachedModes = st.cached_modes || [];
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
        } else if (!state.busy) {
          setStatus("", false, null);
        }
      }
    }
    updateUI();
    sendHeartbeat(false);
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
    const cacheBadge = el("span", {
      className: "ktv-badge cached",
      textContent: "已就緒",
      style: "display:none",
    });

    const modePureBtn = el("button", {
      className: "ktv-mode-btn selected",
      type: "button",
      title: "完全去除主唱及和音，純伴奏",
      textContent: "純伴奏",
      onClick: () => handleModeSelect("pure"),
    });

    const modeKaraokeBtn = el("button", {
      className: "ktv-mode-btn",
      type: "button",
      title: "去除主唱並保留副歌和音",
      textContent: "保留和音",
      onClick: () => handleModeSelect("karaoke"),
    });

    const toggleBtn = el("button", {
      className: "ktv-primary-btn",
      type: "button",
      textContent: "開始唱K",
      onClick: () => {
        if (state.active) {
          deactivateKaraoke(true);
          sendHeartbeat();
        } else {
          startKaraokeFlow();
        }
      },
    });

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

    // Pitch shift controls (±6 semitones)
    const pitchVal = el("span", { className: "ktv-val", textContent: "原調" });
    const pitchDown = el("button", {
      className: "ktv-mini-btn",
      type: "button",
      textContent: "−1",
      onClick: () => {
        engine.setSemitones(engine.semitones - 1);
        updateUI();
      },
    });
    const pitchReset = el("button", {
      className: "ktv-mini-btn",
      type: "button",
      textContent: "重設",
      onClick: () => {
        engine.setSemitones(0);
        updateUI();
      },
    });
    const pitchUp = el("button", {
      className: "ktv-mini-btn",
      type: "button",
      textContent: "+1",
      onClick: () => {
        engine.setSemitones(engine.semitones + 1);
        updateUI();
      },
    });

    // Fine sync offset controls
    const syncVal = el("span", { className: "ktv-val", textContent: "0ms" });
    const adjustSync = (delta) => {
      state.syncOffsetMs = Math.max(-300, Math.min(300, state.syncOffsetMs + delta));
      syncAudioNow(true);
      updateUI();
    };

    // On-demand KTV Lyrics toggle + timing + force AI align
    const lyricsToggleBtn = el("button", {
      className: "ktv-mini-btn",
      type: "button",
      title: "段片本身無字幕先需要開（自動優先讀字幕，無字幕則用 AI 對準純人聲逐字生成）",
      textContent: "關",
      onClick: () => toggleLyrics(),
    });
    const lyricsOffsetVal = el("span", { className: "ktv-val", textContent: "0.0s" });
    const adjustLyricsOffset = (delta) => {
      state.lyricsOffsetSec = Math.round((state.lyricsOffsetSec + delta) * 10) / 10;
      renderLyricsFrame();
      updateUI();
    };
    const lyricsAiBtn = el("button", {
      className: "ktv-mini-btn",
      type: "button",
      title: "強制用本地 AI 針對呢條片嘅純人聲重新逐字對齊（適合 Live 演唱會版）",
      textContent: "AI 重對齊",
      onClick: () => requestLyrics(true),
    });
    const lyricsExtraWrap = el(
      "span",
      { style: "display:none;align-items:center;gap:6px;" },
      [
        el("button", {
          className: "ktv-mini-btn",
          type: "button",
          textContent: "-0.2s",
          onClick: () => adjustLyricsOffset(-0.2),
        }),
        lyricsOffsetVal,
        el("button", {
          className: "ktv-mini-btn",
          type: "button",
          textContent: "+0.2s",
          onClick: () => adjustLyricsOffset(0.2),
        }),
        lyricsAiBtn,
      ]
    );

    const mixerRow = el("div", { className: "ktv-row ktv-mixer-row" }, [
      el("div", { className: "ktv-control-card" }, [
        el("span", { className: "ktv-label" }, [ICONS.sliders(), "導唱音量"]),
        vocalSlider,
        vocalVal,
        preset0,
        preset25,
        preset100,
      ]),
      el("div", { className: "ktv-control-card" }, [
        el("span", { className: "ktv-label" }, [ICONS.music(), "升降 Key"]),
        pitchDown,
        pitchVal,
        pitchUp,
        pitchReset,
      ]),
      el("div", { className: "ktv-control-card" }, [
        el("span", { className: "ktv-label" }, [ICONS.captions(), "KTV 歌詞"]),
        lyricsToggleBtn,
        lyricsExtraWrap,
      ]),
      el("div", { className: "ktv-control-card", title: "若伴奏與畫面有微細時間差可在此微調" }, [
        el("span", { className: "ktv-label" }, [ICONS.clock(), "同步微調"]),
        el("button", {
          className: "ktv-mini-btn",
          type: "button",
          textContent: "-25ms",
          onClick: () => adjustSync(-25),
        }),
        syncVal,
        el("button", {
          className: "ktv-mini-btn",
          type: "button",
          textContent: "+25ms",
          onClick: () => adjustSync(25),
        }),
      ]),
    ]);

    const topRow = el("div", { className: "ktv-row" }, [
      el("div", { className: "ktv-brand" }, [
        dot,
        ICONS.mic(),
        el("span", { textContent: "唱K 去人聲" }),
        deviceBadge,
        cacheBadge,
      ]),
      el("div", { className: "ktv-controls-group" }, [
        el("div", { className: "ktv-mode-switch" }, [modePureBtn, modeKaraokeBtn]),
        toggleBtn,
      ]),
    ]);

    const panel = el("div", { id: "yt-ai-ktv-panel" }, [topRow, progressWrap, mixerRow]);

    ui = {
      panel,
      dot,
      deviceBadge,
      cacheBadge,
      modePureBtn,
      modeKaraokeBtn,
      toggleBtn,
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
      syncVal,
      lyricsToggleBtn,
      lyricsExtraWrap,
      lyricsOffsetVal,
      lyricsAiBtn,
    };

    return panel;
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
      state.syncOffsetMs = 0;
      state.lyricsOffsetSec = 0;
      state.lyricsLines = [];
      state.lyricsSourceLabel = "";
      disableLyricsOverlay();
      state.cachedModes = [];
      state.videoId = newVideoId;

      if (!newVideoId) {
        const existing = document.getElementById("yt-ai-ktv-panel");
        if (existing) existing.remove();
        return;
      }
      mountPanel();
      refreshBackendAndVideoStatus();
    } else if (newVideoId) {
      mountPanel();
    }
  }

  function sendHeartbeat(leaving = false) {
    if (!state.backendOnline && !leaving) return;
    apiRequest("/api/heartbeat", "POST", {
      client_id: CLIENT_ID,
      video_id: state.videoId,
      singing: state.active,
      leaving,
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

  // Load saved default mode preference
  try {
    chrome.storage.local.get(["defaultMode"], (res) => {
      if (res && (res.defaultMode === "pure" || res.defaultMode === "karaoke")) {
        state.mode = res.defaultMode;
        updateUI();
      }
    });
  } catch (_e) {}

  // Initial kick-off
  setTimeout(onNavigate, 600);
})();
