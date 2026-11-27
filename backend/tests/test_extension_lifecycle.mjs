import fs from "fs";
import path from "path";
import vm from "vm";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1")), "..", "..");

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function waitFor(pred, timeoutMs = 2500, label = "condition") {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    if (pred()) return;
    await sleep(10);
  }
  throw new Error(`Timed out waiting for ${label}`);
}

class FakeClassList {
  constructor(el) {
    this._el = el;
    this._set = new Set();
  }
  _syncFromString(s) {
    this._set = new Set(String(s || "").split(/\s+/).filter(Boolean));
  }
  add(...cls) {
    for (const c of cls) if (c) this._set.add(c);
  }
  remove(...cls) {
    for (const c of cls) this._set.delete(c);
  }
  contains(c) {
    return this._set.has(c);
  }
  toggle(c, force) {
    if (force === undefined) {
      if (this._set.has(c)) {
        this._set.delete(c);
        return false;
      }
      this._set.add(c);
      return true;
    }
    if (force) this._set.add(c);
    else this._set.delete(c);
    return !!force;
  }
}

class FakeElement {
  constructor(tag) {
    this.tagName = String(tag || "div").toUpperCase();
    this.nodeType = 1;
    this.children = [];
    this.parentElement = null;
    this.attributes = {};
    this.dataset = {};
    this.style = {};
    this.listeners = new Map();
    this.classList = new FakeClassList(this);
    this._textContent = "";
    this.disabled = false;
    this.value = "";
    this.isConnected = false;
  }
  get id() {
    return this.attributes.id || "";
  }
  set id(v) {
    this.attributes.id = String(v);
  }
  get className() {
    return Array.from(this.classList._set).join(" ");
  }
  set className(v) {
    this.classList._syncFromString(v);
  }
  get textContent() {
    return this._textContent;
  }
  set textContent(v) {
    this._textContent = String(v ?? "");
  }
  get firstChild() {
    return this.children[0] || null;
  }
  setAttribute(k, v) {
    this.attributes[k] = String(v);
    if (k === "class") this.classList._syncFromString(v);
  }
  getAttribute(k) {
    return this.attributes[k] ?? null;
  }
  _setConnected(flag) {
    this.isConnected = flag;
    for (const ch of this.children) {
      if (ch && ch._setConnected) ch._setConnected(flag);
    }
  }
  appendChild(ch) {
    if (ch && ch.parentElement) ch.remove();
    if (ch) {
      ch.parentElement = this;
      if (ch._setConnected) ch._setConnected(this.isConnected);
    }
    this.children.push(ch);
    return ch;
  }
  insertBefore(ch, ref) {
    if (ch && ch.parentElement) ch.remove();
    const idx = ref ? this.children.indexOf(ref) : -1;
    if (ch) {
      ch.parentElement = this;
      if (ch._setConnected) ch._setConnected(this.isConnected);
    }
    if (idx >= 0) this.children.splice(idx, 0, ch);
    else this.children.push(ch);
    return ch;
  }
  remove() {
    if (this.parentElement) {
      const idx = this.parentElement.children.indexOf(this);
      if (idx >= 0) this.parentElement.children.splice(idx, 1);
      this.parentElement = null;
    }
    this._setConnected(false);
  }
  addEventListener(ev, fn) {
    if (!this.listeners.has(ev)) this.listeners.set(ev, []);
    this.listeners.get(ev).push(fn);
  }
  removeEventListener(ev, fn) {
    const arr = this.listeners.get(ev);
    if (!arr) return;
    const i = arr.indexOf(fn);
    if (i >= 0) arr.splice(i, 1);
  }
  dispatch(ev, payload = {}) {
    const arr = (this.listeners.get(ev) || []).slice();
    for (const fn of arr) fn({ target: this, ...payload });
  }
  findAll(pred, out = []) {
    if (pred(this)) out.push(this);
    for (const ch of this.children) {
      if (ch && ch.findAll) ch.findAll(pred, out);
    }
    return out;
  }
  querySelector(sel) {
    return this.querySelectorAll(sel)[0] || null;
  }
  querySelectorAll(sel) {
    if (sel.startsWith("#")) {
      const id = sel.slice(1);
      return this.findAll((n) => n.id === id);
    }
    if (sel.startsWith(".")) {
      const cls = sel.slice(1);
      return this.findAll((n) => n.classList.contains(cls));
    }
    return [];
  }
}

async function runLifecycleSuite(contentJsPath, label) {
  const code = fs.readFileSync(contentJsPath, "utf8");
  const timers = new Set();
  const intervals = new Set();
  const rafs = new Set();

  // Build simulated YouTube DOM
  const body = new FakeElement("body");
  body._setConnected(true);
  const below = new FakeElement("div");
  below.id = "below";
  const aboveFold = new FakeElement("div");
  aboveFold.id = "above-the-fold";
  below.appendChild(aboveFold);
  const moviePlayer = new FakeElement("div");
  moviePlayer.id = "movie_player";
  const video = new FakeElement("video");
  video.className = "html5-main-video";
  video.currentTime = 12.5;
  video.paused = false;
  video.ended = false;
  video.readyState = 4;
  video.playbackRate = 1.0;
  video.volume = 0.85;
  video.muted = false;
  moviePlayer.appendChild(video);
  body.appendChild(moviePlayer);
  body.appendChild(below);

  const locationObj = {
    origin: "https://www.youtube.com",
    pathname: "/watch",
    search: "?v=y9C3lI2afmA",
  };
  const winListeners = new Map();
  const storageListeners = [];

  // Mock CacheStorage
  const cacheStores = new Map();
  let cacheDeletedNames = [];
  const cachesMock = {
    async open(name) {
      if (!cacheStores.has(name)) cacheStores.set(name, new Map());
      const store = cacheStores.get(name);
      return {
        async match(url) {
          const buf = store.get(String(url));
          if (!buf) return undefined;
          return {
            async arrayBuffer() {
              return buf.slice(0);
            },
          };
        },
        async put(url, resp) {
          const buf = await resp.arrayBuffer();
          store.set(String(url), buf.slice(0));
        },
        async keys() {
          return Array.from(store.keys()).map((u) => ({ url: u }));
        },
        async delete(key) {
          const u = typeof key === "string" ? key : key.url;
          return store.delete(u);
        },
      };
    },
    async delete(name) {
      cacheDeletedNames.push(name);
      return cacheStores.delete(name);
    },
  };

  // Mock Response for Cache API put()
  class FakeResponse {
    constructor(bodyBuf, init = {}) {
      this._buf = bodyBuf instanceof ArrayBuffer ? bodyBuf.slice(0) : new ArrayBuffer(0);
      this.status = init.status || 200;
      this.ok = this.status >= 200 && this.status < 300;
    }
    async arrayBuffer() {
      return this._buf.slice(0);
    }
  }

  // Mock WebAudio AudioContext
  const startedSources = [];
  let ctxCreated = null;
  class FakeAudioContext {
    constructor() {
      this.state = "running";
      this._t0 = Date.now();
      this.destination = { id: "dest" };
      this.audioWorklet = {
        async addModule() {
          throw new Error("no worklet in test");
        },
      };
      ctxCreated = this;
    }
    get currentTime() {
      return (Date.now() - this._t0) / 1000.0;
    }
    async resume() {
      this.state = "running";
    }
    createGain() {
      return {
        gain: { value: 1.0, setTargetAtTime(v) { this.value = v; } },
        connect() {},
        disconnect() {},
      };
    }
    createScriptProcessor() {
      return { connect() {}, disconnect() {}, onaudioprocess: null };
    }
    createBufferSource() {
      const src = {
        buffer: null,
        playbackRate: { value: 1.0 },
        onended: null,
        stopped: false,
        connect() {},
        disconnect() {},
        start(when, offset) {
          startedSources.push({
            tag: this.buffer ? this.buffer._tag : "unknown",
            when,
            offset,
            rate: this.playbackRate.value,
            src,
          });
        },
        stop() {
          this.stopped = true;
        },
      };
      return src;
    }
    async decodeAudioData(ab) {
      const view = new Uint8Array(ab);
      const tag = view[0] === 0x11 ? "instrumental" : view[0] === 0x22 ? "vocals" : "other";
      return { duration: 225.0, _tag: tag };
    }
  }

  // Programmable fetch behavior
  const fetchCounts = new Map();
  let fetchOverrides = {};
  const makeFakeMp3 = (markerByte) => {
    const u8 = new Uint8Array(16384);
    u8[0] = markerByte;
    return u8.buffer;
  };

  async function fakeFetch(url, _opts) {
    const u = String(url);
    const pathOnly = u.replace(/^https?:\/\/[^/]+/, "");
    fetchCounts.set(pathOnly, (fetchCounts.get(pathOnly) || 0) + 1);
    const ov = fetchOverrides[pathOnly];
    if (typeof ov === "function") {
      return await ov(fetchCounts.get(pathOnly));
    }
    const isVoc = pathOnly.endsWith("/vocals");
    await sleep(isVoc ? 140 : 15);
    return new FakeResponse(makeFakeMp3(isVoc ? 0x22 : 0x11), { status: 200 });
  }

  let portShouldFail = false;
  let lastStatusVid = "";
  const chromeMock = {
    runtime: {
      lastError: null,
      getURL: (p) => `chrome-extension://test/${p}`,
      sendMessage(msg, cb) {
        if (!cb) return;
        setTimeout(() => {
          if (msg.type === "AUTH_GET_STATE") {
            cb({ ok: true, token: "tok_test", deviceId: "dev_test" });
          } else if (msg.type === "CATALOG_GET_IDS") {
            cb({ ok: true, ids: ["y9C3lI2afmA", "oA9jBgA_PMU", "Pk-Ymt6D3DY"] });
          } else if (msg.type === "CATALOG_ADD_ID") {
            cb({ ok: true, ids: ["y9C3lI2afmA", "oA9jBgA_PMU", "Pk-Ymt6D3DY"] });
          } else if (msg.type === "API_REQUEST") {
            if (msg.path === "/api/health") {
              cb({ ok: true, status: 200, data: { status: "ok", auth_required: false, authenticated: true, gpu_online: true } });
            } else if (msg.path.startsWith("/api/status/")) {
              lastStatusVid = msg.path.slice("/api/status/".length).split("?")[0];
              cb({ ok: true, status: 200, data: { status: "ready", cached_modes: ["pure"], progress: 100 } });
            } else if (msg.path === "/api/separate") {
              cb({ ok: true, status: 200, data: { status: "ready", progress: 100 } });
            } else if (msg.path === "/api/lyrics") {
              cb({
                ok: true,
                status: 200,
                data: {
                  status: "ready",
                  source_label: "同步歌詞",
                  lines: [
                    { start: 10.0, end: 14.0, text: "最好不見", speaker: "male", words: [{ word: "最", start: 10.0, end: 11.0 }, { word: "好", start: 11.0, end: 12.0 }, { word: "不", start: 12.0, end: 13.0 }, { word: "見", start: 13.0, end: 14.0 }] },
                    { start: 14.1, end: 18.0, text: "最好但求痛快", speaker: "female", words: [{ word: "最", start: 14.1, end: 15.0 }, { word: "好", start: 15.0, end: 18.0 }] },
                  ],
                },
              });
            } else {
              cb({ ok: true, status: 200, data: { status: "ok" } });
            }
          } else {
            cb({ ok: true });
          }
        }, 2);
      },
      connect({ name }) {
        const msgLs = [];
        const discLs = [];
        return {
          name,
          onMessage: { addListener: (fn) => msgLs.push(fn) },
          onDisconnect: { addListener: (fn) => discLs.push(fn) },
          disconnect() {},
          postMessage(_m) {
            setTimeout(() => {
              if (portShouldFail) {
                for (const fn of msgLs) fn({ type: "error", error: "HTTP 502" });
              } else {
                for (const fn of msgLs) fn({ type: "done" });
              }
            }, 5);
          },
        };
      },
    },
    storage: {
      onChanged: {
        addListener(fn) {
          storageListeners.push(fn);
        },
      },
    },
  };

  const docMock = {
    body,
    createElement: (tag) => new FakeElement(tag),
    createElementNS: (_ns, tag) => new FakeElement(tag),
    createTextNode: (t) => ({ nodeType: 3, textContent: String(t) }),
    getElementById: (id) => body.findAll((n) => n.id === id)[0] || null,
    querySelector: (sel) => {
      if (sel === "#below #above-the-fold") return aboveFold;
      if (sel === "#below") return below;
      if (sel === "#movie_player") return moviePlayer;
      if (sel === "video.html5-main-video" || sel === "video") return video;
      if (sel === ".ad-showing, .ad-interrupting") return null;
      return body.querySelector(sel);
    },
    querySelectorAll: () => [],
  };

  const sandbox = {
    console,
    Math,
    Number,
    String,
    Array,
    Set,
    Map,
    RegExp,
    Date,
    Promise,
    Error,
    Uint8Array,
    Float32Array,
    Intl,
    URL,
    URLSearchParams,
    AbortController,
    Response: FakeResponse,
    atob: (s) => Buffer.from(s, "base64").toString("binary"),
    crypto: {
      getRandomValues(arr) {
        for (let i = 0; i < arr.length; i++) arr[i] = (i * 37 + 13) & 255;
        return arr;
      },
    },
    setTimeout(fn, ms) {
      const id = setTimeout(() => {
        timers.delete(id);
        fn();
      }, Math.min(ms, 80));
      timers.add(id);
      return id;
    },
    clearTimeout(id) {
      timers.delete(id);
      clearTimeout(id);
    },
    setInterval(fn, ms) {
      const id = setInterval(fn, Math.max(25, Math.min(ms, 200)));
      intervals.add(id);
      return id;
    },
    clearInterval(id) {
      intervals.delete(id);
      clearInterval(id);
    },
    requestAnimationFrame(fn) {
      const id = setTimeout(() => {
        rafs.delete(id);
        fn();
      }, 16);
      rafs.add(id);
      return id;
    },
    cancelAnimationFrame(id) {
      rafs.delete(id);
      clearTimeout(id);
    },
    document: docMock,
    location: locationObj,
    chrome: chromeMock,
    caches: cachesMock,
    fetch: fakeFetch,
    AudioContext: FakeAudioContext,
    MutationObserver: class { observe() {} disconnect() {} },
    getComputedStyle: () => ({ position: "relative" }),
    addEventListener(ev, fn) {
      if (!winListeners.has(ev)) winListeners.set(ev, []);
      winListeners.get(ev).push(fn);
    },
  };
  sandbox.window = sandbox;

  const triggerNavigate = async (newVid) => {
    lastStatusVid = "";
    locationObj.search = `?v=${newVid}`;
    for (const fn of winListeners.get("yt-navigate-finish") || []) fn();
    await waitFor(() => lastStatusVid === newVid, 2000, `navigate to ${newVid}`);
    await sleep(15);
  };

  try {
    vm.runInNewContext(code, sandbox);

    // Wait for panel mount + initial backend health check
    await waitFor(() => docMock.getElementById("yt-ai-ktv-panel") !== null, 2000, "panel mount");
    const panel = docMock.getElementById("yt-ai-ktv-panel");
    const toggleBtn = panel.findAll((n) => n.classList.contains("ktv-primary-btn"))[0];
    const toggleLabel = toggleBtn.children[toggleBtn.children.length - 1];
    const mixerRow = panel.findAll((n) => n.classList.contains("ktv-mixer-row"))[0];
    const preset25 = panel.findAll((n) => n.tagName === "BUTTON" && n.textContent === "導唱")[0];
    const lyricsBtn = panel.findAll((n) => n.tagName === "BUTTON" && n.attributes.title === "開關同步歌詞")[0];

    await waitFor(() => !toggleBtn.disabled, 2000, "backend online & button enabled");

    // SCENARIO 1: Default 0% guide vocal cold start (exact regression test for 4cc7bf3 infinite 95% hang!)
    startedSources.length = 0;
    toggleBtn.dispatch("click");
    // Must activate & start instrumental IMMEDIATELY (within 100ms, BEFORE 140ms vocals finishes!)
    await waitFor(() => panel.classList.contains("ktv-active"), 100, "S1 instant 0%-vocal activation");
    if (toggleBtn.disabled) throw new Error("S1 FAILED: toggleBtn stayed disabled after activation");
    if (toggleLabel.textContent !== "關閉唱K") throw new Error(`S1 FAILED: expected 關閉唱K, got ${toggleLabel.textContent}`);
    if (!video.muted) throw new Error("S1 FAILED: YouTube video not muted");
    if (!mixerRow.classList.contains("visible")) throw new Error("S1 FAILED: mixer row hidden");
    if (!startedSources.some((s) => s.tag === "instrumental")) throw new Error("S1 FAILED: instrumental did not start");

    // Wait for background vocals decode -> must hot-attach sample-accurately!
    await waitFor(() => startedSources.some((s) => s.tag === "vocals"), 500, "S1 background vocals hot-attach");

    // SCENARIO 7: Enable lyrics overlay & verify HK KTV speaker attribute rendering
    lyricsBtn.dispatch("click");
    await waitFor(() => {
      const ov = docMock.getElementById("yt-ai-ktv-lyrics-overlay");
      return ov && ov.classList.contains("visible") && ov.findAll((n) => n.dataset && n.dataset.speaker === "male").length > 0;
    }, 1000, "S7 lyrics overlay render with male speaker color");

    // Close karaoke -> must restore video mute & stop all sources
    toggleBtn.dispatch("click");
    if (panel.classList.contains("ktv-active")) throw new Error("S1 close FAILED: still active");
    if (video.muted !== false) throw new Error("S1 close FAILED: video mute not restored");
    if (!startedSources.every((s) => s.src.stopped)) throw new Error("S1 close FAILED: audio source leaked running");

    // SCENARIO 2: Background vocals failure (both direct retries + port fail) -> clicking 導唱 25% must auto-retry & hot-attach!
    let vocAttempts = 0;
    fetchOverrides["/api/audio/oA9jBgA_PMU/pure/vocals"] = async () => {
      vocAttempts++;
      if (vocAttempts <= 2) throw new Error("transient Drive 502");
      await sleep(15);
      return new FakeResponse(makeFakeMp3(0x22), { status: 200 });
    };
    portShouldFail = true;
    await triggerNavigate("oA9jBgA_PMU");
    await waitFor(() => !toggleBtn.disabled, 1500, "S2 navigate ready");
    startedSources.length = 0;
    toggleBtn.dispatch("click");
    await waitFor(() => panel.classList.contains("ktv-active"), 500, "S2 activation despite vocals fail");
    await sleep(60); // let 1st vocals fail settle
    if (startedSources.some((s) => s.tag === "vocals")) throw new Error("S2 FAILED: vocals should have failed on attempt 1");
    portShouldFail = false;
    preset25.dispatch("click");
    await waitFor(() => startedSources.some((s) => s.tag === "vocals"), 500, "S2 vocals auto-retry on 導唱 click");

    // SCENARIO 3: Hover prefetch -> click (0 duplicate fetch!) + Cache API instant hit on revisit
    fetchOverrides = {};
    await triggerNavigate("Pk-Ymt6D3DY");
    await waitFor(() => !toggleBtn.disabled, 1500, "S3 navigate ready");
    const pkInstPath = "/api/audio/Pk-Ymt6D3DY/pure/instrumental";
    fetchCounts.delete(pkInstPath);
    toggleBtn.dispatch("pointerenter");
    await sleep(5);
    toggleBtn.dispatch("click");
    await waitFor(() => panel.classList.contains("ktv-active"), 500, "S3 hover-prefetch activation");
    if ((fetchCounts.get(pkInstPath) || 0) !== 1) {
      throw new Error(`S3 FAILED: hover prefetch duplicated fetch (count=${fetchCounts.get(pkInstPath)})`);
    }
    await sleep(20); // allow Cache API put() to commit
    // Navigate away (clears in-memory instBuffer) and back -> must hit Cache API with 0 new network fetch!
    await triggerNavigate("y9C3lI2afmA");
    await triggerNavigate("Pk-Ymt6D3DY");
    await waitFor(() => !toggleBtn.disabled, 1500, "S3 revisit ready");
    toggleBtn.dispatch("click");
    await waitFor(() => panel.classList.contains("ktv-active"), 500, "S3 Cache API replay activation");
    if ((fetchCounts.get(pkInstPath) || 0) !== 1) {
      throw new Error(`S3 FAILED: Cache API replay made unexpected network fetch (count=${fetchCounts.get(pkInstPath)})`);
    }

    // SCENARIO 5: Total network failure -> state.busy MUST reset & button MUST re-enable (never hang!)
    const failPath = "/api/audio/FailVid1234/pure/instrumental";
    fetchOverrides[failPath] = async () => { throw new Error("offline"); };
    portShouldFail = true;
    await triggerNavigate("FailVid1234");
    await waitFor(() => !toggleBtn.disabled, 1500, "S5 navigate ready");
    toggleBtn.dispatch("click");
    await waitFor(() => !toggleBtn.disabled && toggleLabel.textContent === "開始唱K", 1500, "S5 busy reset after error");
    if (panel.classList.contains("ktv-active")) throw new Error("S5 FAILED: became active on failed audio");

    // SCENARIO 6: Cross-tab token clear MUST wipe Browser Cache API!
    cacheDeletedNames = [];
    for (const fn of storageListeners) fn({ ktv_access_token: { oldValue: "tok_test", newValue: "" } }, "local");
    await waitFor(() => cacheDeletedNames.includes("yt-ai-ktv-audio-v1"), 500, "S6 Cache API wipe on logout");

    console.log(`[PASS] Extension runtime lifecycle (${label}): ALL 6 published flows verified (0%-vocal instant play, bg vocals hot-attach & retry, hover-prefetch, Cache API replay, error reset, logout wipe, lyrics)`);
  } finally {
    for (const id of timers) clearTimeout(id);
    for (const id of intervals) clearInterval(id);
    for (const id of rafs) clearTimeout(id);
  }
}

await runLifecycleSuite(path.join(ROOT, "extension", "content.js"), "local-dev extension/content.js");
const prodPath = path.join(ROOT, "extension-prod", "content.js");
if (fs.existsSync(prodPath)) {
  await runLifecycleSuite(prodPath, "chrome-store extension-prod/content.js");
}
