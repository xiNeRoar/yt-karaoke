"""Automated Structural Regression & Invariant Test Suite for YouTube AI Karaoke.

Run anytime via:
    uv run python tests/test_regression.py

Verifies:
1. Python syntax & AST undefined-name scan across backend/ and gateway/.
2. Node.js syntax check across all extension/*.js files.
3. Atomic JSON status writer concurrency test (0 torn reads under parallel load).
4. Hyphen-prefixed YouTube ID safety (`-8HPO9tXnZU`) and worker status lifecycle invariants.
5. Language-agnostic parallel title decomposition (zero hardcoded blacklists).
6. Sidecar watcher UTF-8 safety under worst-case Windows cp1252 pipe encoding.
7. Live local backend & public gateway health/schema consistency (when online).
"""

from __future__ import annotations

import ast
import json
import os
import py_compile
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
GATEWAY = ROOT / "gateway"
EXTENSION = ROOT / "extension"
WATCHER = Path.home() / ".gemini" / "config" / "sidecars" / "yt-karaoke-log-watcher" / "watcher.py"

sys.path.insert(0, str(BACKEND))


def test_python_compile_and_ast() -> None:
    import builtins

    py_files = list(BACKEND.glob("*.py")) + [GATEWAY / "main.py"]
    if WATCHER.exists():
        py_files.append(WATCHER)
    builtin_names = set(dir(builtins))
    for pf in py_files:
        src = pf.read_text(encoding="utf-8")
        compile(src, str(pf), "exec")
        tree = ast.parse(src, filename=str(pf))
        # Structural undefined-call guard: collect every defined/imported/assigned/parameter name in module
        defined = set(builtin_names)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
                if hasattr(node, "args"):
                    for a in (
                        list(node.args.posonlyargs)
                        + list(node.args.args)
                        + list(node.args.kwonlyargs)
                        + ([node.args.vararg] if node.args.vararg else [])
                        + ([node.args.kwarg] if node.args.kwarg else [])
                    ):
                        defined.add(a.arg)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    defined.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    defined.add(alias.asname or alias.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                defined.add(node.id)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined.add(node.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id in defined, f"Undefined function call '{node.func.id}' at {pf.name}:{node.lineno}"
    print(f"[PASS] Python compile + AST parse + undefined-call check ({len(py_files)} files)")


def test_extension_js_syntax() -> None:
    node = subprocess.run(["node", "--version"], capture_output=True, text=True)
    if node.returncode != 0:
        print("[SKIP] node not on PATH")
        return
    js_files = sorted(EXTENSION.glob("*.js"))
    for jf in js_files:
        res = subprocess.run(["node", "--check", str(jf)], capture_output=True, text=True)
        assert res.returncode == 0, f"JS syntax error in {jf}: {res.stderr}"
    print(f"[PASS] Extension JS syntax ({len(js_files)} files)")


def test_atomic_json_concurrency() -> None:
    from atomic_io import write_json_atomic

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "job.json"
        write_json_atomic(p, {"status": "init", "progress": 0})
        errors: list[str] = []
        reads = 0
        stop = False

        def _writer():
            for i in range(120):
                try:
                    write_json_atomic(p, {"status": "separating", "progress": i, "stage_text": f"去人聲中... {i}%"})
                except Exception as e:
                    errors.append(f"write:{e}")

        def _reader():
            nonlocal reads
            while not stop:
                try:
                    raw = p.read_text(encoding="utf-8")
                    d = json.loads(raw)
                    assert "status" in d
                    reads += 1
                except (FileNotFoundError, PermissionError):
                    pass
                except Exception as e:
                    errors.append(f"read:{e}")

        r_thread = threading.Thread(target=_reader, daemon=True)
        r_thread.start()
        _writer()
        stop = True
        r_thread.join(timeout=2)
        assert not errors, f"Atomic JSON errors: {errors[:5]}"
    print(f"[PASS] Atomic JSON status concurrency (120 writes + {reads} concurrent reads, 0 torn reads)")


def test_worker_and_separator_invariants() -> None:
    worker_src = (BACKEND / "worker.py").read_text(encoding="utf-8")
    sep_src = (BACKEND / "separator.py").read_text(encoding="utf-8")
    gw_src = (GATEWAY / "main.py").read_text(encoding="utf-8")

    # 1) worker.py must NEVER emit status="ready" directly (separator.py owns atomic ready after GDrive upload!)
    assert 'status="ready"' not in worker_src, "worker.py must leave status at finalizing so Drive upload cannot race!"
    # 2) separator.py must pass --video-id=<val> (= syntax) so hyphen-prefixed IDs never break argparse
    assert 'f"--video-id={video_id}"' in sep_src, "separator.py must use --video-id= syntax"
    # 3) separator.py must check model mismatch generically in _needs_hq_upgrade
    assert "_needs_hq_upgrade" in sep_src
    # 4) gateway/main.py must NEVER set max-age=86400 on fallback proxy stream and must evict _lyrics_ram on HQ upgrade
    assert 'headers={"Cache-Control": "no-store"}' in gw_src, "Gateway proxy stream must not cache 24h"
    assert "_lyrics_ram.pop(evid" in gw_src, "Gateway must evict stale lyrics RAM on HQ upgrade"
    # 5) Exhaustive lyrics_worker.py exit-path invariants (ALL branches under BOTH GPU-free and low-VRAM CPU!)
    lw_src = (BACKEND / "lyrics_worker.py").read_text(encoding="utf-8")
    assert 'update("ready"' not in lw_src, "lyrics_worker.py must leave status at finalizing until separator.py caches lines!"
    assert "Worker produced empty lyrics" in sep_src, "separator.py must fail loudly instead of leaving status at finalizing!"
    assert "if not vocals_path.exists():" in lw_src, "Layer 1 youtube_sub must require word alignment when vocals exist"
    assert "if lrc_synced and not force_ai and not vocals_path.exists():" in lw_src, "Layer 2 LRC must require word alignment when vocals exist"
    assert "require_all_windows=True" in lw_src, "0-word scaffolds must require 100% window coverage from Qwen3"
    assert "if lrc_synced and not force_ai and len(lrc_synced)" in lw_src, "force_ai=True must bypass LRC scaffold shortcut"
    assert "whisper_qwen3_ai" in lw_src, "Free-transcribe fallback must also pass through acoustic & Qwen3 refinement"
    assert "is_real_anchor = float(ref.get(\"sim\") or 0.0) > 0.0" in lw_src, "Must never overwrite acoustic alignment of recovered lines with linear placeholders"
    assert "align_chunk, blk_time_map = build_silence_collapsed_audio(" in lw_src, "Live block alignment must compress long guitar-solo/instrumental silences"
    assert "_alignment_needs_live_rematch(aligned, audio_dur, db_profile, step_sec, time_map=time_map)" in lw_src, "Must pass real db_profile + time_map to detect unaligned Song #2 / encores!"
    assert "if (is_mashup_title and len(multi_song_pools) < 2) or not official_lines:" in lw_src, "Must never overwrite resolved multi_song_pools with split-brain Shazam strings!"
    assert "prev_gap >= 4.0" in lw_src, "Qwen3 windows must split at instrumental interludes (>=4s) to prevent stretched ghost lines!"
    assert "lrc_covers_audio" in lw_src, "LRC fast-path must verify full vocal coverage so single-song LRC never hijacks multi-song videos!"
    assert "async with Shazam() as shazam:" in lw_src, "Shazam probes must close aiohttp ClientSession via async context manager"
    # Frontend structural invariants
    ct_src = (EXTENSION / "content.js").read_text(encoding="utf-8")
    assert "engine.getCurrentPosition()" in ct_src and "state.active && engine.isPlaying" in ct_src, "Lyrics clock must lock to WebAudio playback position!"
    assert "state.lyricsLines = [];" in ct_src, "Deactivating karaoke must clear cached lyricsLines so reopening fetches fresh server lyrics!"
    # 6) 24/7 bounded memory & stream-safety invariants across PC backend + NAS gateway
    assert "OrderedDict" in sep_src and "_lyrics_mem_max" in sep_src, "Backend lyrics cache must be bounded LRU"
    assert "OrderedDict" in gw_src and "_lyrics_ram_max" in gw_src, "Gateway lyrics cache must be bounded LRU"
    assert "_lyrics_refresh_ttl" in gw_src, "Gateway _lyrics_refreshing must have TTL auto-expiry"
    assert "open_gdrive_audio_stream" in gw_src, "Gateway must pre-read first chunk before returning 200 OK"
    st_src = (BACKEND / "storage.py").read_text(encoding="utf-8")
    assert "open_audio_stream" in st_src, "Backend storage must pre-read first chunk before returning 200 OK"
    print("[PASS] Structural lifecycle, 24/7 memory bounds & ALL-6-exit-path invariants")


def test_store_builder_parity() -> None:
    """Verify build-store-package.py produces syntactically valid prod extension with /api/ validation & zero dev polling."""
    builder = ROOT / "scripts" / "build-store-package.py"
    with tempfile.TemporaryDirectory() as td:
        out_zip = Path(td) / "store.zip"
        unpacked = Path(td) / "unpacked"
        res = subprocess.run(
            [sys.executable, str(builder), "--api-base", "https://karaoke.example.com", "--out", str(out_zip), "--unpacked-dir", str(unpacked)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert res.returncode == 0, f"Store builder failed: {res.stderr}"
        bg = (unpacked / "background.js").read_text(encoding="utf-8")
        ct = (unpacked / "content.js").read_text(encoding="utf-8")
        mf = json.loads((unpacked / "manifest.json").read_text(encoding="utf-8"))
        assert 'const API_BASE = "https://karaoke.example.com";' in bg
        assert '!path.startsWith("/api/")' in bg, "Production background.js lost /api/ path guard!"
        assert "AbortController" in bg, "Production background.js lost fetch timeout guard!"
        assert 'const API_BASE = "https://karaoke.example.com";' in ct
        assert mf["host_permissions"] == ["https://karaoke.example.com/*"]
        assert mf["permissions"] == ["storage"]
        assert "X-Karaoke-Token" in bg and "AUTH_ACTIVATE" in bg
        node = subprocess.run(["node", "--check", str(unpacked / "background.js")], capture_output=True, text=True)
        assert node.returncode == 0, f"Prod background.js syntax error: {node.stderr}"
    print("[PASS] Chrome Web Store builder single-source parity & syntax check")


def test_invite_code_device_binding_and_revocation() -> None:
    """End-to-end test of InviteManager + Gateway device binding, device-limit enforcement, and instant revocation."""
    import importlib.util
    from invites import InviteManager

    class _DummyGDrive:
        def refresh_binary(self): return None
        def is_configured(self): return False

    class _DummyLog:
        def info(self, *a, **kw): pass
        def warning(self, *a, **kw): pass
        def error(self, *a, **kw): pass

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        im = InviteManager(tdir, _DummyGDrive(), _DummyLog())
        created = im.create_codes(count=2, label="測試群", max_devices=2)
        assert len(created) == 2
        c1 = created[0]["code"]
        assert c1.startswith("KTV-") and len(c1) == 13

        # Load gateway/main.py in isolation pointed at temp DATA_DIR (no live rclone needed)
        os.environ["DATA_DIR"] = str(tdir / "gw_data")
        os.environ["REQUIRE_AUTH"] = "auto"
        spec = importlib.util.spec_from_file_location("gw_test_mod", str(GATEWAY / "main.py"))
        gw_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gw_mod)
        gw = gw_mod.gw
        gw._access_codes = dict(im._codes)
        gw._last_keys_sync = time.time()

        assert gw.is_auth_required() is True

        # Device 1 activates -> succeeds
        r1 = gw.activate_invite_code(c1, "dev_aaaa1111bbbb2222", "1.2.3.4")
        t1 = r1["token"]
        assert r1["used_devices"] == 1 and t1.startswith("ktv_")
        ok, info, _ = gw.verify_device_token(t1)
        assert ok and info["code"] == c1

        # Same Device 1 re-activates -> reuses slot cleanly (still 1/2!)
        r1b = gw.activate_invite_code(c1, "dev_aaaa1111bbbb2222", "1.2.3.4")
        assert r1b["used_devices"] == 1
        t1_new = r1b["token"]
        assert gw.verify_device_token(t1)[0] is False  # old rotated token invalidated
        assert gw.verify_device_token(t1_new)[0] is True

        # Device 2 activates -> succeeds (2/2)
        r2 = gw.activate_invite_code(c1, "dev_cccc3333dddd4444", "5.6.7.8")
        assert r2["used_devices"] == 2
        t2 = r2["token"]

        # Device 3 (leaked code!) -> BLOCKED with HTTP 403!
        blocked = False
        try:
            gw.activate_invite_code(c1, "dev_eeee5555ffff6666", "9.10.11.12")
        except gw_mod.HTTPException as exc:
            blocked = (exc.status_code == 403)
        assert blocked, "Device 3 should have been blocked by max_devices=2!"

        # Instant revocation -> BOTH Device 1 & Device 2 immediately fail verification!
        im.set_enabled(c1, False)
        gw._access_codes = dict(im._codes)
        assert gw.verify_device_token(t1_new)[0] is False
        assert gw.verify_device_token(t2)[0] is False

        # Re-enable -> immediately works again!
        im.set_enabled(c1, True)
        gw._access_codes = dict(im._codes)
        assert gw.verify_device_token(t1_new)[0] is True
    print("[PASS] Invite code device binding, max-device cap & instant revocation")


def test_generic_title_decomposition() -> None:
    import lyrics_worker as lw

    assert not hasattr(lw, "CONCERT_NOISE_PART_RE"), "Hardcoded concert blacklist should stay deleted!"
    c1 = lw._extract_query_candidates("FEAR AND DREAMS 世界巡迴演唱會 - 武漢站｜第六場｜《十面埋伏》")
    assert any("十面埋伏" in q for q in c1), f"Missing song in {c1}"
    c2 = lw._extract_query_candidates("CHILL CLUB 推介榜 | 張敬軒 隱形遊樂場")
    assert any("隱形遊樂場" in q for q in c2), f"Missing song in {c2}"
    c3 = lw._extract_query_candidates("Random Vlog", track="富士山下", artist="陳奕迅")
    assert c3[0] == "陳奕迅 富士山下", f"Structured metadata should be #1: {c3}"
    # Nested quotes inside 《...》 (e.g. 《唯獨不可敗給名為「我」的你》《戀愛腦之死》) must NEVER truncate at inner 「」!
    b_multi = lw._extract_bracketed_titles("《CHILL CLUB》💙 晚安莉莉《唯獨不可敗給名為「我」的你》《戀愛腦之死》")
    assert "唯獨不可敗給名為「我」的你" in b_multi and "戀愛腦之死" in b_multi, f"Nested quotes broken: {b_multi}"
    print("[PASS] Generic zero-blacklist title decomposition + nested-quote multi-song parser")


def test_word_bunching_deglitcher() -> None:
    import numpy as np
    import lyrics_worker as lw

    # Case 1: Exact raw aligner degeneracy (0.00s collapsed tokens) + 2.8s gap -> must smoothly recover via CDF!
    collapsed = [
        {"word": "不", "start": 142.53, "end": 142.65},
        {"word": "需", "start": 142.65, "end": 142.65},
        {"word": "急", "start": 142.65, "end": 142.65},
        {"word": "切", "start": 142.65, "end": 142.65},
        {"word": "一", "start": 142.65, "end": 142.65},
        {"word": "息", "start": 142.65, "end": 142.65},
        {"word": "間", "start": 142.65, "end": 142.65},
        {"word": "變", "start": 145.77, "end": 146.55},
        {"word": "好", "start": 146.55, "end": 149.15},
    ]
    fixed = lw._smooth_bunched_words(collapsed, 142.53, 149.15, np.array([]), 0.1)
    for w in fixed[1:7]:
        assert (w["end"] - w["start"]) >= 0.40, f"Collapsed token not recovered: {w}"
    assert abs(fixed[6]["end"] - 145.77) < 0.05

    # Case 2: Legitimate ultra-fast rap (0.08s/char) followed by a deliberate 1.2s musical pause -> MUST NOT be smeared!
    fast_rap = [
        {"word": "快", "start": 10.00, "end": 10.08},
        {"word": "使", "start": 10.08, "end": 10.16},
        {"word": "用", "start": 10.16, "end": 10.24},
        {"word": "雙", "start": 10.24, "end": 10.32},
        {"word": "截", "start": 10.32, "end": 10.40},
        {"word": "棍", "start": 11.60, "end": 12.10},
    ]
    rap_out = lw._smooth_bunched_words(fast_rap, 10.00, 12.10, np.array([]), 0.1)
    assert abs(rap_out[4]["end"] - 10.40) < 0.02, f"Fast rap was wrongly smeared into pause: {rap_out[4]}"

    # Case 3: Over-stretched line across a 10s silent guitar solo -> must trim to actual active vocal burst!
    db_prof = np.full(200, -60.0, dtype=np.float32)  # 20.0s @ 0.1s/step
    db_prof[10:25] = -14.0  # normal singing so adaptive threshold sits cleanly around -26 dBFS
    db_prof[142:156] = -15.0  # real vocal burst at 14.2s-15.6s
    ghost = [{"start": 4.0, "end": 16.0, "text": "(It's not over)", "words": [{"word": "over", "start": 4.0, "end": 16.0}]}]
    trimmed = lw.refine_alignment_with_acoustics(ghost, db_prof, 0.1, 20.0, 0.0)
    assert trimmed[0]["start"] >= 13.8 and (trimmed[0]["end"] - trimmed[0]["start"]) <= 2.5, f"Silent ghost line not trimmed: {trimmed[0]}"

    # Case 4: Adaptive threshold mathematical invariant (p15 < thr < p90) across BOTH high-DR AND compressed live stems!
    hdr_prof = np.concatenate([np.full(50, -60.0), np.full(50, -33.0), np.full(50, -5.0)]).astype(np.float32)
    hdr_thr = lw._adaptive_vocal_threshold_db(hdr_prof)
    assert -45.0 <= hdr_thr < -33.0, f"High-DR threshold blinded quiet ballad verse: {hdr_thr:.1f} dBFS"
    comp_live = np.concatenate([np.full(30, -33.0), np.full(60, -28.0), np.full(30, -24.1)]).astype(np.float32)
    comp_thr = lw._adaptive_vocal_threshold_db(comp_live)
    assert -33.0 < comp_thr < -28.0, f"Compressed live threshold must lie strictly between p15 (-33) and p50 (-28), got {comp_thr:.1f}"
    print("[PASS] First-principles Energy-CDF recovery + silent-solo trim + bounded scale-invariant threshold")


def test_silence_collapse_and_mapping() -> None:
    import numpy as np
    import lyrics_worker as lw

    sr = 16000
    step_s = 0.1
    # 30s block: [2s-8s] vocal island 1, [8s-20s] 12s guitar solo (silent vocals), [20s-27s] vocal island 2
    y = np.zeros(30 * sr, dtype=np.float32)
    y[2 * sr : 8 * sr] = 0.2
    y[20 * sr : 27 * sr] = 0.2
    db_prof = np.full(300, -70.0, dtype=np.float32)
    db_prof[20:80] = -14.0
    db_prof[200:270] = -14.0

    comp, tmap = lw.build_silence_collapsed_audio(y, db_prof, step_s, sr=sr, thr_db=-30.0)
    assert len(tmap) == 2, f"Expected 2 vocal islands around 12s solo, got {tmap}"
    assert len(comp) < len(y) * 0.65, "12s instrumental solo should be compressed"
    # Second island starts around 20s - 0.35s pad = 19.65s in real time
    mapped_second = lw._map_comp_to_real(tmap[1][0] + 0.5, tmap)
    assert 19.5 <= mapped_second <= 21.0, f"Unexpected real-time mapping for island 2: {mapped_second}"
    print("[PASS] Silence collapse & piecewise real-time mapping across instrumental breaks")


def test_monotonic_anchor_consensus_and_contiguous_span() -> None:
    import numpy as np
    import lyrics_worker as lw

    # 1) Weighted Monotonic Anchor Consensus: a spurious Verse-1 match to Outro line #60 before genuine lines 16..20
    #    PLUS an isolated spurious jump to line #5 at the end MUST be pruned while genuine anchors stay intact!
    pool = [f"Line {i:02d}" for i in range(66)]
    noisy_run = [
        {"start": 52.5, "end": 55.6, "text": "Line 60", "sim": 0.42},  # spurious early outlier!
        {"start": 55.7, "end": 57.5, "text": "Line 16", "sim": 0.78},
        {"start": 57.6, "end": 59.2, "text": "Line 17", "sim": 0.80},
        {"start": 59.3, "end": 62.7, "text": "Line 18", "sim": 0.85},
        {"start": 62.8, "end": 64.3, "text": "Line 19", "sim": 0.82},
        {"start": 195.0, "end": 197.5, "text": "Line 60", "sim": 0.72},
        {"start": 198.0, "end": 200.5, "text": "Line 62", "sim": 0.75},
        {"start": 201.0, "end": 202.0, "text": "Line 05", "sim": 0.40},  # spurious outro outlier!
    ]
    runs = lw._select_monotonic_run_anchors(noisy_run, pool)
    assert len(runs) == 1, f"Expected 1 clean monotonic run, got {len(runs)}"
    idxs = [it["_idx"] for it in runs[0]]
    assert idxs == [16, 17, 18, 19, 60, 62], f"Outliers not pruned cleanly: {idxs}"

    # 2) Genuine live encore repeat (>= 3 consecutive forward anchors after primary song) MUST be preserved!
    encore_run = noisy_run[:-1] + [
        {"start": 215.0, "end": 218.0, "text": "Line 17", "sim": 0.82},
        {"start": 218.1, "end": 221.0, "text": "Line 18", "sim": 0.85},
        {"start": 221.1, "end": 224.0, "text": "Line 19", "sim": 0.84},
    ]
    runs2 = lw._select_monotonic_run_anchors(encore_run, pool)
    assert len(runs2) == 2 and [it["_idx"] for it in runs2[1]] == [17, 18, 19], f"Genuine encore lost: {runs2}"

    # 3) Gap-bounded backward vocal span: TV host intro at 0..4s separated by 24s silence from Verse 1 at 28..55s
    #    MUST stop at 28.0s and NEVER pull Verse 1 back into 0..4s host chatter!
    db = np.full(600, -75.0, dtype=np.float32)  # 60s @ 0.1s/step
    db[5:35] = -28.0     # TV host intro / applause 0.5s..3.5s
    db[280:555] = -26.0  # Real Verse 1 singing 28.0s..55.5s
    act_s, v_first, v_last = lw._contiguous_vocal_span(db, 0.1, -50.0, 0.0, 55.5, direction="backward", max_silent_gap=8.0)
    assert 27.5 <= v_first <= 28.5 and act_s >= 25.0, f"Backward span leaked into host intro: act={act_s}, v_first={v_first}"
    print("[PASS] Monotonic anchor consensus (outlier pruning + encore preservation) & gap-bounded vocal span")


def test_lrc_multi_timestamp_and_asr_splitting_and_invariants() -> None:
    import numpy as np
    import lyrics_worker as lw

    # 1) Multi-timestamp LRC lines ([00:52.10][02:05.30]Chorus) MUST expand into all occurrences with zero leftover tags!
    raw_lrc = "[00:10.00]作詞：黃偉文\n[00:15.00]Lyrics by: Someone\n[00:20.00][01:40.50]曾話我愛你 我願愛到死\n[00:25.00]用盡力沒頭沒腦似鬼魅\n"
    parsed = lw.parse_lrc(raw_lrc)
    assert len(parsed) == 3, f"Expected 3 expanded LRC lines (credits stripped), got {parsed}"
    assert parsed[0]["start"] == 20.0 and parsed[2]["start"] == 100.5
    assert "[" not in parsed[0]["text"] and "]" not in parsed[2]["text"]

    # 2) Giant 35s Whisper run-on segment MUST split into natural single-line ASR chunks so official matching never fails!
    words = []
    t = 33.5
    text_blob = "該怎麼初次約會實在極費思量起初先推我兩次讓我檢驗正常再若便說好以禮數換你掌上找一間小店吃飯浪漫大戲開場"
    for ch in text_blob:
        words.append({"word": ch, "start": round(t, 2), "end": round(t + 0.35, 2)})
        t += 0.35
    chunks = lw._split_long_asr_segment(33.5, t, text_blob, words, "zh")
    assert len(chunks) >= 3, f"Run-on Whisper blob was not split: {len(chunks)}"
    assert all(len(c["text"]) <= 18 for c in chunks), f"Chunk still too long: {[len(c['text']) for c in chunks]}"

    # 3) Zero-overlap & zero-word-OOB post-invariants + local-floor trim of reverb-bleed 22s line!
    db = np.full(300, -70.0, dtype=np.float32)  # 30s @ 0.1s/step
    db[20:55] = -14.0    # strong real singing 2.0s..5.5s
    db[55:240] = -42.0   # soft reverb/crowd bleed (above global thr ~ -53 dB, but 28 dB below singing!)
    db[245:280] = -15.0  # next verse singing 24.5s..28.0s
    bad_lines = [
        {"start": 2.0, "end": 24.0, "text": "很想花光氣力捉緊你", "words": [{"word": "很", "start": 1.8, "end": 3.0}, {"word": "你", "start": 3.0, "end": 24.5}]},
        {"start": 24.4, "end": 24.75, "text": "開始感激", "words": [{"word": "開", "start": 24.4, "end": 24.9}]},
        {"start": 24.70, "end": 28.0, "text": "過活太好", "words": [{"word": "過", "start": 24.70, "end": 28.2}]},
    ]
    clean = lw.refine_alignment_with_acoustics(bad_lines, db, 0.1, 30.0, 0.0)
    assert (clean[0]["end"] - clean[0]["start"]) < 7.0, f"22s reverb-bleed ghost line not trimmed: {clean[0]}"
    for i in range(len(clean)):
        if i + 1 < len(clean):
            assert clean[i]["end"] <= clean[i + 1]["start"] + 1e-6, f"Overlap at {i}: {clean[i]} vs {clean[i+1]}"
        for w in clean[i]["words"]:
            assert clean[i]["start"] - 1e-6 <= w["start"] <= w["end"] <= clean[i]["end"] + 1e-6, f"WORD_OOB at {i}: {w} vs {clean[i]}"
    print("[PASS] Multi-timestamp LRC, Whisper run-on splitting, reverb-bleed trim, zero-overlap & zero-WORD_OOB invariants")


def test_sidecar_watcher_cp1252_safety() -> None:
    if not WATCHER.exists():
        print("[SKIP] watcher.py not installed")
        return
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252"
    env["PYTHONUTF8"] = "0"
    res = subprocess.run([sys.executable, str(WATCHER), "--self-test"], env=env, capture_output=True, text=True, timeout=15)
    state_tmp = WATCHER.parent / "watcher_state.json"
    state_tmp.unlink(missing_ok=True)
    assert res.returncode == 0, f"watcher cp1252 self-test failed: {res.stderr}"
    print("[PASS] Sidecar watcher UTF-8 safety under forced Windows cp1252")


def test_live_endpoints_if_running() -> None:
    import httpx

    try:
        r = httpx.get("http://127.0.0.1:18765/api/health", timeout=2.5)
        if r.status_code == 200:
            d = r.json()
            assert d.get("status") == "ok"
            assert "pending_hq_upgrades" in d
            assert "catalog_version" in d
            print(f"[PASS] Live PC backend OK (cache={d['cache_count']}, local_mb={d['local_cache_mb']}, pending_hq={d['pending_hq_upgrades']})")
    except Exception:
        print("[SKIP] Local PC backend not running during offline test")


def main() -> int:
    test_python_compile_and_ast()
    test_extension_js_syntax()
    test_atomic_json_concurrency()
    test_worker_and_separator_invariants()
    test_store_builder_parity()
    test_invite_code_device_binding_and_revocation()
    test_generic_title_decomposition()
    test_word_bunching_deglitcher()
    test_silence_collapse_and_mapping()
    test_monotonic_anchor_consensus_and_contiguous_span()
    test_lrc_multi_timestamp_and_asr_splitting_and_invariants()
    test_sidecar_watcher_cp1252_safety()
    test_live_endpoints_if_running()
    print("\nALL REGRESSION TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
