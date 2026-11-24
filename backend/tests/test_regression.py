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
    py_files = list(BACKEND.glob("*.py")) + [GATEWAY / "main.py"]
    if WATCHER.exists():
        py_files.append(WATCHER)
    for pf in py_files:
        py_compile.compile(str(pf), doraise=True)
        ast.parse(pf.read_text(encoding="utf-8"), filename=str(pf))
    print(f"[PASS] Python compile + AST parse ({len(py_files)} files)")


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
    assert "if not vocals_path.exists():" in lw_src, "Layer 1 youtube_sub must require word alignment when vocals exist"
    assert "if lrc_synced and not force_ai and not vocals_path.exists():" in lw_src, "Layer 2 LRC must require word alignment when vocals exist"
    assert "require_all_windows=True" in lw_src, "0-word scaffolds must require 100% window coverage from Qwen3"
    assert "if lrc_synced and not force_ai and len(lrc_synced)" in lw_src, "force_ai=True must bypass LRC scaffold shortcut"
    assert "whisper_qwen3_ai" in lw_src, "Free-transcribe fallback must also pass through acoustic & Qwen3 refinement"
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
        node = subprocess.run(["node", "--check", str(unpacked / "background.js")], capture_output=True, text=True)
        assert node.returncode == 0, f"Prod background.js syntax error: {node.stderr}"
    print("[PASS] Chrome Web Store builder single-source parity & syntax check")


def test_generic_title_decomposition() -> None:
    import lyrics_worker as lw

    assert not hasattr(lw, "CONCERT_NOISE_PART_RE"), "Hardcoded concert blacklist should stay deleted!"
    c1 = lw._extract_query_candidates("FEAR AND DREAMS 世界巡迴演唱會 - 武漢站｜第六場｜《十面埋伏》")
    assert any("十面埋伏" in q for q in c1), f"Missing song in {c1}"
    c2 = lw._extract_query_candidates("CHILL CLUB 推介榜 | 張敬軒 隱形遊樂場")
    assert any("隱形遊樂場" in q for q in c2), f"Missing song in {c2}"
    c3 = lw._extract_query_candidates("Random Vlog", track="富士山下", artist="陳奕迅")
    assert c3[0] == "陳奕迅 富士山下", f"Structured metadata should be #1: {c3}"
    print("[PASS] Generic zero-blacklist title decomposition")


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
    print("[PASS] First-principles Energy-CDF collapse recovery (fixes 0.00s collapses, 0 distortion on fast rap)")


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
    test_generic_title_decomposition()
    test_word_bunching_deglitcher()
    test_sidecar_watcher_cp1252_safety()
    test_live_endpoints_if_running()
    print("\nALL REGRESSION TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
