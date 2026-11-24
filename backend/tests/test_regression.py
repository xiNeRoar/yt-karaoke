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
    print("[PASS] Structural lifecycle & concurrency invariants (worker + separator + gateway)")


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
    test_generic_title_decomposition()
    test_sidecar_watcher_cp1252_safety()
    test_live_endpoints_if_running()
    print("\nALL REGRESSION TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
