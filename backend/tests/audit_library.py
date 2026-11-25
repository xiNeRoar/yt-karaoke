"""Automated Full-Library Lyrics Quality Auditor for YouTube AI Karaoke.

Run anytime against local PC backend or public NAS gateway:
    uv run python tests/audit_library.py
    uv run python tests/audit_library.py --base http://127.0.0.1:18765 --fail-on-defect
"""

from __future__ import annotations

import argparse
import re
import sys
import httpx


def audit_song(vid: str, title: str, dur: float, ly: dict) -> list[str]:
    flags: list[str] = []
    st = ly.get("status")
    lines = ly.get("lines") or []
    if st != "ready" or not lines:
        return [f"NO_LYRICS({st})"]

    first_s = float(lines[0]["start"])
    last_e = float(lines[-1]["end"])
    if dur > 60 and first_s > min(90.0, dur * 0.35):
        flags.append(f"LATE_START({first_s:.0f}s/{dur:.0f}s)")
    if dur > 60 and (dur - last_e) > max(50.0, dur * 0.25):
        flags.append(f"EARLY_END({last_e:.0f}s/{dur:.0f}s)")
    if dur > 90 and len(lines) < max(8, dur / 18.0):
        flags.append(f"SPARSE({len(lines)}L/{dur:.0f}s)")

    zero_w = 0
    crushed = 0
    giant = []
    overlaps = 0
    word_oob = 0
    halluc = 0
    for i, ln in enumerate(lines):
        s, e = float(ln["start"]), float(ln["end"])
        txt = (ln.get("text") or "").strip()
        ws = ln.get("words") or []
        nch = len(re.sub(r"\s+", "", txt))
        if not ws:
            zero_w += 1
        else:
            if float(ws[0]["start"]) < s - 0.15 or float(ws[-1]["end"]) > e + 0.15:
                word_oob += 1
            if sum(1 for w in ws if (float(w["end"]) - float(w["start"])) <= 0.02) >= max(2, len(ws) // 2):
                crushed += 1
        if nch >= 4 and (e - s) <= 0.45:
            crushed += 1
        if (e - s) > 10.5:
            giant.append((i, round(e - s, 1), txt[:16]))
        if i > 0 and s < float(lines[i - 1]["end"]) - 0.05:
            overlaps += 1
        if re.search(r"字幕|訂閱|按讚|謝謝觀看|謝謝收看|Subscribe", txt, re.I):
            halluc += 1

    if zero_w:
        flags.append(f"0W={zero_w}")
    if crushed:
        flags.append(f"CRUSHED={crushed}")
    if giant:
        flags.append(f"GIANT={giant[:3]}")
    if overlaps:
        flags.append(f"OVERLAP={overlaps}")
    if word_oob:
        flags.append(f"WORD_OOB={word_oob}")
    if halluc:
        flags.append(f"HALLUC={halluc}")
    return flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18765")
    ap.add_argument("--fail-on-defect", action="store_true")
    args = ap.parse_args()

    cache = httpx.get(f"{args.base}/api/cache?limit=500", timeout=20).json()
    items = cache.get("items") or []
    print(f"=== AUDITING {len(items)} CACHED SONGS @ {args.base} ===")
    defects = 0
    for idx, it in enumerate(items):
        vid = it["video_id"]
        title = (it.get("title") or vid)[:42]
        dur = float(it.get("duration") or 0)
        ly = httpx.get(f"{args.base}/api/lyrics/{vid}", timeout=15).json()
        flags = audit_song(vid, title, dur, ly)
        src = ly.get("source") or ""
        lines = ly.get("lines") or []
        if flags and not (len(flags) == 1 and flags[0].startswith("NO_LYRICS")):
            defects += 1
        tag = "OK" if not flags else "WARN: " + " | ".join(flags)
        print(f"[{idx+1:02d}] {vid} ({dur:5.0f}s) [{src:18s}] {len(lines):3d}L | {title:42s} -> {tag}")

    print(f"\n=== AUDIT DONE: {defects} generated songs with defects ===")
    return 1 if (args.fail_on_defect and defects > 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
