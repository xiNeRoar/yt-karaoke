"""Build a production Chrome Web Store .zip (and optional unpacked folder) pointing to your own HTTPS gateway.

Usage:
    python scripts/build-store-package.py --api-base https://karaoke.yourdomain.com
    python scripts/build-store-package.py --api-base https://karaoke.yourdomain.com --unpacked-dir extension-prod
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXT_SRC = ROOT / "extension"


def build_package(api_base: str, out_zip: Path, unpacked_dir: Path | None = None) -> None:
    api_base = api_base.rstrip("/")
    if not (api_base.startswith("https://") or api_base.startswith("http://")):
        raise SystemExit("ERROR: --api-base must start with https:// or http://")

    staging = unpacked_dir if unpacked_dir else (ROOT / "backend" / "tmp" / "_store_build")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(EXT_SRC, staging)

    # 1. Patch manifest.json with strictly minimal Chrome Web Store permissions:
    #    - Only "storage" (stores the user's bound invite device token locally; zero sensitive permissions)
    #    - host_permissions strictly limited to the karaoke gateway origin only
    #      (content_scripts.matches already handles injecting the UI on https://www.youtube.com/*)
    m_path = staging / "manifest.json"
    manifest = json.loads(m_path.read_text(encoding="utf-8"))
    manifest["permissions"] = ["storage"]
    manifest["host_permissions"] = [f"{api_base}/*"]
    m_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 2. Patch API_BASE in background.js directly from extension/background.js (100% single-source parity!)
    #    IS_LOCAL_DEV evaluates to false for https:// origins so dev auto-reload & chrome.tabs.reload are disabled cleanly.
    bg_path = staging / "background.js"
    bg_txt = bg_path.read_text(encoding="utf-8")
    bg_replaced = bg_txt.replace("http://127.0.0.1:18765", api_base)
    if api_base not in bg_replaced:
        raise SystemExit("ERROR: failed to patch API_BASE in background.js")
    bg_path.write_text(bg_replaced, encoding="utf-8")

    # 3. Patch API_BASE in content.js
    c_path = staging / "content.js"
    c_txt = c_path.read_text(encoding="utf-8")
    c_replaced = c_txt.replace("http://127.0.0.1:18765", api_base)
    if api_base not in c_replaced:
        raise SystemExit("ERROR: failed to patch API_BASE in content.js")
    c_path.write_text(c_replaced, encoding="utf-8")

    # 3. Pack clean Chrome Web Store .zip
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(staging.rglob("*")):
            if p.is_file() and not p.name.startswith("."):
                zf.write(p, p.relative_to(staging).as_posix())

    if not unpacked_dir:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"Built {out_zip} ({out_zip.stat().st_size} bytes) -> {api_base}")
    if unpacked_dir:
        print(f"Unpacked production extension ready at: {unpacked_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", required=True, help="Your HTTPS karaoke gateway origin, e.g. https://karaoke.yourdomain.com")
    ap.add_argument("--out", default=str(ROOT / "yt-karaoke-chrome-store.zip"), help="Output .zip path")
    ap.add_argument("--unpacked-dir", default="", help="Optional folder to keep unpacked production extension")
    args = ap.parse_args()
    build_package(args.api_base, Path(args.out), Path(args.unpacked_dir) if args.unpacked_dir else None)
