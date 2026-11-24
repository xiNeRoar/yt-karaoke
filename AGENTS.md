# Project Engineering Standards — YouTube AI Karaoke

1. **Zero Band-Aid Patches (Mandatory)**:
   - Never add hardcoded keyword blacklists (e.g., show names, channel names, concert keywords), per-song hacks, or arbitrary magic thresholds to patch individual failures.
   - Always fix the underlying root cause structurally and generically across all exit paths (e.g., adaptive acoustic/energy thresholds, universal structural parsing, single-source build scripts, bounded 24/7 caches, atomic file writes).
2. **All-Layer Verification**:
   - After any fix, verify all affected layers (`backend/`, `gateway/`, `extension/`, `scripts/build-store-package.py`, `tray-app/`) and run `uv run python backend/tests/test_regression.py`.
   - Never claim a live service is updated unless the running process/container/package has been verified directly.
3. **Never Disrupt Active Singing**:
   - Check `/api/health` (`active_singing == 0` and `queue_count == 0`) before restarting the local PC backend.
4. **Zero Secret Leaks**:
   - Keep `.admin_token`, `.worker_secret`, `cookies.txt`, `.yt_firefox_profile/`, `config.json`, `access_keys.json`, `device_bindings.json`, `rclone.conf`, `*.local.yml`, `extension-prod/`, and `*.zip` out of Git. Public `extension/` must stay pointed at `http://127.0.0.1:18765`.
