# Privacy Policy — YouTube AI 唱K 去人聲

Last updated: 2026-11-23

## Data Collection & Usage
- This Chrome Extension **does not collect, store, sell, or transmit** any personal information, browsing history, YouTube cookies, account credentials, or authentication tokens.
- When you actively click "開始唱K" on a YouTube watch page (`https://www.youtube.com/watch?v=...`), the extension sends only the public 11-character YouTube video ID and selected vocal-removal mode (`pure` or `karaoke`) to the configured karaoke processing backend to fetch separated instrumental and vocal audio stems.
- Default mode preference (`pure` or `karaoke`) is saved locally in your browser using `chrome.storage.local`.

## Permissions Justification
- `storage`: Saves your preferred default accompaniment mode locally.
- `host_permissions` (`https://www.youtube.com/*` and backend origin): Injects the karaoke control bar onto YouTube watch pages and loads separated audio stems from the backend.

## Contact
For questions or privacy requests, please open an issue on the project's GitHub repository.
