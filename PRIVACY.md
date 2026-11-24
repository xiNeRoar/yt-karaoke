# Privacy Policy — YouTube AI 唱K 去人聲 (YouTube AI Karaoke)

Last updated: 2026-11-24

## Data Collection & Usage
- This Chrome Extension **does not collect, store, sell, or transmit** any personal information, browsing history, YouTube cookies, account credentials, or authentication tokens.
- When you actively click **"開始唱K"** or **"KTV 歌詞"** on a YouTube watch page (`https://www.youtube.com/watch?v=...`), the extension sends **only the public 11-character YouTube video ID** and an anonymous random per-tab session ID to the configured karaoke backend to retrieve separated instrumental/vocal tracks and synchronized lyrics.
- No audio is recorded from your microphone, and no data is shared with advertisers or third-party analytics services.

## Permissions Justification
- This extension requests **zero** special browser API permissions (no `storage`, `tabs`, `cookies`, or `webRequest`).
- `host_permissions` (only the configured karaoke server origin): Streams separated instrumental/vocal accompaniment tracks and synchronized lyrics for in-browser playback (never downloads or saves video/audio files to disk).
- `content_scripts` (`https://www.youtube.com/*`): Displays the karaoke control bar and KTV lyrics overlay inside the YouTube player while leaving video playback and ads intact.

## Self-Hosting
This project is 100% open source (MIT License). You may run the entire backend and gateway on your own PC or NAS without connecting to any external hosted server.

## Contact
For questions or privacy requests, please open an issue on the project's GitHub repository: https://github.com/xiNeRoar/yt-karaoke
