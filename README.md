# YouTube AI 唱K 去人聲 (Chrome Extension + Local GPU Backend)

專為精準去人聲而設的 YouTube 唱K 工具，使用開源界頂級 **BS-RoFormer** 及 **Mel-Band RoFormer Karaoke** 模型在你的 RTX 顯卡上免費高速分離伴奏與人聲軌。

## 主要功能
- **兩種專業去人聲模式自由切換**：
  - **純伴奏 (全去人聲)**：使用 `BS-RoFormer (SDR 12.97)`，徹底清除主唱與和音，乾淨無殘留。
  - **KTV (保留和音)**：使用 `Mel-Band RoFormer Karaoke`，只去主唱、保留副歌和音，更有卡拉OK氣氛。
- **導唱音量滑桿 (0% – 100%)**：隨時微調原唱導唱大小，或一鍵切換 `伴奏 / 導唱 / 原唱`。
- **即時升降 Key (±6 半音)**：原速升降調，原調時自動 Bypass 保持 100% 無損音質。
- **後台智能防爆 VRAM 保護**：
  - 每次拆歌完成**立即釋放 VRAM**（實測待機僅佔 ~9 MB VRAM）。
  - 預設開啟 **自動保護模式 (`KARAOKE_DEVICE=auto`)**：若偵測到本機 MiniMax 生片模型或打機佔用緊顯卡（剩餘 VRAM < 4.5GB），後台會自動改用 CPU 處理，唔會搶顯存令其他程式 Crash。
  - 亦可在 `start-backend.bat` 內以環境變數 `KARAOKE_DEVICE=auto|gpu|cpu` 控制後台運算裝置。
- **本地快取秒開**：每首歌按 YouTube `videoId` 快取，唱過一次下次即開即唱（0% GPU）。

---

## 安裝與使用步驟

### 1. 啟動桌面控制台程式 (`.exe`)
雙擊桌面嘅 **`YouTube AI 唱K.exe`**（會自動啟動後台服務並常駐於右下角 Windows **System Tray 系統列**）：
- 介面可隨時開關唱 K 服務、監察剩餘 VRAM、切換 `智能防爆 / 強制 GPU / 純 CPU`、清空快取或查看日誌。
- 關閉視窗會自動縮小至系統列，按右鍵或點一下系統列圖示即可操作。

### 2. 載入 Chrome Extension（一鍵輔助）
直接喺桌面控制台撳 **「🧩 安裝 / 載入 Chrome 插件」**（會自動開 Chrome 擴充功能頁同揀中 `extension` 資料夾）：
1. 開啟 Chrome 右上角 **「開發人員模式 (Developer mode)」**。
2. 將檔案總管揀中嘅 `extension` 資料夾直接 **拖曳 (Drag & Drop)** 入 Chrome 頁面即完成安裝！

### 3. 去 YouTube 唱 K！
打開任何 YouTube MV 影片，影片標題上方會出現 **「AI 唱K 去人聲」** 控制列：
1. 揀選 `純伴奏` 或 `保留和音`。
2. 撳 **`🎤 開始唱K`**，等十幾秒拆軌完成就會自動靜音原聲並同步播伴奏！
3. 自由調整 **導唱音量**、**升降 Key (±6)** 同 **同步微調 (±25ms)**。
