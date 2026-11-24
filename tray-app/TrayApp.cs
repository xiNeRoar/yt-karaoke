using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Net;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using System.Windows.Forms;

namespace YtKaraokeTray
{
    static class Program
    {
        private const string MutexName = "Global\\YtAiKaraokeTrayController_SingleInstance";
        private const string ShowEventName = "Global\\YtAiKaraokeTrayController_ShowWindow";

        [DllImport("user32.dll")]
        private static extern bool SetProcessDPIAware();

        [STAThread]
        static void Main()
        {
            try { SetProcessDPIAware(); } catch { }
            ServicePointManager.Expect100Continue = false;

            bool createdNew;
            using (var mutex = new Mutex(true, MutexName, out createdNew))
            {
                if (!createdNew)
                {
                    try
                    {
                        using (var ev = EventWaitHandle.OpenExisting(ShowEventName)) ev.Set();
                    }
                    catch { }
                    return;
                }

                Application.EnableVisualStyles();
                Application.SetCompatibleTextRenderingDefault(false);
                Application.Run(new MainForm(ShowEventName));
            }
        }
    }

    sealed class ProcessJobTracker : IDisposable
    {
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
        private static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string lpName);

        [DllImport("kernel32.dll")]
        private static extern bool SetInformationJobObject(IntPtr hJob, int infoType, IntPtr lpJobObjectInfo, uint cbJobObjectInfoLength);

        [DllImport("kernel32.dll")]
        private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

        [DllImport("kernel32.dll")]
        private static extern bool CloseHandle(IntPtr hObject);

        [StructLayout(LayoutKind.Sequential)]
        struct JOBOBJECT_BASIC_LIMIT_INFORMATION
        {
            public long PerProcessUserTimeLimit;
            public long PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize;
            public UIntPtr MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass;
            public uint SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct IO_COUNTERS
        {
            public ulong ReadOperationCount;
            public ulong WriteOperationCount;
            public ulong OtherOperationCount;
            public ulong ReadTransferCount;
            public ulong WriteTransferCount;
            public ulong OtherTransferCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        {
            public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
            public IO_COUNTERS IoInfo;
            public UIntPtr ProcessMemoryLimit;
            public UIntPtr JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed;
            public UIntPtr PeakJobMemoryUsed;
        }

        private const int JobObjectExtendedLimitInformation = 9;
        private const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000;
        private IntPtr _handle;

        public ProcessJobTracker()
        {
            _handle = CreateJobObject(IntPtr.Zero, null);
            if (_handle == IntPtr.Zero) return;
            var info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            int len = Marshal.SizeOf(typeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION));
            IntPtr ptr = Marshal.AllocHGlobal(len);
            try
            {
                Marshal.StructureToPtr(info, ptr, false);
                SetInformationJobObject(_handle, JobObjectExtendedLimitInformation, ptr, (uint)len);
            }
            finally { Marshal.FreeHGlobal(ptr); }
        }

        public void AddProcess(Process p)
        {
            if (_handle != IntPtr.Zero && p != null && !p.HasExited)
            {
                try { AssignProcessToJobObject(_handle, p.Handle); } catch { }
            }
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                CloseHandle(_handle);
                _handle = IntPtr.Zero;
            }
        }
    }

    public class MainForm : Form
    {
        private static readonly string ProjectRoot = ResolveProjectRoot();
        private static readonly string BackendDir = Path.Combine(ProjectRoot, "backend");
        private static readonly string ExtensionDir = Path.Combine(ProjectRoot, "extension");
        private static readonly string PythonExe = Path.Combine(BackendDir, @".venv\Scripts\python.exe");
        private static readonly string MainPy = Path.Combine(BackendDir, "main.py");
        private static readonly string TokenPath = Path.Combine(BackendDir, ".admin_token");
        private const string ApiBase = "http://127.0.0.1:18765";

        private static string ResolveProjectRoot()
        {
            string exeDir = AppDomain.CurrentDomain.BaseDirectory;
            string parent = Directory.GetParent(exeDir.TrimEnd('\\')) != null ? Directory.GetParent(exeDir.TrimEnd('\\')).FullName : "";
            if (!string.IsNullOrEmpty(parent) && Directory.Exists(Path.Combine(parent, "backend"))) return parent;
            if (Directory.Exists(Path.Combine(exeDir, "backend"))) return exeDir;
            return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), @".gemini\antigravity\scratch\yt-karaoke");
        }

        private readonly ProcessJobTracker _jobTracker = new ProcessJobTracker();
        private readonly EventWaitHandle _showWindowEvent;
        private readonly CancellationTokenSource _cts = new CancellationTokenSource();
        private readonly JavaScriptSerializer _json = new JavaScriptSerializer();

        private Process _backendProcess;
        private readonly object _procLock = new object();
        private bool _isExiting = false;
        private bool _trayHintShown = false;
        private bool _backendOnline = false;
        private bool _starting = false;
        private string _deviceMode = "auto";
        private string _storageMode = "local";
        private bool _gdriveConnected = false;
        private bool _pauseNewJobs = false;
        private bool _limitsInitialized = false;
        private bool _logsExpanded = false;
        private string _lastCacheSig = "";
        private string _lastCatalogVersion = "";

        // Tray
        private NotifyIcon _trayIcon;
        private ContextMenuStrip _trayMenu;
        private ToolStripMenuItem _trayStatsItem;
        private ToolStripMenuItem _trayToggleItem;
        private ToolStripMenuItem _trayPauseItem;
        private ToolStripMenuItem _trayModeAuto;
        private ToolStripMenuItem _trayModeGpu;
        private ToolStripMenuItem _trayModeCpu;
        private Icon _iconRunning;
        private Icon _iconStopped;

        // UI
        private Label _lblStatusPill;
        private Button _btnToggleServer;
        private Button _btnAutoUpdateExt;
        private Label _lblUsersNum;
        private Label _lblUsersSub;
        private Label _lblCacheNum;
        private Label _lblCacheSub;
        private Label _lblVramNum;
        private Label _lblVramSub;

        private Button _btnAuto;
        private Button _btnGpu;
        private Button _btnCpu;
        private Button _btnPauseNew;
        private NumericUpDown _numMaxQueue;
        private NumericUpDown _numRateLimit;
        private NumericUpDown _numMaxClients;
        private Button _btnApplyLimits;

        private Label _lblGdriveStatus;
        private Button _btnConnectGdrive;
        private Button _btnMigrateGdrive;
        private Button _btnToggleStorage;

        private ListView _lvSongs;
        private Button _btnOpenSong;
        private Button _btnCopyLink;
        private Button _btnDelSong;
        private Button _btnClearAll;
        private Button _btnToggleLogs;
        private TextBox _txtLogs;
        private System.Windows.Forms.Timer _pollTimer;
        private float _dpiScale = 1.0f;
        private Font _fontModeActive;
        private Font _fontModeNormal;

        private int S(int px)
        {
            return (int)Math.Round(px * _dpiScale);
        }

        // Studio Monochrome Dark Palette
        private static readonly Color BgDark = Color.FromArgb(15, 15, 15);
        private static readonly Color CardDark = Color.FromArgb(24, 24, 24);
        private static readonly Color BorderSubtle = Color.FromArgb(46, 46, 46);
        private static readonly Color BtnNeutral = Color.FromArgb(40, 40, 40);
        private static readonly Color TextPrimary = Color.FromArgb(241, 241, 241);
        private static readonly Color TextMuted = Color.FromArgb(156, 156, 156);
        private static readonly Color AccentWhite = Color.FromArgb(241, 241, 241);
        private static readonly Color AccentGreen = Color.FromArgb(34, 150, 72);
        private static readonly Color AccentRed = Color.FromArgb(196, 42, 42);
        private static readonly Color AccentAmber = Color.FromArgb(195, 132, 8);

        public MainForm(string showEventName)
        {
            using (var g = Graphics.FromHwnd(IntPtr.Zero))
            {
                _dpiScale = Math.Max(1.0f, g.DpiX / 96.0f);
            }
            _fontModeActive = new Font("Microsoft JhengHei UI", 10f, FontStyle.Bold);
            _fontModeNormal = new Font("Microsoft JhengHei UI", 10f, FontStyle.Regular);

            _showWindowEvent = new EventWaitHandle(false, EventResetMode.AutoReset, showEventName);
            _iconRunning = CreateTrayIcon(true);
            _iconStopped = CreateTrayIcon(false);

            BuildUI();
            BuildTray();
            StartShowEventListener();

            _pollTimer = new System.Windows.Forms.Timer { Interval = 2500 };
            _pollTimer.Tick += async (s, e) => await PollAllAsync();
            _pollTimer.Start();

            this.Shown += async (s, e) =>
            {
                await PollAllAsync();
                if (!_backendOnline) StartBackend();
            };
        }

        private Icon CreateTrayIcon(bool running)
        {
            using (var bmp = new Bitmap(32, 32))
            using (var g = Graphics.FromImage(bmp))
            {
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.Clear(Color.Transparent);
                using (var bg = new SolidBrush(running ? Color.FromArgb(28, 28, 28) : Color.FromArgb(58, 58, 58)))
                using (var ring = new Pen(running ? Color.FromArgb(230, 230, 230) : Color.FromArgb(130, 130, 130), 1.5f))
                {
                    g.FillEllipse(bg, 1, 1, 30, 30);
                    g.DrawEllipse(ring, 2, 2, 28, 28);
                }
                using (var white = new SolidBrush(Color.White))
                using (var pen = new Pen(Color.White, 2.0f))
                {
                    g.FillRectangle(white, 13, 7, 6, 9);
                    g.FillEllipse(white, 13, 5, 6, 5);
                    g.FillEllipse(white, 13, 13, 6, 5);
                    g.DrawArc(pen, 10, 10, 12, 11, 0, 180);
                    g.DrawLine(pen, 16, 21, 16, 25);
                    g.DrawLine(pen, 12, 25, 20, 25);
                }
                Color dotColor = running ? Color.FromArgb(46, 204, 113) : Color.FromArgb(231, 76, 60);
                using (var border = new SolidBrush(Color.FromArgb(15, 15, 15)))
                using (var dot = new SolidBrush(dotColor))
                {
                    g.FillEllipse(border, 21, 21, 11, 11);
                    g.FillEllipse(dot, 23, 23, 7, 7);
                }
                return Icon.FromHandle(bmp.GetHicon());
            }
        }

        private void BuildUI()
        {
            AutoScaleMode = AutoScaleMode.None;
            Text = "YouTube 唱K 控制台";
            Icon = _iconRunning;
            ClientSize = new Size(S(760), S(816));
            FormBorderStyle = FormBorderStyle.FixedSingle;
            MaximizeBox = false;
            StartPosition = FormStartPosition.CenterScreen;
            BackColor = BgDark;
            ForeColor = TextPrimary;
            Font = new Font("Microsoft JhengHei UI", 10f);

            int pad = 22;
            int w = 716;

            // Header Row
            Controls.Add(new Label
            {
                Text = "YouTube 唱K 控制台",
                Font = new Font("Microsoft JhengHei UI", 16f, FontStyle.Bold),
                ForeColor = TextPrimary,
                Location = new Point(S(pad), S(14)),
                AutoSize = true
            });

            _lblStatusPill = new Label
            {
                Text = "檢查中",
                Font = new Font("Microsoft JhengHei UI", 10f, FontStyle.Bold),
                ForeColor = Color.White,
                BackColor = BtnNeutral,
                TextAlign = ContentAlignment.MiddleCenter,
                Size = new Size(S(96), S(30)),
                Location = new Point(S(pad + w - 96), S(16))
            };
            Controls.Add(_lblStatusPill);

            // Top Action Row: Start/Stop + Auto-Update Chrome Extension
            _btnToggleServer = MakeButton("啟動唱K 服務", pad, 58, 352, 44, AccentWhite, Color.FromArgb(15, 15, 15));
            _btnToggleServer.Font = new Font("Microsoft JhengHei UI", 11f, FontStyle.Bold);
            _btnToggleServer.Click += (s, e) =>
            {
                if (_backendOnline || _starting) StopBackend();
                else StartBackend();
            };
            Controls.Add(_btnToggleServer);

            _btnAutoUpdateExt = MakeButton("一鍵自動更新 Chrome 插件", pad + 364, 58, 352, 44, BtnNeutral, TextPrimary);
            _btnAutoUpdateExt.Font = new Font("Microsoft JhengHei UI", 10.5f, FontStyle.Bold);
            _btnAutoUpdateExt.Click += async (s, e) => await TriggerAutoReloadExtensionAsync();
            Controls.Add(_btnAutoUpdateExt);

            // KPI Row (3 Cards)
            int kGap = 12;
            int kw = 230;
            int kwLast = w - (kw + kGap) * 2;

            var cardUsers = MakeCard(pad, 114, kw, 92);
            cardUsers.Controls.Add(MakeSectionLabel("在線連線 / 排隊", 14, 10));
            _lblUsersNum = new Label { Text = "0 人", Font = new Font("Microsoft JhengHei UI", 18f, FontStyle.Bold), ForeColor = TextPrimary, Location = new Point(S(14), S(32)), AutoSize = true };
            _lblUsersSub = new Label { Text = "0 唱K · 排隊 0", Font = new Font("Microsoft JhengHei UI", 9.5f), ForeColor = TextMuted, Location = new Point(S(14), S(66)), Size = new Size(S(kw - 24), S(20)) };
            cardUsers.Controls.AddRange(new Control[] { _lblUsersNum, _lblUsersSub });
            Controls.Add(cardUsers);

            var cardCacheKpi = MakeCard(pad + kw + kGap, 114, kw, 92);
            cardCacheKpi.Controls.Add(MakeSectionLabel("已快取伴奏", 14, 10));
            _lblCacheNum = new Label { Text = "0 首", Font = new Font("Microsoft JhengHei UI", 18f, FontStyle.Bold), ForeColor = TextPrimary, Location = new Point(S(14), S(32)), AutoSize = true };
            _lblCacheSub = new Label { Text = "本地 0 MB", Font = new Font("Microsoft JhengHei UI", 9.5f), ForeColor = TextMuted, Location = new Point(S(14), S(66)), Size = new Size(S(kw - 24), S(20)) };
            cardCacheKpi.Controls.AddRange(new Control[] { _lblCacheNum, _lblCacheSub });
            Controls.Add(cardCacheKpi);

            var cardVramKpi = MakeCard(pad + (kw + kGap) * 2, 114, kwLast, 92);
            cardVramKpi.Controls.Add(MakeSectionLabel("剩餘 VRAM", 14, 10));
            _lblVramNum = new Label { Text = "—", Font = new Font("Microsoft JhengHei UI", 18f, FontStyle.Bold), ForeColor = TextPrimary, Location = new Point(S(14), S(32)), AutoSize = true };
            _lblVramSub = new Label { Text = "主服務佔 0 MB", Font = new Font("Microsoft JhengHei UI", 9.5f), ForeColor = TextMuted, Location = new Point(S(14), S(66)), Size = new Size(S(kwLast - 24), S(20)) };
            cardVramKpi.Controls.AddRange(new Control[] { _lblVramNum, _lblVramSub });
            Controls.Add(cardVramKpi);

            // Multi-User Anti-Overload & GPU Control Card
            var cardOverload = MakeCard(pad, 218, w, 134);
            cardOverload.Controls.Add(MakeSectionLabel("多用戶防爆負載控制（確保幾十至百人連線唔會隊爆電腦）", 14, 10));

            _btnAuto = MakeButton("動態防爆", 14, 38, 116, 36, AccentWhite, Color.FromArgb(15, 15, 15));
            _btnGpu = MakeButton("強制 GPU", 138, 38, 116, 36, BtnNeutral, TextPrimary);
            _btnCpu = MakeButton("純 CPU", 262, 38, 104, 36, BtnNeutral, TextPrimary);
            _btnAuto.Click += async (s, e) => await SetDeviceModeAsync("auto");
            _btnGpu.Click += async (s, e) => await SetDeviceModeAsync("gpu");
            _btnCpu.Click += async (s, e) => await SetDeviceModeAsync("cpu");

            _btnPauseNew = MakeButton("接受新歌拆軌中", 378, 38, 324, 36, AccentGreen, Color.White);
            _btnPauseNew.Font = new Font("Microsoft JhengHei UI", 10f, FontStyle.Bold);
            _btnPauseNew.Click += async (s, e) => await TogglePauseNewJobsAsync();

            cardOverload.Controls.AddRange(new Control[] { _btnAuto, _btnGpu, _btnCpu, _btnPauseNew });

            // Numeric limits row
            cardOverload.Controls.Add(new Label { Text = "排隊上限:", Font = new Font("Microsoft JhengHei UI", 10f), ForeColor = TextMuted, Location = new Point(S(14), S(90)), AutoSize = true });
            _numMaxQueue = MakeNum(94, 87, 58, 1, 20, 3);
            cardOverload.Controls.Add(new Label { Text = "每人/10分:", Font = new Font("Microsoft JhengHei UI", 10f), ForeColor = TextMuted, Location = new Point(S(164), S(90)), AutoSize = true });
            _numRateLimit = MakeNum(256, 87, 58, 1, 50, 5);
            cardOverload.Controls.Add(new Label { Text = "人數上限:", Font = new Font("Microsoft JhengHei UI", 10f), ForeColor = TextMuted, Location = new Point(S(326), S(90)), AutoSize = true });
            _numMaxClients = MakeNum(406, 87, 70, 5, 500, 100);

            _btnApplyLimits = MakeButton("套用限流設定", 488, 84, 214, 36, BtnNeutral, TextPrimary);
            _btnApplyLimits.Font = new Font("Microsoft JhengHei UI", 10f);
            _btnApplyLimits.Click += async (s, e) => await ApplyOverloadLimitsAsync();

            cardOverload.Controls.AddRange(new Control[] { _numMaxQueue, _numRateLimit, _numMaxClients, _btnApplyLimits });
            Controls.Add(cardOverload);

            // Google Drive Cloud Storage Card
            var cardGdrive = MakeCard(pad, 364, w, 92);
            cardGdrive.Controls.Add(MakeSectionLabel("Google Drive 雲端快取（自動上傳並刪除本地檔・0 本地容量）", 14, 10));

            _lblGdriveStatus = new Label
            {
                Text = "Google Drive：檢查中...",
                ForeColor = TextMuted,
                Font = new Font("Microsoft JhengHei UI", 10f),
                Location = new Point(S(14), S(40)),
                Size = new Size(S(274), S(36)),
                TextAlign = ContentAlignment.MiddleLeft
            };
            cardGdrive.Controls.Add(_lblGdriveStatus);

            _btnConnectGdrive = MakeButton("連接 Google Drive", 294, 40, 152, 36, AccentWhite, Color.FromArgb(15, 15, 15));
            _btnConnectGdrive.Font = new Font("Microsoft JhengHei UI", 9.8f, FontStyle.Bold);
            _btnConnectGdrive.Click += async (s, e) => await ConnectGoogleDriveAsync();

            _btnMigrateGdrive = MakeButton("遷移本地上雲端", 454, 40, 138, 36, BtnNeutral, TextPrimary);
            _btnMigrateGdrive.Font = new Font("Microsoft JhengHei UI", 9.8f);
            _btnMigrateGdrive.Click += async (s, e) => await MigrateToGdriveAsync();

            _btnToggleStorage = MakeButton("本地模式", 600, 40, 102, 36, BtnNeutral, TextPrimary);
            _btnToggleStorage.Font = new Font("Microsoft JhengHei UI", 9.8f);
            _btnToggleStorage.Click += async (s, e) => await ToggleStorageModeAsync();

            cardGdrive.Controls.AddRange(new Control[] { _btnConnectGdrive, _btnMigrateGdrive, _btnToggleStorage });
            Controls.Add(cardGdrive);

            // Cached Songs List Card (with clickable YouTube links!)
            var cardSongs = MakeCard(pad, 468, w, 278);
            cardSongs.Controls.Add(MakeSectionLabel("已快取歌曲清單（雙擊即開 YouTube 連結）", 14, 10));

            _lvSongs = new ListView
            {
                Location = new Point(S(14), S(38)),
                Size = new Size(S(w - 28), S(182)),
                View = View.Details,
                FullRowSelect = true,
                MultiSelect = false,
                GridLines = false,
                BorderStyle = BorderStyle.FixedSingle,
                BackColor = Color.FromArgb(18, 18, 18),
                ForeColor = TextPrimary,
                Font = new Font("Microsoft JhengHei UI", 10f),
                SmallImageList = new ImageList { ImageSize = new Size(1, S(26)) }
            };
            _lvSongs.Columns.Add("歌名", S(290));
            _lvSongs.Columns.Add("模式", S(84));
            _lvSongs.Columns.Add("位置", S(76));
            _lvSongs.Columns.Add("容量", S(84));
            _lvSongs.Columns.Add("YouTube 連結", S(130));
            _lvSongs.DoubleClick += (s, e) => OpenSelectedSongLink();
            cardSongs.Controls.Add(_lvSongs);

            _btnOpenSong = MakeButton("開啟 YouTube", 14, 230, 134, 36, AccentWhite, Color.FromArgb(15, 15, 15));
            _btnOpenSong.Font = new Font("Microsoft JhengHei UI", 10f, FontStyle.Bold);
            _btnOpenSong.Click += (s, e) => OpenSelectedSongLink();

            _btnCopyLink = MakeButton("複製連結", 156, 230, 100, 36, BtnNeutral, TextPrimary);
            _btnCopyLink.Font = new Font("Microsoft JhengHei UI", 10f);
            _btnCopyLink.Click += (s, e) => CopySelectedSongLink();

            _btnDelSong = MakeButton("刪除此歌", 264, 230, 100, 36, Color.FromArgb(72, 28, 28), TextPrimary);
            _btnDelSong.Font = new Font("Microsoft JhengHei UI", 10f);
            _btnDelSong.Click += async (s, e) => await DeleteSelectedSongAsync();

            _btnClearAll = MakeButton("清空全部", 372, 230, 100, 36, Color.FromArgb(72, 28, 28), TextPrimary);
            _btnClearAll.Font = new Font("Microsoft JhengHei UI", 10f);
            _btnClearAll.Click += async (s, e) => await ClearAllCacheAsync();

            _btnToggleLogs = MakeButton("日誌 ▼", 602, 230, 100, 36, BtnNeutral, TextPrimary);
            _btnToggleLogs.Font = new Font("Microsoft JhengHei UI", 10f);
            _btnToggleLogs.Click += (s, e) => ToggleLogsDrawer();

            cardSongs.Controls.AddRange(new Control[] { _btnOpenSong, _btnCopyLink, _btnDelSong, _btnClearAll, _btnToggleLogs });
            Controls.Add(cardSongs);

            // Bottom Row
            var btnHideTray = MakeButton("縮小至系統列 (System Tray)", pad, 758, 516, 40, BtnNeutral, TextPrimary);
            btnHideTray.Font = new Font("Microsoft JhengHei UI", 10f);
            btnHideTray.Click += (s, e) => MinimizeToTray();
            Controls.Add(btnHideTray);

            var btnQuit = MakeButton("完全結束", pad + 526, 758, 190, 40, Color.FromArgb(72, 28, 28), TextPrimary);
            btnQuit.Font = new Font("Microsoft JhengHei UI", 10f);
            btnQuit.Click += (s, e) => QuitApplication();
            Controls.Add(btnQuit);

            // Expandable Logs
            _txtLogs = new TextBox
            {
                Multiline = true,
                ReadOnly = true,
                ScrollBars = ScrollBars.Vertical,
                BackColor = Color.FromArgb(10, 10, 10),
                ForeColor = Color.FromArgb(200, 200, 200),
                Font = new Font("Consolas", 9.5f),
                BorderStyle = BorderStyle.FixedSingle,
                Location = new Point(S(pad), S(812)),
                Size = new Size(S(w), S(156)),
                Visible = false
            };
            Controls.Add(_txtLogs);
        }

        private NumericUpDown MakeNum(int x, int y, int w, int min, int max, int val)
        {
            return new NumericUpDown
            {
                Location = new Point(S(x), S(y)),
                Size = new Size(S(w), S(28)),
                Minimum = min,
                Maximum = max,
                Value = val,
                Font = new Font("Microsoft JhengHei UI", 10f),
                BackColor = Color.FromArgb(35, 35, 35),
                ForeColor = TextPrimary,
                BorderStyle = BorderStyle.FixedSingle
            };
        }

        private Panel MakeCard(int x, int y, int w, int h)
        {
            var p = new Panel { Location = new Point(S(x), S(y)), Size = new Size(S(w), S(h)), BackColor = CardDark };
            p.Paint += (s, e) =>
            {
                using (var pen = new Pen(BorderSubtle, 1)) e.Graphics.DrawRectangle(pen, 0, 0, p.Width - 1, p.Height - 1);
            };
            return p;
        }

        private Label MakeSectionLabel(string text, int x, int y)
        {
            return new Label
            {
                Text = text,
                ForeColor = TextMuted,
                Font = new Font("Microsoft JhengHei UI", 10f, FontStyle.Bold),
                Location = new Point(S(x), S(y)),
                AutoSize = true
            };
        }

        private Button MakeButton(string text, int x, int y, int w, int h, Color bg, Color fg)
        {
            var btn = new Button
            {
                Text = text,
                Location = new Point(S(x), S(y)),
                Size = new Size(S(w), S(h)),
                FlatStyle = FlatStyle.Flat,
                BackColor = bg,
                ForeColor = fg,
                Font = new Font("Microsoft JhengHei UI", 10f),
                Cursor = Cursors.Hand
            };
            btn.FlatAppearance.BorderSize = 0;
            return btn;
        }

        private void BuildTray()
        {
            _trayMenu = new ContextMenuStrip();

            _trayStatsItem = new ToolStripMenuItem("連線：0 人 | 快取：0 首");
            _trayStatsItem.Enabled = false;
            _trayMenu.Items.Add(_trayStatsItem);
            _trayMenu.Items.Add(new ToolStripSeparator());

            var openItem = new ToolStripMenuItem("開啟控制台視窗");
            openItem.Font = new Font(openItem.Font, FontStyle.Bold);
            openItem.Click += (s, e) => RestoreFromTray();
            _trayMenu.Items.Add(openItem);

            _trayToggleItem = new ToolStripMenuItem("停止唱K 服務", null, (s, e) =>
            {
                if (_backendOnline || _starting) StopBackend();
                else StartBackend();
            });
            _trayMenu.Items.Add(_trayToggleItem);

            _trayPauseItem = new ToolStripMenuItem("暫停新歌拆軌（僅播快取）", null, async (s, e) => await TogglePauseNewJobsAsync());
            _trayMenu.Items.Add(_trayPauseItem);

            _trayMenu.Items.Add(new ToolStripMenuItem("一鍵自動更新 Chrome 插件", null, async (s, e) => await TriggerAutoReloadExtensionAsync()));

            _trayMenu.Items.Add(new ToolStripSeparator());

            var modeMenu = new ToolStripMenuItem("運算模式");
            _trayModeAuto = new ToolStripMenuItem("動態防爆 VRAM", null, async (s, e) => await SetDeviceModeAsync("auto"));
            _trayModeGpu = new ToolStripMenuItem("強制 GPU", null, async (s, e) => await SetDeviceModeAsync("gpu"));
            _trayModeCpu = new ToolStripMenuItem("純 CPU", null, async (s, e) => await SetDeviceModeAsync("cpu"));
            modeMenu.DropDownItems.AddRange(new ToolStripItem[] { _trayModeAuto, _trayModeGpu, _trayModeCpu });
            _trayMenu.Items.Add(modeMenu);

            _trayMenu.Items.Add(new ToolStripSeparator());
            _trayMenu.Items.Add(new ToolStripMenuItem("完全結束", null, (s, e) => QuitApplication()));

            _trayIcon = new NotifyIcon
            {
                Icon = _iconStopped,
                Text = "YouTube 唱K — 啟動中...",
                ContextMenuStrip = _trayMenu,
                Visible = true
            };
            _trayIcon.MouseClick += (s, e) =>
            {
                if (e.Button == MouseButtons.Left)
                {
                    if (Visible && WindowState != FormWindowState.Minimized) MinimizeToTray();
                    else RestoreFromTray();
                }
            };
        }

        private void SetTrayText(string text)
        {
            if (string.IsNullOrEmpty(text)) return;
            _trayIcon.Text = text.Length > 60 ? text.Substring(0, 60) : text;
        }

        private string ReadAdminToken()
        {
            try { if (File.Exists(TokenPath)) return File.ReadAllText(TokenPath, Encoding.UTF8).Trim(); }
            catch { }
            return "";
        }

        private void StartShowEventListener()
        {
            Task.Run(() =>
            {
                while (!_cts.IsCancellationRequested)
                {
                    try { if (_showWindowEvent.WaitOne(500) && !IsDisposed) BeginInvoke((Action)RestoreFromTray); }
                    catch { break; }
                }
            });
        }

        private void ToggleLogsDrawer()
        {
            _logsExpanded = !_logsExpanded;
            _txtLogs.Visible = _logsExpanded;
            _btnToggleLogs.Text = _logsExpanded ? "日誌 ▲" : "日誌 ▼";
            ClientSize = new Size(ClientSize.Width, _logsExpanded ? S(984) : S(816));
        }

        private void AppendLog(string line)
        {
            if (string.IsNullOrEmpty(line) || IsDisposed) return;
            try
            {
                BeginInvoke((Action)(() =>
                {
                    if (_txtLogs.TextLength > 20000) _txtLogs.Text = _txtLogs.Text.Substring(_txtLogs.TextLength - 12000);
                    _txtLogs.AppendText(line + Environment.NewLine);
                }));
            }
            catch { }
        }

        private void StartBackend()
        {
            lock (_procLock)
            {
                if (_backendProcess != null && !_backendProcess.HasExited) return;
                if (!File.Exists(PythonExe) || !File.Exists(MainPy))
                {
                    MessageBox.Show("搵唔到 Backend Python 環境：\n" + PythonExe, "啟動失敗", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
                _starting = true;
                UpdateOfflineUI();

                var psi = new ProcessStartInfo
                {
                    FileName = PythonExe,
                    Arguments = "-u \"" + MainPy + "\"",
                    WorkingDirectory = BackendDir,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    StandardOutputEncoding = Encoding.UTF8,
                    StandardErrorEncoding = Encoding.UTF8
                };
                psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
                psi.EnvironmentVariables["KARAOKE_DEVICE"] = _deviceMode;
                psi.EnvironmentVariables["KARAOKE_MIN_FREE_VRAM_MB"] = "1800";

                var proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
                proc.OutputDataReceived += (s, e) => AppendLog(e.Data);
                proc.ErrorDataReceived += (s, e) => AppendLog(e.Data);
                proc.Exited += (s, e) =>
                {
                    _starting = false;
                    _backendOnline = false;
                    AppendLog("[System] Backend process exited.");
                    if (!IsDisposed) BeginInvoke((Action)UpdateOfflineUI);
                };

                proc.Start();
                _jobTracker.AddProcess(proc);
                proc.BeginOutputReadLine();
                proc.BeginErrorReadLine();
                _backendProcess = proc;
                AppendLog("[System] Backend started (PID " + proc.Id + ").");
            }
        }

        private void StopBackend()
        {
            lock (_procLock)
            {
                _starting = false;
                if (_backendProcess != null)
                {
                    try
                    {
                        if (!_backendProcess.HasExited)
                        {
                            var kill = Process.Start(new ProcessStartInfo
                            {
                                FileName = "taskkill",
                                Arguments = "/T /F /PID " + _backendProcess.Id,
                                CreateNoWindow = true,
                                UseShellExecute = false
                            });
                            if (kill != null) kill.WaitForExit(3000);
                        }
                    }
                    catch { }
                    finally
                    {
                        try { _backendProcess.Dispose(); } catch { }
                        _backendProcess = null;
                    }
                }
                _backendOnline = false;
            }
            AppendLog("[System] Backend stopped.");
            UpdateOfflineUI();
        }

        private async Task PollAllAsync()
        {
            try
            {
                string healthRaw = await HttpRequestAsync("GET", "/api/health", null, 1800, false);
                if (healthRaw == null)
                {
                    _backendOnline = false;
                    UpdateOfflineUI();
                    return;
                }

                var h = _json.Deserialize<Dictionary<string, object>>(healthRaw);
                if (h == null || Convert.ToString(h["status"]) != "ok")
                {
                    _backendOnline = false;
                    UpdateOfflineUI();
                    return;
                }

                _backendOnline = true;
                _starting = false;
                _deviceMode = GetStr(h, "device_mode", "auto");
                _storageMode = GetStr(h, "storage_mode", "local");
                _gdriveConnected = GetBool(h, "gdrive_connected");
                _pauseNewJobs = GetBool(h, "pause_new_jobs");

                int activeClients = GetInt(h, "active_clients");
                int activeSinging = GetInt(h, "active_singing");
                int queueCount = GetInt(h, "queue_count");
                int cacheCount = GetInt(h, "cache_count");
                double localMb = GetDbl(h, "local_cache_mb");
                double gdriveMb = GetDbl(h, "gdrive_cache_mb");
                double freeVram = GetDbl(h, "vram_free_mb");
                double totalVram = GetDbl(h, "vram_total_mb");

                if (!_limitsInitialized)
                {
                    _limitsInitialized = true;
                    _numMaxQueue.Value = Math.Max(1, Math.Min(20, GetInt(h, "max_queue_size", 3)));
                    _numRateLimit.Value = Math.Max(1, Math.Min(50, GetInt(h, "rate_limit_per_10min", 5)));
                    _numMaxClients.Value = Math.Max(5, Math.Min(500, GetInt(h, "max_clients", 100)));
                }

                _lblStatusPill.Text = "運行中";
                _lblStatusPill.BackColor = AccentGreen;
                _btnToggleServer.Text = "停止唱K 服務";
                _btnToggleServer.BackColor = AccentRed;
                _btnToggleServer.ForeColor = Color.White;
                _trayToggleItem.Text = "停止唱K 服務";
                _trayIcon.Icon = _iconRunning;

                _lblUsersNum.Text = activeClients + " 人";
                _lblUsersSub.Text = string.Format("{0} 唱K · 排隊 {1}", activeSinging, queueCount);

                _lblCacheNum.Text = cacheCount + " 首";
                _lblCacheSub.Text = _gdriveConnected
                    ? string.Format("GDrive {0:F0}MB · 本地 {1:F0}MB", gdriveMb, localMb)
                    : string.Format("本地 {0:F1} MB", localMb);

                _lblVramNum.Text = totalVram > 0 ? string.Format("{0:F1} GB", freeVram / 1024.0) : "CPU";
                _lblVramSub.Text = totalVram > 0 ? string.Format("總計 {0:F0} GB · 主服務 0 MB", totalVram / 1024.0) : "無 CUDA 顯卡";

                bool migRunning = GetBool(h, "migration_running");
                string migMsg = GetStr(h, "migration_message", "");

                if (migRunning)
                {
                    _lblGdriveStatus.Text = string.IsNullOrEmpty(migMsg) ? "正在遷移上 Google Drive..." : migMsg;
                    _lblGdriveStatus.ForeColor = Color.FromArgb(250, 204, 21);
                    _btnMigrateGdrive.Enabled = false;
                    _btnMigrateGdrive.Text = "遷移中...";
                }
                else
                {
                    _lblGdriveStatus.Text = _gdriveConnected
                        ? (_storageMode == "gdrive" ? "Google Drive：雲端模式 (0 本地)" : "Google Drive：已連接")
                        : "Google Drive：未連接";
                    _lblGdriveStatus.ForeColor = _gdriveConnected ? Color.FromArgb(74, 222, 128) : TextMuted;
                    _btnMigrateGdrive.Enabled = _gdriveConnected && localMb > 0;
                    _btnMigrateGdrive.Text = localMb > 0 ? "遷移本地上雲端" : "已全部上雲端";
                }
                _btnConnectGdrive.Text = _gdriveConnected ? "重新同步 Drive" : "連接 Google Drive";
                _btnToggleStorage.Enabled = _gdriveConnected;
                _btnToggleStorage.Text = _storageMode == "gdrive" ? "雲端模式" : "本地模式";

                _btnPauseNew.Text = _pauseNewJobs ? "已暫停新拆歌（僅播快取）" : "接受新歌拆軌中";
                _btnPauseNew.BackColor = _pauseNewJobs ? AccentAmber : AccentGreen;
                _trayPauseItem.Text = _pauseNewJobs ? "恢復接受新歌拆軌" : "暫停新歌拆軌（僅播快取）";

                _trayStatsItem.Text = string.Format("連線：{0} 人 | 排隊：{1} | 快取：{2} 首", activeClients, queueCount, cacheCount);
                SetTrayText(string.Format("YouTube 唱K — 連線 {0} 人 | 快取 {1} 首", activeClients, cacheCount));
                UpdateModeButtons();

                // Refresh cached songs list ONLY when catalog_version changes (zero wasted polling at 10,000+ songs!)
                string catVer = GetStr(h, "catalog_version", "");
                if (string.IsNullOrEmpty(_lastCatalogVersion) || string.IsNullOrEmpty(catVer) || catVer != _lastCatalogVersion)
                {
                    string cacheRaw = await HttpRequestAsync("GET", "/api/cache?limit=250", null, 3000, false);
                    if (cacheRaw != null)
                    {
                        _lastCatalogVersion = catVer;
                        if (cacheRaw != _lastCacheSig)
                        {
                            _lastCacheSig = cacheRaw;
                            PopulateSongsList(cacheRaw);
                        }
                    }
                }
            }
            catch { }
        }

        private void PopulateSongsList(string cacheRaw)
        {
            try
            {
                var doc = _json.Deserialize<Dictionary<string, object>>(cacheRaw);
                if (doc == null || !doc.ContainsKey("items")) return;
                var items = doc["items"] as System.Collections.ArrayList;
                if (items == null) return;

                string prevKey = _lvSongs.SelectedItems.Count > 0 ? Convert.ToString(_lvSongs.SelectedItems[0].Tag) : null;
                _lvSongs.BeginUpdate();
                _lvSongs.Items.Clear();

                foreach (var obj in items)
                {
                    var d = obj as Dictionary<string, object>;
                    if (d == null) continue;
                    string vid = GetStr(d, "video_id", "");
                    string mode = GetStr(d, "mode", "pure");
                    string title = GetStr(d, "title", vid);
                    string storage = GetStr(d, "storage", "local");
                    double sizeMb = GetDbl(d, "size_mb");
                    string url = GetStr(d, "url", "https://www.youtube.com/watch?v=" + vid);

                    var lvi = new ListViewItem(title);
                    lvi.SubItems.Add(mode == "pure" ? "純伴奏" : "保留和音");
                    lvi.SubItems.Add(storage == "gdrive" ? "GDrive" : "本地");
                    lvi.SubItems.Add(string.Format("{0:F1} MB", sizeMb));
                    lvi.SubItems.Add(url);
                    lvi.Tag = vid + "|" + mode + "|" + url;
                    _lvSongs.Items.Add(lvi);

                    if (prevKey != null && Convert.ToString(lvi.Tag) == prevKey)
                    {
                        lvi.Selected = true;
                    }
                }
                _lvSongs.EndUpdate();
            }
            catch { }
        }

        private void UpdateOfflineUI()
        {
            if (IsDisposed) return;
            if (_starting)
            {
                _lblStatusPill.Text = "啟動中";
                _lblStatusPill.BackColor = AccentAmber;
                _btnToggleServer.Text = "啟動中（點擊取消）";
                _btnToggleServer.BackColor = AccentAmber;
                _btnToggleServer.ForeColor = Color.White;
                _trayToggleItem.Text = "取消啟動";
                _trayIcon.Icon = _iconStopped;
                SetTrayText("YouTube 唱K — 啟動中...");
            }
            else
            {
                _lblStatusPill.Text = "已停止";
                _lblStatusPill.BackColor = BtnNeutral;
                _btnToggleServer.Text = "啟動唱K 服務";
                _btnToggleServer.BackColor = AccentWhite;
                _btnToggleServer.ForeColor = Color.FromArgb(15, 15, 15);
                _trayToggleItem.Text = "啟動唱K 服務";
                _trayIcon.Icon = _iconStopped;
                _lblUsersNum.Text = "0 人";
                _lblUsersSub.Text = "服務已停止";
                _trayStatsItem.Text = "服務已停止";
                SetTrayText("YouTube 唱K — 已停止");
            }
            UpdateModeButtons();
        }

        private void UpdateModeButtons()
        {
            SetModeBtnStyle(_btnAuto, _deviceMode == "auto");
            SetModeBtnStyle(_btnGpu, _deviceMode == "gpu");
            SetModeBtnStyle(_btnCpu, _deviceMode == "cpu");
            _trayModeAuto.Checked = _deviceMode == "auto";
            _trayModeGpu.Checked = _deviceMode == "gpu";
            _trayModeCpu.Checked = _deviceMode == "cpu";
        }

        private void SetModeBtnStyle(Button btn, bool selected)
        {
            btn.BackColor = selected ? AccentWhite : BtnNeutral;
            btn.ForeColor = selected ? Color.FromArgb(15, 15, 15) : TextPrimary;
            btn.Font = selected ? _fontModeActive : _fontModeNormal;
        }

        private async Task SetDeviceModeAsync(string mode)
        {
            _deviceMode = mode;
            UpdateModeButtons();
            if (_backendOnline)
            {
                await HttpRequestAsync("POST", "/api/settings", "{\"device_mode\":\"" + mode + "\"}", 2500, true);
                await PollAllAsync();
            }
        }

        private async Task TogglePauseNewJobsAsync()
        {
            if (!_backendOnline) return;
            bool next = !_pauseNewJobs;
            await HttpRequestAsync("POST", "/api/settings", "{\"pause_new_jobs\":" + (next ? "true" : "false") + "}", 2500, true);
            await PollAllAsync();
        }

        private async Task ApplyOverloadLimitsAsync()
        {
            if (!_backendOnline) return;
            string payload = string.Format(
                "{{\"max_queue_size\":{0},\"rate_limit_per_10min\":{1},\"max_clients\":{2}}}",
                (int)_numMaxQueue.Value,
                (int)_numRateLimit.Value,
                (int)_numMaxClients.Value
            );
            string resp = await HttpRequestAsync("POST", "/api/settings", payload, 2500, true);
            if (resp != null)
            {
                _btnApplyLimits.Text = "已套用 ✓";
                await Task.Delay(1200);
                _btnApplyLimits.Text = "套用限流設定";
                await PollAllAsync();
            }
        }

        private async Task TriggerAutoReloadExtensionAsync()
        {
            if (!_backendOnline)
            {
                MessageBox.Show("請先啟動唱K 服務。", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }
            _btnAutoUpdateExt.Enabled = false;
            _btnAutoUpdateExt.Text = "正在通知 Chrome 自動更新...";
            string resp = await HttpRequestAsync("POST", "/api/admin/reload-extension", "{}", 3000, true);
            await Task.Delay(1200);
            _btnAutoUpdateExt.Text = resp != null ? "已觸發自動更新 ✓" : "更新失敗";
            await Task.Delay(1500);
            _btnAutoUpdateExt.Text = "一鍵自動更新 Chrome 插件";
            _btnAutoUpdateExt.Enabled = true;
        }

        private string FindRcloneExe()
        {
            string[] roots =
            {
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Microsoft\WinGet\Links"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Microsoft\WinGet\Packages"),
                @"C:\Program Files\rclone"
            };
            foreach (var r in roots)
            {
                if (!Directory.Exists(r)) continue;
                string direct = Path.Combine(r, "rclone.exe");
                if (File.Exists(direct)) return direct;
                try
                {
                    var found = Directory.GetFiles(r, "rclone.exe", SearchOption.AllDirectories);
                    if (found.Length > 0) return found[0];
                }
                catch { }
            }
            return "rclone.exe";
        }

        private async Task ConnectGoogleDriveAsync()
        {
            if (_gdriveConnected)
            {
                _btnConnectGdrive.Enabled = false;
                _btnConnectGdrive.Text = "同步中...";
                await HttpRequestAsync("POST", "/api/admin/gdrive/sync", "{}", 30000, true);
                _lastCacheSig = ""; _lastCatalogVersion = "";
                await PollAllAsync();
                _btnConnectGdrive.Enabled = true;
                return;
            }

            string rclone = FindRcloneExe();
            _btnConnectGdrive.Enabled = false;
            _btnConnectGdrive.Text = "請喺瀏覽器授權...";

            await Task.Run(() =>
            {
                try
                {
                    // Least-privilege scope=drive.file: can ONLY access files created by YT Karaoke!
                    var p = Process.Start(new ProcessStartInfo
                    {
                        FileName = rclone,
                        Arguments = "config create ytkaraoke_gdrive drive scope=drive.file",
                        UseShellExecute = false,
                        CreateNoWindow = true
                    });
                    if (p != null) p.WaitForExit(180000);
                }
                catch (Exception ex)
                {
                    AppendLog("[GDrive] Auth error: " + ex.Message);
                }
            });

            await HttpRequestAsync("POST", "/api/admin/gdrive/sync", "{}", 30000, true);
            _lastCacheSig = ""; _lastCatalogVersion = "";
            await PollAllAsync();
            _btnConnectGdrive.Enabled = true;

            if (_gdriveConnected)
            {
                var ans = MessageBox.Show(
                    "Google Drive 連接成功！（只限存取本程式建立嘅伴奏資料夾）\n\n要唔要即刻將現有本地快取歌曲全部遷移上 Google Drive 並釋放本地硬碟空間？",
                    "Google Drive 已連接",
                    MessageBoxButtons.YesNo,
                    MessageBoxIcon.Information
                );
                if (ans == DialogResult.Yes)
                {
                    await MigrateToGdriveAsync();
                }
            }
        }

        private async Task MigrateToGdriveAsync()
        {
            if (!_backendOnline || !_gdriveConnected) return;
            _btnMigrateGdrive.Enabled = false;
            _btnMigrateGdrive.Text = "啟動遷移...";
            string resp = await HttpRequestAsync("POST", "/api/admin/gdrive/migrate", "{}", 10000, true);
            if (resp == null)
            {
                MessageBox.Show("啟動遷移失敗，請打開「日誌」查看原因。", "錯誤", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                _btnMigrateGdrive.Enabled = true;
                _btnMigrateGdrive.Text = "遷移本地上雲端";
                return;
            }
            _lastCacheSig = ""; _lastCatalogVersion = "";
            await PollAllAsync();
        }

        private async Task ToggleStorageModeAsync()
        {
            if (!_backendOnline || !_gdriveConnected) return;
            string next = _storageMode == "gdrive" ? "local" : "gdrive";
            await HttpRequestAsync("POST", "/api/settings", "{\"storage_mode\":\"" + next + "\"}", 2500, true);
            await PollAllAsync();
        }

        private string[] GetSelectedSongParts()
        {
            if (_lvSongs.SelectedItems.Count == 0) return null;
            string tag = Convert.ToString(_lvSongs.SelectedItems[0].Tag);
            if (string.IsNullOrEmpty(tag)) return null;
            return tag.Split('|');
        }

        private void OpenSelectedSongLink()
        {
            var parts = GetSelectedSongParts();
            if (parts == null || parts.Length < 3) return;
            try { Process.Start(new ProcessStartInfo(parts[2]) { UseShellExecute = true }); } catch { }
        }

        private void CopySelectedSongLink()
        {
            var parts = GetSelectedSongParts();
            if (parts == null || parts.Length < 3) return;
            try { Clipboard.SetText(parts[2]); } catch { }
        }

        private async Task DeleteSelectedSongAsync()
        {
            var parts = GetSelectedSongParts();
            if (parts == null || parts.Length < 2 || !_backendOnline) return;
            await HttpRequestAsync("DELETE", "/api/cache/" + parts[0] + "?mode=" + parts[1], null, 15000, true);
            _lastCacheSig = ""; _lastCatalogVersion = "";
            await PollAllAsync();
        }

        private async Task ClearAllCacheAsync()
        {
            if (!_backendOnline) return;
            var res = MessageBox.Show("確定清空全部快取（包括本地及 Google Drive 伴奏）？", "清空快取", MessageBoxButtons.YesNo, MessageBoxIcon.Warning);
            if (res != DialogResult.Yes) return;
            await HttpRequestAsync("DELETE", "/api/cache", null, 30000, true);
            _lastCacheSig = ""; _lastCatalogVersion = "";
            await PollAllAsync();
        }

        private Task<string> HttpRequestAsync(string method, string path, string jsonBody, int timeoutMs, bool includeAdminToken)
        {
            return Task.Run(() =>
            {
                try
                {
                    var req = (HttpWebRequest)WebRequest.Create(ApiBase + path);
                    req.Method = method;
                    req.Timeout = timeoutMs;
                    req.ReadWriteTimeout = timeoutMs;
                    if (includeAdminToken)
                    {
                        string tok = ReadAdminToken();
                        if (!string.IsNullOrEmpty(tok)) req.Headers["X-Admin-Token"] = tok;
                    }
                    if (jsonBody != null)
                    {
                        byte[] bytes = Encoding.UTF8.GetBytes(jsonBody);
                        req.ContentType = "application/json";
                        req.ContentLength = bytes.Length;
                        using (var s = req.GetRequestStream()) s.Write(bytes, 0, bytes.Length);
                    }
                    using (var resp = (HttpWebResponse)req.GetResponse())
                    using (var sr = new StreamReader(resp.GetResponseStream(), Encoding.UTF8))
                    {
                        return sr.ReadToEnd();
                    }
                }
                catch { return null; }
            });
        }

        private static string GetStr(Dictionary<string, object> d, string k, string defVal)
        {
            return d.ContainsKey(k) && d[k] != null ? Convert.ToString(d[k]) : defVal;
        }

        private static int GetInt(Dictionary<string, object> d, string k, int defVal = 0)
        {
            try { return d.ContainsKey(k) && d[k] != null ? Convert.ToInt32(d[k]) : defVal; } catch { return defVal; }
        }

        private static double GetDbl(Dictionary<string, object> d, string k, double defVal = 0)
        {
            try { return d.ContainsKey(k) && d[k] != null ? Convert.ToDouble(d[k]) : defVal; } catch { return defVal; }
        }

        private static bool GetBool(Dictionary<string, object> d, string k, bool defVal = false)
        {
            try { return d.ContainsKey(k) && d[k] != null ? Convert.ToBoolean(d[k]) : defVal; } catch { return defVal; }
        }

        private void MinimizeToTray()
        {
            Hide();
            if (!_trayHintShown)
            {
                _trayHintShown = true;
                _trayIcon.ShowBalloonTip(2200, "YouTube 唱K", "已縮小至系統列（System Tray），點一下圖示即可重新打開。", ToolTipIcon.Info);
            }
        }

        private void RestoreFromTray()
        {
            Show();
            WindowState = FormWindowState.Normal;
            Activate();
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            if (!_isExiting && e.CloseReason == CloseReason.UserClosing)
            {
                e.Cancel = true;
                MinimizeToTray();
                return;
            }
            base.OnFormClosing(e);
        }

        private void QuitApplication()
        {
            _isExiting = true;
            _cts.Cancel();
            _pollTimer.Stop();
            StopBackend();
            _trayIcon.Visible = false;
            _trayIcon.Dispose();
            _jobTracker.Dispose();
            Application.Exit();
        }
    }
}
