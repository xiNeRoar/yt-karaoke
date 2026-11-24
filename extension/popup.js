function sendBg(payload) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(payload, (resp) => {
      if (chrome.runtime.lastError) resolve({ ok: false });
      else resolve(resp || { ok: false });
    });
  });
}

function apiRequest(path, method = "GET", body = null) {
  return sendBg({ type: "API_REQUEST", path, method, body });
}

async function refresh() {
  const pill = document.getElementById("status-pill");
  const authCard = document.getElementById("auth-card");
  const songsCard = document.getElementById("songs-card");
  const songCount = document.getElementById("song-count");
  const songList = document.getElementById("song-list");
  const switchBtn = document.getElementById("switch-code-btn");

  const health = await apiRequest("/api/health");
  if (!health.ok || !health.data || health.data.status !== "ok") {
    pill.textContent = "未連線";
    pill.className = "pill";
    authCard.classList.remove("visible");
    songsCard.style.display = "block";
    songCount.textContent = "0 首";
    songList.textContent = "";
    const msg = document.createElement("div");
    msg.className = "hint";
    msg.textContent = "暫時未能連線至唱K 服務。";
    songList.appendChild(msg);
    return;
  }

  if (health.data.auth_required && health.data.authenticated === false) {
    pill.textContent = "需要邀請碼";
    pill.className = "pill warn";
    authCard.classList.add("visible");
    songsCard.style.display = "none";
    return;
  }

  authCard.classList.remove("visible");
  songsCard.style.display = "block";
  switchBtn.style.display = health.data.auth_required ? "inline-block" : "none";

  if (health.data.gpu_online === false) {
    pill.textContent = "AI 休息中 · 歌庫在線";
    pill.className = "pill online";
  } else {
    pill.textContent = "已就緒";
    pill.className = "pill online";
  }

  const cacheRes = await apiRequest("/api/cache?limit=50");
  songList.textContent = "";
  if (cacheRes.ok && cacheRes.data) {
    const items = (cacheRes.data.items || []).filter((i) => i.mode === "pure");
    const totalCount = typeof cacheRes.data.count === "number" ? cacheRes.data.count : items.length;
    songCount.textContent = `${totalCount} 首`;
    if (items.length === 0) {
      const empty = document.createElement("div");
      empty.className = "hint";
      empty.textContent = "仲未有唱過嘅歌。去 YouTube 開首 MV 試吓啦！";
      songList.appendChild(empty);
    } else {
      for (const item of items) {
        const link = document.createElement("a");
        link.className = "song-item";
        link.href = `https://www.youtube.com/watch?v=${encodeURIComponent(item.video_id)}`;
        link.target = "_blank";

        const title = document.createElement("span");
        title.className = "song-title";
        title.title = item.title;
        title.textContent = item.title;

        link.appendChild(title);
        songList.appendChild(link);
      }
    }
  }
}

document.getElementById("invite-btn").addEventListener("click", async () => {
  const input = document.getElementById("invite-input");
  const btn = document.getElementById("invite-btn");
  const errEl = document.getElementById("invite-err");
  const code = (input.value || "").trim();
  if (!code) {
    errEl.textContent = "請先輸入邀請碼";
    return;
  }
  btn.disabled = true;
  btn.textContent = "驗證中...";
  errEl.textContent = "";
  const res = await sendBg({ type: "AUTH_ACTIVATE", code });
  btn.disabled = false;
  btn.textContent = "啟用唱K 服務";
  if (res.ok) {
    input.value = "";
    await refresh();
  } else {
    errEl.textContent = res.error || "邀請碼無效或已達裝置上限";
  }
});

document.getElementById("invite-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("invite-btn").click();
});

document.getElementById("switch-code-btn").addEventListener("click", async () => {
  await sendBg({ type: "AUTH_CLEAR" });
  await refresh();
});

refresh();
