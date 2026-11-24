function apiRequest(path, method = "GET", body = null) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ type: "API_REQUEST", path, method, body }, (resp) => {
      if (chrome.runtime.lastError) resolve({ ok: false });
      else resolve(resp || { ok: false });
    });
  });
}

async function refresh() {
  const pill = document.getElementById("status-pill");
  const songCount = document.getElementById("song-count");
  const songList = document.getElementById("song-list");

  const health = await apiRequest("/api/health");
  if (!health.ok || !health.data || health.data.status !== "ok") {
    pill.textContent = "未連線";
    pill.className = "pill";
    songCount.textContent = "0 首";
    songList.textContent = "";
    const msg = document.createElement("div");
    msg.className = "hint";
    msg.textContent = "暫時未能連線至唱K 服務。";
    songList.appendChild(msg);
    return;
  }

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

refresh();
