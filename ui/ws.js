const ws = new WebSocket(`ws://${location.host}/ws`);

ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);

  if (msg.type !== "state") return;

  document.getElementById("stage").innerText = msg.stage;

  document.getElementById("progress-bar").style.width =
    `${Math.round(msg.progress * 100)}%`;

  if (msg.vram) {
    document.getElementById("vram").innerText =
      `VRAM: ${msg.vram.used.toFixed(1)} / ${msg.vram.total.toFixed(1)} GB`;
  }
};
