const promptInput = document.getElementById("prompt");
const prompt2Input = document.getElementById("prompt_2");
const negativePromptInput = document.getElementById("negative_prompt");
const base_model = document.getElementById("base_model");
const second_pass_model = document.getElementById("second_pass_model");
const second_pass_mode = document.getElementById("second_pass_mode");
const steps = document.getElementById("steps");
const cfg = document.getElementById("cfg");
const width = document.getElementById("width");
const height = document.getElementById("height");
const num_images = document.getElementById("num_images");
const secondPassSection = document.getElementById("second-pass-section");

let lastSeed = null;
let previewState = {
  original: null,
  upscaled: null,
  mode: "original",
  wipe: 50,
};

const STORAGE_KEY = "webbduck_state";

function saveState() {
  const state = {
    prompt: promptInput.value,
    prompt_2: prompt2Input.value,
    negative_prompt: negativePromptInput.value,
    base_model: base_model.value,
    second_pass_model: second_pass_model.value,
    second_pass_mode: second_pass_mode.value,
    steps: steps.value,
    cfg: cfg.value,
    width: width.value,
    height: height.value,
    num_images: num_images.value,
    seed: getSeed(),
    experimental_compress: document.getElementById("experimental_compress").checked,
    loras: Array.from(document.querySelectorAll(".lora-card")).map(card => ({
      name: card.dataset.lora,
      weight: parseFloat(card.querySelector("input[type=range]").value),
      enabled: card.querySelector("input[type=checkbox]").checked,
    }))
  };

  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function loadState() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function updateSecondPassVisibility() {
  if (!secondPassSection || !second_pass_mode) return;

  const model = second_pass_model.value;
  const enabled = model && model !== "None";

  secondPassSection.style.display = enabled ? "block" : "none";

  if (!enabled) {
    second_pass_mode.value = "auto";
  }
}

second_pass_model.addEventListener("change", () => {
  updateSecondPassVisibility();
  saveState();
});

const seedMode = document.getElementById("seed_mode");
const seedInput = document.getElementById("seed_input");

seedMode.onchange = () => {
  seedInput.disabled = seedMode.value !== "manual";
};

async function populateSelect(id, url, includeNone = true) {
  const data = await fetch(url).then(r => r.json());
  const el = document.getElementById(id);

  el.innerHTML = includeNone ? `<option value="None">None</option>` : "";

  data.forEach(item => {
    if (typeof item === "string") {
      el.innerHTML += `<option value="${item}">${item}</option>`;
    } else {
      el.innerHTML += `
        <option value="${item.name}">
          ${item.name}${item.description ? " — " + item.description : ""}
        </option>`;
    }
  });
}

async function updateTokenCount(el, which) {
  const fd = new FormData();
  fd.append("text", el.value);
  fd.append("base_model", base_model.value);
  fd.append("which", which);

  const r = await fetch("/tokenize", { method: "POST", body: fd });
  const data = await r.json();

  el.dataset.tokens = data.tokens;
  el.dataset.over = data.over;

  el.style.outline = data.tokens > 77 ? "2px solid #ff7b72" : "";
}

[
  [promptInput, "prompt"],
  [prompt2Input, "prompt_2"],
  [negativePromptInput, "negative"],
].forEach(([el, which]) => {
  let t;
  el.addEventListener("input", () => {
    clearTimeout(t);
    t = setTimeout(() => updateTokenCount(el, which), 200);
  });
});

function renderPreview() {
  const el = document.getElementById("preview");

  if (!previewState.original) {
    el.innerHTML = "<span>No image yet</span>";
    return;
  }

  let content = "";

  if (previewState.mode === "wipe" && previewState.upscaled) {
    content = `
      <div class="preview-wipe" id="wipeContainer" style="--wipe:${previewState.wipe}%">
        <div class="wipe-canvas">
          <img class="wipe-img base" src="${previewState.original}">
          <img class="wipe-img overlay" src="${previewState.upscaled}">
        </div>
        <div class="wipe-handle"></div>
      </div>
    `;
  }
  else if (previewState.mode === "compare" && previewState.upscaled) {
    content = `
      <div class="preview-compare">
        <img src="${previewState.original}">
        <img src="${previewState.upscaled}">
      </div>
    `;
  }
  else {
    const img =
      previewState.mode === "upscaled" && previewState.upscaled
        ? previewState.upscaled
        : previewState.original;

    content = `
      <div class="preview-inner">
        <img src="${img}">
      </div>
    `;
  }

  el.innerHTML = `
    ${content}

    <div class="preview-toolbar">
      <button data-mode="original">Original</button>
      <button data-mode="upscaled" ${!previewState.upscaled ? "disabled" : ""}>
        Upscaled
      </button>
      <button data-mode="compare" ${!previewState.upscaled ? "disabled" : ""}>
        Side-by-side
      </button>
      <button data-mode="wipe" ${!previewState.upscaled ? "disabled" : ""}>
        Wipe
      </button>
      <button id="upscaleBtn">🔼 Upscale ×2</button>
    </div>
  `;

  document.querySelectorAll(".preview-toolbar button[data-mode]")
    .forEach(btn => {
      btn.onclick = () => {
        previewState.mode = btn.dataset.mode;
        renderPreview();
      };
    });

  const container = document.getElementById("wipeContainer");
  const handle = container?.querySelector(".wipe-handle");

  if (container) {
    let dragging = false;

    const updateFromX = (x) => {
      const rect = container.getBoundingClientRect();
      let pct = ((x - rect.left) / rect.width) * 100;
      pct = Math.max(0, Math.min(100, pct));

      previewState.wipe = pct;
      container.style.setProperty("--wipe", pct + "%");
    };

    container.onmousedown = (e) => {
      e.preventDefault();
      dragging = true;
      updateFromX(e.clientX);
    };

    container.onmousemove = (e) => {
      if (dragging) updateFromX(e.clientX);
    };

    window.onmouseup = () => dragging = false;

    container.ontouchstart = (e) => {
      e.preventDefault();
      dragging = true;
      updateFromX(e.touches[0].clientX);
    };

    container.ontouchmove = (e) => {
      if (dragging) updateFromX(e.touches[0].clientX);
    };

    window.ontouchend = () => dragging = false;
  }

  document.getElementById("upscaleBtn").onclick = runUpscale;
}

base_model.addEventListener("change", loadLoras);

const loraSelect = document.getElementById("lora-select");
const loraSelected = document.getElementById("lora-selected");

let loraDefaults = {};
let modelDefaults = {};

function applyModelDefaults() {
  const name = base_model.value;
  const defs = modelDefaults[name];
  if (!defs) return;

  if (defs.steps) steps.value = defs.steps;
  if (defs.cfg) cfg.value = defs.cfg;
  if (defs.width) width.value = defs.width;
  if (defs.height) height.value = defs.height;

  // Trigger input event to update local storage
  saveState();
}

async function loadLoras() {
  const baseModel = base_model.value;
  const loras = await fetch(
    `/models/${encodeURIComponent(baseModel)}/loras`
  ).then(r => r.json());

  loraDefaults = {};

  loraSelect.innerHTML =
    `<option value="" disabled selected>➕ Add LoRA…</option>`;

  loras.forEach(l => {
    loraDefaults[l.name] = l.weight ?? 1.0;

    const opt = document.createElement("option");
    opt.value = l.name;
    opt.textContent = l.name;
    loraSelect.appendChild(opt);
  });
}

function addLoraCard(name, weight = 1.0, enabled = true) {
  const card = document.createElement("div");
  card.className = "lora-card";
  card.dataset.lora = name;

  card.innerHTML = `
    <div class="lora-card-header">
      <input type="checkbox" ${enabled ? "checked" : ""}>
      <div class="lora-name">${name}</div>
      <div class="lora-remove">✕</div>
    </div>

    <div class="lora-weight">
      <input
        type="range"
        min="0"
        max="2"
        step="0.05"
        value="${weight}"
      >
      <span>${Number(weight).toFixed(2)}</span>
    </div>
  `;

  const slider = card.querySelector("input[type=range]");
  const value = card.querySelector("span");

  slider.oninput = () => {
    value.textContent = Number(slider.value).toFixed(2);
    saveState();
  };

  card.querySelector("input[type=checkbox]").onchange = saveState;

  card.querySelector(".lora-remove").onclick = () => {
    card.remove();
    saveState();
  };

  loraSelected.appendChild(card);
}

loraSelect.onchange = () => {
  const name = loraSelect.value;
  if (!name) return;

  if (document.querySelector(`[data-lora="${name}"]`)) {
    loraSelect.value = "";
    return;
  }

  const saved = loadState()?.loras?.find(l => l.name === name);

  addLoraCard(
    name,
    saved?.weight ?? loraDefaults[name] ?? 1.0,
    saved?.enabled ?? true
  );
  loraSelect.value = "";
  saveState();
};

async function send(endpoint) {
  try {
    const d = new FormData();

    ["steps", "cfg", "width", "height", "num_images"].forEach(id => {
      const el = document.getElementById(id);
      if (el) d.append(id, el.value);
    });

    const compress = document.getElementById("experimental_compress").checked;
    d.append("experimental_compress", compress);
    d.append("prompt", promptInput.value);
    d.append("prompt_2", prompt2Input.value);
    d.append("negative_prompt", negativePromptInput.value);
    d.append("base_model", base_model.value);
    d.append("second_pass_model", second_pass_model.value);
    d.append("second_pass_mode", second_pass_mode.value);

    if (seedMode.value === "reuse" && lastSeed !== null) {
      d.append("seed", lastSeed);
    }

    if (seedMode.value === "manual" && seedInput.value) {
      d.append("seed", seedInput.value);
    }

    const loras = [];

    document.querySelectorAll(".lora-card").forEach(card => {
      const enabled = card.querySelector("input[type=checkbox]").checked;
      if (!enabled) return;

      loras.push({
        name: card.dataset.lora,
        weight: parseFloat(card.querySelector("input[type=range]").value)
      });
    });

    d.append("loras", JSON.stringify(loras));

    const r = await fetch(endpoint, { method: "POST", body: d });
    const data = await r.json();

    if (data.seed !== undefined) {
      setSeed(data.seed);
      saveState();
    }

    if (data?.type === "validation_error") {
      alert(data.error);
      throw new Error(data.error);
    }

    return data;

  } catch (e) {
    console.error("Generation failed:", e);
    alert("Generation failed – check console");
  }
}

function getSeed() {
  const v = seedInput.value;
  return v === "" ? null : Number(v);
}

function setSeed(v) {
  seedInput.value = v ?? "";
  lastSeed = v;
}

async function runUpscale() {
  if (!previewState.original) return;

  const btn = document.getElementById("upscaleBtn");
  btn.disabled = true;
  btn.innerText = "🔼 Upscaling…";

  const fd = new FormData();
  fd.append("image", previewState.original);
  fd.append("scale", 2);

  try {
    const r = await fetch("/upscale", {
      method: "POST",
      body: fd
    });

    const data = await r.json();

    previewState.upscaled = data.image;
    previewState.mode = "upscaled";

    renderPreview();

  } catch (e) {
    alert("Upscale failed – check server logs");
    console.error(e);
  } finally {
    btn.disabled = false;
    btn.innerText = "🔼 Upscale ×2";
  }
}

document.getElementById("testBtn").onclick = async () => {
  const r = await send("/test");
  if (!r) return;

  previewState = {
    original: r.images[0],
    upscaled: null,
    mode: "original",
  };

  renderPreview();
};

document.getElementById("genBtn").onclick = async () => {
  const r = await send("/generate");
  if (!r) return;

  const g = document.getElementById("gallery");
  g.innerHTML = "";

  r.images.forEach(i => {
    const img = document.createElement("img");
    img.src = i;
    img.onclick = () => {
      previewState = {
        original: i,
        upscaled: null,
        mode: "original",
      };

      renderPreview();
    };
    g.appendChild(img);
  });
};

[
  promptInput,
  prompt2Input,
  negativePromptInput,
  steps,
  cfg,
  width,
  height,
  num_images,
  second_pass_mode,
].forEach(el => el.addEventListener("input", saveState));

base_model.addEventListener("change", () => {
  applyModelDefaults();
  loadLoras();
  saveState();
});
second_pass_model.addEventListener("change", saveState);
document.getElementById("experimental_compress").addEventListener("change", saveState);

window.addEventListener("DOMContentLoaded", async () => {

  // Load base models manually to capture defaults
  const models = await fetch("/models").then(r => r.json());
  base_model.innerHTML = "";

  models.forEach(m => {
    // Handle both old string format (just in case) and new object format
    const name = typeof m === "string" ? m : m.name;
    const defaults = typeof m === "string" ? {} : (m.defaults || {});

    modelDefaults[name] = defaults;

    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    base_model.appendChild(opt);
  });

  await populateSelect("second_pass_model", "/second_pass_models", true);


  const state = loadState();

  if (state) {
    if (state.base_model) base_model.value = state.base_model;
    second_pass_model.value = state.second_pass_model || "None";
    second_pass_mode.value = state.second_pass_mode || "auto";

    updateSecondPassVisibility();

    promptInput.value = state.prompt ?? "";
    prompt2Input.value = state.prompt_2 ?? "";
    negativePromptInput.value = state.negative_prompt ?? "";
    steps.value = state.steps ?? 30;
    cfg.value = state.cfg ?? 7.5;
    width.value = state.width ?? 1024;
    height.value = state.height ?? 1024;
    num_images.value = state.num_images ?? 4;
    setSeed(state.seed);

    if (state.experimental_compress) {
      document.getElementById("experimental_compress").checked = true;
    }

    await loadLoras();

    (state.loras || []).forEach(l => {
      addLoraCard(
        l.name,
        l.weight ?? 1.0,
        l.enabled ?? true
      );
    });
  } else {
    await loadLoras();
  }

  updateSecondPassVisibility();
  saveState();
});