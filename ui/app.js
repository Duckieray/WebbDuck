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
const scheduler = document.getElementById("scheduler");
const secondPassSection = document.getElementById("second-pass-section");

// Image Upload Elements
const imageUploadArea = document.getElementById("image-upload-area");
const inputImageInput = document.getElementById("input_image");
const inputImagePreview = document.getElementById("input_image_preview");
const uploadPlaceholder = document.getElementById("upload_placeholder");
const clearInputImageBtn = document.getElementById("clear_input_image");
const img2imgSettings = document.getElementById("img2img-settings");
const strengthSlider = document.getElementById("strength");
const strengthVal = document.getElementById("strength_val");
const baseModelLabel = document.getElementById("base-model-label");
const secondPassModelSection = document.getElementById("second-pass-model-section");
let selectedImageFile = null;

// Caption controls
const captionControls = document.getElementById("caption-controls");
const captionStyleSelect = document.getElementById("caption_style");
const captionBtn = document.getElementById("caption_btn");
let captioningAvailable = false;

// Refinement Strength
const refinementStrength = document.getElementById("refinement_strength");
const refinementStrengthVal = document.getElementById("refinement_strength_val");

refinementStrength.oninput = () => {
  refinementStrengthVal.textContent = refinementStrength.value;
  saveState();
};

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
    })),
    scheduler: scheduler.value,
    refinement_strength: refinementStrength.value,
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


// --- MASK EDITOR LOGIC ---
let currentMaskBlob = null;

const maskOverlay = document.getElementById("mask_overlay");
const maskCanvas = document.getElementById("mask_canvas");
const maskCtx = maskCanvas.getContext("2d");
const maskBrushSize = document.getElementById("mask_brush_size");
const maskBrushPreview = document.getElementById("mask_brush_preview");
const maskClearBtn = document.getElementById("mask_clear_btn");
const maskInvertBtn = document.getElementById("mask_invert_btn");
const maskCancelBtn = document.getElementById("mask_cancel_btn");
const maskSaveBtn = document.getElementById("mask_save_btn");
const editMaskBtn = document.getElementById("edit_mask_btn");

let isDrawing = false;
let lastX = 0;
let lastY = 0;

function updateBrushPreview() {
  const size = maskBrushSize.value;
  maskBrushPreview.style.width = size + "px";
  maskBrushPreview.style.height = size + "px";
}

maskBrushSize.addEventListener("input", updateBrushPreview);
updateBrushPreview();

function openMaskEditor() {
  if (!selectedImageFile) return;

  // Use the main preview area as the mask editor
  previewState.mode = "mask";
  renderPreview();
}

function closeMaskEditor() {
  maskOverlay.style.display = "none";
}

editMaskBtn.onclick = (e) => {
  e.stopPropagation();
  openMaskEditor();
};

maskCancelBtn.onclick = closeMaskEditor;

maskClearBtn.onclick = () => {
  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
};

maskInvertBtn.onclick = () => {
  // Invert alpha/white logic? 
  // We are drawing WHITE on TRANSPARENT.
  // Inverting would mean making transparent->white and white->transparent.

  const w = maskCanvas.width;
  const h = maskCanvas.height;

  const imageData = maskCtx.getImageData(0, 0, w, h);
  const data = imageData.data;

  for (let i = 0; i < data.length; i += 4) {
    // Alpha determines the mask
    // Standard: 0 (Transparent) -> Unmasked, 255 (Opaque White) -> Masked
    // Invert: 0 -> 255, 255 -> 0
    // We only care about Alpha really if we use white brush

    const alpha = data[i + 3];
    data[i + 3] = 255 - alpha;

    // Ensure color is white
    data[i] = 255;
    data[i + 1] = 255;
    data[i + 2] = 255;
  }

  maskCtx.putImageData(imageData, 0, 0);
};

// Drawing Logic
function getPos(e) {
  const rect = maskCanvas.getBoundingClientRect();
  const scaleX = maskCanvas.width / rect.width;
  const scaleY = maskCanvas.height / rect.height;

  // Standardize mouse/touch
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;

  return {
    x: (clientX - rect.left) * scaleX,
    y: (clientY - rect.top) * scaleY
  };
}

function draw(e) {
  if (!isDrawing) return;
  e.preventDefault(); // Prevent scrolling on touch

  const { x, y } = getPos(e);

  maskCtx.beginPath();
  maskCtx.moveTo(lastX, lastY);
  maskCtx.lineTo(x, y);
  maskCtx.strokeStyle = "rgba(255, 255, 255, 1)";
  maskCtx.lineCap = "round";
  maskCtx.lineJoin = "round";
  maskCtx.lineWidth = maskBrushSize.value; // Use scale?
  // Since we scaled coordinates, standard line width applies to internal resolution
  // We might want to scale linewidth if canvas resolution is huge
  // but `value` is screen pixels.
  // Actually line width is in canvas coordinate space.
  const rect = maskCanvas.getBoundingClientRect();
  const scale = maskCanvas.width / rect.width;
  maskCtx.lineWidth = maskBrushSize.value * scale;

  maskCtx.stroke();

  lastX = x;
  lastY = y;
}

maskCanvas.addEventListener("mousedown", (e) => {
  isDrawing = true;
  const pos = getPos(e);
  lastX = pos.x;
  lastY = pos.y;
  // Dot logic
  draw(e);
});

maskCanvas.addEventListener("mousemove", draw);
window.addEventListener("mouseup", () => isDrawing = false);

// Touch support
maskCanvas.addEventListener("touchstart", (e) => {
  isDrawing = true;
  const pos = getPos(e);
  lastX = pos.x;
  lastY = pos.y;
  draw(e);
}, { passive: false });
maskCanvas.addEventListener("touchmove", draw, { passive: false });
window.addEventListener("touchend", () => isDrawing = false);

maskSaveBtn.onclick = () => {
  maskCanvas.toBlob((blob) => {
    currentMaskBlob = blob;
    // Visual indicator that mask is present?
    editMaskBtn.style.color = "var(--green)";
    closeMaskEditor();
  });
};

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

  if (id === "scheduler" && !includeNone) {
    // Set default if available
    if (el.options.length > 0) el.selectedIndex = 0;
  }
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

  // --- MASK EDITING MODE ---
  if (previewState.mode === "mask" && selectedImageFile) {
    el.innerHTML = `
      <div class="preview-mask-editor">
        <canvas id="preview_mask_canvas"></canvas>
      </div>

      <div class="preview-toolbar mask-toolbar-inline">
        <div class="mask-tools-inline">
          <label>Brush</label>
          <input type="range" id="preview_brush_size" min="5" max="100" value="30">
        </div>
        <div class="mask-tools-inline">
          <label>Mode</label>
          <select id="preview_inpainting_fill">
            <option value="replace">Replace Masked</option>
            <option value="keep">Keep Masked</option>
          </select>
        </div>
        <div class="mask-tools-inline">
          <label>Blur</label>
          <input type="range" id="preview_mask_blur" min="0" max="64" value="8" style="width:60px">
        </div>
        <button id="maskClearBtn">Clear</button>
        <button id="maskInvertBtn">Invert</button>
        <button id="maskSaveBtn" class="primary">Save Mask</button>
        <button id="maskCancelBtn">Cancel</button>
      </div>
    `;

    const canvas = document.getElementById("preview_mask_canvas");
    const ctx = canvas.getContext("2d");
    const brushSlider = document.getElementById("preview_brush_size");

    // Load image and size canvas to fill preview
    const img = new Image();
    img.onload = () => {
      const container = el.querySelector(".preview-mask-editor");
      const maxW = container.clientWidth;
      const maxH = container.clientHeight - 10; // Leave some padding

      let w = img.width;
      let h = img.height;
      const ratio = w / h;

      // Scale UP to fill the container (not down)
      if (ratio > maxW / maxH) {
        // Image is wider than container ratio
        w = maxW;
        h = w / ratio;
      } else {
        // Image is taller than container ratio
        h = maxH;
        w = h * ratio;
      }

      canvas.width = w;
      canvas.height = h;
      canvas.style.width = w + "px";
      canvas.style.height = h + "px";
      canvas.style.backgroundImage = `url(${inputImagePreview.src})`;
      canvas.style.backgroundSize = "100% 100%";
      ctx.clearRect(0, 0, w, h);
    };
    img.src = inputImagePreview.src;

    // Drawing state
    let drawing = false;
    let lx = 0, ly = 0;

    function getPos(e) {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const cx = e.touches ? e.touches[0].clientX : e.clientX;
      const cy = e.touches ? e.touches[0].clientY : e.clientY;
      return { x: (cx - rect.left) * scaleX, y: (cy - rect.top) * scaleY };
    }

    function drawLine(e) {
      if (!drawing) return;
      e.preventDefault();
      const { x, y } = getPos(e);
      ctx.beginPath();
      ctx.moveTo(lx, ly);
      ctx.lineTo(x, y);
      ctx.strokeStyle = "rgba(255,255,255,1)";
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      const rect = canvas.getBoundingClientRect();
      ctx.lineWidth = brushSlider.value * (canvas.width / rect.width);
      ctx.stroke();
      lx = x; ly = y;
    }

    canvas.onmousedown = (e) => { drawing = true; const p = getPos(e); lx = p.x; ly = p.y; drawLine(e); };
    canvas.onmousemove = drawLine;
    window.addEventListener("mouseup", () => drawing = false);
    canvas.ontouchstart = (e) => { drawing = true; const p = getPos(e); lx = p.x; ly = p.y; drawLine(e); };
    canvas.ontouchmove = drawLine;
    window.addEventListener("touchend", () => drawing = false);

    document.getElementById("maskClearBtn").onclick = () => ctx.clearRect(0, 0, canvas.width, canvas.height);

    document.getElementById("maskInvertBtn").onclick = () => {
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      for (let i = 0; i < imgData.data.length; i += 4) {
        imgData.data[i + 3] = 255 - imgData.data[i + 3];
        imgData.data[i] = imgData.data[i + 1] = imgData.data[i + 2] = 255;
      }
      ctx.putImageData(imgData, 0, 0);
    };

    document.getElementById("maskSaveBtn").onclick = () => {
      canvas.toBlob((blob) => {
        currentMaskBlob = blob;
        // Store settings from inline controls
        window._maskInpaintingFill = document.getElementById("preview_inpainting_fill").value;
        window._maskBlur = document.getElementById("preview_mask_blur").value;
        editMaskBtn.style.color = "var(--green)";
        previewState.mode = "original";
        renderPreview();
      });
    };

    document.getElementById("maskCancelBtn").onclick = () => {
      previewState.mode = "original";
      renderPreview();
    };

    return;
  }

  // --- NORMAL PREVIEW MODES ---
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
  if (defs.scheduler) scheduler.value = defs.scheduler;

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
    d.append("scheduler", scheduler.value);
    d.append("strength", strengthSlider.value);
    d.append("refinement_strength", refinementStrength.value);

    if (selectedImageFile) {
      d.append("image", selectedImageFile);
      // In Img2Img, we ignore the second pass model selection to effectively "disable" it
      // and use the base model as the refiner/processor
      d.set("second_pass_model", "None");

      // Mask support
      if (currentMaskBlob) {
        d.append("mask", currentMaskBlob, "mask.png");
        // Use stored values from preview editor, or fall back to overlay elements
        const fill = window._maskInpaintingFill || document.getElementById("inpainting_fill")?.value || "replace";
        const blur = window._maskBlur || document.getElementById("mask_blur")?.value || "8";
        d.append("inpainting_fill", fill);
        d.append("mask_blur", blur);
      }
    }

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

    if (!r.ok) {
      const text = await r.text();
      console.error("Server error:", r.status, text);
      alert(`Server Error ${r.status}: ${text.slice(0, 200)}`);
      return;
    }

    const data = await r.json();

    if (data.seed !== undefined) {
      setSeed(data.seed);
      saveState();
    }

    if (data?.type === "validation_error" || data?.error) {
      alert(data.error);
      throw new Error(data.error);
    }

    return data;

  } catch (e) {
    console.error("Generation failed:", e);
    alert(`Client Error: ${e.message}\n${e.stack || ""}`);
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

    // Add "Fix / Inpaint" overlay button
    const container = document.createElement("div");
    container.style.position = "relative";
    container.appendChild(img);

    const sendBtn = document.createElement("button");
    sendBtn.innerText = "🎨 Fix";
    sendBtn.className = "gallery-action";
    sendBtn.onclick = async (e) => {
      e.stopPropagation();
      // Fetch blob and set as input
      const blob = await fetch(i).then(r => r.blob());
      const file = new File([blob], "generated.png", { type: "image/png" });
      handleImageSelect(file);

      // Open Editor Immediately
      setTimeout(() => editMaskBtn.click(), 100);

      // Populate inputs if meta available? (Maybe later)
      // For now, just setting the image is main goal.
    };

    container.appendChild(sendBtn);
    g.appendChild(container);
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


// Image Upload Logic
function handleImageSelect(file) {
  if (!file || !file.type.startsWith("image/")) return;

  selectedImageFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    inputImagePreview.src = e.target.result;
    inputImagePreview.style.display = "block";
    uploadPlaceholder.style.display = "none";
    clearInputImageBtn.style.display = "block";
    clearInputImageBtn.style.display = "block";
    img2imgSettings.style.display = "block";
    // Force flex display for centering icon
    editMaskBtn.style.display = "flex";

    // Img2Img Mode: Hide second pass, rename Base Model
    secondPassModelSection.style.display = "none";
    secondPassSection.style.display = "none";
    baseModelLabel.textContent = "Model";

    // Ensure we don't accidentally send a second pass model
    // (We don't change the value to allow restoring it, but we can handle in send())

    // Show caption controls if captioning is available
    if (captioningAvailable && captionControls) {
      captionControls.style.display = "block";
    }
  };
  reader.readAsDataURL(file);
}

imageUploadArea.onclick = (e) => {
  if (e.target !== clearInputImageBtn) {
    inputImageInput.click();
  }
};

inputImageInput.onchange = (e) => {
  if (e.target.files && e.target.files[0]) {
    handleImageSelect(e.target.files[0]);
  }
};

imageUploadArea.ondragover = (e) => {
  e.preventDefault();
  imageUploadArea.style.borderColor = "var(--accent)";
};

imageUploadArea.ondragleave = (e) => {
  e.preventDefault();
  imageUploadArea.style.borderColor = "";
};

imageUploadArea.ondrop = (e) => {
  e.preventDefault();
  imageUploadArea.style.borderColor = "";
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    handleImageSelect(e.dataTransfer.files[0]);
  }
};

clearInputImageBtn.onclick = (e) => {
  e.stopPropagation();
  selectedImageFile = null;
  inputImageInput.value = "";
  inputImagePreview.src = "";
  inputImagePreview.style.display = "none";
  uploadPlaceholder.style.display = "block";
  clearInputImageBtn.style.display = "none";
  img2imgSettings.style.display = "none";

  // Restore Text2Img Mode
  secondPassModelSection.style.display = "block";
  updateSecondPassVisibility(); // Restore second pass section if model selected
  baseModelLabel.textContent = "Base Model";

  // Clear Mask
  currentMaskBlob = null;
  editMaskBtn.style.display = "none";
  editMaskBtn.style.color = "";

  // Hide caption controls
  if (captionControls) {
    captionControls.style.display = "none";
  }
};

strengthSlider.oninput = () => {
  strengthVal.textContent = strengthSlider.value;
  saveState();
};

// Caption generation function
async function generateCaption() {
  if (!selectedImageFile || !captioningAvailable) return;

  const style = captionStyleSelect.value;
  captionBtn.disabled = true;
  captionBtn.textContent = "🔍 Captioning...";

  try {
    const fd = new FormData();
    fd.append("image", selectedImageFile);
    fd.append("style", style);

    const r = await fetch("/caption", { method: "POST", body: fd });
    const data = await r.json();

    if (data.error) {
      console.error("Caption error:", data.error);
      alert("Captioning failed: " + data.error);
      return;
    }

    // Populate prompt with caption
    promptInput.value = data.caption;
    saveState();

    // Update token count
    updateTokenCount(promptInput, "prompt");

  } catch (e) {
    console.error("Caption failed:", e);
    alert("Captioning failed: " + e.message);
  } finally {
    captionBtn.disabled = false;
    captionBtn.textContent = "🔍 Caption";
  }
}

if (captionBtn) {
  captionBtn.onclick = generateCaption;
}

window.addEventListener("DOMContentLoaded", async () => {

  // Check if captioning is available
  try {
    const captionerData = await fetch("/captioners").then(r => r.json());
    captioningAvailable = captionerData.available;
    console.log("Captioning available:", captioningAvailable, "Plugins:", captionerData.captioners);
  } catch (e) {
    console.log("Captioner check failed:", e);
    captioningAvailable = false;
  }

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
  await populateSelect("scheduler", "/schedulers", false);


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

    if (state.scheduler) scheduler.value = state.scheduler;
    if (state.refinement_strength) {
      refinementStrength.value = state.refinement_strength;
      refinementStrengthVal.textContent = state.refinement_strength;
    }

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