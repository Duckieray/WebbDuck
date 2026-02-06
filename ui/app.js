import { saveState, loadState, setSeed } from './modules/state.js';
import { populateSelect, updateTokenCount } from './modules/utils.js';
import { initMaskEditor } from './modules/mask-editor.js';
import { renderPreview, setPreviewImages, setPreviewMode, getPreviewState } from './modules/preview.js';
import { loadLoras, initLoraEvents, getModelDefaults, setModelDefaults } from './modules/lora.js';
import { loadGallery, openPhotoSwipe, getPswpInstance } from './modules/gallery.js';
import { initUploadHandling, generateCaption, checkCaptionerAvailability, getSelectedImageFile } from './modules/upload.js';
import { send, runUpscale } from './modules/generation.js';

// --- Initialization ---

// Initialize Mask Editor
const { openMaskEditor } = initMaskEditor(null, getSelectedImageFile);

// Initialize Upload Handling
const { handleImageSelect } = initUploadHandling();

// Listen for clear event
document.addEventListener('image-cleared', () => {
  updateSecondPassVisibility();
});

// --- State Loading & Defaults ---
window.addEventListener("DOMContentLoaded", async () => {
  // Check Captioner
  await checkCaptionerAvailability();

  // Load Models
  const models = await fetch("/models").then(r => r.json());
  const base_model = document.getElementById("base_model");
  const modelDefaults = {};

  base_model.innerHTML = "";
  models.forEach(m => {
    const name = typeof m === "string" ? m : m.name;
    const defaults = typeof m === "string" ? {} : (m.defaults || {});
    modelDefaults[name] = defaults;

    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    base_model.appendChild(opt);
  });
  setModelDefaults(modelDefaults);

  await populateSelect("second_pass_model", "/second_pass_models", true);
  await populateSelect("scheduler", "/schedulers", false);

  // Load Saved State
  const state = loadState();
  if (state) {
    if (state.base_model) base_model.value = state.base_model;

    const setVal = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
    setVal("second_pass_model", state.second_pass_model || "None");
    setVal("second_pass_mode", state.second_pass_mode || "auto");
    setVal("prompt", state.prompt ?? "");
    setVal("prompt_2", state.prompt_2 ?? "");
    setVal("negative_prompt", state.negative_prompt ?? "");
    setVal("steps", state.steps ?? 30);
    setVal("cfg", state.cfg ?? 7.5);
    setVal("width", state.width ?? 1024);
    setVal("height", state.height ?? 1024);
    setVal("num_images", state.num_images ?? 4);
    setVal("scheduler", state.scheduler);

    setSeed(state.seed);

    if (state.refinement_strength) {
      document.getElementById("refinement_strength").value = state.refinement_strength;
      document.getElementById("refinement_strength_val").textContent = state.refinement_strength;
    }

    if (state.experimental_compress) {
      document.getElementById("experimental_compress").checked = true;
    }
  }

  // Load LoRAs
  await loadLoras();
  // Re-apply saved loras
  if (state && state.loras) {
    const { addLoraCard } = await import('./modules/lora.js');
    state.loras.forEach(l => addLoraCard(l.name, l.weight, l.enabled));
  }

  updateSecondPassVisibility();
  initLoraEvents();

  // Start tabs
  initTabs();
});

// --- Event Listeners ---

// Button Handlers
document.getElementById("testBtn").onclick = async () => {
  const r = await send("/test");
  if (!r) return;
  setPreviewImages(r.images[0]);
  renderPreview(null, handleUpscaleClick);
};

document.getElementById("genBtn").onclick = async () => {
  const r = await send("/generate");
  if (!r) return;

  // Clear main gallery in UI to show new batch immediately? 
  // Or simpler: just append to gallery logic?
  // The original code replaced global gallery div content with new images.

  const g = document.getElementById("gallery");
  g.innerHTML = "";

  r.images.forEach(i => {
    const img = document.createElement("img");
    img.src = i;
    img.onclick = () => {
      setPreviewImages(i);
      renderPreview(null, handleUpscaleClick);
    };

    const container = document.createElement("div");
    container.style.position = "relative";
    container.appendChild(img);

    const sendBtn = document.createElement("button");
    sendBtn.innerText = "🎨 Fix";
    sendBtn.className = "gallery-action";
    sendBtn.onclick = async (e) => {
      e.stopPropagation();
      const blob = await fetch(i).then(r => r.blob());
      const file = new File([blob], "generated.png", { type: "image/png" });
      handleImageSelect(file);
      setTimeout(() => document.getElementById("edit_mask_btn").click(), 100);
    };

    container.appendChild(sendBtn);
    container.appendChild(sendBtn);
    g.appendChild(container);
  });

  // Auto-refresh main gallery
  loadGallery(true);
};

const handleUpscaleClick = async () => {
  const state = getPreviewState();
  if (!state.original) return;
  const newImg = await runUpscale(state.original); // Pass logic
  if (newImg) {
    setPreviewImages(state.original, newImg);
    renderPreview(null, handleUpscaleClick);
  }
};

document.getElementById("caption_btn").onclick = generateCaption;

document.getElementById("refinement_strength").oninput = (e) => {
  document.getElementById("refinement_strength_val").textContent = e.target.value;
  saveState();
};

document.getElementById("strength").oninput = (e) => {
  document.getElementById("strength_val").textContent = e.target.value;
  saveState();
};

// Input listeners for state saving
[
  "prompt", "prompt_2", "negative_prompt", "steps", "cfg", "width", "height",
  "num_images", "second_pass_mode", "experimental_compress"
].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener("input", saveState);
});

// Base model change
document.getElementById("base_model").addEventListener("change", async () => {
  const val = document.getElementById("base_model").value;
  const defaults = getModelDefaults()[val];
  if (defaults) {
    if (defaults.steps) document.getElementById("steps").value = defaults.steps;
    if (defaults.cfg) document.getElementById("cfg").value = defaults.cfg;
    if (defaults.width) document.getElementById("width").value = defaults.width;
    if (defaults.height) document.getElementById("height").value = defaults.height;
    if (defaults.scheduler) document.getElementById("scheduler").value = defaults.scheduler;
  }
  await loadLoras(); // refresh lora list
  saveState();
});

document.getElementById("second_pass_model").addEventListener("change", () => {
  updateSecondPassVisibility();
  saveState();
});

// Tokenizer updates
[
  ["prompt", "prompt"],
  ["prompt_2", "prompt_2"],
  ["negative_prompt", "negative"],
].forEach(([id, which]) => {
  const el = document.getElementById(id);
  if (el) {
    let t;
    el.addEventListener("input", () => {
      clearTimeout(t);
      t = setTimeout(() => updateTokenCount(el, which, document.getElementById("base_model").value), 200);
    });
  }
});

// Seed Logic
const seedModeSelect = document.getElementById("seed_mode");
const seedInput = document.getElementById("seed_input");
if (seedModeSelect) {
  seedModeSelect.onchange = () => {
    seedInput.disabled = seedModeSelect.value !== "manual";
  };
}


// --- Helper Functions ---

function updateSecondPassVisibility() {
  const section = document.getElementById("second-pass-section");
  const modelSel = document.getElementById("second_pass_model");
  const modeSel = document.getElementById("second_pass_mode");

  if (!section || !modelSel) return;
  const enabled = modelSel.value && modelSel.value !== "None";
  section.style.display = enabled ? "block" : "none";
  if (!enabled) modeSel.value = "auto";
}

function initTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  const views = document.querySelectorAll(".view");
  tabs.forEach(tab => {
    tab.onclick = () => {
      const target = tab.dataset.tab;
      tabs.forEach(t => t.classList.remove("active"));
      views.forEach(v => {
        v.classList.remove("active");
        v.style.display = "none";
      });
      tab.classList.add("active");
      const activeView = document.getElementById(target + "-view");
      activeView.classList.add("active");
      activeView.style.display = "";

      if (target === "gallery") {
        loadGallery();
      }
    };
  });
}

// Gallery Refresh
const refreshBtn = document.getElementById("refresh_gallery_btn");
if (refreshBtn) {
  refreshBtn.onclick = () => loadGallery(true);
}


// --- DELETE MODAL LOGIC (Global) ---
const modal = document.getElementById("confirmation-modal");
const modalDeleteImgBtn = document.getElementById("modal-delete-img");
const modalDeleteRunBtn = document.getElementById("modal-delete-run");
const modalCancelBtn = document.getElementById("modal-cancel");

function hideModal() { if (modal) modal.style.display = "none"; }
if (modalCancelBtn) modalCancelBtn.onclick = (e) => { e.stopPropagation(); hideModal(); };

// Delete Image Handler
if (modalDeleteImgBtn) modalDeleteImgBtn.onclick = async (e) => {
  e.stopPropagation();
  const pswp = getPswpInstance();
  const path = pswp ? pswp.currSlide.data.src : null;

  if (!path) return;

  const formData = new FormData();
  formData.append("path", path);
  try {
    const res = await fetch("/delete_image", { method: "POST", body: formData });
    if (res.ok) {
      if (pswp) pswp.close();
      hideModal();
      // Invalidate gallery?
      // Ideally we'd remove the element from DOM without reload, but reload is safer for sync
      loadGallery();
    } else {
      alert("Failed to delete image");
    }
  } catch (e) { console.error(e); }
};

// Delete Run Handler
if (modalDeleteRunBtn) modalDeleteRunBtn.onclick = async (e) => {
  e.stopPropagation();
  const pswp = getPswpInstance();
  const path = pswp ? pswp.currSlide.data.src : null;
  if (!path) return;

  const formData = new FormData();
  formData.append("path", path);
  try {
    const res = await fetch("/delete_run", { method: "POST", body: formData });
    if (res.ok) {
      if (pswp) pswp.close();
      hideModal();
      loadGallery();
    } else {
      alert("Failed to delete run");
    }
  } catch (e) { console.error(e); }
};

// Bind "Trash" button in Lightbox (PhotoSwipe Custom UI)
// This button is dynamically inside the PhotoSwipe container, but its ID 'lightbox-delete' is unique.
// However, the elements are MOVED into PhotoSwipe. 
const lightboxDelete = document.getElementById("lightbox-delete");
if (lightboxDelete) {
  lightboxDelete.onclick = (e) => {
    e.stopPropagation();
    if (modal) modal.style.display = "flex";
  };
}
const lightboxRegen = document.getElementById("lightbox-regen");
if (lightboxRegen) {
  lightboxRegen.onclick = (e) => {
    e.stopPropagation();
    const pswp = getPswpInstance();
    if (pswp && pswp.currSlide.data.meta) {
      const meta = pswp.currSlide.data.meta;
      // Restore settings
      const setVal = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
      if (meta.prompt) setVal("prompt", meta.prompt);
      if (meta.prompt_2) setVal("prompt_2", meta.prompt_2);
      if (meta.negative_prompt) setVal("negative_prompt", meta.negative_prompt);
      if (meta.base_model) setVal("base_model", meta.base_model);
      if (meta.steps) setVal("steps", meta.steps);
      if (meta.cfg) setVal("cfg", meta.cfg);
      if (meta.width) setVal("width", meta.width);
      if (meta.height) setVal("height", meta.height);
      if (meta.scheduler) setVal("scheduler", meta.scheduler);

      // Random seed
      setVal("seed_mode", "random");
      const seedIn = document.getElementById("seed_input");
      if (seedIn) { seedIn.disabled = true; seedIn.value = ""; }

      saveState();

      // Notify user visually
      const originalText = lightboxRegen.innerText;
      lightboxRegen.innerText = "✅ Restored!";
      setTimeout(() => lightboxRegen.innerText = originalText, 1000);

      // Auto-start? The old code did auto-start.
      console.log("Restored settings from gallery. Click Generate to run.");
    }
  };
}

// Copy Prompt Global Helper
window.copyPromptToInput = function () {
  const file = getSelectedImageFile();
  if (!file) {
    alert("Please upload an image first.");
    return;
  }
  // Future: Parse PNG Info / EXIF here.
  alert("Metadata extraction from uploads is not yet implemented on the server.");
};

// Helper to update toggle button text
function updateToggleBtnText() {
  const btn = document.getElementById("lightbox-info-toggle");
  const info = document.querySelector(".lightbox-info");
  if (btn && info) {
    const isHidden = info.style.display === "none" || info.classList.contains("collapsed");
    btn.innerText = isHidden ? "Show Info" : "Hide Info";
    // Also ensure button stays above
    btn.style.zIndex = "2010";
  }
}

// Global listen for toggle click to update state
document.addEventListener("click", (e) => {
  if (e.target && e.target.id === "lightbox-info-toggle") {
    const info = document.querySelector(".lightbox-info");
    if (info) {
      if (info.style.display === 'none') {
        info.style.display = 'block';
      } else {
        info.style.display = 'none';
      }
      updateToggleBtnText();
    }
  }
});
