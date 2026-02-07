/**
 * WebbDuck Main Application
 * Entry point that initializes all modules and UI interactions
 */

import * as api from './core/api.js';
import { MaskEditor } from './modules/MaskEditor.js';
import { LoraManager } from './modules/LoraManager.js';
import { initState, getState, setState, setSeed, setLastUsedSeed, syncFromDOM, syncToDOM } from './core/state.js';
import { on, emit, Events, initWebSocket } from './core/events.js';
import { ProgressManager } from './modules/ProgressManager.js';
import { $, $$, byId, listen, show, hide, toggleClass, populateSelect, toast, debounce, buildFormData } from './core/utils.js';
import { LightboxManager } from './modules/LightboxManager.js';
import { GalleryManager } from './modules/GalleryManager.js';

// ═══════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', async () => {
    try {
        console.log('🦆 WebbDuck UI Initializing...');

        // Initialize state from localStorage
        initState();

        // Initialize Managers
        window.progressManager = new ProgressManager();
        window.maskEditor = new MaskEditor();
        window.loraManager = new LoraManager();
        window.lightboxManager = new LightboxManager({
            onUpscale: (src) => { /* Upscale handler */ },
            onInpaint: (src) => { /* Inpaint handler */ }
        });
        window.galleryManager = new GalleryManager();
        window.galleryManager.init();
        window.galleryManager.load();

        // Setup event listeners
        setupNavigation();
        setupSliders();
        setupPresetChips();
        setupFormHandlers();
        setupGenerationButtons();
        setupUploadHandling();
        setupUploadHandling();

        // Load initial data
        await loadModels();
        await loadSchedulers();
        await loadModels();
        await loadSchedulers();

        // Sync state to DOM
        syncToDOM();

        // Initialize Mask Editor
        window.maskEditor = new MaskEditor();

        // Initialize LoRA Manager
        window.loraManager = new LoraManager();

        // Initialize Lightbox Manager
        window.lightboxManager = new LightboxManager({
            onUpscale: (src, cb) => startUpscale(src, cb),
            onInpaint: (src) => {
                // Switch to studio, set image as input, set mode to inpaint?
                // Existing sendToInpaint does this
                if (typeof sendToInpaint === 'function') {
                    sendToInpaint(src);
                } else {
                    console.error('sendToInpaint not defined');
                }
            },
            onRegenerate: handleRegenerateFromLightbox,
            onRegenerate: handleRegenerateFromLightbox,
            onDelete: (src, type) => window.galleryManager.handleDelete(src, type)
        });

        // Initialize Progress Manager
        window.progressManager = new ProgressManager();

        // Start WebSocket Connection for real-time updates
        initWebSocket();

        // Count tokens for pre-filled prompt
        const promptEl = byId('prompt');
        if (promptEl && promptEl.value.trim()) {
            const count = await countTokens(promptEl.value);
            const counterEl = byId('token-count');
            if (counterEl) {
                counterEl.textContent = `${count} tokens`;
                toggleClass(counterEl, 'warning', count > 60);
                toggleClass(counterEl, 'danger', count > 75);
            }
        }

        // Display last used seed if available
        const state = getState();
        if (state.lastSeed) {
            const lastSeedEl = byId('last-seed');
            if (lastSeedEl) lastSeedEl.textContent = state.lastSeed;
        }

        console.log('🦆 WebbDuck UI Ready!');
    } catch (fatalError) {
        console.error('CRITICAL UI INIT FAILURE:', fatalError);
        alert('CRITICAL UI ERROR: ' + fatalError.message);
    }
});

// ═══════════════════════════════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════════════════════════════

function setupNavigation() {
    // Desktop tabs
    $$('.nav-tab').forEach(tab => {
        listen(tab, 'click', () => switchView(tab.dataset.view));
    });

    // Mobile tabs
    $$('.mobile-tab').forEach(tab => {
        listen(tab, 'click', () => switchView(tab.dataset.view));
    });
}

function switchView(viewName) {
    // Update tabs
    $$('.nav-tab, .mobile-tab').forEach(tab => {
        toggleClass(tab, 'active', tab.dataset.view === viewName);
    });

    // Update views
    $$('.view').forEach(view => {
        toggleClass(view, 'active', view.id === `view-${viewName}`);
    });

    // Save state
    setState({ view: viewName });

    // If switching to gallery, refresh it
    if (viewName === 'gallery') {
        window.galleryManager.load();
        emit(Events.VIEW_CHANGE, viewName);
    }
}

// ═══════════════════════════════════════════════════════════════
// SLIDERS
// ═══════════════════════════════════════════════════════════════

function setupSliders() {
    const sliderMappings = [
        { slider: 'steps', display: 'steps-value' },
        { slider: 'cfg', display: 'cfg-value' },
        { slider: 'batch', display: 'batch-value' },
        { slider: 'second_pass_steps', display: 'second-steps-value' },
        { slider: 'second_pass_blend', display: 'blend-value' },
        { slider: 'denoising_strength', display: 'denoise-value' },
    ];

    sliderMappings.forEach(({ slider, display }) => {
        const sliderEl = byId(slider);
        const displayEl = byId(display);
        if (sliderEl && displayEl) {
            listen(sliderEl, 'input', () => {
                displayEl.textContent = sliderEl.value;
            });
        }
    });
}

// ═══════════════════════════════════════════════════════════════
// PRESET CHIPS (Dimension shortcuts)
// ═══════════════════════════════════════════════════════════════

function setupPresetChips() {
    $$('.preset-chip[data-width]').forEach(chip => {
        listen(chip, 'click', () => {
            const width = chip.dataset.width;
            const height = chip.dataset.height;

            byId('width').value = width;
            byId('height').value = height;

            // Update active state
            $$('.preset-chip[data-width]').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
        });
    });
}

// ═══════════════════════════════════════════════════════════════
// FORM HANDLERS
// ═══════════════════════════════════════════════════════════════

function setupFormHandlers() {
    // Auto-save form state on change
    const saveState = debounce(() => syncFromDOM(), 500);

    ['prompt', 'negative', 'width', 'height', 'steps', 'cfg', 'scheduler', 'batch'].forEach(id => {
        const el = byId(id);
        if (el) listen(el, 'input', saveState);
    });

    // Token counting for prompt
    const promptEl = byId('prompt');
    if (promptEl) {
        listen(promptEl, 'input', debounce(async () => {
            const count = await countTokens(promptEl.value);
            const counterEl = byId('token-count');
            if (counterEl) {
                counterEl.textContent = `${count} tokens`;
                toggleClass(counterEl, 'warning', count > 60);
                toggleClass(counterEl, 'danger', count > 75);
            }
        }, 300));
    }

    // Randomize seed button
    listen(byId('randomize-seed'), 'click', () => {
        byId('seed_input').value = '';
        setSeed(null);
    });

    // Model change triggers LoRA reload
    listen(byId('base_model'), 'change', async () => {
        const modelName = byId('base_model').value;
        if (modelName) {
            await window.loraManager.loadForModel(modelName);
            emit(Events.MODEL_CHANGE, modelName);
        }
    });
}

async function countTokens(prompt) {
    if (!prompt.trim()) return 0;
    try {
        const baseModel = byId('base_model')?.value;
        if (!baseModel) return 0;
        const result = await api.tokenize(prompt, baseModel);
        return result.tokens || 0;
    } catch (e) {
        console.warn('Token count failed:', e);
        return 0;
    }
}

// ═══════════════════════════════════════════════════════════════
// GENERATION
// ═══════════════════════════════════════════════════════════════

let isGenerating = false;

function setupGenerationButtons() {
    listen(byId('btn-test'), 'click', () => startGeneration('test'));
    listen(byId('btn-generate'), 'click', () => startGeneration('generate'));
    listen(byId('cancel-generation'), 'click', cancelGeneration);
}

async function startGeneration(mode) {
    if (isGenerating) return;

    isGenerating = true;
    window.progressManager.showProgress('Starting...', 0);
    emit(Events.GENERATION_START, mode);

    try {
        const formData = collectFormData();

        const endpoint = mode === 'test' ? api.testGenerate : api.generate;
        const result = await endpoint(formData);

        handleGenerationResult(result);
        emit(Events.GENERATION_COMPLETE, result);

    } catch (error) {
        console.error('Generation error:', error);
        toast(error.message || 'Generation failed', 'error');
        emit(Events.GENERATION_ERROR, error);

    } finally {
        isGenerating = false;
        // Ensure progress is hidden if error/complete
        window.progressManager.hideProgress();
    }
}

function collectFormData() {
    const formData = new FormData();
    const state = getState();

    // Basic params
    formData.append('prompt', byId('prompt').value);
    formData.append('negative', byId('negative').value);

    // Use state for width/height if available (set synchronously on upload), fallback to DOM
    formData.append('width', state.width || byId('width').value);
    formData.append('height', state.height || byId('height').value);

    formData.append('steps', byId('steps').value);
    formData.append('cfg', byId('cfg').value);
    formData.append('scheduler', byId('scheduler').value);
    formData.append('num_images', byId('batch').value);
    formData.append('base_model', byId('base_model').value);

    // LoRAs
    const loras = window.loraManager.getSelected();
    if (loras.length > 0) {
        // Send as JSON for backend to parse
        formData.append('loras', JSON.stringify(loras));

        // Also send individually for legacy compatibility if needed
        loras.forEach((lora, i) => {
            formData.append(`lora_model_${i + 1}`, lora.name);
            formData.append(`lora_weight_${i + 1}`, lora.weight);
        });
    }

    // Seed
    const seedVal = byId('seed_input').value;
    if (seedVal) formData.append('seed', seedVal);

    // Second pass
    if (byId('second_pass_enabled').checked) {
        formData.append('second_pass_model', byId('second_pass_model').value);
        formData.append('second_pass_steps', byId('second_pass_steps').value);
        formData.append('second_pass_blend', byId('second_pass_blend').value);
    }

    // Add uploaded image if present
    const inputImage = window._uploadedImage;
    if (inputImage) {
        formData.append('image', inputImage);
        formData.append('denoising_strength', byId('denoising_strength').value);
    }

    // Add mask if present
    const maskBlob = window._maskBlob;
    if (maskBlob) {
        formData.append('mask', maskBlob);
        // Include mask settings
        const state = getState();
        formData.append('inpainting_fill', state.inpaintMode || 'replace');
        formData.append('mask_blur', byId('mask_blur') ? byId('mask_blur').value : 8);
    }

    return formData;
}

function handleGenerationResult(result) {
    if (result.images && result.images.length > 0) {
        // Show first image in preview
        const previewImg = byId('preview-image');
        const placeholder = byId('preview-placeholder');

        previewImg.src = result.images[0];

        // Robust display toggling via inline styles
        previewImg.style.display = 'block';
        previewImg.classList.remove('hidden');

        placeholder.style.display = 'none';
        placeholder.classList.add('hidden');

        // Force layout repaint
        void previewImg.offsetWidth;

        // Store metadata for lightbox
        // API usually returns parameter/meta object mixed in result
        if (result) {
            previewImg.dataset.meta = JSON.stringify(result.meta || result);
        }

        // Update batch strip
        updateBatchStrip(result.images);

        // Update seed display
        if (result.seed !== undefined) {
            setLastUsedSeed(result.seed);
            const seedEl = byId('last-seed');
            if (seedEl) seedEl.textContent = result.seed;
        }

        toast(`Generated ${result.images.length} image(s)`, 'success');
    }
}

function updateBatchStrip(images) {
    const strip = byId('batch-strip');
    if (!strip) return;

    strip.innerHTML = images.map((img, i) => `
    <div class="image-item" data-index="${i}" style="width: 60px; height: 60px;">
      <img src="${img}" alt="Generated ${i + 1}" />
    </div>
  `).join('');

    show(strip);

    // Click to select
    strip.querySelectorAll('.image-item').forEach(item => {
        listen(item, 'click', () => {
            const img = item.querySelector('img').src;
            byId('preview-image').src = img;
        });
    });
}



function cancelGeneration() {
    // TODO: Implement actual cancel via API
    isGenerating = false;
    window.progressManager.hideProgress();
    toast('Generation cancelled', 'warning');
    emit(Events.GENERATION_CANCEL);
}

// ═══════════════════════════════════════════════════════════════
// IMAGE UPLOAD
// ═══════════════════════════════════════════════════════════════

function setupUploadHandling() {
    const dropZone = byId('upload-drop');
    const fileInput = byId('input-image');

    if (!dropZone || !fileInput) return;

    // Click to upload
    listen(dropZone, 'click', () => fileInput.click());

    // File input change
    listen(fileInput, 'change', (e) => {
        if (e.target.files.length > 0) {
            handleImageUpload(e.target.files[0]);
        }
    });

    // Drag and drop
    listen(dropZone, 'dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    listen(dropZone, 'dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    listen(dropZone, 'drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleImageUpload(e.dataTransfer.files[0]);
        }
    });

    // Clear button
    listen(byId('clear-upload'), 'click', clearUploadedImage);

    // Caption button
    listen(byId('caption-btn'), 'click', async () => {
        const file = window._uploadedImage;
        if (!file) {
            toast('No image to caption', 'error');
            return;
        }

        toast('Generating prompt from image...', 'info');
        try {
            const formData = new FormData();
            formData.append('image', file);
            formData.append('style', 'art'); // Use 'art' style for better SDXL prompts

            const result = await api.caption(formData);
            if (result && result.caption) {
                const promptEl = byId('prompt');
                if (promptEl) {
                    promptEl.value = result.caption;
                    promptEl.dispatchEvent(new Event('input'));
                }
                toast('Caption generated!', 'success');
            } else {
                toast('No caption returned', 'error');
            }
        } catch (e) {
            console.error('Caption error:', e);
            toast('Caption failed: ' + e.message, 'error');
        }
    });

    // Edit Mask button
    listen(byId('edit-mask-btn'), 'click', () => {
        const file = window._uploadedImage;
        if (!file) {
            toast('Upload an image first', 'error');
            return;
        }

        // Open mask editor with current preview image
        // The MaskEditor handles UI and Canvas setup
        if (window.maskEditor) {
            const src = byId('preview-img').src;
            window.maskEditor.open(src);
            toast('Draw to mask areas', 'info');
        }
    });

    // Inpaint mode toggler
    listen(byId('inpaint-replace'), 'click', () => {
        setState({ inpaintMode: 'replace' });
        byId('inpaint-replace').classList.add('active');
        byId('inpaint-keep').classList.remove('active');
    });

    listen(byId('inpaint-keep'), 'click', () => {
        setState({ inpaintMode: 'keep' });
        byId('inpaint-keep').classList.add('active');
        byId('inpaint-replace').classList.remove('active');
    });


}

async function handleImageUpload(file) {
    if (!file.type.startsWith('image/')) {
        toast('Please upload an image file', 'error');
        return;
    }

    window._uploadedImage = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewImg = byId('preview-img');
        previewImg.src = e.target.result;

        // Auto-set resolution to match image
        const img = new Image();
        img.onload = () => {
            let w = img.width;
            let h = img.height;
            const MAX_DIM = 2048; // Max resolution ~2k

            // Scale down if too large, preserving aspect ratio
            if (w > MAX_DIM || h > MAX_DIM) {
                const ratio = Math.min(MAX_DIM / w, MAX_DIM / h);
                w = Math.round(w * ratio);
                h = Math.round(h * ratio);
            }

            // Snap to nearest 8
            w = Math.round(w / 8) * 8;
            h = Math.round(h / 8) * 8;

            const wInput = byId('width');
            const hInput = byId('height');

            if (wInput) {
                wInput.value = w;
                wInput.dispatchEvent(new Event('input'));
                wInput.dispatchEvent(new Event('change'));
            }
            if (hInput) {
                hInput.value = h;
                hInput.dispatchEvent(new Event('input'));
                hInput.dispatchEvent(new Event('change'));
            }

            // Force update state to ensure generation uses new values
            if (typeof setState === 'function') {
                setState({ width: w, height: h });
            }

            toast(`Resolution set to ${w}x${h}`, 'info');
        };
        img.src = e.target.result;

        hide(byId('upload-drop'));
        show(byId('upload-preview'));
        show(byId('denoise-group'));
    };
    reader.readAsDataURL(file);

    emit(Events.IMAGE_UPLOAD, file);
}

function clearUploadedImage() {
    window._uploadedImage = null;
    window._maskBlob = null;

    byId('input-image').value = '';
    byId('preview-img').src = '';

    show(byId('upload-drop'));
    hide(byId('upload-preview'));
    hide(byId('denoise-group'));
    hide(byId('inpaint-options'));

    emit(Events.IMAGE_CLEAR);
}

// ═══════════════════════════════════════════════════════════════
// DATA LOADING
// ═══════════════════════════════════════════════════════════════

async function loadModels() {
    try {
        const models = await api.getModels();
        populateSelect('base_model', models, false);

        // Also load second pass models
        const secondPassModels = await api.getSecondPassModels();
        populateSelect('second_pass_model', secondPassModels, true);

        // Load LoRAs for first model (models are objects with .name property)
        if (models && models.length > 0) {
            const firstModelName = typeof models[0] === 'string' ? models[0] : models[0].name;
            await window.loraManager.loadForModel(firstModelName);
        }
    } catch (error) {
        console.warn('Model load warning:', error);
        // Don't show toast for initial load failures to avoid annoyance, just log
    }
}



async function loadSchedulers() {
    try {
        const schedulers = await api.getSchedulers();
        populateSelect('scheduler', schedulers, false);
    } catch (error) {
        console.error('Failed to load schedulers:', error);
    }
}



// ═══════════════════════════════════════════════════════════════
// PREVIEW TOOLBAR
// ═══════════════════════════════════════════════════════════════

function setupPreviewToolbarV2() {
    // Zoom/Lightbox
    listen('#preview-zoom', 'click', (e) => {
        if (e) e.stopPropagation();
        const img = byId('preview-image');
        if (!img || !img.src || img.style.display === 'none') return;

        let meta = {};
        try {
            meta = img.dataset.meta ? JSON.parse(img.dataset.meta) : {};
        } catch (e) { /* ignore */ }

        const item = {
            src: img.src,
            width: img.naturalWidth || 1024,
            height: img.naturalHeight || 1024,
            msrc: img.src,
            alt: 'Preview',
            meta: meta,
            originalSrc: img.src
        };
        openLightbox([item], 0);
    });

    // Upscale
    listen('#preview-upscale', 'click', () => {
        const img = byId('preview-image');
        if (!img || !img.src) return;
        startUpscale(img.src);
    });

    // Inpaint
    listen('#preview-inpaint', 'click', () => {
        const img = byId('preview-image');
        if (!img || !img.src) return;
        sendToInpaint(img.src);
    });

    // Download
    listen('#preview-download', 'click', () => {
        const img = byId('preview-image');
        if (!img || !img.src) return;

        const a = document.createElement('a');
        a.href = img.src;
        a.download = `webbduck-${Date.now()}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });
}

// Global helper for Upscale
async function startUpscale(imageSrc, onSuccess) {
    toast('Upscaling...', 'info');
    try {
        const formData = new FormData();
        if (imageSrc.startsWith('data:')) {
            const blob = await (await fetch(imageSrc)).blob();
            formData.append('image', blob, 'upscale.png');
        } else {
            formData.append('image', imageSrc);
        }
        formData.append('scale', 2);

        const result = await fetch('/upscale', { method: 'POST', body: formData });
        if (result.ok) {
            const data = await result.json();
            if (data.upscaled) {
                toast('Upscale Success!', 'success');
                if (onSuccess) onSuccess(data.upscaled);
            }
        } else {
            toast('Upscale failed', 'error');
        }
    } catch (e) {
        console.error('Upscale error:', e);
        toast('Upscale failed', 'error');
    }
}

// Global helper for Inpaint
async function sendToInpaint(imageSrc) {
    try {
        toast('Loading image for inpaint...', 'info');
        const response = await fetch(imageSrc);
        const blob = await response.blob();
        const file = new File([blob], "inpaint_src.png", { type: "image/png" });

        // Use existing upload handler
        await handleImageUpload(file);

        // Switch to studio
        const studioTab = document.querySelector('[data-view="studio"]');
        if (studioTab) studioTab.click();

        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });

        toast('Image loaded for inpaint!', 'success');

    } catch (e) {
        console.error('Inpaint error:', e);
        toast('Failed to load image', 'error');
    }
}

// ═══════════════════════════════════════════════════════════════
// LIGHTBOX ACTION HELPERS
// ═══════════════════════════════════════════════════════════════

function handleRegenerateFromLightbox(curr) {
    if (!curr.meta) {
        toast('No metadata available', 'error');
        return;
    }

    // Helper to set value and trigger events
    const setVal = (id, val) => {
        const el = byId(id);
        if (el && val !== undefined) el.value = val;
        if (el) el.dispatchEvent(new Event('change'));
        if (el && el.type === 'range') el.dispatchEvent(new Event('input'));
    };

    const m = curr.meta;
    setVal('prompt', m.prompt);
    setVal('negative', m.negative || m.negative_prompt);
    setVal('steps', m.steps);
    setVal('cfg', m.cfg);
    setVal('width', m.width);
    setVal('height', m.height);
    setVal('base_model', m.base_model || m.model);
    setVal('scheduler', m.scheduler);

    // Set seed to random for variation
    byId('seed_input').value = '';
    if (typeof setSeed === 'function') setSeed(null);

    toast('Settings copied! Generating...', 'success');

    // Trigger generation
    const genBtn = byId('btn-generate');
    if (genBtn) {
        // Switch to studio view
        const studioTab = document.querySelector('[data-view="studio"]');
        if (studioTab) studioTab.click();

        // Allow UI to update then click
        setTimeout(() => genBtn.click(), 100);
    }
}



