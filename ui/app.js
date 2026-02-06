/**
 * WebbDuck Main Application
 * Entry point that initializes all modules and UI interactions
 */

import * as api from './core/api.js';
import { initState, getState, setState, setSeed, setLastUsedSeed, syncFromDOM, syncToDOM } from './core/state.js';
import { on, emit, Events } from './core/events.js';
import { $, $$, byId, listen, show, hide, toggleClass, populateSelect, toast, debounce, buildFormData } from './core/utils.js';
import PhotoSwipeLightbox from './lib/photoswipe-lightbox.esm.js';
import PhotoSwipe from './lib/photoswipe.esm.js';

// ═══════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', async () => {
    console.log('🦆 WebbDuck UI Initializing...');

    // Initialize state from localStorage
    initState();

    // Setup event listeners
    setupNavigation();
    setupSliders();
    setupPresetChips();
    setupFormHandlers();
    setupGenerationButtons();
    setupUploadHandling();
    setupGallerySearch();
    setupPreviewToolbarV2();
    initLightboxActions();
    initCompareSlider();

    // Load initial data
    await loadModels();
    await loadSchedulers();
    loadGallery(); // Pre-load gallery data

    // Sync state to DOM
    syncToDOM();

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
        loadGallery();
    }

    emit(Events.VIEW_CHANGE, viewName);
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
            await loadLoras(modelName);
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
    showProgress();
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
        hideProgress();
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
    const loras = getSelectedLoras();
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

function showProgress() {
    const progressEl = byId('generation-progress');
    const placeholder = byId('preview-placeholder');
    const previewImg = byId('preview-image');

    hide(placeholder);
    hide(previewImg);
    show(progressEl);

    // Update status
    byId('status-indicator').classList.remove('ready');
    byId('status-indicator').classList.add('busy');
    byId('status-text').textContent = 'Generating...';
}

function hideProgress() {
    const progressEl = byId('generation-progress');
    hide(progressEl);

    // Update status
    byId('status-indicator').classList.remove('busy');
    byId('status-indicator').classList.add('ready');
    byId('status-text').textContent = 'Ready';
}

function cancelGeneration() {
    // TODO: Implement actual cancel via API
    isGenerating = false;
    hideProgress();
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

        const maskOverlay = byId('mask_overlay');
        if (!maskOverlay) return;

        // Load image into mask canvas
        const maskCanvas = byId('mask_canvas');
        const wrapper = byId('mask_canvas_wrapper');

        if (maskCanvas && wrapper) {
            const img = new Image();
            img.onload = () => {
                // Determine optimal display size (max 80% viewport)
                const vw = window.innerWidth * 0.8;
                const vh = window.innerHeight * 0.7;
                const ratio = Math.min(vw / img.width, vh / img.height, 1);

                const dispW = Math.round(img.width * ratio);
                const dispH = Math.round(img.height * ratio);

                // Set canvas resolution to full image size
                maskCanvas.width = img.width;
                maskCanvas.height = img.height;

                // Set wrapper display size and background
                wrapper.style.width = `${dispW}px`;
                wrapper.style.height = `${dispH}px`;
                wrapper.style.margin = 'auto'; // Center in flex container
                wrapper.style.backgroundImage = `url(${img.src})`;
                wrapper.style.backgroundSize = 'contain';
                wrapper.style.backgroundRepeat = 'no-repeat';
                wrapper.style.backgroundPosition = 'center';
                wrapper.style.boxShadow = '0 0 20px rgba(0,0,0,0.5)';
                wrapper.style.border = '1px solid rgba(255,255,255,0.2)';

                // Clear any previous mask
                const ctx = maskCanvas.getContext('2d');
                ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);

                // Store the image for the mask editor
                window._maskSourceImage = img;

                maskOverlay.classList.remove('hidden');
                toast('Draw to mask areas', 'info');
            };
            img.src = byId('preview-img').src;
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

    // MASK EDITOR LOGIC
    // -----------------------------------------------------------
    const maskCanvas = byId('mask_canvas');
    const maskCtx = maskCanvas ? maskCanvas.getContext('2d') : null;
    let isDrawing = false;
    let isErasing = false;
    let lastX = 0;
    let lastY = 0;

    if (maskCanvas && maskCtx) {
        // Drawing helpers
        const getPos = (e) => {
            const rect = maskCanvas.getBoundingClientRect();
            const scaleX = maskCanvas.width / rect.width;
            const scaleY = maskCanvas.height / rect.height;
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;
            return {
                x: (clientX - rect.left) * scaleX,
                y: (clientY - rect.top) * scaleY
            };
        };

        const draw = (e) => {
            if (!isDrawing) return;
            e.preventDefault();
            const { x, y } = getPos(e);

            maskCtx.beginPath();
            maskCtx.moveTo(lastX, lastY);
            maskCtx.lineTo(x, y);

            if (isErasing) {
                maskCtx.globalCompositeOperation = 'destination-out';
                maskCtx.strokeStyle = 'rgba(0,0,0,1)'; // Color doesn't matter for dest-out
            } else {
                maskCtx.globalCompositeOperation = 'source-over';
                maskCtx.strokeStyle = 'rgba(255, 255, 255, 1)';
            }

            maskCtx.lineCap = 'round';
            maskCtx.lineJoin = 'round';
            maskCtx.lineWidth = (byId('mask_brush_size')?.value || 30) * (maskCanvas.width / maskCanvas.getBoundingClientRect().width);
            maskCtx.stroke();

            // Reset to default
            maskCtx.globalCompositeOperation = 'source-over';

            lastX = x;
            lastY = y;
        };

        // Drawing events
        listen(maskCanvas, 'mousedown', (e) => {
            isDrawing = true;
            const { x, y } = getPos(e);
            lastX = x;
            lastY = y;
            draw(e);
        });
        listen(maskCanvas, 'mousemove', draw);
        listen(window, 'mouseup', () => isDrawing = false);

        // Touch events
        listen(maskCanvas, 'touchstart', (e) => {
            isDrawing = true;
            const { x, y } = getPos(e);
            lastX = x;
            lastY = y;
            draw(e);
        });
        listen(maskCanvas, 'touchmove', draw);
        listen(window, 'touchend', () => isDrawing = false);
    }

    // Mask UI Controls
    listen(byId('mask_cancel_btn'), 'click', () => byId('mask_overlay').classList.add('hidden'));

    listen(byId('mask_erase_btn'), 'click', () => {
        isErasing = !isErasing;
        const btn = byId('mask_erase_btn');
        if (isErasing) {
            btn.textContent = '✏️ Draw';
            btn.classList.add('btn-primary');
            btn.classList.remove('btn-secondary');
        } else {
            btn.textContent = '🧽 Erase';
            btn.classList.remove('btn-primary');
            btn.classList.add('btn-secondary');
        }
    });

    listen(byId('mask_clear_btn'), 'click', () => {
        if (maskCanvas) {
            const ctx = maskCanvas.getContext('2d');
            ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
        }
    });

    listen(byId('mask_save_btn'), 'click', () => {
        if (maskCanvas) {
            maskCanvas.toBlob((blob) => {
                window._maskBlob = blob;
                byId('mask_overlay').classList.add('hidden');
                byId('edit-mask-btn').style.color = 'var(--green)';
                toast('Mask saved!', 'success');
                show(byId('inpaint-options'));
            });
        }
    });

    listen(byId('mask_invert_btn'), 'click', () => {
        if (!maskCanvas) return;
        const ctx = maskCanvas.getContext('2d');
        const imageData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
        const data = imageData.data;
        for (let i = 0; i < data.length; i += 4) {
            // Invert alpha: 0 -> 255, 255 -> 0
            // If alpha > 0, make it 0. If alpha == 0, make it 255.
            // Simplified: fully transparent becomes fully white, fully white becomes transparent
            const alpha = data[i + 3];
            data[i + 3] = 255 - alpha;
            data[i] = 255; // R
            data[i + 1] = 255; // G
            data[i + 2] = 255; // B
        }
        ctx.putImageData(imageData, 0, 0);
    });

    // Mask Sliders
    const brushSlider = byId('mask_brush_size');
    const brushPreview = byId('mask_brush_preview');
    if (brushSlider && brushPreview) {
        listen(brushSlider, 'input', (e) => {
            const size = e.target.value;
            brushPreview.style.width = `${size}px`;
            brushPreview.style.height = `${size}px`;
        });
    }

    const blurSlider = byId('mask_blur');
    const blurVal = byId('mask_blur_val');
    if (blurSlider && blurVal) {
        listen(blurSlider, 'input', (e) => {
            blurVal.textContent = e.target.value;
        });
    }
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
        if (models.length > 0) {
            const firstModelName = typeof models[0] === 'string' ? models[0] : models[0].name;
            await loadLoras(firstModelName);
        }
    } catch (error) {
        console.error('Failed to load models:', error);
        toast('Failed to load models', 'error');
    }
}

async function loadLoras(modelName) {
    try {
        const loras = await api.getLoras(modelName);
        const select = byId('lora-select');
        if (!select) return;

        // Clear previous cache
        availableLorasMap.clear();

        // Preserve first option and add loras
        select.innerHTML = '<option value="">➕ Add LoRA...</option>';
        loras.forEach(lora => {
            const name = typeof lora === 'string' ? lora : lora.name;

            // Store metadata
            if (typeof lora === 'string') {
                availableLorasMap.set(name, { name, strength_default: 1.0 });
            } else {
                availableLorasMap.set(name, lora);
            }

            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            select.appendChild(opt);
        });

        // Setup selection handler
        select.onchange = () => {
            if (select.value) {
                addLoraCard(select.value);
                select.value = ''; // Reset to placeholder
            }
        };
    } catch (error) {
        console.error('Failed to load LoRAs:', error);
    }
}

// Track selected LoRAs
const selectedLoras = new Map();
// Cache available LoRAs for metadata (defaults)
const availableLorasMap = new Map();

function addLoraCard(loraName, weight = null) {
    if (selectedLoras.has(loraName)) {
        toast(`${loraName} already added`, 'info');
        return;
    }

    // Use provided weight, or look up default, or fallback to 1.0
    if (weight === null) {
        const info = availableLorasMap.get(loraName);
        weight = info && info.strength_default !== undefined ? info.strength_default : 1.0;
    }

    selectedLoras.set(loraName, weight);

    const container = byId('lora-selected');
    if (!container) return;

    const card = document.createElement('div');
    card.className = 'lora-card';
    card.dataset.lora = loraName;
    card.innerHTML = `
        <div class="lora-card-header">
            <span class="lora-card-name">${loraName}</span>
            <button class="btn btn-ghost btn-icon btn-sm lora-remove" title="Remove">✕</button>
        </div>
        <div class="lora-card-slider">
            <input type="range" class="slider lora-weight" min="0" max="2" step="0.1" value="${weight}" />
            <span class="lora-weight-val">${weight.toFixed(1)}</span>
        </div>
    `;

    // Weight slider handler
    const slider = card.querySelector('.lora-weight');
    const valDisplay = card.querySelector('.lora-weight-val');
    slider.oninput = () => {
        const val = parseFloat(slider.value);
        valDisplay.textContent = val.toFixed(1);
        selectedLoras.set(loraName, val);
    };

    // Remove handler
    card.querySelector('.lora-remove').onclick = () => {
        selectedLoras.delete(loraName);
        card.remove();
    };

    container.appendChild(card);
}

function getSelectedLoras() {
    return Array.from(selectedLoras.entries()).map(([name, weight]) => ({ name, weight }));
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
// GALLERY
// ═══════════════════════════════════════════════════════════════

let galleryData = [];
let galleryPage = 0;
const SESSIONS_PER_PAGE = 30;

async function loadGallery() {
    try {
        galleryPage = 0;
        const data = await api.getGallery();
        // API returns array directly, not {sessions: [...]}
        galleryData = Array.isArray(data) ? data : (data.sessions || []);
        // Sort newest first
        galleryData.sort((a, b) => {
            const tsA = (a.meta?.timestamp || a.timestamp || 0);
            const tsB = (b.meta?.timestamp || b.timestamp || 0);
            return tsB - tsA;
        });

        // Render with current search term if exists
        const searchTerm = byId('gallery-search')?.value || '';
        renderGallery(searchTerm);
    } catch (error) {
        console.error('Failed to load gallery:', error);
        toast('Failed to load gallery', 'error');
    }
}

// Search filter state
let currentSearchTerm = '';

function renderGallery(filterText = '') {
    currentSearchTerm = filterText.toLowerCase();
    const container = byId('gallery-sessions');
    const emptyState = byId('gallery-empty');
    const countEl = byId('gallery-count');

    // Filter data
    let filteredData = galleryData;
    if (currentSearchTerm) {
        filteredData = galleryData.filter(session => {
            const prompt = (session.meta?.prompt || session.prompt || '').toLowerCase();
            return prompt.includes(currentSearchTerm);
        });
    }

    if (!filteredData.length) {
        hide(container);
        show(emptyState);
        countEl.textContent = '0 images';
        // If searching but no results
        if (currentSearchTerm) {
            byId('gallery-empty').innerHTML = `
                <div class="gallery-empty-content">
                    <h3>No matches found</h3>
                    <p>Try a different search term</p>
                </div>
            `;
        }
        return;
    }

    show(container);
    hide(emptyState);

    // Count total images
    const totalImages = filteredData.reduce((sum, session) => sum + (session.images?.length || 0), 0);
    countEl.textContent = `${totalImages} image${totalImages !== 1 ? 's' : ''}`;

    // Render only current page of sessions
    const startIdx = 0;
    const endIdx = (galleryPage + 1) * SESSIONS_PER_PAGE;
    const sessionsToRender = filteredData.slice(startIdx, endIdx);

    container.innerHTML = sessionsToRender.map(session => renderSession(session)).join('');

    // Add "Load More" button if there are more sessions
    if (endIdx < filteredData.length) {
        container.innerHTML += `
            <div class="gallery-load-more">
                <button class="btn btn-secondary" id="load-more-btn">
                    Load More (${filteredData.length - endIdx} more sessions)
                </button>
            </div>
        `;
        listen(byId('load-more-btn'), 'click', loadMoreSessions);
    }

    // Setup click handlers
    container.querySelectorAll('.session-header').forEach(header => {
        listen(header, 'click', (e) => {
            if (!e.target.closest('button')) {
                header.closest('.session-group').classList.toggle('collapsed');
            }
        });
    });

    container.querySelectorAll('.image-item').forEach((item, index) => {
        listen(item, 'click', (e) => {
            if (e) e.stopPropagation();
            // Recalculate index based on full filtered list if needed, but here we just pass elements
            // Note: lightbox needs context of all images in current view
            const allImages = Array.from(container.querySelectorAll('.image-item img'));
            const clickedImg = item.querySelector('img');
            const realIndex = allImages.indexOf(clickedImg);
            openLightbox(allImages, realIndex);
        });
    });
}

function loadMoreSessions() {
    galleryPage++;
    renderGallery(currentSearchTerm);
}

function setupGallerySearch() {
    const searchInput = byId('gallery-search');
    if (!searchInput) return;

    let timeout;
    listen(searchInput, 'input', (e) => {
        const val = e.target.value; // Capture value immediately
        clearTimeout(timeout);
        timeout = setTimeout(() => {
            galleryPage = 0;
            renderGallery(val);
        }, 300);
    });
}

function renderSession(session) {
    const images = session.images || [];
    const variants = session.variants || {}; // Map of original filename -> upscaled URL
    // Meta is nested in session.meta
    const meta = session.meta || {};
    const prompt = meta.prompt || session.prompt || 'No prompt';
    const model = meta.base_model || session.model || 'Unknown';
    const seed = meta.seed || session.seed || '-';
    const timestamp = meta.timestamp || session.timestamp || Date.now() / 1000;

    // Embed full session metadata for lightbox
    const sessionJson = encodeURIComponent(JSON.stringify(session));

    return `
    <div class="session-group" data-json="${sessionJson}">
      <div class="session-header">
        <div class="session-header-left">
          <div class="session-expand">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="6 9 12 15 18 9"></polyline>
            </svg>
          </div>
          <div class="session-info">
            <div class="session-prompt">${escapeHtml(prompt.substring(0, 100))}</div>
            <div class="session-meta">
              <span class="session-meta-item">${model}</span>
              <span class="session-meta-item">Seed: ${seed}</span>
              <span class="session-meta-item">${timeAgo(timestamp)}</span>
            </div>
          </div>
        </div>
        <div class="session-header-right">
          <span class="session-image-count">${images.length} image${images.length !== 1 ? 's' : ''}</span>
        </div>
      </div>
      <div class="session-body">
        <div class="image-grid">
          ${images.map((img, i) => {
        const url = img.url || img;
        // Extract filename from URL to check variants
        const filename = url.split('/').pop();
        const variantUrl = variants[filename];
        return `
            <div class="image-item" data-src="${url}" data-index="${i}" ${variantUrl ? `data-variant="${variantUrl}"` : ''}>
              <img src="${url}" alt="Generated image" loading="lazy" />
              ${variantUrl ? '<span class="hd-badge">HD</span>' : ''}
            </div>
          `;
    }).join('')}
        </div>
      </div>
    </div>
  `;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function timeAgo(timestamp) {
    const seconds = Math.floor((Date.now() - timestamp * 1000) / 1000);
    if (seconds < 60) return 'Just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
}

/**
 * Open PhotoSwipe lightbox with full features
 */
let currentPswpInstance = null;

function openLightbox(dataSource, startIndex = 0) {
    let items;

    // Check if dataSource is DOM elements (NodeList or Array of Elements)
    if (dataSource.length > 0 && (dataSource[0].tagName || dataSource[0] instanceof Element)) {
        items = Array.from(dataSource).map(img => {
            const item = img.closest('.image-item');
            const sessionGroup = img.closest('.session-group');
            const sessionMeta = sessionGroup ? extractSessionMeta(sessionGroup) : {};

            // Check for pre-existing upscaled variant from data attribute
            const variantUrl = item?.dataset.variant || null;

            return {
                src: img.src.replace('/thumbs/', '/').replace('_thumb', ''),
                width: img.naturalWidth || 1024,
                height: img.naturalHeight || 1024,
                msrc: img.src,
                alt: img.alt || '',
                meta: sessionMeta,
                originalSrc: img.src.replace('/thumbs/', '/').replace('_thumb', ''),
                variant: variantUrl, // Pre-loaded upscaled version
                isShowingVariant: false
            };
        });
    } else {
        // Assume already formatted data objects
        items = dataSource;
    }

    const lightbox = new PhotoSwipeLightbox({
        dataSource: items,
        pswpModule: PhotoSwipe,
        index: startIndex,
        bgOpacity: 0.95,
        showHideAnimationType: 'fade',
        closeOnVerticalDrag: true,
        padding: { top: 20, bottom: 200, left: 20, right: 20 },
        // Disable default tap actions to prevent conflicts
        imageClickAction: false,
        tapAction: false,
        wheelToZoom: true
    });

    // Manual DOM Injection Strategy
    lightbox.on('openingAnimationStart', () => {
        const pswpEl = lightbox.pswp.element;
        const infoPanel = byId('lightbox-info');
        const toggleBtn = byId('lightbox-info-toggle');
        const comp = byId('lightbox-comparison');

        // Move UI elements INTO the PhotoSwipe container
        if (pswpEl) {
            if (infoPanel) {
                pswpEl.appendChild(infoPanel);
                infoPanel.style.display = 'block';
                infoPanel.classList.remove('hidden');
                // force-visible removed to allow toggling

                // Stop propagation to prevent closing
                listen(infoPanel, 'pointerdown', (e) => e.stopPropagation());
                listen(infoPanel, 'mousedown', (e) => e.stopPropagation());
                listen(infoPanel, 'click', (e) => e.stopPropagation());
            }
            if (toggleBtn) {
                pswpEl.appendChild(toggleBtn);
                toggleBtn.style.display = 'block';
            }
            if (comp) {
                pswpEl.appendChild(comp);
            }
        }
    });

    lightbox.on('destroy', () => {
        const infoPanel = byId('lightbox-info');
        const toggleBtn = byId('lightbox-info-toggle');
        const comp = byId('lightbox-comparison');
        const viewHdBtn = byId('lightbox-view-hd');
        const compareBtn = byId('lightbox-compare');

        // Safely move back to body and hide
        if (infoPanel) {
            document.body.appendChild(infoPanel);
            infoPanel.classList.add('hidden');
        }
        if (toggleBtn) {
            document.body.appendChild(toggleBtn);
            toggleBtn.style.display = 'none';
        }
        if (comp) {
            document.body.appendChild(comp);
            comp.style.display = 'none';
        }

        // Reset button text states to default
        if (viewHdBtn) {
            viewHdBtn.textContent = '👁️ View HD';
            viewHdBtn.style.display = 'none';
        }
        if (compareBtn) {
            compareBtn.textContent = '↔ Compare';
            compareBtn.style.display = 'none';
        }

        currentPswpInstance = null;
    });

    // Removed closingAnimationStart to prevent premature hiding

    // Ensure content is updated
    lightbox.on('contentActivate', ({ content }) => {
        if (content && content.data) {
            try {
                updateLightboxMeta(content.data.meta || content.data);
                updateLightboxButtons(content.data);
            } catch (e) {
                console.error('Failed to update lightbox meta:', e);
            }
        }
    });

    // Update metadata panel on slide change
    lightbox.on('change', () => {
        const curr = lightbox.pswp.currSlide.data;
        updateLightboxMeta(curr.meta);
        updateLightboxButtons(curr);
    });

    lightbox.init();
    lightbox.loadAndOpen(startIndex);

    // Store instance for external access
    currentPswpInstance = lightbox;
}

function extractSessionMeta(sessionGroup) {
    // Extract metadata from session group DOM via data attribute
    const json = sessionGroup.dataset.json;
    if (json) {
        try {
            const session = JSON.parse(decodeURIComponent(json));
            return session.meta || session;
        } catch (e) {
            console.warn('Failed to parse session meta', e);
        }
    }

    // Fallback to DOM scraping
    const promptEl = sessionGroup.querySelector('.session-prompt');
    const metaItems = sessionGroup.querySelectorAll('.session-meta-item');

    return {
        prompt: promptEl?.textContent || '',
        base_model: metaItems[0]?.textContent || '',
        seed: metaItems[1]?.textContent?.replace('Seed: ', '') || '',
        timestamp: metaItems[2]?.textContent || ''
    };
}

function updateLightboxMeta(meta) {
    if (!meta) return;

    const setText = (id, text) => {
        const el = byId(id);
        if (el) el.textContent = text || '--';
    };

    setText('lightbox-prompt', meta.prompt || 'No prompt');
    setText('lightbox-negative', meta.negative || meta.negative_prompt || 'None');
    setText('lightbox-model', meta.base_model || meta.model || 'Unknown');
    setText('lightbox-seed', meta.seed || '--');

    const settings = [];
    if (meta.steps) settings.push(`Steps: ${meta.steps}`);
    if (meta.cfg) settings.push(`CFG: ${meta.cfg}`);
    if (meta.scheduler) settings.push(meta.scheduler);
    if (meta.width && meta.height) settings.push(`${meta.width}×${meta.height}`);

    setText('lightbox-settings', settings.join(' · ') || '--');
}

function updateLightboxButtons(curr) {
    const viewHdBtn = byId('lightbox-view-hd');
    const compareBtn = byId('lightbox-compare');

    if (viewHdBtn) {
        viewHdBtn.style.display = curr.variant ? 'inline-flex' : 'none';
    }
    if (compareBtn) {
        compareBtn.style.display = curr.variant ? 'inline-flex' : 'none';
    }
}

// Setup lightbox action buttons
function initLightboxActions() {
    // Toggle info panel
    listen(byId('lightbox-info-toggle'), 'click', () => {
        const info = byId('lightbox-info');
        if (info) {
            info.style.display = info.style.display === 'none' ? 'block' : 'none';
        }
    });

    // Regenerate button
    listen(byId('lightbox-regen'), 'click', async () => {
        if (!currentPswpInstance?.pswp) return;
        const curr = currentPswpInstance.pswp.currSlide.data;
        if (!curr.meta) {
            toast('No metadata available', 'error');
            return;
        }

        // Copy settings to form
        const setVal = (id, val) => {
            const el = byId(id);
            if (el && val !== undefined) el.value = val;
            // Trigger change event for sliders/selects
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
    });

    // Upscale button
    listen(byId('lightbox-upscale'), 'click', () => {
        if (!currentPswpInstance?.pswp) return;
        const curr = currentPswpInstance.pswp.currSlide.data;
        startUpscale(curr.src, (upscaledUrl) => {
            curr.variant = upscaledUrl;
            updateLightboxButtons(curr);
        });
    });

    // Inpaint button - send gallery image to inpainting
    listen(byId('lightbox-inpaint'), 'click', () => {
        if (!currentPswpInstance?.pswp) return;
        const curr = currentPswpInstance.pswp.currSlide.data;

        // Close lightbox first
        currentPswpInstance.pswp.close();

        // Send image to inpaint workflow
        sendToInpaint(curr.src);
    });

    // View HD toggle
    listen(byId('lightbox-view-hd'), 'click', () => {
        if (!currentPswpInstance?.pswp) return;
        const curr = currentPswpInstance.pswp.currSlide.data;
        if (!curr.variant) return;

        const btn = byId('lightbox-view-hd');

        if (curr.isShowingVariant) {
            curr.src = curr.originalSrc;
            curr.isShowingVariant = false;
            btn.textContent = '👁️ View HD';
        } else {
            curr.src = curr.variant;
            curr.isShowingVariant = true;
            btn.textContent = '↩️ Original';
        }

        currentPswpInstance.pswp.refreshSlideContent(currentPswpInstance.pswp.currSlide.index);
    });

    // Compare slider toggle
    listen(byId('lightbox-compare'), 'click', () => {
        if (!currentPswpInstance?.pswp) return;
        const curr = currentPswpInstance.pswp.currSlide.data;
        if (!curr.variant) return;

        const compContainer = byId('lightbox-comparison');
        const compOriginal = byId('comp-original');
        const compModified = byId('comp-modified');
        const compHandle = byId('comp-handle');
        const btn = byId('lightbox-compare');

        const isOpen = compContainer.style.display !== 'none';

        if (isOpen) {
            compContainer.style.display = 'none';
            btn.textContent = '↔ Compare';
        } else {
            compContainer.style.display = 'flex';
            compOriginal.src = curr.originalSrc;
            compModified.src = curr.variant;
            compHandle.style.left = '50%';
            byId('comp-modified').style.clipPath = 'inset(0 0 0 50%)';
            btn.textContent = '✕ Close';

            initCompareSlider();
        }
    });

    // Delete button
    listen(byId('lightbox-delete'), 'click', () => {
        toggleClass(byId('confirmation-modal'), 'hidden', false);
    });

    // Modal cancel
    listen(byId('modal-cancel'), 'click', () => {
        toggleClass(byId('confirmation-modal'), 'hidden', true);
    });

    // Modal delete image
    listen(byId('modal-delete-img'), 'click', async () => {
        if (!currentPswpInstance?.pswp) return;
        const curr = currentPswpInstance.pswp.currSlide.data;

        try {
            const formData = new FormData();
            formData.append('path', curr.src);
            const res = await fetch('/delete_image', { method: 'POST', body: formData });

            if (res.ok) {
                currentPswpInstance.pswp.close();
                toggleClass(byId('confirmation-modal'), 'hidden', true);
                loadGallery();
                toast('Image deleted', 'success');
            } else {
                toast('Failed to delete', 'error');
            }
        } catch (e) {
            toast('Delete failed', 'error');
        }
    });

    // Modal delete run
    listen(byId('modal-delete-run'), 'click', async () => {
        if (!currentPswpInstance?.pswp) return;
        const curr = currentPswpInstance.pswp.currSlide.data;

        try {
            const formData = new FormData();
            formData.append('path', curr.src);
            const res = await fetch('/delete_run', { method: 'POST', body: formData });

            if (res.ok) {
                currentPswpInstance.pswp.close();
                toggleClass(byId('confirmation-modal'), 'hidden', true);
                loadGallery();
                toast('Run deleted', 'success');
            } else {
                toast('Failed to delete', 'error');
            }
        } catch (e) {
            toast('Delete failed', 'error');
        }
    });
}

// Compare slider drag handling
function initCompareSlider() {
    const compContainer = byId('lightbox-comparison');
    const compHandle = byId('comp-handle');
    const compModified = byId('comp-modified');

    if (!compContainer) return;

    let isDragging = false;

    const setSliderPos = (pct) => {
        pct = Math.max(0, Math.min(100, pct));
        compHandle.style.left = `${pct}%`;
        compModified.style.clipPath = `inset(0 0 0 ${pct}%)`;
    };

    const onMove = (e) => {
        if (!isDragging) return;
        e.preventDefault();
        const rect = compContainer.getBoundingClientRect();
        const x = (e.clientX || e.touches?.[0]?.clientX || 0) - rect.left;
        setSliderPos((x / rect.width) * 100);
    };

    const onUp = () => {
        isDragging = false;
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onUp);
    };

    compContainer.addEventListener('pointerdown', (e) => {
        isDragging = true;
        onMove(e);
        window.addEventListener('pointermove', onMove);
        window.addEventListener('pointerup', onUp);
    });
}

// Setup gallery refresh button
listen(byId('refresh-gallery'), 'click', loadGallery);

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
