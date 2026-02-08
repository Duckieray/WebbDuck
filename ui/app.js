import * as api from './core/api.js';
import { initState, getState, setState, setSeed, setLastUsedSeed, syncFromDOM, syncToDOM } from './core/state.js';
import { emit, on, Events, initWebSocket } from './core/events.js';
import { $$, byId, listen, show, hide, toggleClass, populateSelect, toast, debounce, toggleSection } from './core/utils.js';
import { ProgressManager } from './modules/ProgressManager.js';
import { MaskEditor } from './modules/MaskEditor.js';
import { LoraManager } from './modules/LoraManager.js';
import { LightboxManager } from './modules/LightboxManager.js';
import { GalleryManager } from './modules/GalleryManager.js';

let isGenerating = false;
const seenCompletedQueueJobs = new Set();
const queueViewStartedAt = Date.now() / 1000;
let latestQueuePayload = null;
const expandedQueueJobs = new Set();

window._uploadedImage = null;
window._maskBlob = null;

document.addEventListener('DOMContentLoaded', async () => {
    try {
        initState();

        window.progressManager = new ProgressManager();
        window.maskEditor = new MaskEditor();
        window.loraManager = new LoraManager();
        window.galleryManager = new GalleryManager();
        window.lightboxManager = new LightboxManager({
            onUpscale: (src, cb) => startUpscale(src, cb),
            onInpaint: (src) => sendToInpaint(src),
            onRegenerate: handleRegenerateFromLightbox,
            onDelete: (src, type) => window.galleryManager.handleDelete(src, type)
        });

        window.galleryManager.init();

        setupNavigation();
        setupCollapsibleSections();
        setupSliders();
        setupPresetChips();
        setupFormHandlers();
        setupGenerationButtons();
        setupUploadHandling();
        setupPreviewToolbar();
        setupQueuePanel();
        setupRealtimeGalleryRefresh();
        setupCatalogWatcher();

        await Promise.all([loadModels(), loadSchedulers(), window.galleryManager.load()]);

        syncToDOM();
        updateActivePresetChip(byId('width')?.value, byId('height')?.value);
        ensureSelectDefaults();

        const savedView = getState('view') || 'studio';
        switchView(savedView);

        const promptEl = byId('prompt');
        if (promptEl?.value?.trim()) {
            await updateTokenCounter(promptEl.value);
        }

        const state = getState();
        if (state.lastSeed) {
            const seedEl = byId('last-seed');
            if (seedEl) seedEl.textContent = state.lastSeed;
        }

        initWebSocket();
    } catch (error) {
        console.error('CRITICAL UI INIT FAILURE:', error);
        alert(`CRITICAL UI ERROR: ${error.message}`);
    }
});

function setupNavigation() {
    $$('.nav-tab').forEach(tab => {
        listen(tab, 'click', () => switchView(tab.dataset.view));
    });

    $$('.mobile-tab').forEach(tab => {
        listen(tab, 'click', () => switchView(tab.dataset.view));
    });
}

function switchView(viewName) {
    const nextView = viewName === 'gallery' ? 'gallery' : 'studio';

    $$('.nav-tab, .mobile-tab').forEach(tab => {
        toggleClass(tab, 'active', tab.dataset.view === nextView);
    });

    $$('.view').forEach(view => {
        toggleClass(view, 'active', view.id === `view-${nextView}`);
    });

    setState({ view: nextView });

    if (nextView === 'gallery' && window.galleryManager) {
        window.galleryManager.load();
        emit(Events.VIEW_CHANGE, nextView);
    }
}

function setupCollapsibleSections() {
    $$('[data-section]').forEach(btn => {
        listen(btn, 'click', () => {
            const id = btn.dataset.section;
            if (id) toggleSection(id);
        });
    });
}

function setupSliders() {
    const pairs = [
        ['steps', 'steps-value'],
        ['cfg', 'cfg-value'],
        ['batch', 'batch-value'],
        ['second_pass_steps', 'second-steps-value'],
        ['second_pass_blend', 'blend-value'],
        ['denoising_strength', 'denoise-value']
    ];

    pairs.forEach(([inputId, outputId]) => {
        const input = byId(inputId);
        const output = byId(outputId);
        if (!input || !output) return;

        const update = () => {
            output.textContent = input.value;
        };

        update();
        listen(input, 'input', update);
    });
}

function setupPresetChips() {
    $$('.preset-chip[data-width]').forEach(chip => {
        listen(chip, 'click', () => {
            byId('width').value = chip.dataset.width;
            byId('height').value = chip.dataset.height;

            updateActivePresetChip(chip.dataset.width, chip.dataset.height);

            syncFromDOM();
        });
    });

    // Keep preset highlight in sync for manual width/height edits and restored state.
    const syncPreset = () => updateActivePresetChip(byId('width')?.value, byId('height')?.value);
    listen(byId('width'), 'input', syncPreset);
    listen(byId('height'), 'input', syncPreset);
    syncPreset();
}

function updateActivePresetChip(width, height) {
    const w = String(width ?? '');
    const h = String(height ?? '');
    let matched = false;

    $$('.preset-chip[data-width]').forEach(chip => {
        const isMatch = chip.dataset.width === w && chip.dataset.height === h;
        chip.classList.toggle('active', isMatch);
        if (isMatch) matched = true;
    });

    const customChip = byId('preset-custom');
    if (customChip) {
        customChip.classList.toggle('active', !matched);
    }
}

function setupFormHandlers() {
    const saveState = debounce(() => syncFromDOM(), 250);

    ['prompt', 'negative', 'width', 'height', 'steps', 'cfg', 'scheduler', 'batch', 'seed_input', 'second_pass_steps', 'second_pass_blend', 'second_pass_enabled', 'second_pass_model', 'denoising_strength'].forEach(id => {
        const el = byId(id);
        if (!el) return;
        listen(el, 'input', saveState);
        listen(el, 'change', saveState);
    });

    const promptEl = byId('prompt');
    if (promptEl) {
        listen(promptEl, 'input', debounce(() => updateTokenCounter(promptEl.value), 300));
    }

    listen(byId('randomize-seed'), 'click', () => {
        const seed = byId('seed_input');
        const nextSeed = generateRandomSeed();
        if (seed) seed.value = String(nextSeed);
        setSeed(nextSeed);
    });

    listen(byId('base_model'), 'change', async () => {
        const modelName = byId('base_model')?.value;
        if (!modelName) return;

        setState({ baseModel: modelName });
        await window.loraManager.loadForModel(modelName);
        emit(Events.MODEL_CHANGE, modelName);
        updateTokenCounter(byId('prompt')?.value || '');
    });

    listen(byId('inpaint-replace'), 'click', () => {
        setState({ inpaintMode: 'replace' });
        byId('inpaint-replace')?.classList.add('active');
        byId('inpaint-keep')?.classList.remove('active');
    });

    listen(byId('inpaint-keep'), 'click', () => {
        setState({ inpaintMode: 'keep' });
        byId('inpaint-keep')?.classList.add('active');
        byId('inpaint-replace')?.classList.remove('active');
    });
}

function generateRandomSeed() {
    const maxSeed = 2147483647;
    if (window.crypto?.getRandomValues) {
        const data = new Uint32Array(1);
        window.crypto.getRandomValues(data);
        return (data[0] % maxSeed) + 1;
    }
    return Math.floor(Math.random() * maxSeed) + 1;
}

async function updateTokenCounter(prompt) {
    const counter = byId('token-count');
    if (!counter) return;

    if (!prompt.trim()) {
        counter.textContent = '0 tokens';
        counter.classList.remove('warning', 'danger');
        counter.removeAttribute('title');
        counter.removeAttribute('aria-label');
        return;
    }

    try {
        const model = byId('base_model')?.value;
        if (!model) return;

        const result = await api.tokenize(prompt, model);
        const count = result.tokens || 0;
        const overLimit = count > 77;
        const nearLimit = count > 60 && !overLimit;
        const tooltipText = 'Prompt exceeds the CLIP token window. Only the first 77 tokens are guaranteed to be parsed.';

        counter.textContent = `${count} tokens`;
        toggleClass(counter, 'warning', nearLimit);
        toggleClass(counter, 'danger', overLimit);
        if (overLimit) {
            counter.title = tooltipText;
            counter.setAttribute('aria-label', tooltipText);
        } else {
            counter.removeAttribute('title');
            counter.removeAttribute('aria-label');
        }
    } catch (error) {
        console.warn('Token count failed:', error);
    }
}

function setupGenerationButtons() {
    listen(byId('btn-test'), 'click', () => startGeneration('test'));
    listen(byId('btn-generate'), 'click', () => startGeneration('generate'));
    listen(byId('cancel-generation'), 'click', cancelGeneration);

    listen(document, 'keydown', (event) => {
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
            event.preventDefault();
            startGeneration('generate');
        }
    });
}

async function startGeneration(mode) {
    try {
        syncFromDOM();
        const formData = collectFormData();

        if (mode === 'test') {
            if (isGenerating) return;
            isGenerating = true;
            window.progressManager?.showProgress('Starting...', 0);
            emit(Events.GENERATION_START, mode);

            const result = await api.testGenerate(formData);
            handleGenerationResult(result);
            emit(Events.GENERATION_COMPLETE, result);
            return;
        }

        formData.append('wait_for_result', 'false');
        const queued = await api.generate(formData);
        const pos = queued?.queue_position;
        toast(pos ? `Queued (position ${pos})` : 'Queued', 'success');
        emit(Events.GENERATION_START, 'queued');
    } catch (error) {
        console.error('Generation error:', error);
        toast(error.message || 'Generation failed', 'error');
        emit(Events.GENERATION_ERROR, error);
    } finally {
        if (mode === 'test') {
            isGenerating = false;
            window.progressManager?.hideProgress();
        }
    }
}

function collectFormData() {
    const formData = new FormData();
    const state = getState();

    formData.append('prompt', byId('prompt')?.value || '');
    const negative = byId('negative')?.value || '';
    formData.append('negative_prompt', negative);
    // Keep legacy key for compatibility with any older handlers.
    formData.append('negative', negative);
    formData.append('width', state.width || byId('width')?.value || 1024);
    formData.append('height', state.height || byId('height')?.value || 1024);
    formData.append('steps', byId('steps')?.value || 30);
    formData.append('cfg', byId('cfg')?.value || 7.5);
    formData.append('scheduler', byId('scheduler')?.value || '');
    formData.append('num_images', byId('batch')?.value || 1);
    formData.append('base_model', byId('base_model')?.value || '');

    const seedVal = byId('seed_input')?.value;
    if (seedVal) formData.append('seed', seedVal);

    if (byId('second_pass_enabled')?.checked) {
        formData.append('second_pass_model', byId('second_pass_model')?.value || 'None');
        formData.append('second_pass_steps', byId('second_pass_steps')?.value || 20);
        formData.append('second_pass_blend', byId('second_pass_blend')?.value || 0.8);
    }

    const loras = window.loraManager?.getSelected() || [];
    if (loras.length > 0) {
        formData.append('loras', JSON.stringify(loras));
        loras.forEach((lora, i) => {
            formData.append(`lora_model_${i + 1}`, lora.name);
            formData.append(`lora_weight_${i + 1}`, lora.weight);
        });
    }

    if (window._uploadedImage) {
        formData.append('image', window._uploadedImage);
        formData.append('denoising_strength', byId('denoising_strength')?.value || 0.75);
    }

    if (window._maskBlob) {
        formData.append('mask', window._maskBlob);
        formData.append('inpainting_fill', getState().inpaintMode || 'replace');
        formData.append('mask_blur', byId('mask_blur')?.value || 8);
    }

    return formData;
}

function handleGenerationResult(result) {
    if (!result?.images?.length) return;

    const preview = byId('preview-image');
    const placeholder = byId('preview-placeholder');

    preview.src = result.images[0];
    preview.classList.remove('hidden');
    placeholder.classList.add('hidden');

    preview.dataset.meta = JSON.stringify(result.meta || result);

    updateBatchStrip(result.images);

    if (result.seed !== undefined) {
        setLastUsedSeed(result.seed);
        byId('last-seed').textContent = result.seed;
    }

    toast(`Generated ${result.images.length} image(s)`, 'success');
}

function updateBatchStrip(images) {
    const strip = byId('batch-strip');
    if (!strip) return;

    strip.innerHTML = images.map((src, i) => {
        return `<button class="image-item" type="button" data-index="${i}"><img src="${src}" alt="Variant ${i + 1}" loading="lazy" /></button>`;
    }).join('');

    show(strip);

    strip.querySelectorAll('.image-item').forEach(item => {
        listen(item, 'click', () => {
            const img = item.querySelector('img');
            if (img) byId('preview-image').src = img.src;
        });
    });
}

function cancelGeneration() {
    isGenerating = false;
    window.progressManager?.hideProgress();
    toast('Generation cancelled', 'warning');
    emit(Events.GENERATION_CANCEL);
}

function setupUploadHandling() {
    const dropZone = byId('upload-drop');
    const fileInput = byId('input-image');

    if (!dropZone || !fileInput) return;

    listen(dropZone, 'click', () => fileInput.click());

    listen(fileInput, 'change', event => {
        const file = event.target.files?.[0];
        if (file) handleImageUpload(file);
    });

    listen(dropZone, 'dragover', event => {
        event.preventDefault();
        dropZone.classList.add('drag-over');
    });

    listen(dropZone, 'dragleave', () => dropZone.classList.remove('drag-over'));

    listen(dropZone, 'drop', event => {
        event.preventDefault();
        dropZone.classList.remove('drag-over');
        const file = event.dataTransfer?.files?.[0];
        if (file) handleImageUpload(file);
    });

    listen(byId('clear-upload'), 'click', clearUploadedImage);

    listen(byId('caption-btn'), 'click', async () => {
        if (!window._uploadedImage) {
            toast('No image to caption', 'error');
            return;
        }

        try {
            toast('Generating prompt from image...', 'info');
            const formData = new FormData();
            formData.append('image', window._uploadedImage);
            formData.append('style', 'art');

            const result = await api.caption(formData);
            if (result?.caption) {
                const prompt = byId('prompt');
                prompt.value = result.caption;
                prompt.dispatchEvent(new Event('input'));
                toast('Caption generated', 'success');
            } else {
                toast('No caption returned', 'warning');
            }
        } catch (error) {
            console.error('Caption error:', error);
            toast(`Caption failed: ${error.message}`, 'error');
        }
    });

    listen(byId('edit-mask-btn'), 'click', () => {
        const src = byId('preview-img')?.src;
        if (!src) {
            toast('Upload an image first', 'error');
            return;
        }

        window.maskEditor?.open(src);
    });
}

async function handleImageUpload(file) {
    if (!file.type.startsWith('image/')) {
        toast('Please upload an image file', 'error');
        return;
    }

    window._uploadedImage = file;
    window._maskBlob = null;

    const reader = new FileReader();
    reader.onload = event => {
        const src = event.target.result;
        const preview = byId('preview-img');
        preview.src = src;

        const probe = new Image();
        probe.onload = () => {
            const fit = fitResolution(probe.width, probe.height, 2048);

            const widthInput = byId('width');
            const heightInput = byId('height');

            widthInput.value = fit.width;
            heightInput.value = fit.height;

            widthInput.dispatchEvent(new Event('input'));
            heightInput.dispatchEvent(new Event('input'));

            setState({ width: fit.width, height: fit.height });
            toast(`Resolution set to ${fit.width}x${fit.height}`, 'info');
        };
        probe.src = src;

        hide(byId('upload-drop'));
        show(byId('upload-preview'));
        show(byId('denoise-group'));
    };

    reader.readAsDataURL(file);
    emit(Events.IMAGE_UPLOAD, file);
}

function fitResolution(width, height, maxDim) {
    let w = width;
    let h = height;

    if (w > maxDim || h > maxDim) {
        const ratio = Math.min(maxDim / w, maxDim / h);
        w = Math.round(w * ratio);
        h = Math.round(h * ratio);
    }

    w = Math.max(512, Math.round(w / 8) * 8);
    h = Math.max(512, Math.round(h / 8) * 8);

    return { width: w, height: h };
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

function setupPreviewToolbar() {
    listen(byId('preview-zoom'), 'click', () => {
        const img = byId('preview-image');
        if (!img?.src) return;

        let meta = {};
        try {
            meta = img.dataset.meta ? JSON.parse(img.dataset.meta) : {};
        } catch (_) {
            meta = {};
        }

        const item = {
            src: img.src,
            width: img.naturalWidth || 1024,
            height: img.naturalHeight || 1024,
            msrc: img.src,
            alt: 'Preview',
            meta,
            originalSrc: img.src
        };

        window.lightboxManager?.open([item], 0);
    });

    listen(byId('preview-upscale'), 'click', () => {
        const img = byId('preview-image');
        if (!img?.src) return;
        startUpscale(img.src);
    });

    listen(byId('preview-inpaint'), 'click', () => {
        const img = byId('preview-image');
        if (!img?.src) return;
        sendToInpaint(img.src);
    });

    listen(byId('preview-download'), 'click', () => {
        const img = byId('preview-image');
        if (!img?.src) return;

        const anchor = document.createElement('a');
        anchor.href = img.src;
        anchor.download = `webbduck-${Date.now()}.png`;
        document.body.appendChild(anchor);
        anchor.click();
        document.body.removeChild(anchor);
    });
}

async function startUpscale(imageSrc, onSuccess) {
    try {
        toast('Upscaling...', 'info');

        if (imageSrc.startsWith('data:')) {
            toast('Upscale requires a saved image from gallery/outputs', 'warning');
            return;
        }

        const formData = new FormData();
        const normalizedImagePath = normalizeImagePathForUpscale(imageSrc);
        formData.append('image', normalizedImagePath);
        formData.append('scale', '2');

        const data = await api.upscale(formData);
        const rawUpscaled = data?.upscaled || data?.image || data?.path || data?.url;
        const upscaledUrl = normalizeUpscaledUrl(rawUpscaled);

        if (upscaledUrl) {
            toast('Upscale complete', 'success');
            if (onSuccess) {
                onSuccess(upscaledUrl);
            } else {
                const preview = byId('preview-image');
                const placeholder = byId('preview-placeholder');
                if (preview) preview.src = upscaledUrl;
                if (preview) preview.classList.remove('hidden');
                if (placeholder) placeholder.classList.add('hidden');
            }
        } else {
            toast('Upscale failed', 'error');
        }
    } catch (error) {
        console.error('Upscale error:', error);
        toast('Upscale failed', 'error');
    }
}

function normalizeImagePathForUpscale(src) {
    try {
        const url = new URL(src, window.location.origin);
        return url.pathname;
    } catch (_) {
        return src;
    }
}

function normalizeUpscaledUrl(pathOrUrl) {
    if (!pathOrUrl) return null;

    if (typeof pathOrUrl !== 'string') return null;
    if (pathOrUrl.startsWith('/')) return pathOrUrl;
    if (/^https?:\/\//i.test(pathOrUrl)) return pathOrUrl;

    const normalized = pathOrUrl.replace(/\\/g, '/');
    const outputMarker = '/outputs/';
    const idx = normalized.lastIndexOf(outputMarker);
    if (idx >= 0) return normalized.slice(idx);

    const lower = normalized.toLowerCase();
    const fallbackIdx = lower.lastIndexOf('outputs/');
    if (fallbackIdx >= 0) return `/${normalized.slice(fallbackIdx)}`;

    return pathOrUrl;
}

async function sendToInpaint(imageSrc) {
    try {
        toast('Loading image for inpaint...', 'info');

        const response = await fetch(imageSrc);
        const blob = await response.blob();
        const file = new File([blob], 'inpaint_source.png', { type: blob.type || 'image/png' });

        await handleImageUpload(file);
        switchView('studio');
        window.scrollTo({ top: 0, behavior: 'smooth' });

        toast('Image loaded for inpaint', 'success');
    } catch (error) {
        console.error('Inpaint transfer error:', error);
        toast('Failed to load image for inpaint', 'error');
    }
}

async function handleRegenerateFromLightbox(curr) {
    if (!curr?.meta) {
        toast('No metadata available', 'error');
        return;
    }

    const meta = curr.meta;

    const setValue = (id, value) => {
        const el = byId(id);
        if (!el || value === undefined || value === null) return;

        el.value = value;
        el.dispatchEvent(new Event('input'));
        el.dispatchEvent(new Event('change'));
    };

    setValue('prompt', meta.prompt);
    setValue('negative', meta.negative || meta.negative_prompt);
    setValue('steps', meta.steps);
    setValue('cfg', meta.cfg);
    setValue('width', meta.width);
    setValue('height', meta.height);
    const baseModel = meta.base_model || meta.model;
    setValue('base_model', baseModel);
    setValue('scheduler', meta.scheduler);

    byId('seed_input').value = '';
    setSeed(null);

    if (baseModel && window.loraManager) {
        try {
            await window.loraManager.loadForModel(baseModel);
        } catch (_) {
            // Ignore and continue with best-effort restoration.
        }
    }

    if (window.loraManager) {
        window.loraManager.clear();
        const loras = Array.isArray(meta.loras) ? meta.loras : [];
        loras.forEach((lora) => {
            if (typeof lora === 'string') {
                window.loraManager.addLora(lora, 1.0);
                return;
            }

            const name = lora?.name || lora?.model;
            if (!name) return;
            const weightRaw = lora?.weight ?? lora?.strength;
            const parsedWeight = Number(weightRaw);
            const weight = Number.isFinite(parsedWeight) ? parsedWeight : 1.0;
            window.loraManager.addLora(name, weight);
        });
    }

    syncFromDOM();
    switchView('studio');
    window.scrollTo({ top: 0, behavior: 'smooth' });

    const generate = byId('btn-generate');
    if (generate) {
        setTimeout(() => generate.click(), 120);
    }

    toast('Regenerate started', 'success');
}

async function loadModels() {
    try {
        const preferredBaseModel = byId('base_model')?.value || getState('baseModel');
        const preferredSecondPassModel = byId('second_pass_model')?.value || getState('secondPassModel');

        const models = await api.getModels();
        populateSelect('base_model', models, false);
        const baseSelect = byId('base_model');
        if (preferredBaseModel && baseSelect && Array.from(baseSelect.options).some(opt => opt.value === preferredBaseModel)) {
            baseSelect.value = preferredBaseModel;
        }

        const secondPassModels = await api.getSecondPassModels();
        populateSelect('second_pass_model', secondPassModels, true);
        const secondPassSelect = byId('second_pass_model');
        if (
            preferredSecondPassModel
            && secondPassSelect
            && Array.from(secondPassSelect.options).some(opt => opt.value === preferredSecondPassModel)
        ) {
            secondPassSelect.value = preferredSecondPassModel;
        }

        const select = byId('base_model');
        const initialModel = select?.value || select?.options?.[0]?.value;

        if (initialModel) {
            await window.loraManager.loadForModel(initialModel);
        }
    } catch (error) {
        console.warn('Failed to load models:', error);
        toast('Failed to load models', 'error');
    }
}

async function loadSchedulers() {
    try {
        const schedulers = await api.getSchedulers();
        populateSelect('scheduler', schedulers, false);
    } catch (error) {
        console.warn('Failed to load schedulers:', error);
        toast('Failed to load schedulers', 'error');
    }
}

function ensureSelectDefaults() {
    const baseModel = byId('base_model');
    if (baseModel && !baseModel.value && baseModel.options.length > 0) {
        baseModel.selectedIndex = 0;
    }

    const scheduler = byId('scheduler');
    if (scheduler && !scheduler.value && scheduler.options.length > 0) {
        scheduler.selectedIndex = 0;
    }

    syncFromDOM();
}

function setupQueuePanel() {
    on(Events.QUEUE_UPDATE, renderQueuePanel);
    listen(byId('open-queue-modal'), 'click', openQueueModal);
    listen(byId('close-queue-modal'), 'click', closeQueueModal);
    listen(byId('close-queue-modal-footer'), 'click', closeQueueModal);
    listen(byId('queue-modal'), 'click', (event) => {
        if (event.target?.id === 'queue-modal') {
            closeQueueModal();
        }
    });
    listen(document, 'keydown', (event) => {
        if (event.key === 'Escape') {
            closeQueueModal();
        }
    });
    refreshQueuePanel();
}

function setupRealtimeGalleryRefresh() {
    on(Events.GENERATION_COMPLETE, () => {
        // Keep gallery up to date immediately after each finished job.
        window.galleryManager?.refreshLatest();
    });
}

async function refreshQueuePanel() {
    try {
        const data = await api.getQueue();
        renderQueuePanel(data);
    } catch (_) {
        const summaryEl = byId('queue-summary');
        if (summaryEl) summaryEl.textContent = 'Queue unavailable';
        updateQueueOpenButton(null);
    }
}

function setupCatalogWatcher() {
    on(Events.CATALOG_UPDATE, async () => {
        const currentBaseModel = byId('base_model')?.value || '';
        try {
            await loadModels();
            const nextBaseModel = byId('base_model')?.value || '';
            if (currentBaseModel && currentBaseModel !== nextBaseModel) {
                toast(`Catalog updated. Active model switched to ${nextBaseModel || 'first available model'}.`, 'info', 2400);
            }
        } catch (error) {
            console.warn('Catalog refresh failed:', error);
        }
    });
}

function renderQueuePanel(data) {
    latestQueuePayload = data || null;
    const summaryEl = byId('queue-summary');
    const listEl = byId('queue-list');
    if (!summaryEl || !listEl) return;

    const jobs = Array.isArray(data?.jobs) ? data.jobs : [];
    const recentCompleted = Array.isArray(data?.recent_completed) ? data.recent_completed : [];
    const queuedCount = data?.queued_count || 0;
    const activeId = data?.active_job_id;
    const runningCount = activeId ? 1 : 0;
    const pendingCount = queuedCount + runningCount;

    summaryEl.textContent = pendingCount > 0
        ? `${queuedCount} queued | ${runningCount} running`
        : 'No jobs queued';

    updateQueueOpenButton(data);

    applyCompletedQueueResults(recentCompleted);

    if (jobs.length === 0) {
        listEl.innerHTML = '<div class="queue-item-empty">Queue is empty.</div>';
        return;
    }

    const recent = jobs.slice(0, 32);
    listEl.innerHTML = recent.map(job => {
        const status = job.status || 'unknown';
        const mode = job.settings?.mode || 'txt2img';
        const prompt = (job.settings?.prompt || '').trim();
        const title = prompt ? escapeHtml(prompt.slice(0, 160)) : '(no prompt)';
        const dims = `${job.settings?.width || '-'}x${job.settings?.height || '-'}`;
        const steps = job.settings?.steps ?? '-';
        const cfg = job.settings?.cfg ?? '-';
        const batch = job.settings?.num_images ?? '-';
        const model = escapeHtml(job.settings?.base_model || '-');
        const scheduler = escapeHtml(job.settings?.scheduler || '-');
        const pos = job.queue_position ? `#${job.queue_position}` : '';
        const startedAt = formatQueueTime(job.started_at || job.created_at);
        const inputThumb = normalizeQueueThumbUrl(job.settings?.input_image_url);
        const seed = job.settings?.seed ?? null;
        const negative = (job.settings?.negative_prompt || '').trim();
        const loras = Array.isArray(job.settings?.loras) ? job.settings.loras : [];
        const thumbHtml = inputThumb
            ? `<img class="queue-item-thumb" src="${escapeHtml(inputThumb)}" alt="Queue input preview" loading="lazy" />`
            : '';
        const canCancel = status === 'queued';
        const statusText = status === 'running' ? 'Running' : status === 'queued' ? 'Queued' : status;
        const isExpanded = expandedQueueJobs.has(job.job_id);
        const hasExtraDetails = seed !== null || negative.length > 0 || loras.length > 0;
        const detailsHtml = hasExtraDetails
            ? `
                <div class="queue-item-details ${isExpanded ? '' : 'hidden'}" data-details-for="${job.job_id}">
                  <div class="queue-detail-row"><span class="queue-detail-label">Seed</span><span class="queue-detail-value">${seed === null ? 'Random' : escapeHtml(String(seed))}</span></div>
                  <div class="queue-detail-row"><span class="queue-detail-label">Negative</span><span class="queue-detail-value">${negative ? escapeHtml(negative.slice(0, 220)) : 'None'}</span></div>
                  <div class="queue-detail-row"><span class="queue-detail-label">LoRAs</span><span class="queue-detail-value">${loras.length > 0 ? escapeHtml(loras.join(', ')) : 'None'}</span></div>
                </div>
              `
            : '';

        return `
              <div class="queue-item status-${status} ${inputThumb ? 'has-thumb' : 'no-thumb'} ${isExpanded ? 'expanded' : ''}" data-job-id="${job.job_id}">
                ${thumbHtml}
                <div class="queue-item-main">
                  <div class="queue-item-badges">
                    <span class="queue-badge status-${status}">${escapeHtml(statusText)}${pos ? ` ${escapeHtml(pos)}` : ''}</span>
                    <span class="queue-badge mode-${escapeHtml(mode)}">${escapeHtml(mode)}</span>
                  </div>
                  <div class="queue-item-title">${title}</div>
                  <div class="queue-item-meta">${escapeHtml(dims)} | steps ${escapeHtml(String(steps))} | cfg ${escapeHtml(String(cfg))} | batch ${escapeHtml(String(batch))}</div>
                  <div class="queue-item-meta">${model} | ${scheduler} | ${escapeHtml(startedAt)}</div>
                  ${detailsHtml}
                </div>
                <div class="queue-item-actions">
                  ${hasExtraDetails ? `<button class="btn btn-ghost btn-sm queue-expand" data-job-id="${job.job_id}" type="button">${isExpanded ? 'Less' : 'More'}</button>` : ''}
                  ${canCancel ? `<button class="btn btn-ghost btn-sm queue-cancel" data-job-id="${job.job_id}" type="button">Cancel</button>` : ''}
                </div>
              </div>
            `;
    }).join('');

    listEl.querySelectorAll('.queue-expand').forEach(btn => {
        listen(btn, 'click', () => {
            const jobId = btn.dataset.jobId;
            if (!jobId) return;
            if (expandedQueueJobs.has(jobId)) {
                expandedQueueJobs.delete(jobId);
            } else {
                expandedQueueJobs.add(jobId);
            }
            renderQueuePanel(latestQueuePayload || data);
        });
    });

    listEl.querySelectorAll('.queue-cancel').forEach(btn => {
        listen(btn, 'click', async () => {
            const jobId = btn.dataset.jobId;
            if (!jobId) return;
            try {
                await api.cancelQueue(jobId);
                toast('Queued job cancelled', 'info');
                // Backend pushes fresh queue state via WebSocket.
            } catch (err) {
                toast(err.message || 'Cancel failed', 'error');
            }
        });
    });
}

function updateQueueOpenButton(data) {
    const countEl = byId('queue-open-count');
    if (!countEl) return;

    const queuedCount = data?.queued_count || 0;
    const runningCount = data?.active_job_id ? 1 : 0;
    const pendingCount = queuedCount + runningCount;

    countEl.textContent = String(pendingCount);
    toggleClass(countEl, 'hidden', pendingCount === 0);
}

function openQueueModal() {
    const modal = byId('queue-modal');
    if (!modal) return;
    modal.classList.remove('hidden');
    modal.classList.add('active');
    modal.setAttribute('aria-hidden', 'false');
    if (latestQueuePayload) {
        renderQueuePanel(latestQueuePayload);
    }
}

function closeQueueModal() {
    const modal = byId('queue-modal');
    if (!modal || modal.classList.contains('hidden')) return;
    modal.classList.remove('active');
    modal.classList.add('hidden');
    modal.setAttribute('aria-hidden', 'true');
}

function applyCompletedQueueResults(jobs) {
    if (!Array.isArray(jobs) || jobs.length === 0) return;

    const completed = jobs
        .filter(job => {
            if (job?.status !== 'completed') return false;
            if (!job?.job_id || seenCompletedQueueJobs.has(job.job_id)) return false;
            if ((job.finished_at || 0) < queueViewStartedAt) return false;
            const images = job?.result?.images;
            return Array.isArray(images) && images.length > 0;
        })
        .sort((a, b) => (a.finished_at || 0) - (b.finished_at || 0));

    completed.forEach(job => {
        seenCompletedQueueJobs.add(job.job_id);
        handleGenerationResult(job.result);
        emit(Events.GENERATION_COMPLETE, job.result);
    });
}

function escapeHtml(text) {
    return String(text)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

function formatQueueTime(unixSeconds) {
    if (!unixSeconds) return 'now';
    const value = Number(unixSeconds);
    if (!Number.isFinite(value) || value <= 0) return 'now';
    const deltaSeconds = Math.max(0, Math.floor(Date.now() / 1000 - value));
    if (deltaSeconds < 60) return 'just now';
    if (deltaSeconds < 3600) return `${Math.floor(deltaSeconds / 60)}m ago`;
    if (deltaSeconds < 86400) return `${Math.floor(deltaSeconds / 3600)}h ago`;
    return `${Math.floor(deltaSeconds / 86400)}d ago`;
}

function normalizeQueueThumbUrl(src) {
    if (!src || typeof src !== 'string') return null;
    if (!src.startsWith('/inputs/')) return null;
    return src;
}
