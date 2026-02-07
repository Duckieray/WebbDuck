import * as api from './core/api.js';
import { initState, getState, setState, setSeed, setLastUsedSeed, syncFromDOM, syncToDOM } from './core/state.js';
import { emit, Events, initWebSocket } from './core/events.js';
import { $$, byId, listen, show, hide, toggleClass, populateSelect, toast, debounce, toggleSection } from './core/utils.js';
import { ProgressManager } from './modules/ProgressManager.js';
import { MaskEditor } from './modules/MaskEditor.js';
import { LoraManager } from './modules/LoraManager.js';
import { LightboxManager } from './modules/LightboxManager.js';
import { GalleryManager } from './modules/GalleryManager.js';

let isGenerating = false;

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

        await Promise.all([loadModels(), loadSchedulers(), window.galleryManager.load()]);

        syncToDOM();
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

            $$('.preset-chip[data-width]').forEach(x => x.classList.remove('active'));
            chip.classList.add('active');

            syncFromDOM();
        });
    });
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
        if (seed) seed.value = '';
        setSeed(null);
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

async function updateTokenCounter(prompt) {
    const counter = byId('token-count');
    if (!counter) return;

    if (!prompt.trim()) {
        counter.textContent = '0 tokens';
        counter.classList.remove('warning', 'danger');
        return;
    }

    try {
        const model = byId('base_model')?.value;
        if (!model) return;

        const result = await api.tokenize(prompt, model);
        const count = result.tokens || 0;

        counter.textContent = `${count} tokens`;
        toggleClass(counter, 'warning', count > 60);
        toggleClass(counter, 'danger', count > 75);
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
    if (isGenerating) return;

    isGenerating = true;
    window.progressManager?.showProgress('Starting...', 0);
    emit(Events.GENERATION_START, mode);

    try {
        syncFromDOM();
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
        window.progressManager?.hideProgress();
    }
}

function collectFormData() {
    const formData = new FormData();
    const state = getState();

    formData.append('prompt', byId('prompt')?.value || '');
    formData.append('negative', byId('negative')?.value || '');
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

        const formData = new FormData();
        if (imageSrc.startsWith('data:')) {
            const blob = await (await fetch(imageSrc)).blob();
            formData.append('image', blob, 'upscale.png');
        } else {
            formData.append('image', imageSrc);
        }
        formData.append('scale', '2');

        const data = await api.upscale(formData);
        if (data?.upscaled) {
            toast('Upscale complete', 'success');
            if (onSuccess) onSuccess(data.upscaled);
        } else {
            toast('Upscale failed', 'error');
        }
    } catch (error) {
        console.error('Upscale error:', error);
        toast('Upscale failed', 'error');
    }
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

function handleRegenerateFromLightbox(curr) {
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
    setValue('base_model', meta.base_model || meta.model);
    setValue('scheduler', meta.scheduler);

    byId('seed_input').value = '';
    setSeed(null);

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
        const models = await api.getModels();
        populateSelect('base_model', models, false);

        const secondPassModels = await api.getSecondPassModels();
        populateSelect('second_pass_model', secondPassModels, true);

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
