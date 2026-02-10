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
const DENOISE_ACTUAL_MIN = 0.75;
const DENOISE_ACTUAL_MAX = 1.00;
let appConfirmResolver = null;
const IS_COARSE_POINTER = window.matchMedia?.('(pointer: coarse)')?.matches ?? false;

window._uploadedImage = null;
window._maskBlob = null;
window._uploadedImageDims = null;
window._smartExtendPlacement = null;
window._smartExtendDrag = null;
window._smartExtendResize = null;
window._previewEditMode = null; // "place" | "mask" | null
window._previewMaskCanvas = null;
window._maskDrawState = null;

function mapDenoiseUiToActual(uiValue) {
    const v = Number.isFinite(uiValue) ? uiValue : 0.85;
    return Math.max(DENOISE_ACTUAL_MIN, Math.min(DENOISE_ACTUAL_MAX, v));
}

function roundTo8(value) {
    return Math.max(8, Math.round(Number(value || 0) / 8) * 8);
}

function buildSmartExtendFractions(srcW, srcH, dstW, dstH, growth) {
    if (dstW <= srcW && dstH <= srcH) return [1.0];
    const g = Math.max(1.05, Math.min(3.00, Number(growth || 1.25)));
    const fractions = [];
    let currW = srcW;
    let currH = srcH;
    let progress = 0.0;
    const spanW = Math.max(1, dstW - srcW);
    const spanH = Math.max(1, dstH - srcH);
    for (let i = 0; i < 32; i++) {
        if (currW >= dstW && currH >= dstH) break;
        let nextW = Math.min(dstW, roundTo8(currW * g));
        let nextH = Math.min(dstH, roundTo8(currH * g));
        if (nextW === currW && currW < dstW) nextW = Math.min(dstW, currW + 8);
        if (nextH === currH && currH < dstH) nextH = Math.min(dstH, currH + 8);
        const pW = dstW > srcW ? (nextW - srcW) / spanW : 1.0;
        const pH = dstH > srcH ? (nextH - srcH) / spanH : 1.0;
        const nextProgress = Math.min(1.0, Math.max(progress + 0.01, pW, pH));
        fractions.push(nextProgress);
        progress = nextProgress;
        currW = roundTo8(srcW + (dstW - srcW) * progress);
        currH = roundTo8(srcH + (dstH - srcH) * progress);
        if (progress >= 0.999) break;
    }
    if (!fractions.length || fractions[fractions.length - 1] < 1.0) fractions.push(1.0);
    return fractions;
}

function estimateAutoRepeatPasses(srcW, srcH, dstW, dstH) {
    if (!(srcW > 0 && srcH > 0)) return 1;
    const scaleW = dstW / srcW;
    const scaleH = dstH / srcH;
    const scale = Math.max(scaleW, scaleH);
    const maxExtraDim = Math.max(0, dstW - srcW, dstH - srcH);
    const repeatByPixels = Math.max(1, Math.ceil(maxExtraDim / 320.0));
    let repeatByScale = 1;
    if (scale > 1.20) repeatByScale = 2;
    if (scale > 1.45) repeatByScale = 3;
    if (scale > 1.80) repeatByScale = 4;
    if (scale > 2.60) repeatByScale = 5;
    return Math.max(1, Math.min(7, Math.max(repeatByPixels, repeatByScale) + (scale > 2.20 ? 1 : 0)));
}

function estimateSmartExtendRuntimeSeconds() {
    const steps = Math.max(1, Number(byId('steps')?.value || 30));
    const batch = Math.max(1, Number(byId('batch')?.value || 1));
    const dstW = Number(byId('width')?.value || 0);
    const dstH = Number(byId('height')?.value || 0);
    const secondPassEnabled = Boolean(byId('second_pass_enabled')?.checked);

    // Base fallback estimate for non-smart runs.
    if (!(window._uploadedImage && byId('smart-extend-enabled')?.checked)) {
        if (!(dstW > 0 && dstH > 0)) return null;
        const mp = (dstW * dstH) / 1_000_000;
        const stepCost = 0.32 * Math.pow(Math.max(0.5, mp), 1.20);
        let perImage = steps * stepCost;
        if (secondPassEnabled) {
            perImage += Math.max(6, Math.floor(steps * 0.7)) * (stepCost * 0.90);
        }
        return perImage * batch;
    }

    const srcW = Number(window._uploadedImageDims?.width || 0);
    const srcH = Number(window._uploadedImageDims?.height || 0);
    if (!(srcW > 0 && srcH > 0 && dstW > 0 && dstH > 0)) return null;
    if (dstW <= srcW && dstH <= srcH) return null;

    const autoStep = Boolean(byId('smart-extend-auto-step')?.checked);
    const growth = Number(byId('smart-extend-step-growth')?.value || 1.25);
    const refineEnabled = Boolean(byId('smart-extend-refine')?.checked);
    const refineEach = Boolean(byId('smart-extend-refine-each-step')?.checked);

    const stepCost = (mp) => 0.32 * Math.pow(Math.max(0.5, mp), 1.20); // sec/step
    const refineCost = (mp) => 0.22 * Math.pow(Math.max(0.5, mp), 1.12); // sec/step seam pass
    let perImageSec = 0;

    if (autoStep) {
        const fractions = buildSmartExtendFractions(srcW, srcH, dstW, dstH, growth);
        for (let i = 0; i < fractions.length; i++) {
            const p = fractions[i];
            const isFinal = i === fractions.length - 1;
            const stageW = isFinal ? dstW : roundTo8(srcW + (dstW - srcW) * p);
            const stageH = isFinal ? dstH : roundTo8(srcH + (dstH - srcH) * p);
            const mp = (stageW * stageH) / 1_000_000;
            const stageSteps = isFinal ? steps : Math.max(12, Math.floor(steps * (0.55 + 0.45 * p)));
            perImageSec += stageSteps * stepCost(mp);
            if (refineEnabled && refineEach && !isFinal) {
                const refineSteps = Math.max(6, Math.floor(stageSteps * 0.30));
                perImageSec += refineSteps * refineCost(mp);
            }
        }
        if (refineEnabled) {
            const finalMp = (dstW * dstH) / 1_000_000;
            const finalRefineSteps = Math.max(8, Math.floor(steps * 0.35));
            perImageSec += finalRefineSteps * refineCost(finalMp);
        }
    } else {
        const repeatPasses = estimateAutoRepeatPasses(srcW, srcH, dstW, dstH);
        const fullMp = (dstW * dstH) / 1_000_000;
        perImageSec += repeatPasses * steps * stepCost(fullMp);
        if (refineEnabled && repeatPasses <= 1) {
            const finalRefineSteps = Math.max(10, Math.floor(steps * 0.45));
            perImageSec += finalRefineSteps * refineCost(fullMp);
        }
    }

    if (secondPassEnabled) {
        const mp = (dstW * dstH) / 1_000_000;
        perImageSec += Math.max(6, Math.floor(steps * 0.7)) * (stepCost(mp) * 0.90);
    }

    return perImageSec * batch;
}

async function maybeShowRuntimePreflightWarning() {
    const seconds = estimateSmartExtendRuntimeSeconds();
    if (!Number.isFinite(seconds) || seconds <= 0) return true;
    const thresholdMin = Math.max(1, Number(byId('long-run-warning-minutes')?.value || getState('longRunWarningMinutes') || 8));
    if (seconds < thresholdMin * 60) return true;

    const minutes = Math.max(1, Math.round(seconds / 60));
    const hours = seconds >= 3600 ? ` (~${(seconds / 3600).toFixed(1)} hours)` : '';
    const msg = `Get some coffee.\n\nThis run is estimated to take about ${minutes} minutes${hours} with current settings.\n\nContinue anyway?`;
    return await showAppConfirmModal({
        title: 'Get Some Coffee',
        message: msg,
        okText: 'Start Run',
        cancelText: 'Cancel',
        showCancel: true,
        danger: false,
    });
}

function setupAppConfirmModal() {
    const modal = byId('app-confirm-modal');
    const btnOk = byId('app-confirm-ok');
    const btnCancel = byId('app-confirm-cancel');
    if (!modal || !btnOk || !btnCancel) return;

    const finish = (result) => {
        if (appConfirmResolver) {
            appConfirmResolver(result);
            appConfirmResolver = null;
        }
        modal.classList.remove('active');
        modal.classList.add('hidden');
        modal.setAttribute('aria-hidden', 'true');
    };

    listen(btnOk, 'click', () => finish(true));
    listen(btnCancel, 'click', () => finish(false));
    listen(modal, 'click', (e) => {
        if (e.target === modal) finish(false);
    });
    listen(document, 'keydown', (e) => {
        if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
            finish(false);
        }
    });
}

function showAppConfirmModal({
    title = 'Please confirm',
    message = 'Are you sure?',
    okText = 'Continue',
    cancelText = 'Cancel',
    showCancel = true,
    danger = false,
} = {}) {
    const modal = byId('app-confirm-modal');
    const titleEl = byId('app-confirm-title');
    const msgEl = byId('app-confirm-message');
    const btnOk = byId('app-confirm-ok');
    const btnCancel = byId('app-confirm-cancel');
    if (!modal || !titleEl || !msgEl || !btnOk || !btnCancel) {
        return Promise.resolve(true);
    }

    titleEl.textContent = title;
    msgEl.textContent = message;
    btnOk.textContent = okText;
    btnCancel.textContent = cancelText;
    btnCancel.classList.toggle('hidden', !showCancel);
    btnOk.classList.toggle('btn-danger', !!danger);
    btnOk.classList.toggle('btn-primary', !danger);

    modal.classList.remove('hidden');
    modal.setAttribute('aria-hidden', 'false');
    requestAnimationFrame(() => modal.classList.add('active'));

    return new Promise((resolve) => {
        appConfirmResolver = resolve;
    });
}

document.addEventListener('DOMContentLoaded', async () => {
    try {
        initState();
        setupAppConfirmModal();

        window.progressManager = new ProgressManager();
        window.maskEditor = new MaskEditor();
        window.loraManager = new LoraManager();
        window.galleryManager = new GalleryManager();
        window.lightboxManager = new LightboxManager({
            onUpscale: (src, cb) => startUpscale(src, cb),
            onInpaint: (src) => sendToInpaint(src),
            onRegenerate: handleRegenerateFromLightbox,
            onStageSettings: handleStageSettingsFromLightbox,
            onDelete: (src, type) => window.galleryManager.handleDelete(src, type),
            onFavorite: (src, currentlyFavorite) => window.galleryManager.handleFavoriteToggle(src, currentlyFavorite),
        });

        window.galleryManager.init();

        setupNavigation();
        setupHelpModals();
        setupMobileStudioToggle();
        setupCollapsibleSections();
        setupSliders();
        setupPresetChips();
        setupFormHandlers();
        setupGenerationButtons();
        setupUploadHandling();
        setupSmartExtendPlacement();
        setupPreviewToolbar();
        setupQueuePanel();
        setupRealtimeGalleryRefresh();
        setupCatalogWatcher();

        await Promise.all([loadModels(), loadSchedulers(), window.galleryManager.load()]);

        syncToDOM();
        updateSmartExtendAutoStepUI();
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
        await showAppConfirmModal({
            title: 'Startup Error',
            message: `A critical UI error occurred:\n${error.message}`,
            okText: 'Close',
            showCancel: false,
            danger: true,
        });
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
    applyMobileStudioPane(getState('mobileStudioPane') || 'controls');

    if (nextView === 'gallery' && window.galleryManager) {
        window.galleryManager.load();
        emit(Events.VIEW_CHANGE, nextView);
    }
}

function setupMobileStudioToggle() {
    const btn = byId('mobile-studio-toggle');
    if (!btn) return;
    window.__webbduckMobileToggleBound = true;

    listen(btn, 'click', () => {
        const studio = byId('view-studio');
        const isPreview = studio?.dataset.mobilePane === 'preview';
        const next = isPreview ? 'controls' : 'preview';
        setState({ mobileStudioPane: next });
        applyMobileStudioPane(next);
    });

    const onResize = debounce(() => {
        applyMobileStudioPane(getState('mobileStudioPane') || 'controls');
    }, 120);
    listen(window, 'resize', onResize);

    applyMobileStudioPane(getState('mobileStudioPane') || 'controls');
}

function applyMobileStudioPane(pane) {
    const studio = byId('view-studio');
    const btn = byId('mobile-studio-toggle');
    const controls = studio?.querySelector('.nova-controls');
    const workspace = studio?.querySelector('.nova-workspace');
    const currentView = getState('view') || 'studio';
    const isMobile = window.matchMedia('(max-width: 860px)').matches;
    if (!studio || !btn) return;

    const showToggle = isMobile && currentView === 'studio';
    btn.classList.toggle('hidden', !showToggle);

    const showPreview = showToggle && pane === 'preview';
    studio.dataset.mobilePane = showPreview ? 'preview' : 'controls';
    studio.classList.toggle('mobile-pane-preview', showPreview);
    if (controls && workspace) {
        controls.classList.toggle('hidden', showToggle && showPreview);
        workspace.classList.toggle('hidden', showToggle && !showPreview);

        if (showToggle) {
            controls.style.setProperty('display', showPreview ? 'none' : 'grid', 'important');
            workspace.style.setProperty('display', showPreview ? 'grid' : 'none', 'important');
            controls.setAttribute('aria-hidden', showPreview ? 'true' : 'false');
            workspace.setAttribute('aria-hidden', showPreview ? 'false' : 'true');
        } else {
            controls.style.removeProperty('display');
            workspace.style.removeProperty('display');
            controls.removeAttribute('aria-hidden');
            workspace.removeAttribute('aria-hidden');
        }
    }
    btn.textContent = showPreview ? 'Settings' : 'Preview';
    btn.setAttribute('aria-pressed', showPreview ? 'true' : 'false');
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
        ['denoising_strength', 'denoise-value'],
        ['smart-extend-feather', 'smart-extend-feather-value'],
        ['smart-extend-step-growth', 'smart-extend-step-growth-value'],
        ['smart-extend-pyramid-trigger-ratio', 'smart-extend-pyramid-trigger-ratio-value'],
        ['smart-extend-refine-width', 'smart-extend-refine-width-value'],
        ['smart-extend-refine-strength', 'smart-extend-refine-strength-value']
    ];

    pairs.forEach(([inputId, outputId]) => {
        const input = byId(inputId);
        const output = byId(outputId);
        if (!input || !output) return;

        const update = () => {
            if (inputId === 'denoising_strength') {
                const uiValue = parseFloat(input.value);
                output.textContent = mapDenoiseUiToActual(uiValue).toFixed(2);
                return;
            }
            if (inputId === 'smart-extend-step-growth') {
                output.textContent = Number(input.value).toFixed(2);
                return;
            }
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

    ['prompt', 'negative', 'width', 'height', 'steps', 'cfg', 'scheduler', 'batch', 'long-run-warning-minutes', 'seed_input', 'second_pass_steps', 'second_pass_blend', 'second_pass_enabled', 'second_pass_model', 'denoising_strength', 'smart-extend-enabled', 'smart-extend-feather', 'smart-extend-auto-step', 'smart-extend-step-growth', 'smart-extend-pyramid-enable', 'smart-extend-pyramid-trigger-ratio', 'smart-extend-refine', 'smart-extend-refine-each-step', 'smart-extend-refine-width', 'smart-extend-refine-strength'].forEach(id => {
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

    const autoStepEl = byId('smart-extend-auto-step');
    if (autoStepEl) {
        listen(autoStepEl, 'change', updateSmartExtendAutoStepUI);
        updateSmartExtendAutoStepUI();
    }
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
        if (!await maybeShowRuntimePreflightWarning()) {
            toast('Run cancelled by user', 'info');
            return;
        }
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
        const denoiseUi = parseFloat(byId('denoising_strength')?.value ?? '0.85');
        const denoise = mapDenoiseUiToActual(denoiseUi).toFixed(2);
        formData.append('strength', denoise);
        // Keep legacy key for compatibility with any older handlers.
        formData.append('denoising_strength', denoise);
        if (byId('smart-extend-enabled')?.checked) {
            formData.append('smart_extend', 'true');
            formData.append('smart_extend_anchor', 'center');
            formData.append('smart_extend_feather', byId('smart-extend-feather')?.value || '8');
            formData.append('smart_extend_auto_step', byId('smart-extend-auto-step')?.checked ? 'true' : 'false');
            formData.append('smart_extend_step_growth', byId('smart-extend-step-growth')?.value || '1.25');
            formData.append('smart_extend_refine', byId('smart-extend-refine')?.checked ? 'true' : 'false');
            formData.append('smart_extend_refine_each_step', byId('smart-extend-refine-each-step')?.checked ? 'true' : 'false');
            formData.append('smart_extend_refine_width', byId('smart-extend-refine-width')?.value || '24');
            formData.append('smart_extend_refine_strength', byId('smart-extend-refine-strength')?.value || '0.28');
            formData.append('smart_extend_pyramid_enable', byId('smart-extend-pyramid-enable')?.checked ? 'true' : 'false');
            formData.append('smart_extend_pyramid_trigger_ratio', byId('smart-extend-pyramid-trigger-ratio')?.value || '2.4');
            const placement = window._smartExtendPlacement;
            if (placement && Number.isFinite(placement.x) && Number.isFinite(placement.y)) {
                formData.append('smart_extend_offset_x', String(Math.round(placement.x)));
                formData.append('smart_extend_offset_y', String(Math.round(placement.y)));
            }
        }
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

    // When showing generated output, temporarily leave edit mode so the
    // smart-extend canvas doesn't visually override the selected result.
    if (window._previewEditMode) {
        window._previewEditMode = null;
        renderSmartExtendCanvas();
    }

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
            if (!img) return;
            const preview = byId('preview-image');
            const placeholder = byId('preview-placeholder');
            if (!preview || !placeholder) return;

            // Same behavior as main result preview: selecting a generated image
            // should display that image directly (not the edit canvas).
            if (window._previewEditMode) {
                window._previewEditMode = null;
                renderSmartExtendCanvas();
            }

            preview.src = img.src;
            preview.classList.remove('hidden');
            placeholder.classList.add('hidden');
        });
    });
}

function updateSmartExtendAutoStepUI() {
    const autoStepEl = byId('smart-extend-auto-step');
    const growthEl = byId('smart-extend-step-growth');
    const growthValueEl = byId('smart-extend-step-growth-value');
    if (!autoStepEl || !growthEl || !growthValueEl) return;

    const isEnabled = autoStepEl.checked;
    growthEl.disabled = !isEnabled;
    growthEl.style.opacity = isEnabled ? '1' : '0.45';
    growthValueEl.style.opacity = isEnabled ? '1' : '0.65';
    growthValueEl.textContent = isEnabled ? Number(growthEl.value || 1.25).toFixed(2) : 'off';
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

    listen(byId('clear-upload'), 'click', clearInputCanvas);
    listen(byId('remove-upload-image'), 'click', clearUploadedImage);
    listen(byId('preview-img'), 'click', () => showInputImageInStudio(true));

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

    listen(byId('edit-mask-btn'), 'click', () => togglePreviewMaskMode());
}

function setupHelpModals() {
    const modal = byId('help-modal');
    const titleEl = byId('help-modal-title');
    const bodyEl = byId('help-modal-body');
    const closeBtn = byId('help-modal-close');
    if (!modal || !titleEl || !bodyEl || !closeBtn) return;

    const sectionHelp = {
        prompt: {
            title: 'Prompt Help',
            html: `
                <h4>Prompt</h4>
                <p>Your prompt is the main idea for the picture. Write what you want to see.</p>
                <ul>
                  <li>Use short, clear words first (example: <code>1girl, glasses, bedroom, soft light</code>).</li>
                  <li>Then add extra details like style, mood, or camera angle.</li>
                  <li>If your prompt is very long, the model may ignore some words.</li>
                  <li><strong>Best range:</strong> about <code>20-70 tokens</code> for clean, stable results.</li>
                </ul>
            `
        },
        negative: {
            title: 'Negative Prompt Help',
            html: `
                <h4>Negative Prompt</h4>
                <p>This is the "do not include" list.</p>
                <ul>
                  <li>Use it to avoid mistakes (like blurry, bad hands, watermark).</li>
                  <li>If you add too many negatives, images can look flat or less creative.</li>
                  <li><strong>Best range:</strong> keep it short, around <code>5-40 tokens</code>.</li>
                </ul>
            `
        },
        parameters: {
            title: 'Parameters Help',
            html: `
                <h4>Parameters</h4>
                <ul>
                  <li><strong>Width / Height:</strong> image size in pixels. Bigger = more detail, but slower and heavier on VRAM. <strong>Best:</strong> <code>832-1344</code> on the long side for SDXL.</li>
                  <li><strong>Aspect Ratio Buttons:</strong> quick size shapes like 1:1 or 9:16.</li>
                  <li><strong>Steps:</strong> how many thinking passes the model does. More steps can improve detail, but take longer. <strong>Best:</strong> <code>25-40</code>.</li>
                  <li><strong>CFG:</strong> how strongly the model follows your prompt. Too low = ignores prompt, too high = can look weird. <strong>Best:</strong> <code>5-8</code>.</li>
                  <li><strong>Scheduler:</strong> the path the model uses while generating. Different schedulers give different speed/texture behavior. <strong>Best starters:</strong> <code>Euler a</code>, <code>DPM++ 2M Karras</code>, <code>UniPC</code>.</li>
                  <li><strong>Seed:</strong> random starting number. Same seed + same settings gives a very similar image. <strong>Best use:</strong> leave random while exploring, lock seed when refining.</li>
                </ul>
            `
        },
        second_pass: {
            title: 'Second Pass Help',
            html: `
                <h4>Second Pass (Refiner)</h4>
                <p>This is an optional cleanup pass after the first image is made.</p>
                <ul>
                  <li><strong>Enable:</strong> turns on the second pass.</li>
                  <li><strong>Refiner Model:</strong> model used for cleanup.</li>
                  <li><strong>Refiner Steps:</strong> how long the cleanup pass runs. <strong>Best:</strong> <code>12-28</code>.</li>
                  <li><strong>Blend:</strong> how much of the refiner result is mixed in. <strong>Best:</strong> <code>0.55-0.85</code>.</li>
                </ul>
            `
        },
        lora: {
            title: 'LoRA Stack Help',
            html: `
                <h4>LoRA Stack</h4>
                <p>LoRAs are mini add-ons that teach specific styles, characters, or details.</p>
                <ul>
                  <li>Add one or more LoRAs from the list.</li>
                  <li>Use the weight slider to control strength. <strong>Best:</strong> <code>0.5-1.0</code> each.</li>
                  <li>Too many strong LoRAs can fight each other and reduce quality.</li>
                  <li><strong>Best count:</strong> <code>1-3</code> LoRAs at once.</li>
                </ul>
            `
        },
        input_image: {
            title: 'Input Image Help',
            html: `
                <h4>Input Image / Inpaint / Outpaint</h4>
                <ul>
                  <li><strong>Upload:</strong> starts img2img flow from an existing picture.</li>
                  <li><strong>Denoising Strength:</strong> low keeps source close, high changes more. Range is <code>0.75-1.00</code>. <strong>Best:</strong> outpaint <code>0.90-0.98</code>.</li>
                  <li><strong>Mask:</strong> paint where edits are allowed.</li>
                  <li><strong>Inpaint Mode:</strong> Replace edits masked area; Keep protects masked area.</li>
                  <li><strong>Smart Extend:</strong> expands canvas and fills new space.</li>
                  <li><strong>Auto Step Outpaint:</strong> grows canvas in smaller passes to keep identity more stable on large expansions. Turn it off for one single pass.</li>
                  <li><strong>Refine Each Step:</strong> runs seam cleanup after every growth step. <strong>Best for quality:</strong> <code>ON</code>.</li>
                  <li><strong>Step Growth:</strong> per-pass growth factor. <strong>Best:</strong> <code>1.15-1.30</code>.</li>
                  <li><strong>Feather:</strong> softens blend edge. <strong>Best:</strong> <code>6-12</code>.</li>
                  <li><strong>Seam Width:</strong> repair band around the edge. <strong>Best:</strong> <code>18-32</code>.</li>
                  <li><strong>Refine Strength:</strong> seam cleanup strength. <strong>Best:</strong> <code>0.20-0.35</code>.</li>
                </ul>
            `
        },
        batch: {
            title: 'Batch Size Help',
            html: `
                <h4>Batch Size</h4>
                <p>How many images to generate in one request.</p>
                <ul>
                  <li>Bigger batch = more images at once, but more VRAM/RAM use.</li>
                  <li>If you get memory errors, lower batch size first.</li>
                  <li><strong>Best:</strong> <code>1-2</code> for stability. Use <code>3-4</code> only if your system has headroom.</li>
                </ul>
            `
        }
    };

    const closeModal = () => {
        modal.classList.remove('active');
        setTimeout(() => {
            modal.classList.add('hidden');
            modal.setAttribute('aria-hidden', 'true');
        }, 220);
    };

    const openModal = (title, html) => {
        titleEl.textContent = title || 'Help';
        bodyEl.innerHTML = html || '<p>No help content available.</p>';
        modal.classList.remove('hidden');
        modal.setAttribute('aria-hidden', 'false');
        void modal.offsetWidth;
        modal.classList.add('active');
    };

    const renderMarkdown = (markdownText) => {
        const escapeHtml = (s) => String(s || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        const lines = String(markdownText || '').replace(/\r/g, '').split('\n');
        const out = [];
        let inCode = false;
        let inUl = false;
        let inOl = false;

        const closeLists = () => {
            if (inUl) { out.push('</ul>'); inUl = false; }
            if (inOl) { out.push('</ol>'); inOl = false; }
        };

        for (const rawLine of lines) {
            const line = rawLine.trimEnd();
            const t = line.trim();

            if (t.startsWith('```')) {
                closeLists();
                if (!inCode) {
                    out.push('<pre><code>');
                    inCode = true;
                } else {
                    out.push('</code></pre>');
                    inCode = false;
                }
                continue;
            }
            if (inCode) {
                out.push(`${escapeHtml(line)}\n`);
                continue;
            }
            if (!t) {
                closeLists();
                continue;
            }

            if (t.startsWith('# ')) {
                closeLists();
                out.push(`<h4>${escapeHtml(t.slice(2))}</h4>`);
                continue;
            }
            if (t.startsWith('## ')) {
                closeLists();
                out.push(`<h4>${escapeHtml(t.slice(3))}</h4>`);
                continue;
            }

            const olMatch = t.match(/^\d+\.\s+(.*)$/);
            if (olMatch) {
                if (inUl) { out.push('</ul>'); inUl = false; }
                if (!inOl) { out.push('<ol>'); inOl = true; }
                out.push(`<li>${escapeHtml(olMatch[1]).replace(/`([^`]+)`/g, '<code>$1</code>')}</li>`);
                continue;
            }

            if (t.startsWith('- ')) {
                if (inOl) { out.push('</ol>'); inOl = false; }
                if (!inUl) { out.push('<ul>'); inUl = true; }
                out.push(`<li>${escapeHtml(t.slice(2)).replace(/`([^`]+)`/g, '<code>$1</code>')}</li>`);
                continue;
            }

            closeLists();
            out.push(`<p>${escapeHtml(t).replace(/`([^`]+)`/g, '<code>$1</code>')}</p>`);
        }

        if (inCode) out.push('</code></pre>');
        closeLists();
        return out.join('');
    };

    const loadGuide = async () => {
        try {
            const res = await fetch('/docs/simple-guide');
            const data = await res.json();
            const html = renderMarkdown(data?.markdown || '');
            openModal('Studio Guide', html || '<p>Guide is empty.</p>');
        } catch (e) {
            console.warn('Failed to load simple guide:', e);
            openModal('Studio Guide', '<p>Could not load guide right now.</p>');
        }
    };

    $$('.section-help-trigger').forEach(el => {
        const open = (evt) => {
            evt.preventDefault();
            evt.stopPropagation();
            const key = el.dataset.helpSection;
            const help = sectionHelp[key];
            if (!help) return;
            openModal(help.title, help.html);
        };
        listen(el, 'click', open);
        listen(el, 'keydown', (evt) => {
            if (evt.key === 'Enter' || evt.key === ' ') open(evt);
        });
    });

    listen(byId('studio-guide-btn'), 'click', loadGuide);
    listen(closeBtn, 'click', closeModal);
    listen(modal, 'click', (evt) => {
        if (evt.target === modal) closeModal();
    });
    listen(document, 'keydown', (evt) => {
        if (evt.key === 'Escape' && !modal.classList.contains('hidden')) {
            closeModal();
        }
    });
}

function showInputImageInStudio(enableEditMode = false) {
    const inputPreview = byId('preview-img');
    const studioPreview = byId('preview-image');
    const placeholder = byId('preview-placeholder');
    if (!inputPreview?.src || !studioPreview || !placeholder) return;
    inputPreview.setAttribute('draggable', 'false');
    studioPreview.setAttribute('draggable', 'false');
    studioPreview.src = inputPreview.src;
    studioPreview.classList.remove('hidden');
    placeholder.classList.add('hidden');
    studioPreview.dataset.meta = JSON.stringify({ source: 'upload' });
    if (enableEditMode && byId('smart-extend-enabled')?.checked && window._uploadedImage) {
        window._previewEditMode = 'place';
        renderSmartExtendCanvas();
    }
}

function setupSmartExtendPlacement() {
    const canvas = byId('preview-edit-canvas');
    const enabled = byId('smart-extend-enabled');
    const centerBtn = byId('smart-extend-center');
    const width = byId('width');
    const height = byId('height');
    if (!canvas || !enabled || !centerBtn || !width || !height) return;

    // Prevent browser touch/image-selection behavior from stealing drag handles on mobile.
    const blockNativeGesture = (evt) => {
        if (window._previewEditMode === 'place' || window._previewEditMode === 'mask') {
            evt.preventDefault();
            evt.stopPropagation();
        }
    };
    canvas.addEventListener('touchstart', blockNativeGesture, { passive: false });
    canvas.addEventListener('touchmove', blockNativeGesture, { passive: false });
    canvas.addEventListener('gesturestart', blockNativeGesture, { passive: false });

    const refresh = () => renderSmartExtendCanvas();
    listen(enabled, 'change', () => {
        if (enabled.checked && window._uploadedImage && window._previewEditMode !== 'mask') {
            window._previewEditMode = 'place';
        } else if (!enabled.checked && window._previewEditMode === 'place') {
            window._previewEditMode = null;
        }
        refresh();
    });
    listen(width, 'input', refresh);
    listen(height, 'input', refresh);

    listen(centerBtn, 'click', () => {
        window._smartExtendPlacement = null;
        setState({ smartExtendOffsetX: null, smartExtendOffsetY: null });
        renderSmartExtendCanvas(true);
    });

    const pointerPos = (evt) => {
        const rect = canvas.getBoundingClientRect();
        const sx = rect.width > 0 ? (canvas.width / rect.width) : 1;
        const sy = rect.height > 0 ? (canvas.height / rect.height) : 1;
        return {
            x: (evt.clientX - rect.left) * sx,
            y: (evt.clientY - rect.top) * sy,
        };
    };

    listen(canvas, 'pointerdown', (evt) => {
        if (window._previewEditMode !== 'place' || !byId('smart-extend-enabled')?.checked) return;
        evt.preventDefault();
        evt.stopPropagation();
        const geom = renderSmartExtendCanvas();
        if (!geom) return;

        const p = pointerPos(evt);
        const handle = hitResizeHandle(geom, p.x, p.y);
        if (handle) {
            canvas.setPointerCapture(evt.pointerId);
            canvas.classList.add('dragging');
            window._smartExtendResize = {
                pointerId: evt.pointerId,
                handle,
                startClientX: p.x,
                startClientY: p.y,
                startW: geom.targetW,
                startH: geom.targetH,
                startX: geom.placeX,
                startY: geom.placeY,
                scale: geom.scale,
                srcW: geom.srcW,
                srcH: geom.srcH,
            };
            evt.preventDefault();
            return;
        }

        const imgLeft = geom.frameX + geom.placeX * geom.scale;
        const imgTop = geom.frameY + geom.placeY * geom.scale;
        const imgW = geom.srcW * geom.scale;
        const imgH = geom.srcH * geom.scale;
        const inside = p.x >= imgLeft && p.x <= (imgLeft + imgW) && p.y >= imgTop && p.y <= (imgTop + imgH);
        if (!inside) return;

        canvas.setPointerCapture(evt.pointerId);
        canvas.classList.add('dragging');
        window._smartExtendDrag = {
            pointerId: evt.pointerId,
            grabDx: p.x - imgLeft,
            grabDy: p.y - imgTop,
            geom,
        };
        evt.preventDefault();
    });

    listen(canvas, 'pointermove', (evt) => {
        if (window._previewEditMode === 'place' || window._previewEditMode === 'mask') {
            evt.preventDefault();
            evt.stopPropagation();
        }
        if (window._previewEditMode === 'mask') {
            drawPreviewMask(evt);
            return;
        }
        const resize = window._smartExtendResize;
        if (resize && resize.pointerId === evt.pointerId) {
            const p = pointerPos(evt);
            const dx = (p.x - resize.startClientX) / resize.scale;
            const dy = (p.y - resize.startClientY) / resize.scale;
            const minW = Math.max(512, resize.srcW);
            const minH = Math.max(512, resize.srcH);
            const maxW = 4096;
            const maxH = 4096;

            let newW = resize.startW;
            let newH = resize.startH;
            let newX = resize.startX;
            let newY = resize.startY;

            if (resize.handle.includes('right')) {
                newW = clamp(Math.roundTo8(resize.startW + dx), minW, maxW);
            }
            if (resize.handle.includes('left')) {
                newW = clamp(Math.roundTo8(resize.startW - dx), minW, maxW);
                newX = resize.startX + (newW - resize.startW);
            }
            if (resize.handle.includes('bottom')) {
                newH = clamp(Math.roundTo8(resize.startH + dy), minH, maxH);
            }
            if (resize.handle.includes('top')) {
                newH = clamp(Math.roundTo8(resize.startH - dy), minH, maxH);
                newY = resize.startY + (newH - resize.startH);
            }

            const maxX = Math.max(0, newW - resize.srcW);
            const maxY = Math.max(0, newH - resize.srcH);
            newX = clamp(Math.roundTo8(newX), 0, maxX);
            newY = clamp(Math.roundTo8(newY), 0, maxY);

            byId('width').value = String(newW);
            byId('height').value = String(newH);
            byId('width').dispatchEvent(new Event('input'));
            byId('height').dispatchEvent(new Event('input'));
            setState({ width: newW, height: newH, smartExtendOffsetX: newX, smartExtendOffsetY: newY });

            window._smartExtendPlacement = { x: newX, y: newY, custom: true };
            renderSmartExtendCanvas();
            evt.preventDefault();
            return;
        }

        if (window._previewEditMode === 'place' && byId('smart-extend-enabled')?.checked) {
            const geom = renderSmartExtendCanvas();
            if (geom) {
                const p = pointerPos(evt);
                const handle = hitResizeHandle(geom, p.x, p.y);
                if (handle) {
                    canvas.style.cursor = resizeHandleCursor(handle);
                } else {
                    const imgLeft = geom.frameX + geom.placeX * geom.scale;
                    const imgTop = geom.frameY + geom.placeY * geom.scale;
                    const imgW = geom.srcW * geom.scale;
                    const imgH = geom.srcH * geom.scale;
                    const inside = p.x >= imgLeft && p.x <= (imgLeft + imgW) && p.y >= imgTop && p.y <= (imgTop + imgH);
                    canvas.style.cursor = inside ? 'grab' : 'default';
                }
            }
        }

        const drag = window._smartExtendDrag;
        if (!drag || drag.pointerId !== evt.pointerId) return;

        const p = pointerPos(evt);
        const maxX = Math.max(0, drag.geom.targetW - drag.geom.srcW);
        const maxY = Math.max(0, drag.geom.targetH - drag.geom.srcH);

        let x = (p.x - drag.geom.frameX - drag.grabDx) / drag.geom.scale;
        let y = (p.y - drag.geom.frameY - drag.grabDy) / drag.geom.scale;

        x = Math.max(0, Math.min(maxX, x));
        y = Math.max(0, Math.min(maxY, y));

        window._smartExtendPlacement = { x, y, custom: true };
        setState({ smartExtendOffsetX: Math.round(x), smartExtendOffsetY: Math.round(y) });
        renderSmartExtendCanvas();
        evt.preventDefault();
    });

    const stopDrag = (evt) => {
        stopPreviewMaskDraw(evt);
        const resize = window._smartExtendResize;
        if (resize && resize.pointerId === evt.pointerId) {
            window._smartExtendResize = null;
            canvas.classList.remove('dragging');
            canvas.style.cursor = 'default';
            return;
        }
        const drag = window._smartExtendDrag;
        if (!drag || drag.pointerId !== evt.pointerId) return;
        canvas.classList.remove('dragging');
        window._smartExtendDrag = null;
        canvas.style.cursor = 'grab';
    };

    listen(canvas, 'pointerup', stopDrag);
    listen(canvas, 'pointercancel', stopDrag);
    listen(canvas, 'pointerdown', (evt) => {
        if (window._previewEditMode === 'mask') {
            evt.preventDefault();
            evt.stopPropagation();
            startPreviewMaskDraw(evt);
        }
    });
    renderSmartExtendCanvas(true);
}

function updateEditInteractionLock() {
    const canvas = byId('preview-edit-canvas');
    const active = Boolean(
        window._previewEditMode
        && canvas
        && !canvas.classList.contains('hidden')
    );
    document.body.classList.toggle('edit-canvas-active', active);
}

function anchorOffset(anchor, srcW, srcH, targetW, targetH) {
    const dx = Math.max(0, targetW - srcW);
    const dy = Math.max(0, targetH - srcH);
    const key = String(anchor || 'center').toLowerCase();
    if (key === 'top') return { x: dx / 2, y: 0 };
    if (key === 'bottom') return { x: dx / 2, y: dy };
    if (key === 'left') return { x: 0, y: dy / 2 };
    if (key === 'right') return { x: dx, y: dy / 2 };
    if (key === 'top_left') return { x: 0, y: 0 };
    if (key === 'top_right') return { x: dx, y: 0 };
    if (key === 'bottom_left') return { x: 0, y: dy };
    if (key === 'bottom_right') return { x: dx, y: dy };
    return { x: dx / 2, y: dy / 2 };
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

Math.roundTo8 = Math.roundTo8 || function (v) {
    return Math.max(8, Math.round(v / 8) * 8);
};

function getResizeHandles(geom) {
    const baseHalf = Math.max(8, Math.min(14, Math.round(Math.min(geom.frameW, geom.frameH) * 0.02)));
    const half = IS_COARSE_POINTER ? Math.max(14, Math.round(baseHalf * 1.35)) : baseHalf;
    const x = geom.frameX;
    const y = geom.frameY;
    const w = geom.frameW;
    const h = geom.frameH;
    return [
        { id: 'top_left', x, y, half },
        { id: 'top_right', x: x + w, y, half },
        { id: 'bottom_left', x, y: y + h, half },
        { id: 'bottom_right', x: x + w, y: y + h, half },
        { id: 'top', x: x + w / 2, y, half },
        { id: 'bottom', x: x + w / 2, y: y + h, half },
        { id: 'left', x, y: y + h / 2, half },
        { id: 'right', x: x + w, y: y + h / 2, half },
    ];
}

function hitResizeHandle(geom, px, py) {
    const hs = getResizeHandles(geom).map((h) => {
        if (!IS_COARSE_POINTER) return h;
        return { ...h, half: Math.max(20, Math.round(h.half * 1.8)) };
    });
    for (const h of hs) {
        if (
            px >= (h.x - h.half) &&
            px <= (h.x + h.half) &&
            py >= (h.y - h.half) &&
            py <= (h.y + h.half)
        ) {
            return h.id;
        }
    }
    return null;
}

function resizeHandleCursor(handle) {
    if (handle === 'top' || handle === 'bottom') return 'ns-resize';
    if (handle === 'left' || handle === 'right') return 'ew-resize';
    if (handle === 'top_left' || handle === 'bottom_right') return 'nwse-resize';
    if (handle === 'top_right' || handle === 'bottom_left') return 'nesw-resize';
    return 'grab';
}

function renderSmartExtendCanvas(resetPlacement = false) {
    const canvas = byId('preview-edit-canvas');
    if (!canvas) return null;
    const previewMain = byId('preview-main');
    if (!previewMain) return null;
    const rectMain = previewMain.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const targetPixelW = Math.max(1, Math.round(rectMain.width * dpr));
    const targetPixelH = Math.max(1, Math.round(rectMain.height * dpr));
    if (canvas.width !== targetPixelW || canvas.height !== targetPixelH) {
        canvas.width = targetPixelW;
        canvas.height = targetPixelH;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    const srcW = window._uploadedImageDims?.width || 0;
    const srcH = window._uploadedImageDims?.height || 0;
    const targetW = parseInt(byId('width')?.value || '0', 10) || srcW;
    const targetH = parseInt(byId('height')?.value || '0', 10) || srcH;

    const offsetLabel = byId('smart-extend-offset');
    const cw = canvas.width;
    const ch = canvas.height;
    canvas.classList.toggle('hidden', !window._uploadedImage || !window._previewEditMode);
    canvas.classList.toggle('mode-place', window._previewEditMode === 'place');
    updateEditInteractionLock();
    ctx.clearRect(0, 0, cw, ch);
    if (!window._uploadedImage || !window._previewEditMode) return null;

    ctx.fillStyle = 'rgba(8,12,20,0.95)';
    ctx.fillRect(0, 0, cw, ch);

    if (!srcW || !srcH || !targetW || !targetH) {
        ctx.fillStyle = '#8fa4c4';
        ctx.font = '12px sans-serif';
        ctx.fillText('Upload an image to place smart extend area', 12, ch / 2);
        if (offsetLabel) offsetLabel.textContent = 'Offset: auto';
        return null;
    }

    const pad = 10;
    const scale = Math.min((cw - pad * 2) / targetW, (ch - pad * 2) / targetH);
    const frameW = targetW * scale;
    const frameH = targetH * scale;
    const frameX = (cw - frameW) / 2;
    const frameY = (ch - frameH) / 2;

    const fallback = anchorOffset(byId('smart-extend-anchor')?.value, srcW, srcH, targetW, targetH);
    if (resetPlacement || !window._smartExtendPlacement || !window._smartExtendPlacement.custom) {
        const saved = getState();
        const savedX = Number(saved.smartExtendOffsetX);
        const savedY = Number(saved.smartExtendOffsetY);
        if (!resetPlacement && Number.isFinite(savedX) && Number.isFinite(savedY)) {
            window._smartExtendPlacement = { x: savedX, y: savedY, custom: true };
        } else {
            window._smartExtendPlacement = { x: fallback.x, y: fallback.y, custom: false };
        }
    }

    const maxX = Math.max(0, targetW - srcW);
    const maxY = Math.max(0, targetH - srcH);
    const placeX = Math.max(0, Math.min(maxX, window._smartExtendPlacement.x));
    const placeY = Math.max(0, Math.min(maxY, window._smartExtendPlacement.y));
    window._smartExtendPlacement.x = placeX;
    window._smartExtendPlacement.y = placeY;

    ctx.fillStyle = 'rgba(18,29,47,0.95)';
    ctx.fillRect(frameX, frameY, frameW, frameH);
    ctx.strokeStyle = 'rgba(148,163,184,0.55)';
    ctx.lineWidth = 1;
    ctx.strokeRect(frameX, frameY, frameW, frameH);

    const imgEl = byId('preview-img');
    const imgX = frameX + placeX * scale;
    const imgY = frameY + placeY * scale;
    const imgW = srcW * scale;
    const imgH = srcH * scale;

    ctx.save();
    ctx.beginPath();
    ctx.rect(imgX, imgY, imgW, imgH);
    ctx.clip();
    if (imgEl && imgEl.complete && imgEl.naturalWidth > 0) {
        ctx.drawImage(imgEl, imgX, imgY, imgW, imgH);
    } else {
        ctx.fillStyle = 'rgba(96,165,250,0.35)';
        ctx.fillRect(imgX, imgY, imgW, imgH);
    }
    ctx.restore();

    // Shade only extension regions so the placed image stays visible.
    ctx.fillStyle = 'rgba(14,165,233,0.16)';
    if (imgY > frameY) {
        ctx.fillRect(frameX, frameY, frameW, imgY - frameY);
    }
    const imgBottom = imgY + imgH;
    const frameBottom = frameY + frameH;
    if (imgBottom < frameBottom) {
        ctx.fillRect(frameX, imgBottom, frameW, frameBottom - imgBottom);
    }
    if (imgX > frameX) {
        ctx.fillRect(frameX, imgY, imgX - frameX, imgH);
    }
    const imgRight = imgX + imgW;
    const frameRight = frameX + frameW;
    if (imgRight < frameRight) {
        ctx.fillRect(imgRight, imgY, frameRight - imgRight, imgH);
    }

    ctx.strokeStyle = 'rgba(96,165,250,0.95)';
    ctx.lineWidth = 2;
    ctx.strokeRect(imgX, imgY, imgW, imgH);

    const handles = getResizeHandles({ frameX, frameY, frameW, frameH });
    ctx.fillStyle = 'rgba(147, 197, 253, 0.95)';
    ctx.strokeStyle = 'rgba(8, 13, 24, 0.98)';
    ctx.lineWidth = 1.5;
    for (const h of handles) {
        ctx.fillRect(h.x - h.half, h.y - h.half, h.half * 2, h.half * 2);
        ctx.strokeRect(h.x - h.half, h.y - h.half, h.half * 2, h.half * 2);
    }

    if (window._previewEditMode === 'place') {
        const hint = 'Drag image to place. Drag frame handles to expand.';
        ctx.fillStyle = 'rgba(191, 219, 254, 0.95)';
        ctx.font = '600 12px Inter, sans-serif';
        const tw = ctx.measureText(hint).width;
        const tx = frameX + Math.max(10, (frameW - tw) / 2);
        const ty = Math.max(18, frameY - 10);
        ctx.fillText(hint, tx, ty);
    }

    if (window._previewEditMode !== 'place') {
        canvas.style.cursor = 'crosshair';
    } else if (!canvas.classList.contains('dragging')) {
        canvas.style.cursor = 'default';
    }

    if (offsetLabel) {
        offsetLabel.textContent = `Offset: x ${Math.round(placeX)}, y ${Math.round(placeY)}`;
    }

    if (window._previewEditMode === 'mask') {
        renderPreviewMaskOverlay(ctx, { frameX, frameY, frameW, frameH, imgX, imgY, imgW, imgH, scale, targetW, targetH });
    }

    return { frameX, frameY, frameW, frameH, scale, srcW, srcH, targetW, targetH, placeX, placeY };
}

function ensurePreviewMaskCanvas(targetW, targetH) {
    if (!window._previewMaskCanvas) {
        window._previewMaskCanvas = document.createElement('canvas');
    }
    if (window._previewMaskCanvas.width !== targetW || window._previewMaskCanvas.height !== targetH) {
        window._previewMaskCanvas.width = targetW;
        window._previewMaskCanvas.height = targetH;
        const c = window._previewMaskCanvas.getContext('2d');
        if (c) c.clearRect(0, 0, targetW, targetH);
    }
}

function renderPreviewMaskOverlay(ctx, geom) {
    ensurePreviewMaskCanvas(geom.targetW, geom.targetH);
    const maskCtx = window._previewMaskCanvas.getContext('2d');
    if (!maskCtx) return;

    ctx.save();
    ctx.globalAlpha = 0.4;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(
        window._previewMaskCanvas,
        0, 0, geom.targetW, geom.targetH,
        geom.frameX, geom.frameY, geom.frameW, geom.frameH
    );
    ctx.restore();
}

function startPreviewMaskDraw(evt) {
    const geom = renderSmartExtendCanvas();
    if (!geom) return;
    const canvas = byId('preview-edit-canvas');
    if (!canvas) return;
    canvas.setPointerCapture(evt.pointerId);
    window._maskDrawState = { pointerId: evt.pointerId, geom };
    drawPreviewMask(evt);
}

function drawPreviewMask(evt) {
    const state = window._maskDrawState;
    if (!state || state.pointerId !== evt.pointerId) return;
    const canvas = byId('preview-edit-canvas');
    if (!canvas || !window._previewMaskCanvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = (evt.clientX - rect.left) * (canvas.width / rect.width);
    const y = (evt.clientY - rect.top) * (canvas.height / rect.height);
    const g = state.geom;
    if (x < g.frameX || x > g.frameX + g.frameW || y < g.frameY || y > g.frameY + g.frameH) return;

    const mx = ((x - g.frameX) / g.frameW) * g.targetW;
    const my = ((y - g.frameY) / g.frameH) * g.targetH;
    const radius = Math.max(4, Number(byId('mask_brush_size')?.value || 24));
    const maskCtx = window._previewMaskCanvas.getContext('2d');
    if (!maskCtx) return;
    maskCtx.fillStyle = 'rgba(255,255,255,1)';
    maskCtx.beginPath();
    maskCtx.arc(mx, my, radius, 0, Math.PI * 2);
    maskCtx.fill();
    renderSmartExtendCanvas();
}

function stopPreviewMaskDraw(evt) {
    const state = window._maskDrawState;
    if (!state || state.pointerId !== evt.pointerId) return;
    window._maskDrawState = null;
}

async function togglePreviewMaskMode() {
    if (!window._uploadedImage || !window._uploadedImageDims) {
        toast('Upload an image first', 'error');
        return;
    }
    const btn = byId('edit-mask-btn');
    if (!btn) return;

    if (window._previewEditMode === 'mask') {
        if (!window._previewMaskCanvas) return;
        const blob = await new Promise((resolve) => window._previewMaskCanvas.toBlob(resolve, 'image/png'));
        if (blob) {
            window._maskBlob = blob;
            show(byId('inpaint-options'));
            toast('Mask saved', 'success');
        }
        window._previewEditMode = byId('smart-extend-enabled')?.checked ? 'place' : null;
        btn.textContent = 'Mask';
        renderSmartExtendCanvas();
        return;
    }

    window._previewEditMode = 'mask';
    btn.textContent = 'Save Mask';
    renderSmartExtendCanvas();
    toast('Draw mask on the main preview, then click Save Mask', 'info');
}

async function handleImageUpload(file) {
    if (!file.type.startsWith('image/')) {
        toast('Please upload an image file', 'error');
        return;
    }

    window._uploadedImage = file;
    window._maskBlob = null;
    window._uploadedImageDims = null;
    window._smartExtendPlacement = null;

    const reader = new FileReader();
    reader.onload = event => {
        const src = event.target.result;
        const preview = byId('preview-img');
        preview.src = src;
        const studioPreview = byId('preview-image');
        const placeholder = byId('preview-placeholder');
        if (studioPreview && placeholder) {
            studioPreview.src = src;
            studioPreview.classList.remove('hidden');
            placeholder.classList.add('hidden');
            studioPreview.dataset.meta = JSON.stringify({ source: 'upload' });
        }

        const probe = new Image();
        probe.onload = () => {
            window._uploadedImageDims = { width: probe.width, height: probe.height };
            window._smartExtendPlacement = null;
            window._previewMaskCanvas = null;
            const fit = fitResolution(probe.width, probe.height, 2048);

            const widthInput = byId('width');
            const heightInput = byId('height');

            widthInput.value = fit.width;
            heightInput.value = fit.height;

            widthInput.dispatchEvent(new Event('input'));
            heightInput.dispatchEvent(new Event('input'));

            setState({ width: fit.width, height: fit.height });
            toast(`Resolution set to ${fit.width}x${fit.height}`, 'info');
            if (byId('smart-extend-enabled')?.checked) {
                window._previewEditMode = 'place';
            }
            renderSmartExtendCanvas(true);
        };
        probe.src = src;

        hide(byId('upload-drop'));
        show(byId('upload-preview'));
        show(byId('denoise-group'));
        show(byId('smart-extend-group'));
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
    window._uploadedImageDims = null;
    window._smartExtendPlacement = null;
    window._previewMaskCanvas = null;
    window._previewEditMode = null;

    byId('input-image').value = '';
    byId('preview-img').src = '';
    const studioPreview = byId('preview-image');
    const placeholder = byId('preview-placeholder');
    if (studioPreview && placeholder) {
        studioPreview.src = '';
        studioPreview.classList.add('hidden');
        placeholder.classList.remove('hidden');
    }

    show(byId('upload-drop'));
    hide(byId('upload-preview'));
    hide(byId('denoise-group'));
    hide(byId('inpaint-options'));
    hide(byId('smart-extend-group'));
    if (byId('smart-extend-enabled')) byId('smart-extend-enabled').checked = false;
    const maskBtn = byId('edit-mask-btn');
    if (maskBtn) maskBtn.textContent = 'Mask';
    setState({ smartExtendOffsetX: null, smartExtendOffsetY: null });
    renderSmartExtendCanvas(true);

    emit(Events.IMAGE_CLEAR);
}

function clearInputCanvas() {
    if (!window._uploadedImage) {
        toast('No uploaded image to clear edits from', 'info');
        return;
    }
    window._maskBlob = null;
    window._previewMaskCanvas = null;
    window._maskDrawState = null;
    window._smartExtendPlacement = null;
    const maskBtn = byId('edit-mask-btn');
    if (maskBtn) maskBtn.textContent = 'Mask';
    hide(byId('inpaint-options'));
    setState({ smartExtendOffsetX: null, smartExtendOffsetY: null });
    showInputImageInStudio(true);
    renderSmartExtendCanvas(true);
    toast('Canvas edits cleared', 'success');
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
    const ok = await applyLightboxSettingsToStudio(curr, { preserveSeed: false });
    if (!ok) return;

    const generate = byId('btn-generate');
    if (generate) {
        setTimeout(() => generate.click(), 120);
    }

    toast('Regenerate started', 'success');
}

async function handleStageSettingsFromLightbox(curr) {
    const ok = await applyLightboxSettingsToStudio(curr, { preserveSeed: true });
    if (!ok) return;
    toast('Settings staged in Studio', 'success');
}

async function applyLightboxSettingsToStudio(curr, options = {}) {
    const preserveSeed = options?.preserveSeed === true;
    if (!curr?.meta) {
        toast('No metadata available', 'error');
        return false;
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

    if (preserveSeed) {
        const stagedSeed = meta.seed !== undefined && meta.seed !== null ? String(meta.seed) : '';
        byId('seed_input').value = stagedSeed;
        setSeed(stagedSeed || null);
    } else {
        byId('seed_input').value = '';
        setSeed(null);
    }

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
    return true;
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
        const modeDetails = Array.isArray(job.settings?.mode_details) ? job.settings.mode_details : [];
        const thumbHtml = inputThumb
            ? `<img class="queue-item-thumb" src="${escapeHtml(inputThumb)}" alt="Queue input preview" loading="lazy" />`
            : '';
        const canCancel = status === 'queued';
        const statusText = status === 'running' ? 'Running' : status === 'queued' ? 'Queued' : status;
        const isExpanded = expandedQueueJobs.has(job.job_id);
        const hasExtraDetails = seed !== null || negative.length > 0 || loras.length > 0 || modeDetails.length > 0;
        const detailsHtml = hasExtraDetails
            ? `
                <div class="queue-item-details ${isExpanded ? '' : 'hidden'}" data-details-for="${job.job_id}">
                  <div class="queue-detail-row"><span class="queue-detail-label">Seed</span><span class="queue-detail-value">${seed === null ? 'Random' : escapeHtml(String(seed))}</span></div>
                  <div class="queue-detail-row"><span class="queue-detail-label">Negative</span><span class="queue-detail-value">${negative ? escapeHtml(negative.slice(0, 220)) : 'None'}</span></div>
                  <div class="queue-detail-row"><span class="queue-detail-label">Mode</span><span class="queue-detail-value">${modeDetails.length > 0 ? escapeHtml(modeDetails.join(' | ')) : 'Default'}</span></div>
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
