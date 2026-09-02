import * as api from './core/api.js';
import { initState, getState, setState, setSeed, setLastUsedSeed, syncFromDOM, syncToDOM } from './core/state.js';
import { emit, on, Events, initWebSocket } from './core/events.js';
import { $$, byId, listen, show, hide, toggleClass, populateSelect, toast, debounce, toggleSection } from './core/utils.js';
import { ProgressManager } from './modules/ProgressManager.js';
import { MaskEditor } from './modules/MaskEditor.js';
import { LoraManager } from './modules/LoraManager.js';
import { EmbeddingManager } from './modules/EmbeddingManager.js';
import { LightboxManager } from './modules/LightboxManager.js';
import { GalleryManager } from './modules/GalleryManager.js';

let isGenerating = false;
const seenCompletedQueueJobs = new Set();
const queueViewStartedAt = Date.now() / 1000;
let latestQueuePayload = null;
const expandedQueueJobs = new Set();
const DENOISE_DETAIL_MIN = 0.75;
const DENOISE_PRESERVE_MIN = 0.30;
const DENOISE_PRESERVE_MAX = 0.50;
const DENOISE_ACTUAL_MAX = 1.00;
const SMART_EXTEND_MAX_RESOLUTION = { width: 1600, height: 2048 };
const SMART_EXTEND_MAX_AREA = SMART_EXTEND_MAX_RESOLUTION.width * SMART_EXTEND_MAX_RESOLUTION.height;
const SMART_EXTEND_FIXED = Object.freeze({
    feather: 12,
    stepGrowth: 1.25,
    autoStep: false,
    refine: true,
    refineEachStep: true,
    refineWidth: 64,
    refineStrength: 0.32,
    pyramidTriggerRatio: 2.4,
});
let appConfirmResolver = null;
const IS_COARSE_POINTER = window.matchMedia?.('(pointer: coarse)')?.matches ?? false;
const webPluginRegistry = new Map();
const webPluginFrames = new Map();
const pendingPluginMessages = new Map();
const DUCKMOTION_PLUGIN_ID = 'duckmotion';

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
    const mode = getDenoiseMode();
    const range = getDenoiseRange(mode);
    const v = Number.isFinite(uiValue) ? uiValue : 0.85;
    return Math.max(range.min, Math.min(range.max, v));
}

function getDenoiseMode() {
    const raw = (byId('denoise-mode')?.value || '').trim().toLowerCase();
    return raw === 'preserve' ? 'preserve' : 'details';
}

function getDenoiseRange(mode) {
    if (mode === 'preserve') {
        return { min: DENOISE_PRESERVE_MIN, max: DENOISE_PRESERVE_MAX };
    }
    return { min: DENOISE_DETAIL_MIN, max: DENOISE_ACTUAL_MAX };
}

function refreshDenoiseControls() {
    const slider = byId('denoising_strength');
    const valueOutput = byId('denoise-value');
    const hint = byId('denoise-range-hint');
    const mode = getDenoiseMode();
    const range = getDenoiseRange(mode);
    if (!slider) return;

    slider.min = range.min.toFixed(2);
    slider.max = range.max.toFixed(2);
    slider.step = '0.01';

    const clamped = mapDenoiseUiToActual(parseFloat(slider.value));
    slider.value = clamped.toFixed(2);
    if (valueOutput) valueOutput.textContent = clamped.toFixed(2);
    const preserveBtn = byId('denoise-mode-preserve');
    const detailsBtn = byId('denoise-mode-details');
    preserveBtn?.classList.toggle('active', mode === 'preserve');
    detailsBtn?.classList.toggle('active', mode !== 'preserve');
    if (hint) {
        hint.textContent = mode === 'preserve'
            ? 'Keep mostly the same uses 0.30-0.50 for subtle edits.'
            : 'Change details uses 0.75-1.00 for stronger variation.';
    }
}

function setupDenoiseModeToggle() {
    const modeInput = byId('denoise-mode');
    const preserveBtn = byId('denoise-mode-preserve');
    const detailsBtn = byId('denoise-mode-details');
    if (!modeInput || !preserveBtn || !detailsBtn) return;

    const setMode = (mode) => {
        modeInput.value = mode === 'preserve' ? 'preserve' : 'details';
        refreshDenoiseControls();
        syncFromDOM();
    };

    listen(preserveBtn, 'click', () => setMode('preserve'));
    listen(detailsBtn, 'click', () => setMode('details'));
}

function roundTo8(value) {
    return Math.max(8, Math.round(Number(value || 0) / 8) * 8);
}

function normalizeDimensionToMultipleOf8(rawValue, fallback = 1024) {
    const n = Number(rawValue);
    const safe = Number.isFinite(n) ? n : Number(fallback);
    return Math.max(8, Math.round(safe / 8) * 8);
}

function normalizeResolutionInputs({ syncState = false } = {}) {
    const widthInput = byId('width');
    const heightInput = byId('height');
    if (!widthInput || !heightInput) return false;

    const prevW = Number(widthInput.value);
    const prevH = Number(heightInput.value);
    const nextW = normalizeDimensionToMultipleOf8(prevW, 1024);
    const nextH = normalizeDimensionToMultipleOf8(prevH, 1024);

    const changed = nextW !== prevW || nextH !== prevH;
    if (changed) {
        widthInput.value = String(nextW);
        heightInput.value = String(nextH);
    }

    if (syncState) {
        setState({ width: nextW, height: nextH });
    }
    return changed;
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

    const autoStep = SMART_EXTEND_FIXED.autoStep;
    const growth = SMART_EXTEND_FIXED.stepGrowth;
    const refineEnabled = SMART_EXTEND_FIXED.refine;
    const refineEach = SMART_EXTEND_FIXED.refineEachStep;

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

function isAboveOutpaintSafetyResolution(width, height) {
    const w = Number(width || 0);
    const h = Number(height || 0);
    if (!(w > 0 && h > 0)) return false;
    const area = w * h;
    return area > SMART_EXTEND_MAX_AREA;
}

function updateSmartExtendSizeWarning(width, height) {
    const warning = byId('smart-extend-size-warning');
    const widthInput = byId('width');
    const heightInput = byId('height');
    const smartExtendOn = Boolean(byId('smart-extend-enabled')?.checked);
    const show = Boolean(window._uploadedImage && smartExtendOn && isAboveOutpaintSafetyResolution(width, height));

    if (warning) {
        warning.classList.toggle('hidden', !show);
        if (show) {
            const currW = Math.round(Number(width) || 0);
            const currH = Math.round(Number(height) || 0);
            const currArea = currW * currH;
            warning.textContent = `Warning: Smart Extend above ${SMART_EXTEND_MAX_RESOLUTION.width}x${SMART_EXTEND_MAX_RESOLUTION.height} may fail or produce artifacts. Current target is ${currW}x${currH}.`;
        }
    }
    widthInput?.classList.toggle('input-danger', show);
    heightInput?.classList.toggle('input-danger', show);
}

async function maybeShowLargeResolutionWarning() {
    const smartExtendOn = Boolean(byId('smart-extend-enabled')?.checked && window._uploadedImage);
    if (!smartExtendOn) return true;

    const width = Number(byId('width')?.value || getState('width') || 0);
    const height = Number(byId('height')?.value || getState('height') || 0);
    if (!isAboveOutpaintSafetyResolution(width, height)) return true;

    const currW = Math.round(width || 0);
    const currH = Math.round(height || 0);
    const currArea = currW * currH;
    return await showAppConfirmModal({
        title: 'Smart Extend Warning',
        message: `Smart Extend above (${SMART_EXTEND_MAX_RESOLUTION.width}x${SMART_EXTEND_MAX_RESOLUTION.height} or ${SMART_EXTEND_MAX_RESOLUTION.height}x${SMART_EXTEND_MAX_RESOLUTION.width}) may fail or produce artifacts.\n\nCurrent target: ${currW}x${currH}.\n\nContinue anyway?`,
        okText: 'Continue',
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

async function loadWebPlugins() {
    let plugins = [];
    try {
        const response = await api.getWebPlugins();
        plugins = Array.isArray(response?.plugins) ? response.plugins : [];
    } catch (error) {
        console.warn('Failed to load web plugins:', error);
        return;
    }

    for (const plugin of plugins) {
        registerWebPlugin(plugin);
    }
    updateOptionalPluginUi();
}

function getWebPluginRecord(pluginId) {
    return webPluginRegistry.get(String(pluginId || '').trim().toLowerCase()) || null;
}

function hasWebPlugin(pluginId) {
    return Boolean(getWebPluginRecord(pluginId));
}

function queuePluginMessage(pluginId, payload) {
    const id = String(pluginId || '').trim().toLowerCase();
    if (!id) return;
    const rows = pendingPluginMessages.get(id) || [];
    rows.push(payload);
    pendingPluginMessages.set(id, rows.slice(-10));
}

function flushPluginMessages(pluginId) {
    const id = String(pluginId || '').trim().toLowerCase();
    const frame = webPluginFrames.get(id);
    if (!frame?.contentWindow) return;
    const rows = pendingPluginMessages.get(id) || [];
    if (!rows.length) return;
    pendingPluginMessages.delete(id);
    for (const payload of rows) {
        try {
            frame.contentWindow.postMessage(payload, window.location.origin);
        } catch (_) {
            // Ignore transient iframe messaging failures.
        }
    }
}

function postMessageToPlugin(pluginId, payload) {
    const id = String(pluginId || '').trim().toLowerCase();
    if (!id) return false;
    const frame = webPluginFrames.get(id);
    if (!frame?.contentWindow) {
        queuePluginMessage(id, payload);
        return false;
    }
    if (frame.dataset.loaded === '1') {
        try {
            frame.contentWindow.postMessage(payload, window.location.origin);
            return true;
        } catch (_) {
            queuePluginMessage(id, payload);
            return false;
        }
    }
    queuePluginMessage(id, payload);
    return true;
}

function switchToPluginView(pluginId) {
    const id = String(pluginId || '').trim().toLowerCase();
    if (!id) return false;
    const viewName = `plugin-${id}`;
    if (!byId(`view-${viewName}`)) return false;
    switchView(viewName);
    return true;
}

function updateOptionalPluginUi() {
    const duckMotionAvailable = hasWebPlugin(DUCKMOTION_PLUGIN_ID);
    const previewBtn = byId('preview-duckmotion');
    const lightboxBtn = byId('lightbox-duckmotion');
    if (previewBtn) toggleClass(previewBtn, 'hidden', !duckMotionAvailable);
    if (lightboxBtn) toggleClass(lightboxBtn, 'hidden', !duckMotionAvailable);
}

async function sendToDuckMotion(imageSrc, curr) {
    if (!hasWebPlugin(DUCKMOTION_PLUGIN_ID)) {
        toast('DuckMotion plugin is not installed', 'warning');
        updateOptionalPluginUi();
        return false;
    }

    if (!imageSrc || typeof imageSrc !== 'string') {
        toast('No image available to send', 'warning');
        return false;
    }

    if (imageSrc.startsWith('data:') || imageSrc.startsWith('blob:')) {
        toast('Send to DuckMotion currently requires a saved image path from outputs/gallery', 'warning');
        return false;
    }

    let normalizedPath = imageSrc;
    try {
        const url = new URL(imageSrc, window.location.origin);
        normalizedPath = `${url.pathname}${url.search || ''}`;
    } catch (_) {
        normalizedPath = imageSrc;
    }

    if (!normalizedPath.startsWith('/outputs/')) {
        toast('Send to DuckMotion currently supports WebbDuck output images', 'warning');
        return false;
    }

    const meta = curr?.meta || curr || {};
    const payload = {
        type: 'webbduck.duckmotion.handoff',
        source: 'webbduck',
        image: { src: normalizedPath },
        meta: {
            prompt: meta.prompt || '',
            negative_prompt: meta.negative_prompt || meta.negative || '',
            seed: meta.seed ?? null,
            width: meta.width ?? null,
            height: meta.height ?? null,
        },
        sent_at: Date.now(),
    };

    switchToPluginView(DUCKMOTION_PLUGIN_ID);
    postMessageToPlugin(DUCKMOTION_PLUGIN_ID, payload);
    toast('Sent image to DuckMotion', 'success');
    return true;
}

function registerWebPlugin(plugin) {
    const id = String(plugin?.id || '').trim();
    if (!id) return;
    const pluginId = id.toLowerCase();
    webPluginRegistry.set(pluginId, { ...plugin, id: pluginId });

    const viewName = String(plugin?.view || `plugin-${id}`).trim();
    const title = String(plugin?.name || id);
    const description = String(plugin?.description || '');
    const uiUrl = String(plugin?.ui_url || '').trim();
    if (!uiUrl) return;

    const desktopTabs = document.querySelector('.nova-tabs');
    if (desktopTabs && !desktopTabs.querySelector(`[data-view="${viewName}"]`)) {
        const tab = document.createElement('button');
        tab.type = 'button';
        tab.className = 'nav-tab';
        tab.dataset.view = viewName;
        tab.textContent = title;
        desktopTabs.appendChild(tab);
    }

    const mobileTabs = document.querySelector('.mobile-tabs');
    if (mobileTabs && !mobileTabs.querySelector(`[data-view="${viewName}"]`)) {
        const tab = document.createElement('button');
        tab.type = 'button';
        tab.className = 'mobile-tab';
        tab.dataset.view = viewName;
        tab.textContent = title;
        mobileTabs.appendChild(tab);
    }

    if (byId(`view-${viewName}`)) {
        updateOptionalPluginUi();
        return;
    }

    const container = document.querySelector('.view-container');
    if (!container) return;

    const section = document.createElement('section');
    section.className = 'view';
    section.id = `view-${viewName}`;

    const shell = document.createElement('div');
    shell.className = 'plugin-shell';

    const header = document.createElement('header');
    header.className = 'plugin-shell-header';

    const heading = document.createElement('h2');
    heading.className = 'plugin-shell-title';
    heading.textContent = title;

    const sub = document.createElement('p');
    sub.className = 'plugin-shell-subtitle';
    sub.textContent = description || 'Plugin workspace';

    header.appendChild(heading);
    header.appendChild(sub);

    const frameWrap = document.createElement('div');
    frameWrap.className = 'plugin-frame-wrap';

    const frame = document.createElement('iframe');
    frame.className = 'plugin-frame';
    frame.src = uiUrl;
    frame.loading = 'lazy';
    frame.title = title;
    frame.referrerPolicy = 'same-origin';
    frame.dataset.pluginId = pluginId;
    webPluginFrames.set(pluginId, frame);
    frame.addEventListener('load', () => {
        frame.dataset.loaded = '1';
        flushPluginMessages(pluginId);
    });

    frameWrap.appendChild(frame);
    shell.appendChild(header);
    shell.appendChild(frameWrap);
    section.appendChild(shell);
    container.appendChild(section);
    updateOptionalPluginUi();
}

function unregisterWebPlugin(pluginId) {
    const id = String(pluginId || '').trim();
    if (!id) return;
    webPluginRegistry.delete(id.toLowerCase());
    webPluginFrames.delete(id.toLowerCase());
    pendingPluginMessages.delete(id.toLowerCase());
    const viewName = `plugin-${id}`;

    document.querySelectorAll(`.nav-tab[data-view="${viewName}"], .mobile-tab[data-view="${viewName}"]`).forEach(el => {
        el.remove();
    });

    const view = byId(`view-${viewName}`);
    if (view) {
        view.remove();
    }

    const currentView = getState('view');
    if (currentView === viewName) {
        switchView('studio');
    }
    updateOptionalPluginUi();
}

async function refreshRemotePluginList() {
    const listEl = byId('remote-plugin-list');
    if (!listEl) return;

    let plugins = [];
    try {
        const response = await api.getRemoteWebPlugins();
        plugins = Array.isArray(response?.plugins) ? response.plugins : [];
    } catch (error) {
        listEl.innerHTML = '<div class="queue-item-empty">Failed to load remote plugins.</div>';
        return;
    }

    renderRemotePluginList(plugins);
}

function renderRemotePluginList(plugins) {
    const listEl = byId('remote-plugin-list');
    if (!listEl) return;

    if (!Array.isArray(plugins) || plugins.length === 0) {
        listEl.innerHTML = '<div class="queue-item-empty">No remote plugins connected.</div>';
        return;
    }

    listEl.innerHTML = '';
    plugins.forEach(plugin => {
        const pluginId = String(plugin?.id || '').trim();
        if (!pluginId) return;

        const row = document.createElement('div');
        row.className = 'remote-plugin-item';

        const textWrap = document.createElement('div');
        const name = document.createElement('div');
        name.className = 'remote-plugin-name';
        name.textContent = String(plugin?.name || pluginId);

        const meta = document.createElement('div');
        meta.className = 'remote-plugin-meta';
        const remoteBase = String(plugin?.remote_base || plugin?.ui_url || '').trim();
        meta.textContent = remoteBase || pluginId;

        textWrap.appendChild(name);
        textWrap.appendChild(meta);

        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'btn btn-danger btn-sm';
        removeBtn.textContent = 'Disconnect';
        removeBtn.dataset.pluginId = pluginId;
        listen(removeBtn, 'click', async () => {
            removeBtn.disabled = true;
            try {
                await api.disconnectRemoteWebPlugin(pluginId);
                unregisterWebPlugin(pluginId);
                await refreshRemotePluginList();
                toast(`Disconnected plugin: ${pluginId}`, 'info');
            } catch (error) {
                toast(error?.message || 'Disconnect failed', 'error');
            } finally {
                removeBtn.disabled = false;
            }
        });

        row.appendChild(textWrap);
        row.appendChild(removeBtn);
        listEl.appendChild(row);
    });
}

function setupRemotePluginSettings() {
    const connectBtn = byId('remote-plugin-connect');
    const input = byId('remote-plugin-base');
    if (!connectBtn || !input) return;

    const connect = async () => {
        const baseUrl = String(input.value || '').trim();
        if (!baseUrl) {
            toast('Enter plugin IP:port or URL', 'warning');
            return;
        }

        connectBtn.disabled = true;
        try {
            const response = await api.connectRemoteWebPlugin(baseUrl);
            const plugin = response?.plugin;
            if (plugin) {
                registerWebPlugin(plugin);
            }
            input.value = '';
            await refreshRemotePluginList();
            toast('Remote plugin connected', 'success');
        } catch (error) {
            toast(error?.message || 'Remote plugin connect failed', 'error');
        } finally {
            connectBtn.disabled = false;
        }
    };

    listen(connectBtn, 'click', connect);
    listen(input, 'keydown', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            connect();
        }
    });
}

document.addEventListener('DOMContentLoaded', async () => {
    try {
        initState();
        setupAppConfirmModal();

        window.progressManager = new ProgressManager();
        window.maskEditor = new MaskEditor();
        window.loraManager = new LoraManager();
        window.embeddingManager = new EmbeddingManager();
        window.galleryManager = new GalleryManager();
        window.lightboxManager = new LightboxManager({
            onUpscale: (src, cb) => startUpscale(src, cb),
            onInpaint: (src) => sendToInpaint(src),
            onDuckMotion: (src, curr) => sendToDuckMotion(src, curr),
            onRegenerate: handleRegenerateFromLightbox,
            onStageSettings: handleStageSettingsFromLightbox,
            onDelete: (src, type) => window.galleryManager.handleDelete(src, type),
            onFavorite: (src, currentlyFavorite) => window.galleryManager.handleFavoriteToggle(src, currentlyFavorite),
        });

        window.galleryManager.init();

        await loadWebPlugins();
        setupNavigation();
        setupHelpModals();
        setupMobileStudioToggle();
        setupCollapsibleSections();
        setupSliders();
        setupDenoiseModeToggle();
        setupPresetChips();
        setupFormHandlers();
        setupGenerationButtons();
        setupUploadHandling();
        setupSmartExtendPlacement();
        setupPreviewToolbar();
        setupQueuePanel();
        setupRealtimeGalleryRefresh();
        setupCatalogWatcher();
        setupIpAdapterManager();

        await Promise.all([loadModels(), loadSchedulers(), window.galleryManager.load()]);

        syncToDOM();
        refreshDenoiseControls();
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
    listen(document, 'click', (event) => {
        const tab = event.target?.closest?.('.nav-tab, .mobile-tab');
        if (!tab) return;
        const view = tab.dataset?.view;
        if (!view) return;
        switchView(view);
    });
}

function switchView(viewName) {
    const requested = String(viewName || '').trim();
    const nextView = byId(`view-${requested}`) ? requested : 'studio';

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
            output.textContent = input.value;
        };

        update();
        listen(input, 'input', update);
    });

    refreshDenoiseControls();
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

    ['prompt', 'negative', 'width', 'height', 'steps', 'cfg', 'scheduler', 'batch', 'long-run-warning-minutes', 'clip-skip-2-enabled', 'seed_input', 'second_pass_steps', 'second_pass_blend', 'second_pass_enabled', 'second_pass_model', 'denoising_strength', 'denoise-mode', 'smart-extend-enabled', 'smart-extend-pyramid-enable', 'smart-extend-advanced-enabled', 'smart-extend-feather', 'smart-extend-auto-step', 'smart-extend-step-growth', 'smart-extend-refine', 'smart-extend-refine-each-step', 'smart-extend-refine-width', 'smart-extend-refine-strength', 'smart-extend-pyramid-trigger-ratio', 'ip-adapter-enabled', 'ip-adapter-type', 'ip-adapter-refs', 'ip-adapter-scale', 'ip-adapter-lora-scale'].forEach(id => {
        const el = byId(id);
        if (!el) return;
        listen(el, 'input', saveState);
        listen(el, 'change', saveState);
    });

    const normalizeDimsAndPersist = () => {
        normalizeResolutionInputs({ syncState: true });
        syncFromDOM();
    };
    listen(byId('width'), 'change', normalizeDimsAndPersist);
    listen(byId('height'), 'change', normalizeDimsAndPersist);
    listen(byId('width'), 'blur', normalizeDimsAndPersist);
    listen(byId('height'), 'blur', normalizeDimsAndPersist);

    const refreshSmartExtendSizeWarning = () => {
        updateSmartExtendSizeWarning(Number(byId('width')?.value || 0), Number(byId('height')?.value || 0));
    };
    listen(byId('width'), 'input', refreshSmartExtendSizeWarning);
    listen(byId('height'), 'input', refreshSmartExtendSizeWarning);
    listen(byId('smart-extend-enabled'), 'change', refreshSmartExtendSizeWarning);
    refreshSmartExtendSizeWarning();

    const updateSmartExtendAdvancedVisibility = () => {
        const enabled = Boolean(byId('smart-extend-advanced-enabled')?.checked);
        toggleClass(byId('smart-extend-advanced-panel'), 'hidden', !enabled);
    };
    listen(byId('smart-extend-advanced-enabled'), 'change', () => {
        updateSmartExtendAdvancedVisibility();
        syncFromDOM();
    });
    updateSmartExtendAdvancedVisibility();

    const updateIPAdapterVisibility = () => {
        const enabled = Boolean(byId('ip-adapter-enabled')?.checked);
        toggleClass(byId('ip-adapter-controls'), 'hidden', !enabled);
    };
    listen(byId('ip-adapter-enabled'), 'change', () => {
        updateIPAdapterVisibility();
        syncFromDOM();
    });
    updateIPAdapterVisibility();

    const updateSliderDisplay = (sliderId, displayId) => {
        const slider = byId(sliderId);
        const display = byId(displayId);
        if (slider && display) {
            const update = () => { display.textContent = parseFloat(slider.value).toFixed(2); };
            listen(slider, 'input', update);
            update();
        }
    };
    updateSliderDisplay('ip-adapter-scale', 'ip-adapter-scale-value');
    updateSliderDisplay('ip-adapter-lora-scale', 'ip-adapter-lora-scale-value');

    const promptEl = byId('prompt');
    if (promptEl) {
        listen(promptEl, 'input', debounce(() => updateTokenCounter(promptEl.value), 300));
    }

    listen(byId('randomize-seed'), 'click', () => {
        const seed = byId('seed_input');
        if (seed) seed.value = '';
        setSeed(null);
        toast('Seed set to Random mode', 'info');
    });

    listen(byId('base_model'), 'change', async () => {
        const modelName = byId('base_model')?.value;
        if (!modelName) return;

        setState({ baseModel: modelName });
        await window.loraManager.loadForModel(modelName);
        await window.embeddingManager.loadForModel(modelName);
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

function readSmartExtendOptions() {
    const advanced = Boolean(byId('smart-extend-advanced-enabled')?.checked);
    if (!advanced) {
        return {
            advanced,
            feather: SMART_EXTEND_FIXED.feather,
            autoStep: SMART_EXTEND_FIXED.autoStep,
            stepGrowth: SMART_EXTEND_FIXED.stepGrowth,
            refine: SMART_EXTEND_FIXED.refine,
            refineEachStep: SMART_EXTEND_FIXED.refineEachStep,
            refineWidth: SMART_EXTEND_FIXED.refineWidth,
            refineStrength: SMART_EXTEND_FIXED.refineStrength,
            pyramidTriggerRatio: SMART_EXTEND_FIXED.pyramidTriggerRatio,
        };
    }
    const state = getState();
    return {
        advanced,
        feather: Number(byId('smart-extend-feather')?.value || state.smartExtendFeather || SMART_EXTEND_FIXED.feather),
        autoStep: Boolean(byId('smart-extend-auto-step')?.checked),
        stepGrowth: Number(byId('smart-extend-step-growth')?.value || state.smartExtendStepGrowth || SMART_EXTEND_FIXED.stepGrowth),
        refine: Boolean(byId('smart-extend-refine')?.checked),
        refineEachStep: Boolean(byId('smart-extend-refine-each-step')?.checked),
        refineWidth: Number(byId('smart-extend-refine-width')?.value || state.smartExtendRefineWidth || SMART_EXTEND_FIXED.refineWidth),
        refineStrength: Number(byId('smart-extend-refine-strength')?.value || state.smartExtendRefineStrength || SMART_EXTEND_FIXED.refineStrength),
        pyramidTriggerRatio: Number(byId('smart-extend-pyramid-trigger-ratio')?.value || state.smartExtendPyramidTriggerRatio || SMART_EXTEND_FIXED.pyramidTriggerRatio),
    };
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
        const tooltipText = 'Prompt exceeds one CLIP window (77 tokens). WebbDuck now chunks long prompts, but very long prompts use more VRAM and can slow generation.';

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
        normalizeResolutionInputs({ syncState: true });
        syncFromDOM();
        if (!await maybeShowLargeResolutionWarning()) {
            toast('Run cancelled by user', 'info');
            return;
        }
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
    const width = normalizeDimensionToMultipleOf8(state.width || byId('width')?.value || 1024, 1024);
    const height = normalizeDimensionToMultipleOf8(state.height || byId('height')?.value || 1024, 1024);
    const widthInput = byId('width');
    const heightInput = byId('height');
    if (widthInput) widthInput.value = String(width);
    if (heightInput) heightInput.value = String(height);
    setState({ width, height });

    formData.append('prompt', byId('prompt')?.value || '');
    const negative = byId('negative')?.value || '';
    formData.append('negative_prompt', negative);
    // Keep legacy key for compatibility with any older handlers.
    formData.append('negative', negative);
    formData.append('width', width);
    formData.append('height', height);
    formData.append('steps', byId('steps')?.value || 30);
    formData.append('cfg', byId('cfg')?.value || 7.5);
    formData.append('scheduler', byId('scheduler')?.value || '');
    formData.append('num_images', byId('batch')?.value || 1);
    formData.append('base_model', byId('base_model')?.value || '');

    const seedVal = byId('seed_input')?.value;
    if (seedVal) formData.append('seed', seedVal);
    if (byId('clip-skip-2-enabled')?.checked) {
        formData.append('clip_skip', '2');
    }

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
    const embeddings = window.embeddingManager?.getSelected() || [];
    if (embeddings.length > 0) {
        formData.append('embeddings', JSON.stringify(embeddings));
    }

    if (byId('ip-adapter-enabled')?.checked) {
        const refsJson = byId('ip-adapter-refs-json')?.value || '[]';
        let refs = [];
        try { refs = JSON.parse(refsJson); } catch { refs = []; }
        if (Array.isArray(refs) && refs.length > 0) {
            const adapterType = byId('ip-adapter-type')?.value || 'faceid_sdxl';
            const presetSelect = byId('ip-adapter-preset-select');
            const presetName = presetSelect?.value || '';
            const payload = {
                enabled: true,
                type: adapterType,
                repo: 'h94/IP-Adapter-FaceID',
                adapter_weight: 'ip-adapter-faceid_sdxl.bin',
                embedder: 'buffalo_l',
                adapter_scale: parseFloat(byId('ip-adapter-scale')?.value || 1.0),
                lora_scale: parseFloat(byId('ip-adapter-lora-scale')?.value || 0.60),
                reference_images: refs,
                reference_mode: 'primary_only',
                preset_name: presetName,
            };
            if (adapterType === 'flux2_native') {
                payload.face_crop = byId('ip-adapter-face-crop')?.value || 'auto';
                payload.flux2_anchor_dup = Boolean(byId('ip-adapter-anchor-dup')?.checked);
                payload.face_focus = Boolean(byId('ip-adapter-face-focus')?.checked);
            }
            formData.append('identity_adapter', JSON.stringify(payload));
        }
    }

    if (window._uploadedImage) {
        formData.append('image', window._uploadedImage);
        const denoiseUi = parseFloat(byId('denoising_strength')?.value ?? '0.85');
        const denoise = mapDenoiseUiToActual(denoiseUi).toFixed(2);
        formData.append('strength', denoise);
        // Keep legacy key for compatibility with any older handlers.
        formData.append('denoising_strength', denoise);
        if (byId('smart-extend-enabled')?.checked) {
            const smartExtendOpts = readSmartExtendOptions();
            formData.append('smart_extend', 'true');
            formData.append('smart_extend_anchor', 'center');
            formData.append('smart_extend_advanced', smartExtendOpts.advanced ? 'true' : 'false');
            formData.append('smart_extend_feather', String(Math.round(smartExtendOpts.feather)));
            formData.append('smart_extend_auto_step', smartExtendOpts.autoStep ? 'true' : 'false');
            formData.append('smart_extend_step_growth', Number(smartExtendOpts.stepGrowth).toFixed(2));
            formData.append('smart_extend_refine', smartExtendOpts.refine ? 'true' : 'false');
            formData.append('smart_extend_refine_each_step', smartExtendOpts.refineEachStep ? 'true' : 'false');
            formData.append('smart_extend_refine_width', String(Math.round(smartExtendOpts.refineWidth)));
            formData.append('smart_extend_refine_strength', Number(smartExtendOpts.refineStrength).toFixed(2));
            formData.append('smart_extend_pyramid_enable', byId('smart-extend-pyramid-enable')?.checked ? 'true' : 'false');
            formData.append('smart_extend_pyramid_trigger_ratio', Number(smartExtendOpts.pyramidTriggerRatio).toFixed(2));
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

async function cancelGeneration() {
    isGenerating = false;
    const activeJobId = latestQueuePayload?.active_job_id;
    if (!activeJobId) {
        window.progressManager?.hideProgress();
        toast('No running job to cancel', 'info');
        emit(Events.GENERATION_CANCEL);
        return;
    }

    try {
        const res = await api.cancelQueue(activeJobId);
        if (res?.status === 'cancelling') {
            toast('Cancellation requested', 'warning');
        } else {
            toast('Generation cancelled', 'warning');
        }
        emit(Events.GENERATION_CANCEL);
    } catch (error) {
        toast(error?.message || 'Cancel failed', 'error');
    }
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
            formData.append('style', 'sd_prompt');

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

    listen(byId('upscale-input-btn'), 'click', async () => {
        if (!window._uploadedImage) {
            toast('No image to upscale', 'error');
            return;
        }
        try {
            toast('Upscaling input image...', 'info');
            const formData = new FormData();
            formData.append('image', window._uploadedImage);
            formData.append('scale', '2');
            const data = await api.upscaleInput(formData);
            const url = data?.image;
            if (url) {
                toast('Upscale complete — saved to gallery', 'success');
                window.galleryManager?.invalidateFilterCache?.();
            } else {
                toast('Upscale returned no result', 'warning');
            }
        } catch (error) {
            console.error('Input upscale error:', error);
            toast(`Upscale failed: ${error.message}`, 'error');
        }
    });
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
        embedding: {
            title: 'Embedding Stack Help',
            html: `
                <h4>Embedding Stack</h4>
                <p>Embeddings (textual inversion) add learned concepts through trigger tokens.</p>
                <ul>
                  <li>Add one or more embeddings from the list.</li>
                  <li>Each card shows the active token used in your prompt.</li>
                  <li>Most embeddings work best when the token appears once in prompt text.</li>
                  <li>Use embeddings and LoRAs together carefully to avoid style conflicts.</li>
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
                  <li><strong>Default mode:</strong> uses tuned defaults (Feather <code>12</code>, Seam Width <code>64</code>, Refine Strength <code>0.32</code>, Pyramid Ratio <code>2.4</code>).</li>
                  <li><strong>Advanced mode:</strong> enable <em>Advanced Smart Extend Controls</em> to override those values manually.</li>
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
    const width = byId('width');
    const height = byId('height');
    if (!canvas || !enabled || !width || !height) return;

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
            const updatedGeom = renderSmartExtendCanvas();
            // Rebase drag deltas every move so resize stays smooth even as scale changes.
            resize.startClientX = p.x;
            resize.startClientY = p.y;
            resize.startW = newW;
            resize.startH = newH;
            resize.startX = newX;
            resize.startY = newY;
            if (updatedGeom && Number.isFinite(updatedGeom.scale) && updatedGeom.scale > 0) {
                resize.scale = updatedGeom.scale;
            }
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
    const baseHalf = Math.max(12, Math.min(22, Math.round(Math.min(geom.frameW, geom.frameH) * 0.03)));
    const half = IS_COARSE_POINTER ? Math.max(22, Math.round(baseHalf * 1.35)) : baseHalf;
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
        return { ...h, half: Math.max(20, Math.round(h.half * (IS_COARSE_POINTER ? 2.0 : 1.5))) };
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

    // Fallback: allow edge drags anywhere near the frame border (not only on corner/edge boxes).
    const edgeTol = IS_COARSE_POINTER ? 30 : 22;
    const withinX = px >= (geom.frameX - edgeTol) && px <= (geom.frameX + geom.frameW + edgeTol);
    const withinY = py >= (geom.frameY - edgeTol) && py <= (geom.frameY + geom.frameH + edgeTol);
    if (!withinX || !withinY) return null;
    const nearLeft = Math.abs(px - geom.frameX) <= edgeTol;
    const nearRight = Math.abs(px - (geom.frameX + geom.frameW)) <= edgeTol;
    const nearTop = Math.abs(py - geom.frameY) <= edgeTol;
    const nearBottom = Math.abs(py - (geom.frameY + geom.frameH)) <= edgeTol;
    if (nearTop && nearLeft) return 'top_left';
    if (nearTop && nearRight) return 'top_right';
    if (nearBottom && nearLeft) return 'bottom_left';
    if (nearBottom && nearRight) return 'bottom_right';
    if (nearLeft) return 'left';
    if (nearRight) return 'right';
    if (nearTop) return 'top';
    if (nearBottom) return 'bottom';
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
    updateSmartExtendSizeWarning(targetW, targetH);

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
    let imgX = frameX + placeX * scale;
    let imgY = frameY + placeY * scale;
    let imgW = srcW * scale;
    let imgH = srcH * scale;
    if (imgW > frameW || imgH > frameH) {
        const fitScale = Math.min(frameW / srcW, frameH / srcH);
        imgW = srcW * fitScale;
        imgH = srcH * fitScale;
        imgX = frameX + (frameW - imgW) / 2;
        imgY = frameY + (frameH - imgH) / 2;
    }
    const riskyOutpaint = Boolean(byId('smart-extend-enabled')?.checked && isAboveOutpaintSafetyResolution(targetW, targetH));

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

    ctx.strokeStyle = riskyOutpaint ? 'rgba(239,68,68,0.95)' : 'rgba(96,165,250,0.95)';
    ctx.lineWidth = 2;
    ctx.strokeRect(imgX, imgY, imgW, imgH);

    const handles = getResizeHandles({ frameX, frameY, frameW, frameH });
    ctx.fillStyle = riskyOutpaint ? 'rgba(254, 202, 202, 0.95)' : 'rgba(147, 197, 253, 0.95)';
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
        if (riskyOutpaint) {
            const warn = 'Large target: outpaint can fail or artifact.';
            ctx.fillStyle = 'rgba(254, 202, 202, 0.95)';
            const ww = ctx.measureText(warn).width;
            const wx = frameX + Math.max(10, (frameW - ww) / 2);
            const wy = Math.min(ch - 12, frameY + frameH + 18);
            ctx.fillText(warn, wx, wy);
        }
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

    listen(byId('preview-duckmotion'), 'click', async () => {
        const img = byId('preview-image');
        if (!img?.src) return;
        await sendToDuckMotion(img.src, { src: img.src });
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
            window.galleryManager?.invalidateFilterCache?.();
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
    if (/^https?:\/\//i.test(pathOrUrl)) return pathOrUrl;

    const normalized = pathOrUrl.replace(/\\/g, '/');
    const outputMarker = '/outputs/';
    const idx = normalized.lastIndexOf(outputMarker);
    if (idx >= 0) return normalized.slice(idx);

    const lower = normalized.toLowerCase();
    const fallbackIdx = lower.lastIndexOf('outputs/');
    if (fallbackIdx >= 0) return `/${normalized.slice(fallbackIdx)}`;

    if (pathOrUrl.startsWith('/')) return pathOrUrl;

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
    const ok = await applyLightboxSettingsToStudio(curr, { preserveSeed: false });
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
            await window.embeddingManager?.loadForModel(baseModel);
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

    if (window.embeddingManager) {
        window.embeddingManager.clear();
        const embeddings = Array.isArray(meta.embeddings) ? meta.embeddings : [];
        embeddings.forEach((embedding) => {
            if (typeof embedding === 'string') {
                window.embeddingManager.addEmbedding(embedding, embedding);
                return;
            }
            const name = embedding?.name || embedding?.model;
            if (!name) return;
            const token = String(embedding?.token || name);
            window.embeddingManager.addEmbedding(name, token);
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
            await window.embeddingManager.loadForModel(initialModel);
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
    listen(byId('open-settings-modal'), 'click', openSettingsModal);
    listen(byId('close-settings-modal'), 'click', closeSettingsModal);
    listen(byId('close-settings-modal-footer'), 'click', closeSettingsModal);
    listen(byId('unload-models-btn'), 'click', handleUnloadModelsClick);
    listen(byId('shutdown-app-btn'), 'click', handleShutdownAppClick);
    setupRemotePluginSettings();
    listen(byId('queue-modal'), 'click', (event) => {
        if (event.target?.id === 'queue-modal') {
            closeQueueModal();
        }
    });
    listen(byId('settings-modal'), 'click', (event) => {
        if (event.target?.id === 'settings-modal') {
            closeSettingsModal();
        }
    });
    listen(document, 'keydown', (event) => {
        if (event.key === 'Escape') {
            closeQueueModal();
            closeSettingsModal();
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

// ── IP-Adapter FaceID Manager ───────────────────────────────────

let _ipAdapterRefs = [];
let _ipAdapterPresets = {};

function setupIpAdapterManager() {
    const grid = byId('ip-adapter-refs-grid');
    const uploadInput = byId('ip-adapter-refs-upload-input');
    const uploadBtn = byId('ip-adapter-refs-upload-btn');
    const presetSelect = byId('ip-adapter-preset-select');
    const saveBtn = byId('ip-adapter-preset-save-btn');
    const deleteBtn = byId('ip-adapter-preset-delete-btn');
    const nameRow = byId('ip-adapter-preset-name-row');
    const nameInput = byId('ip-adapter-preset-name-input');
    const confirmSave = byId('ip-adapter-preset-confirm-save');

    // Load from saved state
    try {
        const saved = JSON.parse(byId('ip-adapter-refs-json')?.value || '[]');
        if (Array.isArray(saved)) _ipAdapterRefs = saved;
    } catch {}

    function renderGrid() {
        grid.innerHTML = '';
        const selectedPaths = new Set(_ipAdapterRefs);

        // Fetch server-side refs and render all
        fetch('/ip-adapter/refs').then(res => res.json()).then(data => {
            const serverImages = data.images || [];
            const allUrls = new Set();

            // Render server images as selectable thumbnails
            for (const img of serverImages) {
                allUrls.add(img.url);
                const isSelected = selectedPaths.has(img.url);
                const thumb = document.createElement('div');
                thumb.className = 'ip-adapter-ref-thumb' + (isSelected ? ' selected' : '');
                thumb.title = img.name;
                thumb.innerHTML = `<img src="${img.url}" />${isSelected ? '<button class="remove-ref">x</button>' : ''}`;
                thumb.addEventListener('click', () => {
                    const url = img.url;
                    const idx = _ipAdapterRefs.indexOf(url);
                    if (idx >= 0) {
                        _ipAdapterRefs.splice(idx, 1);
                    } else {
                        _ipAdapterRefs.push(url);
                    }
                    syncRefsState();
                    renderGrid();
                });
                if (isSelected) {
                    const rmBtn = thumb.querySelector('.remove-ref');
                    rmBtn.addEventListener('click', async (e) => {
                        e.stopPropagation();
                        _ipAdapterRefs = _ipAdapterRefs.filter(p => p !== img.url);
                        syncRefsState();
                        renderGrid();
                        try {
                            await fetch(`/ip-adapter/refs/${encodeURIComponent(img.name)}`, { method: 'DELETE' });
                        } catch {}
                    });
                }
                grid.appendChild(thumb);
            }

            // Also show any selected refs not on server (e.g., external paths)
            for (const p of _ipAdapterRefs) {
                if (!allUrls.has(p)) {
                    const isSelected = true;
                    const thumb = document.createElement('div');
                    thumb.className = 'ip-adapter-ref-thumb selected';
                    thumb.innerHTML = `<img src="${p}" onerror="this.parentElement.remove()" /><button class="remove-ref">x</button>`;
                    const rmBtn = thumb.querySelector('.remove-ref');
                    rmBtn.addEventListener('click', async (e) => {
                        e.stopPropagation();
                        _ipAdapterRefs = _ipAdapterRefs.filter(x => x !== p);
                        syncRefsState();
                        renderGrid();
                    });
                    grid.appendChild(thumb);
                }
            }

            if (grid.children.length === 0) {
                grid.innerHTML = '<span style="color:var(--text-muted);font-size:12px;align-self:center;">No reference images. Upload face photos below.</span>';
            }
        }).catch(() => {
            grid.innerHTML = '<span style="color:var(--text-muted);font-size:12px;align-self:center;">Could not load ref images.</span>';
        });
    }

    function syncRefsState() {
        byId('ip-adapter-refs-json').value = JSON.stringify(_ipAdapterRefs);
        setState({ identityAdapterRefs: _ipAdapterRefs });
    }

    // Upload button
    uploadBtn.addEventListener('click', () => uploadInput.click());

    uploadInput.addEventListener('change', async () => {
        const files = uploadInput.files;
        if (!files || files.length === 0) return;
        toast(`Uploading ${files.length} image(s)...`, 'info');
        let uploaded = 0;
        for (const file of files) {
            const formData = new FormData();
            formData.append('file', file);
            try {
                const res = await fetch('/ip-adapter/refs/upload', { method: 'POST', body: formData });
                if (!res.ok) {
                    console.warn('Upload returned', res.status, await res.text());
                    continue;
                }
                const data = await res.json();
                if (data.url) {
                    _ipAdapterRefs.push(data.url);
                    uploaded++;
                }
            } catch (err) {
                console.error('Upload failed:', err);
                toast(`Upload failed: ${err.message}`, 'error');
            }
        }
        uploadInput.value = '';
        syncRefsState();
        renderGrid();
        if (uploaded > 0) toast(`${uploaded} image(s) uploaded`, 'success');
        else toast('Upload failed — check console', 'error');
    });

    // Presets
    async function loadPresets() {
        try {
            const res = await fetch('/ip-adapter/presets');
            const data = await res.json();
            _ipAdapterPresets = data.presets || {};
            renderPresetSelect();
        } catch (err) {
            console.error('Load presets failed:', err);
        }
    }

    function renderPresetSelect() {
        presetSelect.innerHTML = '<option value="">-- Load Preset --</option>';
        for (const name of Object.keys(_ipAdapterPresets).sort()) {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            presetSelect.appendChild(opt);
        }
    }

    presetSelect.addEventListener('change', () => {
        const name = presetSelect.value;
        if (!name || !_ipAdapterPresets[name]) return;
        const preset = _ipAdapterPresets[name];
        _ipAdapterRefs = Array.isArray(preset.refs) ? [...preset.refs] : [];
        syncRefsState();
        renderGrid();
        if (preset.adapter_scale != null) {
            byId('ip-adapter-scale').value = preset.adapter_scale;
            byId('ip-adapter-scale-value').textContent = preset.adapter_scale;
        }
        if (preset.lora_scale != null) {
            byId('ip-adapter-lora-scale').value = preset.lora_scale;
            byId('ip-adapter-lora-scale-value').textContent = preset.lora_scale;
        }
        if (preset.type) {
            byId('ip-adapter-type').value = preset.type;
        }
        if (preset.face_crop) {
            byId('ip-adapter-face-crop').value = preset.face_crop;
        }
        byId('ip-adapter-anchor-dup').checked = Boolean(preset.flux2_anchor_dup);
        byId('ip-adapter-face-focus').checked = Boolean(preset.face_focus);
    });

    saveBtn.addEventListener('click', () => {
        nameRow.classList.toggle('hidden');
        nameInput.value = '';
        nameInput.focus();
    });

    confirmSave.addEventListener('click', async () => {
        const name = nameInput.value.trim();
        if (!name) return;
        const payload = {
            name,
            type: byId('ip-adapter-type')?.value || 'faceid_sdxl',
            refs: _ipAdapterRefs,
            adapter_scale: parseFloat(byId('ip-adapter-scale')?.value || 1.0),
            lora_scale: parseFloat(byId('ip-adapter-lora-scale')?.value || 0.60),
            face_crop: byId('ip-adapter-face-crop')?.value || 'auto',
            flux2_anchor_dup: Boolean(byId('ip-adapter-anchor-dup')?.checked),
            face_focus: Boolean(byId('ip-adapter-face-focus')?.checked),
        };
        try {
            const res = await fetch('/ip-adapter/presets', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (res.ok) {
                nameRow.classList.add('hidden');
                await loadPresets();
                presetSelect.value = name;
            }
        } catch (err) {
            console.error('Save preset failed:', err);
        }
    });

    deleteBtn.addEventListener('click', async () => {
        const name = presetSelect.value;
        if (!name) return;
        try {
            await fetch(`/ip-adapter/presets/${encodeURIComponent(name)}`, { method: 'DELETE' });
            await loadPresets();
        } catch (err) {
            console.error('Delete preset failed:', err);
        }
    });

    // Load initial data
    loadPresets();
    renderGrid();

    // Also reload when section becomes visible (user opens it)
    const section = byId('section-ip-adapter');
    if (section) {
        const observer = new MutationObserver(() => {
            if (!section.classList.contains('collapsed')) {
                loadPresets();
                renderGrid();
            }
        });
        observer.observe(section, { attributes: true, attributeFilter: ['class'] });
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
        const embeddings = Array.isArray(job.settings?.embeddings) ? job.settings.embeddings : [];
        const modeDetails = Array.isArray(job.settings?.mode_details) ? job.settings.mode_details : [];
        const thumbHtml = inputThumb
            ? `<img class="queue-item-thumb" src="${escapeHtml(inputThumb)}" alt="Queue input preview" loading="lazy" />`
            : '';
        const canCancel = status === 'queued' || status === 'running' || status === 'cancelling';
        const statusText = status === 'running'
            ? 'Running'
            : status === 'queued'
                ? 'Queued'
                : status === 'cancelling'
                    ? 'Cancelling'
                    : status;
        const isExpanded = expandedQueueJobs.has(job.job_id);
        const hasExtraDetails = seed !== null || negative.length > 0 || loras.length > 0 || embeddings.length > 0 || modeDetails.length > 0;
        const detailsHtml = hasExtraDetails
            ? `
                <div class="queue-item-details ${isExpanded ? '' : 'hidden'}" data-details-for="${job.job_id}">
                  <div class="queue-detail-row"><span class="queue-detail-label">Seed</span><span class="queue-detail-value">${seed === null ? 'Random' : escapeHtml(String(seed))}</span></div>
                  <div class="queue-detail-row"><span class="queue-detail-label">Negative</span><span class="queue-detail-value">${negative ? escapeHtml(negative.slice(0, 220)) : 'None'}</span></div>
                  <div class="queue-detail-row"><span class="queue-detail-label">Mode</span><span class="queue-detail-value">${modeDetails.length > 0 ? escapeHtml(modeDetails.join(' | ')) : 'Default'}</span></div>
                  <div class="queue-detail-row"><span class="queue-detail-label">LoRAs</span><span class="queue-detail-value">${loras.length > 0 ? escapeHtml(loras.join(', ')) : 'None'}</span></div>
                  <div class="queue-detail-row"><span class="queue-detail-label">Embeddings</span><span class="queue-detail-value">${embeddings.length > 0 ? escapeHtml(embeddings.join(', ')) : 'None'}</span></div>
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
                  ${canCancel ? `<button class="btn btn-ghost btn-sm queue-cancel" data-job-id="${job.job_id}" type="button">${status === 'cancelling' ? 'Cancelling...' : 'Cancel'}</button>` : ''}
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
            btn.disabled = true;
            try {
                const res = await api.cancelQueue(jobId);
                if (res?.status === 'cancelling') {
                    toast('Cancellation requested for running job', 'info');
                } else {
                    toast('Queued job cancelled', 'info');
                }
                // Backend pushes fresh queue state via WebSocket.
            } catch (err) {
                toast(err.message || 'Cancel failed', 'error');
            } finally {
                btn.disabled = false;
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

function openSettingsModal() {
    const modal = byId('settings-modal');
    if (!modal) return;
    modal.classList.remove('hidden');
    modal.classList.add('active');
    modal.setAttribute('aria-hidden', 'false');
    void refreshRemotePluginList();
}

function closeSettingsModal() {
    const modal = byId('settings-modal');
    if (!modal || modal.classList.contains('hidden')) return;
    modal.classList.remove('active');
    modal.classList.add('hidden');
    modal.setAttribute('aria-hidden', 'true');
}

async function handleUnloadModelsClick() {
    if (!await showAppConfirmModal({
        title: 'Unload Models',
        message: 'Unload all generation models from VRAM now?\n\nThis frees GPU memory and the next run will reload the selected model.',
        okText: 'Unload',
        cancelText: 'Cancel',
        showCancel: true,
    })) {
        return;
    }

    try {
        await api.unloadAllModels();
        toast('Models unloaded from memory', 'success');
    } catch (err) {
        toast(err?.message || 'Unload failed', 'error');
    }
}

async function handleShutdownAppClick() {
    if (!await showAppConfirmModal({
        title: 'Shut Down App',
        message: 'Shut down WebbDuck now?\n\nThis stops the server process.',
        okText: 'Shut Down',
        cancelText: 'Cancel',
        showCancel: true,
        danger: true,
    })) {
        return;
    }

    try {
        await api.shutdownApp();
        toast('Shutting down server...', 'warning', 6000);
    } catch (err) {
        toast(err?.message || 'Shutdown failed', 'error');
    }
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
