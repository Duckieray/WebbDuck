/**
 * WebbDuck State Management
 * Centralized state with localStorage persistence
 */

const STORAGE_KEY = 'webbduck_state_v2';
const DENOISE_DETAIL_MIN = 0.75;
const DENOISE_PRESERVE_MIN = 0.30;
const DENOISE_PRESERVE_MAX = 0.50;
const DENOISE_ACTUAL_MAX = 1.00;

// Default state values
const DEFAULT_STATE = {
    prompt: '',
    negative: '',
    width: 1024,
    height: 1024,
    steps: 30,
    cfg: 7.5,
    seed: null,
    scheduler: '',
    batch: 1,
    longRunWarningMinutes: 8,
    baseModel: '',
    secondPassEnabled: false,
    secondPassModel: 'None',
    secondPassSteps: 20,
    secondPassBlend: 0.8,
    denoisingStrength: 0.85,
    denoisingMode: 'details', // 'details' | 'preserve'
    denoisingScaleVersion: 5,
    smartExtendEnabled: false,
    smartExtendAdvanced: false,
    smartExtendAnchor: 'center',
    smartExtendFeather: 12,
    smartExtendAutoStep: false,
    smartExtendStepGrowth: 1.25,
    smartExtendRefine: true,
    smartExtendRefineEachStep: true,
    smartExtendRefineWidth: 64,
    smartExtendRefineStrength: 0.32,
    smartExtendPyramidTriggerRatio: 2.4,
    smartExtendOffsetX: null,
    smartExtendOffsetY: null,
    clipSkip2: false,
    selectedLoras: [],
    selectedEmbeddings: [],
    inpaintMode: 'replace', // 'replace' or 'keep'
    view: 'studio',
    identityAdapterEnabled: false,
    identityAdapterType: 'faceid_sdxl',
    identityAdapterRefs: [],
    identityAdapterScale: 1.0,
    identityAdapterLoraScale: 0.60,
    identityAdapterFaceCrop: 'auto',
    identityAdapterAnchorDup: false,
    identityAdapterFaceFocus: false,
};

// Current state
let state = { ...DEFAULT_STATE };

function normalizeDimensionToMultipleOf8(rawValue, fallback = 1024) {
    const raw = Number(rawValue);
    const safe = Number.isFinite(raw) ? raw : Number(fallback);
    return Math.max(8, Math.round(safe / 8) * 8);
}

function normalizeDenoisingMode(rawMode) {
    return rawMode === 'preserve' ? 'preserve' : 'details';
}

function denoiseRangeForMode(mode) {
    if (normalizeDenoisingMode(mode) === 'preserve') {
        return { min: DENOISE_PRESERVE_MIN, max: DENOISE_PRESERVE_MAX };
    }
    return { min: DENOISE_DETAIL_MIN, max: DENOISE_ACTUAL_MAX };
}

function normalizeDenoisingStrength(rawValue, opts = {}) {
    const mode = normalizeDenoisingMode(opts.mode);
    const scaleVersion = Number(opts.scaleVersion ?? DEFAULT_STATE.denoisingScaleVersion);
    const raw = Number(rawValue);
    if (!Number.isFinite(raw)) return DEFAULT_STATE.denoisingStrength;

    // Legacy state versions stored denoise as normalized 0..1.
    if (scaleVersion < 3 && raw >= 0 && raw <= 1) {
        const migrated = DENOISE_DETAIL_MIN + (DENOISE_ACTUAL_MAX - DENOISE_DETAIL_MIN) * raw;
        const range = denoiseRangeForMode(mode);
        return Math.max(range.min, Math.min(range.max, migrated));
    }
    const range = denoiseRangeForMode(mode);
    return Math.max(range.min, Math.min(range.max, raw));
}

function parseRefsJson(raw) {
    if (!raw) return [];
    try {
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed : [];
    } catch { return []; }
}

// Subscribers for state changes
const subscribers = new Map();

/**
 * Initialize state from localStorage
 */
export function initState() {
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
            const parsed = JSON.parse(saved);
            state = { ...DEFAULT_STATE, ...parsed };
            const inferredMode = parsed.denoisingMode
                ?? ((Boolean(parsed.denoisingFullControl) && Number(parsed.denoisingStrength) <= DENOISE_PRESERVE_MAX)
                    ? 'preserve'
                    : 'details');
            state.denoisingMode = normalizeDenoisingMode(inferredMode);
            state.denoisingStrength = normalizeDenoisingStrength(parsed.denoisingStrength, {
                mode: state.denoisingMode,
                scaleVersion: parsed.denoisingScaleVersion,
            });
            if ((parsed.denoisingScaleVersion ?? 1) < 5) {
                state.denoisingScaleVersion = 5;
                localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
            }
        }
    } catch (error) {
        console.warn('Failed to load state from localStorage:', error);
    }
    return state;
}

/**
 * Get current state or a specific key
 */
export function getState(key) {
    if (key) {
        return state[key];
    }
    return { ...state };
}

/**
 * Update state and persist
 */
export function setState(updates) {
    const prevState = { ...state };
    state = { ...state, ...updates };

    // Persist to localStorage
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (error) {
        console.warn('Failed to save state to localStorage:', error);
    }

    // Notify subscribers
    for (const [key, callbacks] of subscribers) {
        if (key in updates) {
            callbacks.forEach(callback => callback(state[key], prevState[key]));
        }
    }
}

/**
 * Subscribe to state changes for a specific key
 */
export function subscribe(key, callback) {
    if (!subscribers.has(key)) {
        subscribers.set(key, new Set());
    }
    subscribers.get(key).add(callback);

    // Return unsubscribe function
    return () => {
        subscribers.get(key).delete(callback);
    };
}

/**
 * Reset state to defaults
 */
export function resetState() {
    state = { ...DEFAULT_STATE };
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (error) {
        console.warn('Failed to reset state in localStorage:', error);
    }
}

// ═══════════════════════════════════════════════════════════════
// SEED MANAGEMENT
// ═══════════════════════════════════════════════════════════════

let lastUsedSeed = null;

/**
 * Get the seed value (null means random)
 */
export function getSeed() {
    return state.seed;
}

/**
 * Set the seed value
 */
export function setSeed(value) {
    const seed = value === '' || value === null ? null : Number(value);
    setState({ seed });
    lastUsedSeed = seed;
}

/**
 * Get the last seed that was actually used in generation
 */
export function getLastUsedSeed() {
    return lastUsedSeed;
}

/**
 * Set the last used seed (from generation response)
 */
export function setLastUsedSeed(seed) {
    lastUsedSeed = seed;
}

// ═══════════════════════════════════════════════════════════════
// LORA MANAGEMENT
// ═══════════════════════════════════════════════════════════════

/**
 * Add a LoRA to selected
 */
export function addLora(name, strength = 1.0) {
    const loras = [...state.selectedLoras];
    const existing = loras.find(l => l.name === name);
    if (!existing) {
        loras.push({ name, strength });
        setState({ selectedLoras: loras });
    }
}

/**
 * Remove a LoRA from selected
 */
export function removeLora(name) {
    const loras = state.selectedLoras.filter(l => l.name !== name);
    setState({ selectedLoras: loras });
}

/**
 * Update a LoRA's strength
 */
export function updateLoraStrength(name, strength) {
    const loras = state.selectedLoras.map(l =>
        l.name === name ? { ...l, strength } : l
    );
    setState({ selectedLoras: loras });
}

/**
 * Clear all selected LoRAs
 */
export function clearLoras() {
    setState({ selectedLoras: [] });
}

// ═══════════════════════════════════════════════════════════════
// STATE SYNC WITH DOM
// ═══════════════════════════════════════════════════════════════

/**
 * Sync state from DOM elements
 */
export function syncFromDOM() {
    const getValue = (id) => document.getElementById(id)?.value ?? '';
    const getChecked = (id) => document.getElementById(id)?.checked ?? false;
    const denoisingMode = normalizeDenoisingMode(getValue('denoise-mode'));
    const denoiseRaw = parseFloat(getValue('denoising_strength'));
    const denoisingStrength = normalizeDenoisingStrength(denoiseRaw, {
        mode: denoisingMode,
        scaleVersion: state.denoisingScaleVersion,
    });

    const width = normalizeDimensionToMultipleOf8(parseInt(getValue('width'), 10), 1024);
    const height = normalizeDimensionToMultipleOf8(parseInt(getValue('height'), 10), 1024);

    setState({
        prompt: getValue('prompt'),
        negative: getValue('negative'),
        width,
        height,
        steps: parseInt(getValue('steps')) || 30,
        cfg: parseFloat(getValue('cfg')) || 7.5,
        scheduler: getValue('scheduler'),
        batch: parseInt(getValue('batch')) || 1,
        longRunWarningMinutes: parseInt(getValue('long-run-warning-minutes')) || 8,
        baseModel: getValue('base_model'),
        secondPassEnabled: getChecked('second_pass_enabled'),
        secondPassModel: getValue('second_pass_model'),
        secondPassSteps: parseInt(getValue('second_pass_steps')) || 20,
        secondPassBlend: parseFloat(getValue('second_pass_blend')) || 0.8,
        denoisingMode,
        denoisingStrength,
        smartExtendEnabled: getChecked('smart-extend-enabled'),
        smartExtendAdvanced: getChecked('smart-extend-advanced-enabled'),
        smartExtendAnchor: getValue('smart-extend-anchor') || 'center',
        smartExtendFeather: parseInt(getValue('smart-extend-feather')) || 12,
        smartExtendAutoStep: getChecked('smart-extend-auto-step'),
        smartExtendStepGrowth: parseFloat(getValue('smart-extend-step-growth')) || 1.25,
        smartExtendRefine: getChecked('smart-extend-refine'),
        smartExtendRefineEachStep: getChecked('smart-extend-refine-each-step'),
        smartExtendRefineWidth: parseInt(getValue('smart-extend-refine-width')) || 64,
        smartExtendRefineStrength: parseFloat(getValue('smart-extend-refine-strength')) || 0.32,
        smartExtendPyramidTriggerRatio: parseFloat(getValue('smart-extend-pyramid-trigger-ratio')) || 2.4,
        smartExtendOffsetX: state.smartExtendOffsetX ?? null,
        smartExtendOffsetY: state.smartExtendOffsetY ?? null,
        clipSkip2: getChecked('clip-skip-2-enabled'),
        identityAdapterEnabled: getChecked('ip-adapter-enabled'),
        identityAdapterType: getValue('ip-adapter-type') || 'faceid_sdxl',
        identityAdapterRefs: parseRefsJson(getValue('ip-adapter-refs-json')),
        identityAdapterScale: parseFloat(getValue('ip-adapter-scale')) || 1.0,
        identityAdapterLoraScale: parseFloat(getValue('ip-adapter-lora-scale')) || 0.60,
        identityAdapterFaceCrop: getValue('ip-adapter-face-crop') || 'auto',
        identityAdapterAnchorDup: getChecked('ip-adapter-anchor-dup'),
        identityAdapterFaceFocus: getChecked('ip-adapter-face-focus'),
    });
}

/**
 * Sync state to DOM elements
 */
export function syncToDOM() {
    const setValue = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.value = value ?? '';
    };
    const setChecked = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.checked = value ?? false;
    };

    setValue('prompt', state.prompt);
    setValue('negative', state.negative);
    setValue('width', state.width);
    setValue('height', state.height);
    setValue('steps', state.steps);
    setValue('cfg', state.cfg);
    setValue('scheduler', state.scheduler);
    setValue('batch', state.batch);
    setValue('long-run-warning-minutes', state.longRunWarningMinutes ?? 8);
    setChecked('clip-skip-2-enabled', Boolean(state.clipSkip2));
    setValue('base_model', state.baseModel);
    setChecked('second_pass_enabled', state.secondPassEnabled);
    setValue('second_pass_model', state.secondPassModel);
    setValue('second_pass_steps', state.secondPassSteps);
    setValue('second_pass_blend', state.secondPassBlend);
    const denoiseMode = normalizeDenoisingMode(state.denoisingMode);
    setValue('denoise-mode', denoiseMode);
    const preserveBtn = document.getElementById('denoise-mode-preserve');
    const detailsBtn = document.getElementById('denoise-mode-details');
    preserveBtn?.classList.toggle('active', denoiseMode === 'preserve');
    detailsBtn?.classList.toggle('active', denoiseMode !== 'preserve');
    const denoiseInput = document.getElementById('denoising_strength');
    if (denoiseInput) {
        const range = denoiseRangeForMode(denoiseMode);
        denoiseInput.min = String(range.min);
        denoiseInput.max = String(range.max);
    }
    setValue('denoising_strength', normalizeDenoisingStrength(state.denoisingStrength, {
        mode: denoiseMode,
        scaleVersion: state.denoisingScaleVersion,
    }));
    setChecked('smart-extend-enabled', state.smartExtendEnabled);
    setChecked('smart-extend-advanced-enabled', Boolean(state.smartExtendAdvanced));
    setValue('smart-extend-anchor', state.smartExtendAnchor || 'center');
    setValue('smart-extend-feather', state.smartExtendFeather ?? 12);
    setChecked('smart-extend-auto-step', Boolean(state.smartExtendAutoStep));
    setValue('smart-extend-step-growth', state.smartExtendStepGrowth ?? 1.25);
    setChecked('smart-extend-refine', Boolean(state.smartExtendRefine ?? true));
    setChecked('smart-extend-refine-each-step', Boolean(state.smartExtendRefineEachStep ?? true));
    setValue('smart-extend-refine-width', state.smartExtendRefineWidth ?? 64);
    setValue('smart-extend-refine-strength', state.smartExtendRefineStrength ?? 0.32);
    setValue('smart-extend-pyramid-trigger-ratio', state.smartExtendPyramidTriggerRatio ?? 2.4);

    setChecked('ip-adapter-enabled', Boolean(state.identityAdapterEnabled));
    setValue('ip-adapter-type', state.identityAdapterType || 'faceid_sdxl');
    setValue('ip-adapter-refs-json', JSON.stringify(state.identityAdapterRefs ?? []));
    setValue('ip-adapter-scale', state.identityAdapterScale ?? 1.0);
    setValue('ip-adapter-lora-scale', state.identityAdapterLoraScale ?? 0.60);
    setValue('ip-adapter-face-crop', state.identityAdapterFaceCrop ?? 'auto');
    setChecked('ip-adapter-anchor-dup', Boolean(state.identityAdapterAnchorDup));
    setChecked('ip-adapter-face-focus', Boolean(state.identityAdapterFaceFocus));
    const ipControls = document.getElementById('ip-adapter-controls');
    if (ipControls) {
        ipControls.classList.toggle('hidden', !Boolean(state.identityAdapterEnabled));
    }

    // Sync Inpaint Mode Buttons
    const replaceBtn = document.getElementById('inpaint-replace');
    const keepBtn = document.getElementById('inpaint-keep');
    if (replaceBtn && keepBtn) {
        if (state.inpaintMode === 'keep') {
            keepBtn.classList.add('active');
            replaceBtn.classList.remove('active');
        } else {
            replaceBtn.classList.add('active');
            keepBtn.classList.remove('active');
        }
    }

    // Update value displays
    updateValueDisplays();
}

/**
 * Update slider value displays
 */
function updateValueDisplays() {
    const displays = {
        'steps': 'steps-value',
        'cfg': 'cfg-value',
        'batch': 'batch-value',
        'second_pass_steps': 'second-steps-value',
        'second_pass_blend': 'blend-value',
        'denoising_strength': 'denoise-value',
        'smart-extend-feather': 'smart-extend-feather-value',
        'smart-extend-step-growth': 'smart-extend-step-growth-value',
        'smart-extend-refine-width': 'smart-extend-refine-width-value',
        'smart-extend-refine-strength': 'smart-extend-refine-strength-value',
        'smart-extend-pyramid-trigger-ratio': 'smart-extend-pyramid-trigger-ratio-value',
    };

    for (const [inputId, displayId] of Object.entries(displays)) {
        const input = document.getElementById(inputId);
        const display = document.getElementById(displayId);
        if (input && display) {
            if (inputId === 'denoising_strength') {
                const modeInput = document.getElementById('denoise-mode');
                const denoiseValue = normalizeDenoisingStrength(input.value, {
                    mode: normalizeDenoisingMode(modeInput?.value),
                    scaleVersion: state.denoisingScaleVersion,
                });
                display.textContent = denoiseValue.toFixed(2);
            } else if (inputId === 'smart-extend-refine-strength') {
                display.textContent = Number(input.value).toFixed(2);
            } else if (inputId === 'smart-extend-step-growth' || inputId === 'smart-extend-pyramid-trigger-ratio') {
                display.textContent = Number(input.value).toFixed(2);
            } else {
                display.textContent = input.value;
            }
        }
    }
}
