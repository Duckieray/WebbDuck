/**
 * WebbDuck State Management
 * Centralized state with localStorage persistence
 */

const STORAGE_KEY = 'webbduck_state_v2';
const DENOISE_ACTUAL_MIN = 0.70;
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
    baseModel: '',
    secondPassEnabled: false,
    secondPassModel: 'None',
    secondPassSteps: 20,
    secondPassBlend: 0.8,
    denoisingStrength: 0.5,
    denoisingScaleVersion: 2,
    smartExtendEnabled: false,
    smartExtendAnchor: 'center',
    smartExtendFeather: 8,
    smartExtendAutoStep: true,
    smartExtendStepGrowth: 1.25,
    smartExtendRefine: true,
    smartExtendRefineEachStep: true,
    smartExtendRefineWidth: 24,
    smartExtendRefineStrength: 0.28,
    smartExtendOffsetX: null,
    smartExtendOffsetY: null,
    selectedLoras: [],
    inpaintMode: 'replace', // 'replace' or 'keep'
    view: 'studio',
};

// Current state
let state = { ...DEFAULT_STATE };

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
            if ((parsed.denoisingScaleVersion ?? 1) < 2) {
                const oldActual = Number(parsed.denoisingStrength);
                if (Number.isFinite(oldActual)) {
                    const normalized = (oldActual - DENOISE_ACTUAL_MIN) / (DENOISE_ACTUAL_MAX - DENOISE_ACTUAL_MIN);
                    state.denoisingStrength = Math.max(0, Math.min(1, normalized));
                } else {
                    state.denoisingStrength = DEFAULT_STATE.denoisingStrength;
                }
                state.denoisingScaleVersion = 2;
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
    const denoiseRaw = parseFloat(getValue('denoising_strength'));
    const denoisingStrength = Number.isFinite(denoiseRaw) ? denoiseRaw : 0.5;

    setState({
        prompt: getValue('prompt'),
        negative: getValue('negative'),
        width: parseInt(getValue('width')) || 1024,
        height: parseInt(getValue('height')) || 1024,
        steps: parseInt(getValue('steps')) || 30,
        cfg: parseFloat(getValue('cfg')) || 7.5,
        scheduler: getValue('scheduler'),
        batch: parseInt(getValue('batch')) || 1,
        baseModel: getValue('base_model'),
        secondPassEnabled: getChecked('second_pass_enabled'),
        secondPassModel: getValue('second_pass_model'),
        secondPassSteps: parseInt(getValue('second_pass_steps')) || 20,
        secondPassBlend: parseFloat(getValue('second_pass_blend')) || 0.8,
        denoisingStrength,
        smartExtendEnabled: getChecked('smart-extend-enabled'),
        smartExtendAnchor: getValue('smart-extend-anchor') || 'center',
        smartExtendFeather: parseInt(getValue('smart-extend-feather')) || 8,
        smartExtendAutoStep: getChecked('smart-extend-auto-step'),
        smartExtendStepGrowth: parseFloat(getValue('smart-extend-step-growth')) || 1.25,
        smartExtendRefine: getChecked('smart-extend-refine'),
        smartExtendRefineEachStep: getChecked('smart-extend-refine-each-step'),
        smartExtendRefineWidth: parseInt(getValue('smart-extend-refine-width')) || 24,
        smartExtendRefineStrength: parseFloat(getValue('smart-extend-refine-strength')) || 0.28,
        smartExtendOffsetX: state.smartExtendOffsetX ?? null,
        smartExtendOffsetY: state.smartExtendOffsetY ?? null,
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
    setValue('base_model', state.baseModel);
    setChecked('second_pass_enabled', state.secondPassEnabled);
    setValue('second_pass_model', state.secondPassModel);
    setValue('second_pass_steps', state.secondPassSteps);
    setValue('second_pass_blend', state.secondPassBlend);
    setValue('denoising_strength', state.denoisingStrength);
    setChecked('smart-extend-enabled', state.smartExtendEnabled);
    setValue('smart-extend-anchor', state.smartExtendAnchor || 'center');
    setValue('smart-extend-feather', state.smartExtendFeather ?? 8);
    setChecked('smart-extend-auto-step', state.smartExtendAutoStep ?? true);
    setValue('smart-extend-step-growth', state.smartExtendStepGrowth ?? 1.25);
    setChecked('smart-extend-refine', state.smartExtendRefine ?? true);
    setChecked('smart-extend-refine-each-step', state.smartExtendRefineEachStep ?? true);
    setValue('smart-extend-refine-width', state.smartExtendRefineWidth ?? 24);
    setValue('smart-extend-refine-strength', state.smartExtendRefineStrength ?? 0.28);

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
    };

    for (const [inputId, displayId] of Object.entries(displays)) {
        const input = document.getElementById(inputId);
        const display = document.getElementById(displayId);
        if (input && display) {
            if (inputId === 'denoising_strength') {
                const uiValue = parseFloat(input.value);
                const clamped = Number.isFinite(uiValue) ? Math.max(0, Math.min(1, uiValue)) : 0.5;
                const actual = DENOISE_ACTUAL_MIN + (DENOISE_ACTUAL_MAX - DENOISE_ACTUAL_MIN) * clamped;
                display.textContent = actual.toFixed(2);
            } else {
                display.textContent = input.value;
            }
        }
    }
}
