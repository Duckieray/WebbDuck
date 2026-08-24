import { byId, toast } from './utils.js';

const PROFILE_EVENT = 'webbduck:model-profile';
const CATALOG_EVENT = 'webbduck:model-catalog';

let catalog = [];
let activeProfile = null;
let initialized = false;
let profileRequestSerial = 0;

function findProfile(name) {
    if (!name) return null;
    return catalog.find(item => item?.name === name) || null;
}

function setHidden(element, hidden) {
    if (!element) return;
    element.classList.toggle('hidden', !!hidden);
    if ('disabled' in element) element.disabled = !!hidden;
}

function sectionForElement(id) {
    const element = byId(id);
    if (!element) return null;
    return element.closest('.section, .nova-block, .form-group') || element;
}

function setCapabilityDisabled(control, disabled) {
    if (!control || control.id === 'base_model') return;
    if (disabled) {
        if (!control.dataset.capabilityPreviousDisabled) {
            control.dataset.capabilityPreviousDisabled = control.disabled ? '1' : '0';
        }
        control.disabled = true;
        return;
    }
    if (control.dataset.capabilityPreviousDisabled === '0') control.disabled = false;
    delete control.dataset.capabilityPreviousDisabled;
}

function setSectionVisible(id, visible) {
    const section = byId(id);
    if (!section) return;
    section.classList.toggle('hidden', !visible);
    section.querySelectorAll('input, select, textarea, button').forEach(control => {
        setCapabilityDisabled(control, !visible);
    });
}

function setControlVisible(id, visible) {
    const element = byId(id);
    if (!element) return;
    const container = sectionForElement(id);
    if (container) container.classList.toggle('hidden', !visible);
    setCapabilityDisabled(element, !visible);
}

function capabilityEnabled(caps, key) {
    return caps?.[key] === true;
}

function capabilityAllowed(caps, key) {
    // Fail open while a profile is loading or when talking to an older/partial
    // catalog. Mature Studio controls should disappear only when the model
    // contract explicitly says that capability is unsupported.
    return caps?.[key] !== false;
}

function capabilityLabel(profile) {
    if (!profile) return '';
    const caps = profile.capabilities || {};
    const workflows = [];
    if (capabilityEnabled(caps, 'text2img')) workflows.push('Text → Image');
    if (capabilityEnabled(caps, 'img2img')) workflows.push('Image → Image');
    if (capabilityEnabled(caps, 'inpaint')) workflows.push('Inpaint');
    if (capabilityEnabled(caps, 'outpaint')) workflows.push('Outpaint');
    const multiple = Number(profile.constraints?.dimension_multiple || 0);
    if (multiple > 1) workflows.push(`${multiple}px grid`);
    return workflows.join(' • ');
}

function ensureProfileSummary() {
    const select = byId('base_model');
    if (!select) return null;
    let summary = byId('model-profile-summary');
    if (!summary) {
        summary = document.createElement('div');
        summary.id = 'model-profile-summary';
        summary.className = 'form-hint';
        summary.style.marginTop = '8px';
        select.insertAdjacentElement('afterend', summary);
    }
    return summary;
}

function neutralizeProductCopy() {
    document.title = 'WebbDuck - Local AI Image Studio';
    const brandSubtitle = document.querySelector('.nova-brand-copy span');
    if (brandSubtitle) brandSubtitle.textContent = 'Local Model-Driven Image Studio';
    const workspaceCopy = document.querySelector('.nova-workspace-header p');
    if (workspaceCopy) workspaceCopy.textContent = 'Checkpoint-driven local image generation with controls that adapt to the selected model.';
}

function snapDimensionInput(input, multiple) {
    if (!input) return false;
    const numeric = Number(input.value);
    if (!Number.isFinite(numeric) || numeric <= 0) return false;

    const min = Number(input.min || multiple);
    const max = Number(input.max || Number.POSITIVE_INFINITY);
    let snapped = Math.max(multiple, Math.round(numeric / multiple) * multiple);
    if (Number.isFinite(min)) snapped = Math.max(min, snapped);
    if (Number.isFinite(max)) snapped = Math.min(max, snapped);
    // Clamp can break the grid when min/max themselves are not multiples.
    snapped = Math.max(multiple, Math.round(snapped / multiple) * multiple);

    if (snapped === numeric) return true;
    input.value = String(snapped);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
}

function applyDimensionConstraints(profile) {
    const multiple = Math.max(1, Number(profile?.constraints?.dimension_multiple || 8));
    for (const id of ['width', 'height']) {
        const input = byId(id);
        if (!input) continue;
        input.step = String(multiple);
        // Resolution is user intent. Model switches may require a different
        // pixel grid, but they should not throw away the current aspect ratio.
        snapDimensionInput(input, multiple);
    }
}

function updateValueOutput(id, value) {
    const output = byId(id);
    if (output) output.textContent = String(value);
}

function applyDefaults(profile, { force = false } = {}) {
    const defaults = profile?.defaults || {};
    const mappings = [
        ['width', 'width', null],
        ['height', 'height', null],
        ['steps', 'steps', 'steps-value'],
        ['cfg', 'cfg', 'cfg-value'],
    ];
    for (const [field, id, outputId] of mappings) {
        if (defaults[field] === undefined || defaults[field] === null) continue;
        const input = byId(id);
        if (!input) continue;

        // Width/height are sticky once the user/form already has a valid value.
        // The selected model contributes only its grid constraint at that point.
        // Steps/CFG remain model defaults because those values are part of the
        // checkpoint's intended inference contract (for example Raw vs Turbo).
        if ((field === 'width' || field === 'height') && Number(input.value) > 0) {
            input.dataset.modelDefaultApplied = profile.name;
            continue;
        }

        if (!force && input.dataset.modelDefaultApplied === profile.name) continue;
        if (id === 'cfg') {
            const numeric = Number(defaults[field]);
            if (Number.isFinite(numeric) && numeric < Number(input.min || 0)) {
                input.min = String(Math.min(0, numeric));
            }
        }
        input.value = String(defaults[field]);
        input.dataset.modelDefaultApplied = profile.name;
        if (outputId) updateValueOutput(outputId, defaults[field]);
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
    }
}

function clearUnsupportedSourceState(caps) {
    if (caps?.img2img === false) {
        window._uploadedImage = null;
        window._uploadedImageDims = null;
        const input = byId('input-image');
        if (input) input.value = '';
    }
    if (caps?.inpaint === false) {
        window._maskBlob = null;
        window._previewMaskCanvas = null;
        window._maskDrawState = null;
        window._previewEditMode = null;
    }
    if (caps?.outpaint === false) {
        window._smartExtendPlacement = null;
        const smart = byId('smart-extend-enabled');
        if (smart) smart.checked = false;
    }
}

function applyOperationControls(profile) {
    const caps = profile?.capabilities || {};
    const supportsSource = capabilityAllowed(caps, 'img2img')
        || capabilityAllowed(caps, 'inpaint')
        || capabilityAllowed(caps, 'outpaint');
    const drop = byId('upload-drop');
    const sourceSection = drop?.closest('.section, .nova-block');
    if (sourceSection) sourceSection.classList.toggle('hidden', !supportsSource);
    setCapabilityDisabled(byId('input-image'), !supportsSource);

    setHidden(byId('inpaint-options'), !capabilityAllowed(caps, 'inpaint'));
    setHidden(byId('smart-extend-group'), !capabilityAllowed(caps, 'outpaint'));
    for (const id of ['preview-inpaint', 'lightbox-inpaint']) {
        const button = byId(id);
        if (button) {
            const available = capabilityAllowed(caps, 'inpaint');
            button.disabled = !available;
            button.classList.toggle('hidden', !available);
        }
    }
    clearUnsupportedSourceState(caps);
}

function applyFeatureSections(profile) {
    const caps = profile?.capabilities || {};
    setSectionVisible('section-negative', capabilityAllowed(caps, 'negative_prompt'));
    setSectionVisible('section-refiner', capabilityAllowed(caps, 'second_pass'));
    setSectionVisible('section-lora', capabilityAllowed(caps, 'lora'));
    setSectionVisible('section-embeddings', capabilityAllowed(caps, 'embeddings'));
    setSectionVisible('section-ip-adapter', capabilityAllowed(caps, 'identity_adapter'));
    setControlVisible('prompt_2', capabilityAllowed(caps, 'prompt_2'));
    setControlVisible('clip_skip', capabilityAllowed(caps, 'clip_skip'));
}

function restoreLegacyStudioControls() {
    // A transient catalog miss must never collapse the mature Studio down to
    // Parameters. Leave the proven controls available until a concrete profile
    // explicitly disables them.
    for (const id of ['section-negative', 'section-refiner', 'section-lora', 'section-embeddings', 'section-ip-adapter']) {
        setSectionVisible(id, true);
    }
    setControlVisible('prompt_2', true);
    setControlVisible('clip_skip', true);
    applyOperationControls({ capabilities: {} });
}

function updateModelOptions() {
    const select = byId('base_model');
    if (!select) return;
    for (const option of Array.from(select.options)) {
        const profile = findProfile(option.value);
        if (!profile) continue;
        option.disabled = profile.supported === false;
        if (profile.supported === false && !option.textContent.includes('runtime unavailable')) {
            option.textContent = `${profile.name} — runtime unavailable`;
        }
        option.title = capabilityLabel(profile);
    }
}

function requestedOperation() {
    if (!activeProfile) return 'text2img';
    if (byId('smart-extend-enabled')?.checked && window._uploadedImage) return 'outpaint';
    if (window._maskBlob && window._uploadedImage) return 'inpaint';
    if (window._uploadedImage) return 'img2img';
    return 'text2img';
}

function validateCurrentOperation(event) {
    const profile = activeProfile;
    if (!profile) return;
    if (profile.supported === false) {
        event.preventDefault();
        event.stopImmediatePropagation();
        toast(`Runtime unavailable for ${profile.name}.`, 'error');
        return;
    }
    const op = requestedOperation();
    const caps = profile.capabilities || {};
    const supported = {
        text2img: capabilityAllowed(caps, 'text2img'),
        img2img: capabilityAllowed(caps, 'img2img'),
        inpaint: capabilityAllowed(caps, 'inpaint'),
        outpaint: capabilityAllowed(caps, 'outpaint'),
    }[op];
    if (!supported) {
        event.preventDefault();
        event.stopImmediatePropagation();
        toast(`${profile.name} does not support the selected ${op} workflow.`, 'warning');
    }
}

function updateGenerateButtons(profile) {
    const unavailable = profile?.supported === false;
    for (const id of ['btn-generate', 'btn-test']) {
        const button = byId(id);
        if (!button) continue;
        button.disabled = unavailable;
        button.title = unavailable ? 'Selected model runtime is unavailable' : '';
    }
}

function applyProfile(profile, options = {}) {
    if (!profile) return;
    activeProfile = profile;
    updateModelOptions();
    applyFeatureSections(profile);
    applyOperationControls(profile);
    applyDimensionConstraints(profile);
    applyDefaults(profile, options);
    updateGenerateButtons(profile);
    const summary = ensureProfileSummary();
    if (summary) {
        summary.textContent = profile.supported === false
            ? 'Model discovered, but its runtime is not available in this build.'
            : capabilityLabel(profile);
    }
}

function cacheProfile(profile) {
    if (!profile?.name) return;
    const existing = catalog.findIndex(item => item?.name === profile.name);
    if (existing >= 0) catalog[existing] = profile;
    else catalog.push(profile);
}

async function refreshSelectedProfile({ forceDefaults = false } = {}) {
    const name = byId('base_model')?.value;
    if (!name) {
        activeProfile = null;
        restoreLegacyStudioControls();
        updateGenerateButtons(null);
        return;
    }

    const cached = findProfile(name);
    if (cached) applyProfile(cached, { force: forceDefaults });
    else {
        activeProfile = null;
        restoreLegacyStudioControls();
        updateGenerateButtons(null);
    }

    const requestSerial = ++profileRequestSerial;
    try {
        const response = await fetch(`/model-catalog/${encodeURIComponent(name)}`, { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const profile = await response.json();
        if (requestSerial !== profileRequestSerial || byId('base_model')?.value !== name) return;
        if (!profile?.name) throw new Error('Model profile response was empty');
        cacheProfile(profile);
        applyProfile(profile, { force: forceDefaults });
    } catch (error) {
        if (requestSerial !== profileRequestSerial) return;
        console.warn(`Could not refresh model profile for ${name}:`, error);
        if (!cached) restoreLegacyStudioControls();
    }
}

function initialize() {
    if (initialized) return;
    initialized = true;
    neutralizeProductCopy();
    const select = byId('base_model');
    if (select) {
        select.addEventListener('change', () => {
            void refreshSelectedProfile({ forceDefaults: true });
        });
        const observer = new MutationObserver(() => {
            updateModelOptions();
            setTimeout(() => { void refreshSelectedProfile(); }, 0);
        });
        observer.observe(select, { childList: true, subtree: true });
    }
    for (const id of ['btn-generate', 'btn-test']) {
        byId(id)?.addEventListener('click', validateCurrentOperation, true);
    }
    void refreshSelectedProfile({ forceDefaults: true });
}

window.addEventListener(CATALOG_EVENT, event => {
    catalog = Array.isArray(event.detail) ? event.detail : [];
    updateModelOptions();
    setTimeout(() => { void refreshSelectedProfile({ forceDefaults: true }); }, 0);
});

window.addEventListener(PROFILE_EVENT, event => {
    const profile = event.detail;
    if (!profile?.name) return;
    cacheProfile(profile);
    if (byId('base_model')?.value === profile.name) applyProfile(profile);
});

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize, { once: true });
} else {
    initialize();
}
