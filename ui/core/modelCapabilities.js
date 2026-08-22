import { byId, toast } from './utils.js';

const PROFILE_EVENT = 'webbduck:model-profile';
const CATALOG_EVENT = 'webbduck:model-catalog';

let catalog = [];
let activeProfile = null;
let initialized = false;

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

function setSectionVisible(id, visible) {
    const section = byId(id);
    if (!section) return;
    section.classList.toggle('hidden', !visible);
    section.querySelectorAll('input, select, textarea, button').forEach(control => {
        if (control.id === 'base_model') return;
        control.disabled = !visible;
    });
}

function setControlVisible(id, visible) {
    const element = byId(id);
    if (!element) return;
    const container = sectionForElement(id);
    if (container) container.classList.toggle('hidden', !visible);
    element.disabled = !visible;
}

function capabilityLabel(profile) {
    if (!profile) return '';
    const caps = profile.capabilities || {};
    const workflows = [];
    if (caps.text2img) workflows.push('Text → Image');
    if (caps.img2img) workflows.push('Image → Image');
    if (caps.inpaint) workflows.push('Inpaint');
    if (caps.outpaint) workflows.push('Outpaint');
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

function applyDimensionConstraints(profile) {
    const multiple = Math.max(1, Number(profile?.constraints?.dimension_multiple || 8));
    for (const id of ['width', 'height']) {
        const input = byId(id);
        if (!input) continue;
        input.step = String(multiple);
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
    if (!caps.img2img) {
        window._uploadedImage = null;
        window._uploadedImageDims = null;
        const input = byId('input-image');
        if (input) input.value = '';
    }
    if (!caps.inpaint) {
        window._maskBlob = null;
        window._previewMaskCanvas = null;
        window._maskDrawState = null;
        window._previewEditMode = null;
    }
    if (!caps.outpaint) {
        window._smartExtendPlacement = null;
        const smart = byId('smart-extend-enabled');
        if (smart) smart.checked = false;
    }
}

function applyOperationControls(profile) {
    const caps = profile?.capabilities || {};
    const supportsSource = !!(caps.img2img || caps.inpaint || caps.outpaint);
    const drop = byId('upload-drop');
    const sourceSection = drop?.closest('.section, .nova-block');
    if (sourceSection) sourceSection.classList.toggle('hidden', !supportsSource);
    if (byId('input-image')) byId('input-image').disabled = !supportsSource;

    setHidden(byId('inpaint-options'), !caps.inpaint);
    setHidden(byId('smart-extend-group'), !caps.outpaint);
    for (const id of ['preview-inpaint', 'lightbox-inpaint']) {
        const button = byId(id);
        if (button) {
            button.disabled = !caps.inpaint;
            button.classList.toggle('hidden', !caps.inpaint);
        }
    }
    clearUnsupportedSourceState(caps);
}

function applyFeatureSections(profile) {
    const caps = profile?.capabilities || {};
    setSectionVisible('section-negative', !!caps.negative_prompt);
    setSectionVisible('section-refiner', !!caps.second_pass);
    setSectionVisible('section-lora', !!caps.lora);
    setSectionVisible('section-embeddings', !!caps.embeddings);
    setSectionVisible('section-ip-adapter', !!caps.identity_adapter);
    setControlVisible('prompt_2', !!caps.prompt_2);
    setControlVisible('clip_skip', !!caps.clip_skip);
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
        text2img: !!caps.text2img,
        img2img: !!caps.img2img,
        inpaint: !!caps.inpaint,
        outpaint: !!caps.outpaint,
    }[op];
    if (!supported) {
        event.preventDefault();
        event.stopImmediatePropagation();
        toast(`${profile.name} does not support the selected ${op} workflow.`, 'warning');
    }
}

function updateGenerateButtons(profile) {
    const unavailable = !profile || profile.supported === false;
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

function applySelectedProfile({ forceDefaults = false } = {}) {
    const name = byId('base_model')?.value;
    const profile = findProfile(name);
    if (profile) applyProfile(profile, { force: forceDefaults });
}

function initialize() {
    if (initialized) return;
    initialized = true;
    const select = byId('base_model');
    if (select) {
        select.addEventListener('change', () => applySelectedProfile({ forceDefaults: true }));
        const observer = new MutationObserver(() => {
            updateModelOptions();
            setTimeout(() => applySelectedProfile(), 0);
        });
        observer.observe(select, { childList: true, subtree: true });
    }
    for (const id of ['btn-generate', 'btn-test']) {
        byId(id)?.addEventListener('click', validateCurrentOperation, true);
    }
    applySelectedProfile({ forceDefaults: true });
}

window.addEventListener(CATALOG_EVENT, event => {
    catalog = Array.isArray(event.detail) ? event.detail : [];
    updateModelOptions();
    setTimeout(() => applySelectedProfile({ forceDefaults: true }), 0);
});

window.addEventListener(PROFILE_EVENT, event => {
    const profile = event.detail;
    if (!profile?.name) return;
    const existing = catalog.findIndex(item => item?.name === profile.name);
    if (existing >= 0) catalog[existing] = profile;
    else catalog.push(profile);
    if (byId('base_model')?.value === profile.name) applyProfile(profile);
});

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize, { once: true });
} else {
    initialize();
}
