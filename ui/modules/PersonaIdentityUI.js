// Generic Identity / Persona presentation layered over WebbDuck's existing
// reference-image and saved-preset manager. The underlying storage/API remains
// backward compatible with SDXL FaceID presets while FLUX.2 uses the same saved
// reference sets as native multi-reference conditioning.

const PROFILE_EVENT = 'webbduck:model-profile';

function byId(id) {
    return document.getElementById(id);
}

function selectedModelLooksFlux() {
    const value = String(byId('base_model')?.value || '').toLowerCase();
    return value.includes('flux');
}

function ensureNativeFluxOption() {
    const select = byId('ip-adapter-type');
    if (!select) return;
    if (Array.from(select.options).some(option => option.value === 'flux2_native')) return;
    const option = document.createElement('option');
    option.value = 'flux2_native';
    option.textContent = 'FLUX.2 Native References';
    select.insertBefore(option, select.firstChild);
}

function setText(selector, text) {
    const element = document.querySelector(selector);
    if (element) element.textContent = text;
}

function ensurePersonaHint() {
    const controls = byId('ip-adapter-controls');
    if (!controls || byId('identity-persona-hint')) return;
    const hint = document.createElement('div');
    hint.id = 'identity-persona-hint';
    hint.className = 'form-help';
    hint.style.margin = '0 0 10px';
    hint.textContent = 'Upload reference photos once, save them as a named persona, and reuse that identity in future generations.';
    controls.insertAdjacentElement('afterbegin', hint);
}

function setScaleControlsVisible(visible) {
    for (const sliderId of ['ip-adapter-scale', 'ip-adapter-lora-scale']) {
        const slider = byId(sliderId);
        if (!slider) continue;
        slider.classList.toggle('hidden', !visible);
        const row = slider.previousElementSibling;
        if (row?.classList?.contains('param-row')) {
            row.classList.toggle('hidden', !visible);
        }
    }
}

function syncProviderForSelectedModel() {
    const select = byId('ip-adapter-type');
    if (!select) return;
    ensureNativeFluxOption();

    const isFlux = selectedModelLooksFlux();
    if (isFlux) {
        select.value = 'flux2_native';
        setScaleControlsVisible(false);
        const hint = byId('identity-persona-hint');
        if (hint) {
            hint.textContent = 'FLUX.2 uses these persona photos directly as native multi-reference identity conditioning. Up to 5 references are supported.';
        }
    } else {
        if (select.value === 'flux2_native') select.value = 'faceid_sdxl';
        setScaleControlsVisible(true);
        const hint = byId('identity-persona-hint');
        if (hint) {
            hint.textContent = 'Upload reference photos once, save them as a named persona, and reuse that identity in future generations.';
        }
    }
    select.dispatchEvent(new Event('change', { bubbles: true }));
}

function relabelIdentityUI() {
    const section = byId('section-ip-adapter');
    if (!section) return false;

    setText('#section-ip-adapter .section-title', 'Identity / Persona');
    setText('label[for="ip-adapter-enabled"] span', 'Use reference identity');
    setText('label[for="ip-adapter-type"]', 'Identity Provider');

    const uploadHelp = byId('ip-adapter-refs-upload-btn')?.nextElementSibling;
    if (uploadHelp) uploadHelp.textContent = 'Add reference photos for this identity';

    const controls = byId('ip-adapter-controls');
    if (controls) {
        const labels = Array.from(controls.querySelectorAll('.form-label'));
        for (const label of labels) {
            if (label.textContent?.trim() === 'Face Model Presets') {
                label.textContent = 'Saved Personas';
            }
        }
    }

    const presetSelect = byId('ip-adapter-preset-select');
    if (presetSelect?.options?.[0]) {
        presetSelect.options[0].textContent = '-- Load Persona --';
    }
    const nameInput = byId('ip-adapter-preset-name-input');
    if (nameInput) nameInput.placeholder = 'Persona name...';

    const saveBtn = byId('ip-adapter-preset-save-btn');
    if (saveBtn) saveBtn.title = 'Save current references as a persona';
    const deleteBtn = byId('ip-adapter-preset-delete-btn');
    if (deleteBtn) deleteBtn.title = 'Delete selected persona';

    ensureNativeFluxOption();
    ensurePersonaHint();
    syncProviderForSelectedModel();
    return true;
}

function initialize() {
    if (!relabelIdentityUI()) {
        const observer = new MutationObserver(() => {
            if (relabelIdentityUI()) observer.disconnect();
        });
        observer.observe(document.documentElement, { childList: true, subtree: true });
    }

    byId('base_model')?.addEventListener('change', () => {
        setTimeout(syncProviderForSelectedModel, 0);
    });

    byId('ip-adapter-preset-select')?.addEventListener('change', () => {
        // Existing preset loading restores the provider saved with the preset.
        // Normalize it back to the provider required by the currently selected
        // architecture so a persona can be reused across SDXL and FLUX.2.
        setTimeout(syncProviderForSelectedModel, 0);
    });

    window.addEventListener(PROFILE_EVENT, () => {
        setTimeout(syncProviderForSelectedModel, 0);
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize, { once: true });
} else {
    initialize();
}
