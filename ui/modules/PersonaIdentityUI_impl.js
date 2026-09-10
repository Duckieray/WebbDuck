// Generic Identity / Persona presentation layered over WebbDuck's existing
// reference-image and saved-preset manager. The underlying storage/API remains
// backward compatible with SDXL FaceID presets while FLUX.2 uses the same saved
// reference sets as native multi-reference conditioning.

const PROFILE_EVENT = 'webbduck:model-profile';

function byId(id) {
    return document.getElementById(id);
}

function selectedModelLooksFlux2() {
    const value = String(byId('base_model')?.value || '').toLowerCase();
    return value.includes('klein') || /flux[._\s-]*2(?:\b|[._\s-])/.test(value);
}

function selectedModelLooksKrea() {
    const value = String(byId('base_model')?.value || '').toLowerCase();
    return value.includes('krea');
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

function ensureKreaOption() {
    const select = byId('ip-adapter-type');
    if (!select) return;
    if (Array.from(select.options).some(option => option.value === 'krea2_identity_edit')) return;
    const option = document.createElement('option');
    option.value = 'krea2_identity_edit';
    option.textContent = 'Krea Identity';
    select.appendChild(option);
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

function setFluxOnlyControlsVisible(visible) {
    for (const el of document.querySelectorAll('.flux2-only')) {
        el.classList.toggle('hidden', !visible);
    }
}

function setKreaOnlyControlsVisible(visible) {
    for (const el of document.querySelectorAll('.krea2-only')) {
        el.classList.toggle('hidden', !visible);
    }
}

function setScaleSliderDefaults(provider) {
    const applyDefaults = (slider, span, min, max, step, fallback) => {
        if (!slider) return;
        slider.min = min;
        slider.max = max;
        slider.step = step;
        let value = parseFloat(slider.value);
        if (!Number.isFinite(value) || value < parseFloat(min) || value > parseFloat(max)) {
            value = fallback;
            slider.value = String(fallback);
        }
        if (span) span.textContent = Number(value).toFixed(2);
    };
    const scale = byId('ip-adapter-scale');
    const lora = byId('ip-adapter-lora-scale');
    // A single 0..1 "Identity Strength" control for every provider. Under the
    // hood: SDXL/FLUX use adapter_scale = slider; Krea maps slider -> ref_boost
    // as 1 + 10*slider (0.1 == the balanced default of 2.0, 1.0 == max 10).
    applyDefaults(scale, byId('ip-adapter-scale-value'), '0', '1', '0.05', provider === 'krea2_identity_edit' ? 0.1 : 1.0);
    applyDefaults(lora, byId('ip-adapter-lora-scale-value'), '0', '1', '0.05', provider === 'krea2_identity_edit' ? 1.0 : 0.60);
    setText('label[for="ip-adapter-scale"]', 'Identity Strength');
    setText('label[for="ip-adapter-lora-scale"]', 'Identity LoRA Scale');
    if (provider === 'krea2_identity_edit') {
        const span = byId('ip-adapter-scale-value');
        if (span) span.textContent = Number(parseFloat(byId('ip-adapter-scale')?.value) || 0.1).toFixed(2);
    }
}

function syncProviderForSelectedModel() {
    const select = byId('ip-adapter-type');
    if (!select) return;
    ensureNativeFluxOption();
    ensureKreaOption();

    const isFlux2 = selectedModelLooksFlux2();
    const isKrea = selectedModelLooksKrea();
    const hint = byId('identity-persona-hint');

    if (isFlux2) {
        select.value = 'flux2_native';
        setScaleControlsVisible(false);
        setFluxOnlyControlsVisible(true);
        setKreaOnlyControlsVisible(false);
        setScaleSliderDefaults('faceid_sdxl');
        if (hint) {
            hint.textContent = 'FLUX.2 uses these persona photos directly as native multi-reference identity conditioning. Up to 5 references are supported. Auto face-crop, anchor boost, and face-focus framing tighten identity likeness.';
        }
    } else if (isKrea) {
        select.value = 'krea2_identity_edit';
        setScaleControlsVisible(true);
        setFluxOnlyControlsVisible(false);
        setKreaOnlyControlsVisible(true);
        setScaleSliderDefaults('krea2_identity_edit');
        if (hint) {
            hint.textContent = 'Krea Identity Edit uses a single anchor reference and the krea2-identity-edit LoRA for identity. Identity Strength is a 0..1 control mapped to the ref_boost likeness dial (0.1 = balanced default, lower = weaker likeness, higher = stronger but can lock the reference composition). Ref count is limited to 1, so the last image you select becomes the anchor.';
        }
    } else {
        if (select.value === 'flux2_native' || select.value === 'krea2_identity_edit') select.value = 'faceid_sdxl';
        setScaleControlsVisible(true);
        setFluxOnlyControlsVisible(false);
        setKreaOnlyControlsVisible(false);
        setScaleSliderDefaults('faceid_sdxl');
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
