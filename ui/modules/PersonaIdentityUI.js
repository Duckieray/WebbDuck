// Preserve the existing Persona UI implementation, then hard-lock identity
// provider ownership to the selected model family. Provider selection is an
// architecture/runtime detail, not a user choice.
import './PersonaIdentityUI_impl.js';

const PROFILE_EVENT = 'webbduck:model-profile';
let lastProvider = null;

const PROVIDERS = Object.freeze({
    sdxl: { value: 'faceid_sdxl', label: 'SDXL FaceID' },
    krea2: { value: 'krea2_identity_edit', label: 'Krea Identity' },
    flux2: { value: 'flux2_native', label: 'FLUX.2 Native References' },
});

function byId(id) {
    return document.getElementById(id);
}

function selectedModelFamily() {
    const value = String(byId('base_model')?.value || '').toLowerCase();
    if (value.includes('krea')) return 'krea2';
    if (value.includes('klein') || /flux[._\s-]*2(?:\b|[._\s-])/.test(value)) return 'flux2';
    return 'sdxl';
}

function requiredProvider() {
    return PROVIDERS[selectedModelFamily()] || PROVIDERS.sdxl;
}

function updateValue(id, value, digits = 2) {
    const input = byId(id);
    if (!input) return;
    input.value = String(value);
    const output = byId(`${id}-value`);
    if (output) output.textContent = Number(value).toFixed(digits);
    input.dispatchEvent(new Event('input', { bubbles: true }));
}

function setScaleControlsVisible(visible) {
    for (const sliderId of ['ip-adapter-scale', 'ip-adapter-lora-scale']) {
        const slider = byId(sliderId);
        if (!slider) continue;
        slider.classList.toggle('hidden', !visible);
        const row = slider.previousElementSibling;
        if (row?.classList?.contains('param-row')) row.classList.toggle('hidden', !visible);
    }
}

function setProviderSpecificControls(provider) {
    const isFlux = provider === 'flux2_native';
    const isKrea = provider === 'krea2_identity_edit';
    for (const el of document.querySelectorAll('.flux2-only')) {
        el.classList.toggle('hidden', !isFlux);
    }
    for (const el of document.querySelectorAll('.krea2-only')) {
        el.classList.toggle('hidden', !isKrea);
    }
    setScaleControlsVisible(!isFlux);
}

function lockProviderToSelectedModel({ initial = false } = {}) {
    const select = byId('ip-adapter-type');
    if (!select) return;

    const required = requiredProvider();
    const previous = select.value;

    // There is deliberately only one option. Saved Personas contain reusable
    // reference photos; their historical provider value must never become a
    // selectable/runtime value after switching architectures.
    const onlyOption = document.createElement('option');
    onlyOption.value = required.value;
    onlyOption.textContent = required.label;
    select.replaceChildren(onlyOption);
    select.value = required.value;
    select.disabled = true;
    select.setAttribute('aria-disabled', 'true');
    select.title = `${required.label} is selected automatically for this model.`;

    const label = document.querySelector('label[for="ip-adapter-type"]');
    if (label) label.textContent = 'Identity Provider (automatic)';

    const transitioned = required.value !== lastProvider;
    setProviderSpecificControls(required.value);

    if (required.value === 'krea2_identity_edit' && (initial || transitioned)) {
        // Upstream Krea2Edit v1.2 baseline: ref_boost ~= 4 and LoRA scale 1.0.
        // app_main maps the 0..1 identity slider as ref_boost = 1 + 10*slider,
        // so 0.30 is the correct UI default for a boost of 4.0.
        updateValue('ip-adapter-scale', 0.30);
        updateValue('ip-adapter-lora-scale', 1.00);
        const grounding = byId('ip-adapter-grounding-px');
        if (grounding) grounding.value = '768';
    }

    const hint = byId('identity-persona-hint');
    if (hint) {
        if (required.value === 'krea2_identity_edit') {
            hint.textContent = 'Krea 2 always uses Krea Identity. Persona references are routed through the Krea2Edit v1.2 identity path; SDXL FaceID and FLUX.2 reference adapters cannot be selected for this model.';
        } else if (required.value === 'flux2_native') {
            hint.textContent = 'FLUX.2 always uses Native References for personas. SDXL FaceID and Krea Identity cannot be selected for this model.';
        } else {
            hint.textContent = 'SDXL always uses the SDXL FaceID adapter for personas. Krea Identity and FLUX.2 Native References cannot be selected for this model.';
        }
    }

    lastProvider = required.value;

    // Notify the existing state/preset machinery when we corrected stale state.
    // Do not dispatch on every sync or we would create an event loop.
    if (previous !== required.value) {
        select.dispatchEvent(new Event('change', { bubbles: true }));
    }
}

function schedule() {
    setTimeout(() => lockProviderToSelectedModel(), 0);
}

function initialize() {
    setTimeout(() => lockProviderToSelectedModel({ initial: true }), 0);
    byId('base_model')?.addEventListener('change', schedule);
    byId('ip-adapter-preset-select')?.addEventListener('change', schedule);
    window.addEventListener(PROFILE_EVENT, schedule);

    // The legacy implementation may add provider options while rebuilding the
    // Persona controls. Prune them immediately whenever the select's children
    // are touched, so no incompatible provider is ever user-selectable.
    const select = byId('ip-adapter-type');
    if (select) {
        const observer = new MutationObserver(() => {
            const required = requiredProvider();
            const options = Array.from(select.options);
            if (options.length !== 1 || options[0]?.value !== required.value) schedule();
        });
        observer.observe(select, { childList: true });
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize, { once: true });
} else {
    initialize();
}
