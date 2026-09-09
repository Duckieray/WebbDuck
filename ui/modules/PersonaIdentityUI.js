// Preserve the existing Persona UI implementation, then correct Krea-specific
// provider defaults after architecture/provider transitions.
import './PersonaIdentityUI_impl.js';

const PROFILE_EVENT = 'webbduck:model-profile';
let lastProvider = null;

function byId(id) {
    return document.getElementById(id);
}

function updateValue(id, value, digits = 2) {
    const input = byId(id);
    if (!input) return;
    input.value = String(value);
    const output = byId(`${id}-value`);
    if (output) output.textContent = Number(value).toFixed(digits);
    input.dispatchEvent(new Event('input', { bubbles: true }));
}

function applyKreaQualityDefaults({ initial = false } = {}) {
    const provider = byId('ip-adapter-type')?.value || '';
    const transitioned = provider !== lastProvider;

    if (provider === 'krea2_identity_edit' && (initial || transitioned)) {
        // Upstream Krea2Edit v1.2 baseline: ref_boost ~= 4 and LoRA scale 1.0.
        // app_main maps the 0..1 identity slider as ref_boost = 1 + 10*slider,
        // so 0.30 is the correct UI default for a boost of 4.0.
        updateValue('ip-adapter-scale', 0.30);
        updateValue('ip-adapter-lora-scale', 1.00);
        const grounding = byId('ip-adapter-grounding-px');
        if (grounding && !String(grounding.value || '').trim()) grounding.value = '768';

        const hint = byId('identity-persona-hint');
        if (hint) {
            hint.textContent = 'Krea Identity uses the Krea2Edit v1.2 recipe: one anchor reference, LoRA 1.0, balanced identity strength (ref_boost 4), grounded Qwen3-VL conditioning, and v1.2.4 FIT geometry.';
        }
    }

    lastProvider = provider;
}

function schedule() {
    setTimeout(() => applyKreaQualityDefaults(), 0);
}

function initialize() {
    setTimeout(() => applyKreaQualityDefaults({ initial: true }), 0);
    byId('base_model')?.addEventListener('change', schedule);
    byId('ip-adapter-type')?.addEventListener('change', schedule);
    window.addEventListener(PROFILE_EVENT, schedule);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize, { once: true });
} else {
    initialize();
}
