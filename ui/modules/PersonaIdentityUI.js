// Persona / identity UI ownership layer.
//
// The selected model architecture owns its identity provider. This module keeps
// that provider as hidden runtime state, presents only controls that are useful
// for the selected provider, and owns reference/preset interaction so selecting
// a thumbnail does not refetch/rebuild the entire reference gallery.

const PROFILE_EVENT = 'webbduck:model-profile';

const PROVIDERS = Object.freeze({
    sdxl: 'faceid_sdxl',
    krea2: 'krea2_identity_edit',
    flux2: 'flux2_native',
});

let serverRefs = [];
let presets = {};
let renderingGrid = false;
let initialized = false;

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

function isKrea() {
    return selectedModelFamily() === 'krea2';
}

function maxReferenceCount() {
    if (isKrea()) return 1;
    if (selectedModelFamily() === 'flux2') return 5;
    return Number.POSITIVE_INFINITY;
}

function uniqueRefs(values) {
    const seen = new Set();
    const refs = [];
    for (const value of Array.isArray(values) ? values : []) {
        const url = String(value || '').trim();
        if (!url || seen.has(url)) continue;
        seen.add(url);
        refs.push(url);
    }
    return refs;
}

function applyReferenceLimit(values) {
    let refs = uniqueRefs(values);
    const max = maxReferenceCount();
    if (Number.isFinite(max) && refs.length > max) {
        // Krea is single-anchor. Keeping the most recently selected/reference
        // at the end matches the existing "last selected becomes anchor" UX.
        refs = refs.slice(-max);
    }
    return refs;
}

function readSelectedRefs() {
    try {
        const parsed = JSON.parse(byId('ip-adapter-refs-json')?.value || '[]');
        return applyReferenceLimit(parsed);
    } catch (_) {
        return [];
    }
}

function writeSelectedRefs(values, { render = false } = {}) {
    const refs = applyReferenceLimit(values);
    const input = byId('ip-adapter-refs-json');
    if (input) {
        input.value = JSON.stringify(refs);
        input.dispatchEvent(new Event('input', { bubbles: true }));
    }
    if (render) renderRefGrid();
    else updateRefSelectionVisuals();
    return refs;
}

function installHiddenProviderField() {
    const current = byId('ip-adapter-type');
    if (!current) return null;
    if (current.tagName === 'INPUT' && current.type === 'hidden') return current;

    const hidden = document.createElement('input');
    hidden.type = 'hidden';
    hidden.id = 'ip-adapter-type';
    hidden.value = requiredProvider();

    const row = current.closest('.param-row');
    if (row) row.replaceWith(hidden);
    else current.replaceWith(hidden);
    return hidden;
}

function setPairVisible(sliderId, visible) {
    const slider = byId(sliderId);
    if (!slider) return;
    slider.classList.toggle('hidden', !visible);
    const row = slider.previousElementSibling;
    if (row?.classList?.contains('param-row')) row.classList.toggle('hidden', !visible);
}

function setProviderControls() {
    const providerField = installHiddenProviderField();
    const provider = requiredProvider();
    if (providerField) providerField.value = provider;

    const krea = provider === PROVIDERS.krea2;
    const flux = provider === PROVIDERS.flux2;

    for (const el of document.querySelectorAll('.krea2-only')) {
        el.classList.toggle('hidden', !krea);
    }
    for (const el of document.querySelectorAll('.flux2-only')) {
        el.classList.toggle('hidden', !flux);
    }

    // Krea's identity recipe is tuned automatically; exposing these sliders was
    // misleading and also allowed stale SDXL values to leak into a Krea run.
    // FLUX.2 already has provider-specific reference controls, so keep the two
    // legacy FaceID sliders visible only for SDXL.
    const showLegacyScales = provider === PROVIDERS.sdxl;
    setPairVisible('ip-adapter-scale', showLegacyScales);
    setPairVisible('ip-adapter-lora-scale', showLegacyScales);

    if (krea) {
        // app_main still serializes these hidden legacy fields. Pin them to the
        // known-good Krea defaults instead of letting stale browser state vary
        // the recipe invisibly: slider 0.30 -> ref_boost 4.0, LoRA scale 1.0.
        const strength = byId('ip-adapter-scale');
        const lora = byId('ip-adapter-lora-scale');
        if (strength) strength.value = '0.30';
        if (lora) lora.value = '1.00';
        const grounding = byId('ip-adapter-grounding-px');
        if (grounding && !grounding.value) grounding.value = '768';
    }

    const uploadInput = byId('ip-adapter-refs-upload-input');
    if (uploadInput) uploadInput.multiple = !krea;

    const currentRefs = readSelectedRefs();
    writeSelectedRefs(currentRefs);

    const hint = byId('identity-persona-hint');
    if (hint) {
        if (krea) {
            hint.textContent = 'Krea 2 uses one Krea Identity reference. Selecting another image replaces the current anchor automatically.';
        } else if (flux) {
            hint.textContent = 'FLUX.2 uses Native References automatically and supports up to 5 selected references.';
        } else {
            hint.textContent = 'SDXL uses the SDXL FaceID adapter automatically. Select the reference photos to use for this persona.';
        }
    }
}

function relabelIdentityUI() {
    const section = byId('section-ip-adapter');
    if (!section) return false;

    const title = section.querySelector('.section-title');
    if (title) title.textContent = 'Identity / Persona';
    const enabledText = section.querySelector('label[for="ip-adapter-enabled"] span');
    if (enabledText) enabledText.textContent = 'Use reference identity';

    const controls = byId('ip-adapter-controls');
    if (controls && !byId('identity-persona-hint')) {
        const hint = document.createElement('div');
        hint.id = 'identity-persona-hint';
        hint.className = 'form-help';
        hint.style.margin = '0 0 10px';
        controls.insertAdjacentElement('afterbegin', hint);
    }

    const labels = controls ? Array.from(controls.querySelectorAll('.form-label')) : [];
    for (const label of labels) {
        if (label.textContent?.trim() === 'Face Model Presets') label.textContent = 'Saved Personas';
        if (label.textContent?.trim() === 'Reference Images') label.textContent = 'Persona References';
    }
    const nameInput = byId('ip-adapter-preset-name-input');
    if (nameInput) nameInput.placeholder = 'Persona name...';
    const uploadHelp = byId('ip-adapter-refs-upload-btn')?.nextElementSibling;
    if (uploadHelp) uploadHelp.textContent = 'Add reference images for this identity';

    setProviderControls();
    return true;
}

async function refreshServerRefs({ rerender = true } = {}) {
    try {
        const response = await fetch('/ip-adapter/refs', { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        serverRefs = Array.isArray(data.images) ? data.images : [];
        if (rerender) renderRefGrid();
    } catch (error) {
        console.warn('Could not load identity references:', error);
        const grid = byId('ip-adapter-refs-grid');
        if (grid && !grid.children.length) {
            grid.textContent = 'Could not load reference images.';
        }
    }
}

function makeRemoveButton() {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'remove-ref';
    button.textContent = 'x';
    button.title = 'Remove reference';
    return button;
}

function appendRefThumb(fragment, url, name, selected, { external = false } = {}) {
    const thumb = document.createElement('div');
    thumb.className = `ip-adapter-ref-thumb${selected ? ' selected' : ''}`;
    thumb.dataset.personaOwned = '1';
    thumb.dataset.refUrl = url;
    thumb.dataset.external = external ? '1' : '0';
    thumb.title = name || url;

    const image = document.createElement('img');
    image.src = url;
    image.alt = name || 'Identity reference';
    image.loading = 'lazy';
    image.decoding = 'async';
    thumb.appendChild(image);

    const remove = makeRemoveButton();
    remove.classList.toggle('hidden', !selected);
    thumb.appendChild(remove);
    fragment.appendChild(thumb);
}

function renderRefGrid() {
    const grid = byId('ip-adapter-refs-grid');
    if (!grid) return;
    renderingGrid = true;
    try {
        const selected = new Set(readSelectedRefs());
        const known = new Set();
        const fragment = document.createDocumentFragment();

        for (const item of serverRefs) {
            const url = String(item?.url || '').trim();
            if (!url || known.has(url)) continue;
            known.add(url);
            appendRefThumb(fragment, url, String(item?.name || ''), selected.has(url));
        }

        for (const url of selected) {
            if (!known.has(url)) appendRefThumb(fragment, url, url.split('/').pop(), true, { external: true });
        }

        if (!fragment.childNodes.length) {
            const empty = document.createElement('span');
            empty.dataset.personaOwned = '1';
            empty.style.cssText = 'color:var(--text-muted);font-size:12px;align-self:center;';
            empty.textContent = 'No reference images yet. Upload one below.';
            fragment.appendChild(empty);
        }
        grid.replaceChildren(fragment);
    } finally {
        renderingGrid = false;
    }
}

function updateRefSelectionVisuals() {
    const grid = byId('ip-adapter-refs-grid');
    if (!grid) return;
    const selected = new Set(readSelectedRefs());
    for (const thumb of grid.querySelectorAll('.ip-adapter-ref-thumb[data-ref-url]')) {
        const active = selected.has(thumb.dataset.refUrl);
        thumb.classList.toggle('selected', active);
        thumb.querySelector('.remove-ref')?.classList.toggle('hidden', !active);
        if (thumb.dataset.external === '1' && !active) thumb.remove();
    }
}

async function deleteServerRef(url) {
    const item = serverRefs.find(row => row?.url === url);
    if (!item?.name) return;
    try {
        await fetch(`/ip-adapter/refs/${encodeURIComponent(item.name)}`, { method: 'DELETE' });
        await refreshServerRefs();
    } catch (error) {
        console.warn('Reference delete failed:', error);
    }
}

function handleGridClick(event) {
    const grid = byId('ip-adapter-refs-grid');
    const thumb = event.target?.closest?.('.ip-adapter-ref-thumb[data-ref-url]');
    if (!grid || !thumb || !grid.contains(thumb)) return;

    // Stop the legacy app_main per-thumbnail listener. That listener called
    // renderGrid(), which cleared the grid and refetched every thumbnail on
    // every click and was the source of the visible reload/glitching.
    event.preventDefault();
    event.stopImmediatePropagation();

    const url = thumb.dataset.refUrl;
    let refs = readSelectedRefs();
    const selected = refs.includes(url);

    if (event.target?.closest?.('.remove-ref')) {
        refs = refs.filter(value => value !== url);
        writeSelectedRefs(refs);
        if (thumb.dataset.external !== '1') void deleteServerRef(url);
        return;
    }

    if (selected) {
        refs = refs.filter(value => value !== url);
    } else if (isKrea()) {
        refs = [url];
    } else {
        refs.push(url);
    }
    writeSelectedRefs(refs);
}

function stopLegacyEvent(event) {
    event.preventDefault();
    event.stopImmediatePropagation();
}

async function handleUploads(event) {
    stopLegacyEvent(event);
    const input = byId('ip-adapter-refs-upload-input');
    let files = Array.from(input?.files || []);
    if (!files.length) return;
    if (isKrea()) files = files.slice(-1);

    let refs = readSelectedRefs();
    let uploaded = 0;
    for (const file of files) {
        const form = new FormData();
        form.append('file', file);
        try {
            const response = await fetch('/ip-adapter/refs/upload', { method: 'POST', body: form });
            if (!response.ok) continue;
            const data = await response.json();
            if (!data?.url) continue;
            uploaded += 1;
            if (isKrea()) refs = [data.url];
            else if (!refs.includes(data.url)) refs.push(data.url);
        } catch (error) {
            console.error('Identity reference upload failed:', error);
        }
    }
    if (input) input.value = '';
    writeSelectedRefs(refs);
    await refreshServerRefs();
    if (uploaded > 0) {
        window.dispatchEvent(new CustomEvent('webbduck:persona-refs-updated'));
    }
}

async function loadPresets({ preserveSelection = true } = {}) {
    const select = byId('ip-adapter-preset-select');
    if (!select) return;
    const previous = preserveSelection ? select.value : '';
    try {
        const response = await fetch('/ip-adapter/presets', { cache: 'no-store' });
        const data = await response.json();
        presets = data.presets || {};
        const fragment = document.createDocumentFragment();
        const first = document.createElement('option');
        first.value = '';
        first.textContent = '-- Load Persona --';
        fragment.appendChild(first);
        for (const name of Object.keys(presets).sort()) {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            fragment.appendChild(option);
        }
        select.replaceChildren(fragment);
        if (previous && presets[previous]) select.value = previous;
    } catch (error) {
        console.warn('Could not load personas:', error);
    }
}

function applyPreset(event) {
    stopLegacyEvent(event);
    const select = byId('ip-adapter-preset-select');
    const preset = presets[select?.value];
    if (!preset) return;

    writeSelectedRefs(Array.isArray(preset.refs) ? preset.refs : [], { render: true });
    setProviderControls();

    const provider = requiredProvider();
    if (provider === PROVIDERS.krea2) {
        if (preset.grounding_px != null && byId('ip-adapter-grounding-px')) {
            byId('ip-adapter-grounding-px').value = String(preset.grounding_px);
            if (byId('ip-adapter-grounding-px-value')) byId('ip-adapter-grounding-px-value').textContent = String(preset.grounding_px);
        }
        if (preset.lora_rank && ['r64', 'r128', 'full'].includes(String(preset.lora_rank))) {
            byId('ip-adapter-lora-rank').value = String(preset.lora_rank);
        } else if (byId('ip-adapter-lora-rank')) {
            byId('ip-adapter-lora-rank').value = 'auto';
        }
        return;
    }

    if (preset.adapter_scale != null && byId('ip-adapter-scale')) {
        byId('ip-adapter-scale').value = String(preset.adapter_scale);
        if (byId('ip-adapter-scale-value')) byId('ip-adapter-scale-value').textContent = Number(preset.adapter_scale).toFixed(2);
    }
    if (preset.lora_scale != null && byId('ip-adapter-lora-scale')) {
        byId('ip-adapter-lora-scale').value = String(preset.lora_scale);
        if (byId('ip-adapter-lora-scale-value')) byId('ip-adapter-lora-scale-value').textContent = Number(preset.lora_scale).toFixed(2);
    }
    if (provider === PROVIDERS.flux2) {
        if (preset.face_crop && byId('ip-adapter-face-crop')) byId('ip-adapter-face-crop').value = preset.face_crop;
        if (byId('ip-adapter-anchor-dup')) byId('ip-adapter-anchor-dup').checked = Boolean(preset.flux2_anchor_dup);
        if (byId('ip-adapter-face-focus')) byId('ip-adapter-face-focus').checked = Boolean(preset.face_focus);
    }
}

function showSaveRow(event) {
    stopLegacyEvent(event);
    const row = byId('ip-adapter-preset-name-row');
    const input = byId('ip-adapter-preset-name-input');
    row?.classList.toggle('hidden');
    if (input) {
        input.value = '';
        input.focus();
    }
}

async function savePreset(event) {
    stopLegacyEvent(event);
    const name = String(byId('ip-adapter-preset-name-input')?.value || '').trim();
    if (!name) return;

    const provider = requiredProvider();
    const payload = {
        name,
        type: provider,
        refs: readSelectedRefs(),
    };

    if (provider === PROVIDERS.krea2) {
        payload.ref_boost = 4.0;
        payload.lora_scale = 1.0;
        payload.grounding_px = parseInt(byId('ip-adapter-grounding-px')?.value || '768', 10);
        payload.fit_mode = 'fit';
        const rank = byId('ip-adapter-lora-rank')?.value;
        if (rank && rank !== 'auto') payload.lora_rank = rank;
    } else {
        payload.adapter_scale = parseFloat(byId('ip-adapter-scale')?.value || '1.0');
        payload.lora_scale = parseFloat(byId('ip-adapter-lora-scale')?.value || '0.60');
        if (provider === PROVIDERS.flux2) {
            payload.face_crop = byId('ip-adapter-face-crop')?.value || 'auto';
            payload.flux2_anchor_dup = Boolean(byId('ip-adapter-anchor-dup')?.checked);
            payload.face_focus = Boolean(byId('ip-adapter-face-focus')?.checked);
        }
    }

    try {
        const response = await fetch('/ip-adapter/presets', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!response.ok) return;
        byId('ip-adapter-preset-name-row')?.classList.add('hidden');
        await loadPresets({ preserveSelection: false });
        if (byId('ip-adapter-preset-select')) byId('ip-adapter-preset-select').value = name;
    } catch (error) {
        console.error('Save persona failed:', error);
    }
}

async function deletePreset(event) {
    stopLegacyEvent(event);
    const select = byId('ip-adapter-preset-select');
    const name = String(select?.value || '').trim();
    if (!name) return;
    try {
        await fetch(`/ip-adapter/presets/${encodeURIComponent(name)}`, { method: 'DELETE' });
        await loadPresets({ preserveSelection: false });
    } catch (error) {
        console.error('Delete persona failed:', error);
    }
}

function handleControlsClick(event) {
    const target = event.target?.closest?.('button');
    if (!target) return;
    if (target.id === 'ip-adapter-refs-upload-btn') {
        stopLegacyEvent(event);
        byId('ip-adapter-refs-upload-input')?.click();
    } else if (target.id === 'ip-adapter-preset-save-btn') {
        showSaveRow(event);
    } else if (target.id === 'ip-adapter-preset-confirm-save') {
        void savePreset(event);
    } else if (target.id === 'ip-adapter-preset-delete-btn') {
        void deletePreset(event);
    }
}

function handleControlsChange(event) {
    if (event.target?.id === 'ip-adapter-refs-upload-input') {
        void handleUploads(event);
    } else if (event.target?.id === 'ip-adapter-preset-select') {
        applyPreset(event);
    }
}

function bindOwnedManager() {
    const controls = byId('ip-adapter-controls');
    const grid = byId('ip-adapter-refs-grid');
    if (!controls || !grid) return;

    controls.addEventListener('click', handleControlsClick, true);
    controls.addEventListener('change', handleControlsChange, true);
    grid.addEventListener('click', handleGridClick, true);

    // app_main has a legacy section-open observer that may rebuild this grid.
    // Reassert our cached, no-refetch rendering if that happens. We explicitly
    // ignore our own children so this observer cannot form a render loop.
    const gridObserver = new MutationObserver(() => {
        if (renderingGrid) return;
        const foreignChild = Array.from(grid.children).some(child => child.dataset?.personaOwned !== '1');
        if (foreignChild) queueMicrotask(renderRefGrid);
    });
    gridObserver.observe(grid, { childList: true });

    const section = byId('section-ip-adapter');
    if (section) {
        const sectionObserver = new MutationObserver(() => {
            if (!section.classList.contains('collapsed')) {
                setTimeout(() => renderRefGrid(), 0);
            }
        });
        sectionObserver.observe(section, { attributes: true, attributeFilter: ['class'] });
    }
}

function syncForModel() {
    setProviderControls();
    renderRefGrid();
}

function initialize() {
    if (initialized) return;
    initialized = true;
    if (!relabelIdentityUI()) return;
    bindOwnedManager();

    byId('base_model')?.addEventListener('change', () => setTimeout(syncForModel, 0));
    window.addEventListener(PROFILE_EVENT, () => setTimeout(syncForModel, 0));

    void Promise.all([
        refreshServerRefs({ rerender: false }),
        loadPresets(),
    ]).then(() => renderRefGrid());
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize, { once: true });
} else {
    initialize();
}
