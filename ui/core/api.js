/**
 * WebbDuck Core API Module
 * Centralized fetch wrappers for all API endpoints
 */

const API_BASE = '';
let modelCatalogCache = null;
let modelCatalogCapabilityAware = false;

/**
 * Generic fetch wrapper with error handling
 */
async function request(url, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${url}`, {
            ...options,
            headers: {
                ...options.headers,
            },
        });

        if (!response.ok) {
            const text = await response.text();
            let message = `HTTP ${response.status}`;
            try {
                const parsed = JSON.parse(text);
                if (parsed.error) message = parsed.error;
                else if (typeof parsed.detail === 'string') message = parsed.detail;
                else if (parsed.detail?.message) message = parsed.detail.message;
            } catch {
                if (text) message = text;
            }
            throw new Error(message);
        }

        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            return await response.json();
        }
        return await response.text();
    } catch (error) {
        console.error(`API Error: ${url}`, error);
        throw error;
    }
}

/**
 * GET request helper
 */
export async function get(url) {
    return request(url, { method: 'GET' });
}

/**
 * POST request with JSON body
 */
export async function post(url, data) {
    return request(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    });
}

/**
 * POST request with FormData
 */
export async function postForm(url, formData) {
    return request(url, {
        method: 'POST',
        body: formData,
    });
}

function normalizeCatalog(catalog) {
    if (!Array.isArray(catalog)) return [];
    return catalog
        .filter(item => item && item.name)
        .map(item => ({
            ...item,
            value: item.name,
            // Existing Studio select rendering already honors `label`, so
            // recognized future runtimes can be visible without architecture
            // names leaking into the UI.
            label: item.supported === false
                ? `${item.name} — runtime unavailable`
                : item.name,
        }));
}

function cachedProfile(modelName) {
    if (!modelName || !Array.isArray(modelCatalogCache)) return null;
    return modelCatalogCache.find(item => item?.name === modelName) || null;
}

export function getCachedModelProfile(modelName) {
    return cachedProfile(modelName);
}

// ═══════════════════════════════════════════════════════════════
// SPECIFIC API ENDPOINTS
// ═══════════════════════════════════════════════════════════════

/**
 * Fetch available models from the architecture-free catalog.
 * Falls back to the legacy endpoint for launch methods that have not mounted
 * the new router yet.
 */
export async function getModels() {
    try {
        const catalog = normalizeCatalog(await get('/model-catalog'));
        modelCatalogCache = catalog;
        modelCatalogCapabilityAware = true;
        return catalog;
    } catch (error) {
        console.warn('Model catalog unavailable; falling back to legacy /models.', error);
        const legacy = await get('/models');
        modelCatalogCache = Array.isArray(legacy) ? legacy : [];
        modelCatalogCapabilityAware = false;
        return legacy;
    }
}

/**
 * Fetch one model profile. The architecture/backend remain server-internal.
 */
export async function getModelProfile(modelName) {
    const cached = cachedProfile(modelName);
    if (modelCatalogCapabilityAware && cached?.capabilities) return cached;
    return get(`/model-catalog/${encodeURIComponent(modelName)}`);
}

/**
 * Fetch LoRAs for a specific model. If the selected model explicitly says it
 * cannot use LoRAs, avoid querying an architecture-shaped legacy endpoint.
 */
export async function getLoras(modelName) {
    const profile = cachedProfile(modelName);
    if (modelCatalogCapabilityAware && profile?.capabilities?.lora === false) return [];
    return get(`/models/${encodeURIComponent(modelName)}/loras`);
}

/**
 * Fetch embeddings for a specific model.
 */
export async function getEmbeddings(modelName) {
    const profile = cachedProfile(modelName);
    if (modelCatalogCapabilityAware && profile?.capabilities?.embeddings === false) return [];
    return get(`/models/${encodeURIComponent(modelName)}/embeddings`);
}

/**
 * Fetch second pass / refiner models. Once the catalog has loaded, derive this
 * list from model capabilities instead of assuming every checkpoint qualifies.
 */
export async function getSecondPassModels() {
    if (modelCatalogCapabilityAware && Array.isArray(modelCatalogCache)) {
        return modelCatalogCache
            .filter(item => item?.supported !== false && item?.capabilities?.second_pass === true)
            .map(item => item.name);
    }
    return get('/second_pass_models');
}

/**
 * Fetch available schedulers
 */
export async function getSchedulers() {
    return get('/schedulers');
}

/**
 * Fetch captioner availability
 */
export async function getCaptioners() {
    return get('/captioners');
}

/**
 * Fetch discovered web plugins.
 */
export async function getWebPlugins() {
    return get('/plugins/web');
}

/**
 * Fetch connected remote web plugins.
 */
export async function getRemoteWebPlugins() {
    return get('/plugins/web/remote');
}

/**
 * Connect a remote web plugin by base URL.
 */
export async function connectRemoteWebPlugin(baseUrl) {
    return post('/plugins/web/remote/connect', { base_url: baseUrl });
}

/**
 * Disconnect a remote web plugin by plugin ID.
 */
export async function disconnectRemoteWebPlugin(pluginId) {
    return request(`/plugins/web/remote/${encodeURIComponent(pluginId)}`, { method: 'DELETE' });
}

/**
 * Generate images (full batch)
 */
export async function generate(formData) {
    return postForm('/generate', formData);
}

/**
 * Test generation (single image)
 */
export async function testGenerate(formData) {
    return postForm('/test', formData);
}

/**
 * Upscale an image
 */
export async function upscale(formData) {
    return postForm('/upscale', formData);
}

export async function upscaleInput(formData) {
    return postForm('/upscale-input', formData);
}

/**
 * Caption an image
 */
export async function caption(formData) {
    return postForm('/caption', formData);
}

/**
 * Tokenize prompt for counting
 */
export async function tokenize(prompt, baseModel) {
    const profile = cachedProfile(baseModel);
    // Token diagnostics are currently implemented by the legacy SDXL runtime.
    // Do not pretend the same tokenizer contract exists for future backends.
    if (modelCatalogCapabilityAware && profile && profile.supported === false) {
        return { tokens: 0, available: false };
    }
    const formData = new FormData();
    formData.append('text', prompt);
    formData.append('base_model', baseModel);
    return postForm('/tokenize', formData);
}

/**
 * Fetch gallery data
 */
export async function getGallery(start = 0, limit = 50) {
    const url = `/gallery?start=${start}&limit=${limit}&_=${Date.now()}`;
    return get(url);
}

/**
 * Search gallery sessions globally via manifest index.
 */
export async function searchGallery(query, start = 0, limit = 2000) {
    const q = encodeURIComponent(query || '');
    return get(`/gallery/search?q=${q}&start=${start}&limit=${limit}&_=${Date.now()}`);
}

/**
 * Filter gallery sessions by tag (`hd`, `favorites`) globally via manifest index.
 */
export async function filterGallery(kind, start = 0, limit = 2000) {
    const k = encodeURIComponent(kind || '');
    return get(`/gallery/filter?kind=${k}&start=${start}&limit=${limit}&_=${Date.now()}`);
}

/**
 * Fetch queued/running job metadata.
 */
export async function getQueue() {
    return get('/queue');
}

/**
 * Cancel a queued job by ID.
 */
export async function cancelQueue(jobId) {
    const formData = new FormData();
    formData.append('job_id', jobId);
    return postForm('/queue/cancel', formData);
}

/**
 * Unload all loaded generation models from memory.
 */
export async function unloadAllModels() {
    const formData = new FormData();
    return postForm('/models/unload_all', formData);
}

/**
 * Shut down the WebbDuck server process.
 */
export async function shutdownApp() {
    const formData = new FormData();
    return postForm('/app/shutdown', formData);
}

/**
 * Favorite/unfavorite an image.
 */
export async function setFavorite(imagePath, favorite = true) {
    const formData = new FormData();
    formData.append('path', imagePath);
    formData.append('favorite', favorite ? 'true' : 'false');
    return postForm('/favorite', formData);
}

/**
 * Delete a single image
 */
export async function deleteImage(imagePath) {
    const formData = new FormData();
    formData.append('path', imagePath);
    return postForm('/delete_image', formData);
}

/**
 * Delete an entire run/session
 */
export async function deleteRun(runId) {
    const formData = new FormData();
    formData.append('path', runId);
    return postForm('/delete_run', formData);
}

/**
 * Delete multiple images in one request
 */
export async function deleteImages(paths = []) {
    return post('/delete_images', { paths: Array.isArray(paths) ? paths : [] });
}
