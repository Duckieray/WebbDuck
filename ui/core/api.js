import './modelCapabilities.js';

/**
 * WebbDuck Core API Module
 * Centralized fetch wrappers for all API endpoints
 */

const API_BASE = '';
let modelCatalogCache = null;

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

export async function get(url) {
    return request(url, { method: 'GET' });
}

export async function post(url, data) {
    return request(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    });
}

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
            label: item.supported === false
                ? `${item.name} — runtime unavailable`
                : item.name,
        }));
}

function cachedProfile(modelName) {
    if (!modelName || !Array.isArray(modelCatalogCache)) return null;
    return modelCatalogCache.find(item => item?.name === modelName) || null;
}

function publishCatalog(catalog) {
    if (typeof window === 'undefined') return;
    window.dispatchEvent(new CustomEvent('webbduck:model-catalog', {
        detail: Array.isArray(catalog) ? catalog : [],
    }));
}

function publishProfile(profile) {
    if (typeof window === 'undefined' || !profile) return;
    window.dispatchEvent(new CustomEvent('webbduck:model-profile', {
        detail: profile,
    }));
}

export function getCachedModelProfile(modelName) {
    return cachedProfile(modelName);
}

// ═══════════════════════════════════════════════════════════════
// MODEL-DRIVEN API
// ═══════════════════════════════════════════════════════════════

export async function getModels() {
    const catalog = normalizeCatalog(await get('/model-catalog'));
    modelCatalogCache = catalog;
    publishCatalog(catalog);
    return catalog;
}

export async function getModelProfile(modelName) {
    const cached = cachedProfile(modelName);
    if (cached?.capabilities) {
        publishProfile(cached);
        return cached;
    }
    const profile = await get(`/model-catalog/${encodeURIComponent(modelName)}`);
    publishProfile(profile);
    return profile;
}

export async function getLoras(modelName) {
    const profile = cachedProfile(modelName);
    if (profile?.capabilities?.lora === false) return [];
    return get(`/models/${encodeURIComponent(modelName)}/loras`);
}

export async function getEmbeddings(modelName) {
    const profile = cachedProfile(modelName);
    if (profile?.capabilities?.embeddings === false) return [];
    return get(`/models/${encodeURIComponent(modelName)}/embeddings`);
}

export async function getSecondPassModels() {
    if (!Array.isArray(modelCatalogCache)) {
        await getModels();
    }
    return (modelCatalogCache || [])
        .filter(item => item?.supported !== false && item?.capabilities?.second_pass === true)
        .map(item => item.name);
}

export async function getSchedulers() {
    return get('/schedulers');
}

export async function getCaptioners() {
    return get('/captioners');
}

export async function getWebPlugins() {
    return get('/plugins/web');
}

export async function getRemoteWebPlugins() {
    return get('/plugins/web/remote');
}

export async function connectRemoteWebPlugin(baseUrl) {
    return post('/plugins/web/remote/connect', { base_url: baseUrl });
}

export async function disconnectRemoteWebPlugin(pluginId) {
    return request(`/plugins/web/remote/${encodeURIComponent(pluginId)}`, { method: 'DELETE' });
}

export async function generate(formData) {
    return postForm('/generate', formData);
}

export async function testGenerate(formData) {
    return postForm('/test', formData);
}

export async function upscale(formData) {
    return postForm('/upscale', formData);
}

export async function upscaleInput(formData) {
    return postForm('/upscale-input', formData);
}

export async function caption(formData) {
    return postForm('/caption', formData);
}

export async function tokenize(prompt, baseModel) {
    const profile = cachedProfile(baseModel);
    if (profile?.capabilities?.tokenize !== true) {
        return { tokens: 0, available: false };
    }
    const formData = new FormData();
    formData.append('text', prompt);
    formData.append('base_model', baseModel);
    return postForm('/tokenize', formData);
}

export async function getGallery(start = 0, limit = 50) {
    const url = `/gallery?start=${start}&limit=${limit}&_=${Date.now()}`;
    return get(url);
}

export async function searchGallery(query, start = 0, limit = 2000) {
    const q = encodeURIComponent(query || '');
    return get(`/gallery/search?q=${q}&start=${start}&limit=${limit}&_=${Date.now()}`);
}

export async function filterGallery(kind, start = 0, limit = 2000) {
    const k = encodeURIComponent(kind || '');
    return get(`/gallery/filter?kind=${k}&start=${start}&limit=${limit}&_=${Date.now()}`);
}

export async function getQueue() {
    return get('/queue');
}

export async function cancelQueue(jobId) {
    const formData = new FormData();
    formData.append('job_id', jobId);
    return postForm('/queue/cancel', formData);
}

export async function unloadAllModels() {
    const formData = new FormData();
    return postForm('/models/unload_all', formData);
}

export async function shutdownApp() {
    const formData = new FormData();
    return postForm('/app/shutdown', formData);
}

export async function setFavorite(imagePath, favorite = true) {
    const formData = new FormData();
    formData.append('path', imagePath);
    formData.append('favorite', favorite ? 'true' : 'false');
    return postForm('/favorite', formData);
}

export async function deleteImage(imagePath) {
    const formData = new FormData();
    formData.append('path', imagePath);
    return postForm('/delete_image', formData);
}

export async function deleteRun(runId) {
    const formData = new FormData();
    formData.append('path', runId);
    return postForm('/delete_run', formData);
}

export async function deleteImages(paths = []) {
    return post('/delete_images', { paths: Array.isArray(paths) ? paths : [] });
}
