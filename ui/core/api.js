/**
 * WebbDuck Core API Module
 * Centralized fetch wrappers for all API endpoints
 */

const API_BASE = '';

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
            const error = await response.text();
            throw new Error(error || `HTTP ${response.status}`);
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

// ═══════════════════════════════════════════════════════════════
// SPECIFIC API ENDPOINTS
// ═══════════════════════════════════════════════════════════════

/**
 * Fetch available models
 */
export async function getModels() {
    return get('/models');
}

/**
 * Fetch LoRAs for a specific model
 */
export async function getLoras(modelName) {
    return get(`/models/${encodeURIComponent(modelName)}/loras`);
}

/**
 * Fetch second pass / refiner models
 */
export async function getSecondPassModels() {
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
    return post('/delete_image', { image_path: imagePath });
}

/**
 * Delete an entire run/session
 */
export async function deleteRun(runId) {
    return post('/delete_run', { run_id: runId });
}
