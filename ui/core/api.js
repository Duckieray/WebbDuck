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
export async function getGallery(since = 0) {
    const url = since ? `/gallery?since=${since}` : '/gallery';
    return get(url);
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
