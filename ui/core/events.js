/**
 * WebbDuck Event Bus
 * Simple pub/sub system for decoupled communication
 */

// Event storage
const events = new Map();

/**
 * Subscribe to an event
 * @param {string} eventName - The event to subscribe to
 * @param {Function} callback - The callback function
 * @returns {Function} Unsubscribe function
 */
export function on(eventName, callback) {
    if (!events.has(eventName)) {
        events.set(eventName, new Set());
    }
    events.get(eventName).add(callback);

    // Return unsubscribe function
    return () => off(eventName, callback);
}

/**
 * Subscribe to an event once
 * @param {string} eventName - The event to subscribe to
 * @param {Function} callback - The callback function
 */
export function once(eventName, callback) {
    const wrapper = (...args) => {
        off(eventName, wrapper);
        callback(...args);
    };
    on(eventName, wrapper);
}

/**
 * Unsubscribe from an event
 * @param {string} eventName - The event to unsubscribe from
 * @param {Function} callback - The callback function to remove
 */
export function off(eventName, callback) {
    if (events.has(eventName)) {
        events.get(eventName).delete(callback);
    }
}

/**
 * Emit an event
 * @param {string} eventName - The event to emit
 * @param {*} data - The data to pass to subscribers
 */
export function emit(eventName, data) {
    if (events.has(eventName)) {
        events.get(eventName).forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error(`Error in event handler for "${eventName}":`, error);
            }
        });
    }
}

/**
 * Clear all listeners for an event
 * @param {string} eventName - The event to clear
 */
export function clear(eventName) {
    if (events.has(eventName)) {
        events.get(eventName).clear();
    }
}

/**
 * Clear all events
 */
export function clearAll() {
    events.clear();
}

// ═══════════════════════════════════════════════════════════════
// PREDEFINED EVENT NAMES
// ═══════════════════════════════════════════════════════════════

export const Events = {
    // View events
    VIEW_CHANGE: 'view:change',

    // Generation events
    GENERATION_START: 'generation:start',
    GENERATION_PROGRESS: 'generation:progress',
    GENERATION_COMPLETE: 'generation:complete',
    GENERATION_ERROR: 'generation:error',
    GENERATION_CANCEL: 'generation:cancel',

    // Image events
    IMAGE_GENERATED: 'image:generated',
    IMAGE_UPSCALED: 'image:upscaled',
    IMAGE_DELETED: 'image:deleted',
    IMAGE_SELECT: 'image:select',

    // Preview events
    PREVIEW_UPDATE: 'preview:update',
    PREVIEW_COMPARE: 'preview:compare',
    PREVIEW_ZOOM: 'preview:zoom',

    // Model events
    MODEL_CHANGE: 'model:change',
    MODEL_LOADED: 'model:loaded',
    LORA_CHANGE: 'lora:change',

    // Gallery events
    GALLERY_REFRESH: 'gallery:refresh',
    GALLERY_SEARCH: 'gallery:search',

    // Upload events
    IMAGE_UPLOAD: 'upload:image',
    IMAGE_CLEAR: 'upload:clear',
    MASK_CREATED: 'mask:created',
    MASK_CLEARED: 'mask:cleared',

    // UI events
    TOAST_SHOW: 'toast:show',
    MODAL_OPEN: 'modal:open',
    MODAL_CLOSE: 'modal:close',
    SECTION_TOGGLE: 'section:toggle',

    // State events
    STATE_CHANGE: 'state:change',
    STATE_RESET: 'state:reset',
};
