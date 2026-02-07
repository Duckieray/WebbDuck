/**
 * WebbDuck Utility Functions
 * Common helper functions used throughout the app
 */

// ═══════════════════════════════════════════════════════════════
// DOM UTILITIES
// ═══════════════════════════════════════════════════════════════

/**
 * Shorthand for querySelector
 */
export function $(selector, parent = document) {
    return parent.querySelector(selector);
}

/**
 * Shorthand for querySelectorAll with array return
 */
export function $$(selector, parent = document) {
    return Array.from(parent.querySelectorAll(selector));
}

/**
 * Get element by ID
 */
export function byId(id) {
    return document.getElementById(id);
}

/**
 * Add event listener shorthand
 */
export function listen(element, event, handler, options) {
    if (typeof element === 'string') {
        element = $(element);
    }
    if (element) {
        element.addEventListener(event, handler, options);
    }
}

/**
 * Toggle class on element
 */
export function toggleClass(element, className, force) {
    if (typeof element === 'string') {
        element = $(element);
    }
    if (element) {
        element.classList.toggle(className, force);
    }
}

/**
 * Show element (remove hidden class)
 */
export function show(element) {
    if (typeof element === 'string') {
        element = $(element);
    }
    if (element) {
        element.classList.remove('hidden');
    }
}

/**
 * Hide element (add hidden class)
 */
export function hide(element) {
    if (typeof element === 'string') {
        element = $(element);
    }
    if (element) {
        element.classList.add('hidden');
    }
}

// ═══════════════════════════════════════════════════════════════
// STRING UTILITIES
// ═══════════════════════════════════════════════════════════════

/**
 * Truncate string with ellipsis
 */
export function truncate(str, maxLength = 50) {
    if (!str || str.length <= maxLength) return str;
    return str.slice(0, maxLength - 3) + '...';
}

/**
 * Format number with commas
 */
export function formatNumber(num) {
    return num.toLocaleString();
}

/**
 * Format timestamp to relative time
 */
export function timeAgo(timestamp) {
    const seconds = Math.floor((Date.now() - timestamp * 1000) / 1000);

    if (seconds < 60) return 'Just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;

    return new Date(timestamp * 1000).toLocaleDateString();
}

/**
 * Format timestamp to date string
 */
export function formatDate(timestamp) {
    return new Date(timestamp * 1000).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    });
}

// ═══════════════════════════════════════════════════════════════
// FILE UTILITIES
// ═══════════════════════════════════════════════════════════════

/**
 * Convert File/Blob to Data URL
 */
export function fileToDataURL(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

/**
 * Convert Data URL to Blob
 */
export function dataURLToBlob(dataURL) {
    const [header, data] = dataURL.split(',');
    const mime = header.match(/:(.*?);/)[1];
    const binary = atob(data);
    const array = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
        array[i] = binary.charCodeAt(i);
    }
    return new Blob([array], { type: mime });
}

/**
 * Download a file from URL
 */
export function downloadFile(url, filename) {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// ═══════════════════════════════════════════════════════════════
// FORM UTILITIES
// ═══════════════════════════════════════════════════════════════

/**
 * Populate a select element with options
 */
export function populateSelect(selectId, options, includeNone = false) {
    const select = byId(selectId);
    if (!select) return;

    select.innerHTML = includeNone ? '<option value="None">None</option>' : '';

    options.forEach(option => {
        const opt = document.createElement('option');
        if (typeof option === 'string') {
            opt.value = option;
            opt.textContent = option;
        } else {
            opt.value = option.value || option.name;
            opt.textContent = option.label || option.name || option.value;
        }
        select.appendChild(opt);
    });
}

/**
 * Build FormData from state and elements
 */
export function buildFormData(data) {
    const formData = new FormData();
    for (const [key, value] of Object.entries(data)) {
        if (value !== null && value !== undefined) {
            if (value instanceof Blob) {
                formData.append(key, value);
            } else {
                formData.append(key, String(value));
            }
        }
    }
    return formData;
}

// ═══════════════════════════════════════════════════════════════
// DEBOUNCE / THROTTLE
// ═══════════════════════════════════════════════════════════════

/**
 * Debounce function
 */
export function debounce(fn, delay = 300) {
    let timeoutId;
    return function (...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn.apply(this, args), delay);
    };
}

/**
 * Throttle function
 */
export function throttle(fn, limit = 100) {
    let inThrottle;
    return function (...args) {
        if (!inThrottle) {
            fn.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// ═══════════════════════════════════════════════════════════════
// ASYNC UTILITIES
// ═══════════════════════════════════════════════════════════════

/**
 * Sleep for a duration
 */
export function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Wait for an element to appear in DOM
 */
export function waitForElement(selector, timeout = 5000) {
    return new Promise((resolve, reject) => {
        const element = $(selector);
        if (element) {
            return resolve(element);
        }

        const observer = new MutationObserver(() => {
            const element = $(selector);
            if (element) {
                observer.disconnect();
                resolve(element);
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true,
        });

        setTimeout(() => {
            observer.disconnect();
            reject(new Error(`Element ${selector} not found within ${timeout}ms`));
        }, timeout);
    });
}

// ═══════════════════════════════════════════════════════════════
// TOAST NOTIFICATIONS
// ═══════════════════════════════════════════════════════════════

/**
 * Show a toast notification
 */
export function toast(message, type = 'info', duration = 3000) {
    const container = byId('toast-container');
    if (!container) return;

    const toastEl = document.createElement('div');
    toastEl.className = `toast toast-${type}`;

    const icons = {
        success: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>',
        error: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>',
        warning: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>',
        info: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>',
    };

    toastEl.innerHTML = `
    <span class="toast-icon">${icons[type] || icons.info}</span>
    <span class="toast-message">${message}</span>
    <button class="toast-close">&times;</button>
  `;

    container.appendChild(toastEl);

    // Trigger animation
    requestAnimationFrame(() => {
        toastEl.classList.add('show');
    });

    // Close button
    toastEl.querySelector('.toast-close').addEventListener('click', () => {
        removeToast(toastEl);
    });

    // Auto remove
    if (duration > 0) {
        setTimeout(() => removeToast(toastEl), duration);
    }

    return toastEl;
}

function removeToast(toastEl) {
    toastEl.classList.remove('show');
    setTimeout(() => toastEl.remove(), 300);
}

// ═══════════════════════════════════════════════════════════════
// SECTION TOGGLE
// ═══════════════════════════════════════════════════════════════

/**
 * Toggle a collapsible section
 */
export function toggleSection(sectionId) {
    const section = byId(sectionId);
    if (section) {
        section.classList.toggle('collapsed');
    }
}

// Make toggleSection available globally for onclick handlers
window.toggleSection = toggleSection;
