/**
 * ImageEditor Module
 * Browser-side crop / rotate / flip / resize / adjust editor for gallery images.
 *
 * All interactive work happens on a <canvas>: no upload round-trip for previews and no
 * GPU involvement. Only the finished result is handed to the host, either to be saved as
 * a new gallery image (POST /image/edit) or to be pushed into the Studio input image.
 * The source image is never modified.
 */

import * as api from '../core/api.js';
import { byId, listen, toast } from '../core/utils.js';

const MAX_PREVIEW_EDGE = 1400;
const HANDLE_HIT_PX = 11;
const MIN_CROP_PX = 8;
const ROOT_ID = 'image-editor-overlay';
const FILTER_SUPPORTED = (() => {
    try {
        const ctx = document.createElement('canvas').getContext('2d');
        return typeof ctx?.filter === 'string';
    } catch (_) {
        return false;
    }
})();

const ASPECT_PRESETS = [
    { key: 'free', label: 'Free', ratio: 0 },
    { key: '1:1', label: '1:1', ratio: 1 },
    { key: '3:4', label: '3:4', ratio: 3 / 4 },
    { key: '4:5', label: '4:5', ratio: 4 / 5 },
    { key: '16:9', label: '16:9', ratio: 16 / 9 },
    { key: '9:16', label: '9:16', ratio: 9 / 16 },
];

const HANDLES = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'];

function defaultState() {
    return {
        rot: 0,
        flipH: false,
        flipV: false,
        // Normalized (0..1) rectangle in oriented-image space. Null means "whole frame".
        crop: null,
        aspect: 'free',
        brightness: 1,
        contrast: 1,
        saturation: 1,
        // Null means "use the cropped size".
        targetW: null,
        targetH: null,
        lockAspect: true,
        snapEight: true,
    };
}

function clamp(value, min, max) {
    if (!Number.isFinite(value)) return min;
    return Math.min(Math.max(value, min), max);
}

function roundTo(value, step) {
    return step > 0 ? Math.round(value / step) * step : Math.round(value);
}

function readFileAsImage(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error('Could not load the image for editing'));
        img.src = url;
    });
}

function canvasToBlob(canvas) {
    return new Promise((resolve, reject) => {
        canvas.toBlob((blob) => {
            if (blob) resolve(blob);
            else reject(new Error('Could not render the edited image'));
        }, 'image/png');
    });
}

function editorMarkup() {
    return `
    <div class="image-editor" role="dialog" aria-modal="true" aria-label="Image editor">
      <header class="image-editor-header">
        <div class="image-editor-titles">
          <h3 class="modal-title">Edit Image</h3>
          <span class="image-editor-source" id="image-editor-source"></span>
        </div>
        <button class="modal-close" id="image-editor-close-x" type="button" aria-label="Close editor">&times;</button>
      </header>

      <div class="image-editor-body">
        <div class="image-editor-stage" id="image-editor-stage">
          <canvas id="image-editor-canvas" class="image-editor-canvas"></canvas>
          <div class="image-editor-loading" id="image-editor-loading">Loading image...</div>
        </div>

        <aside class="image-editor-panel">
          <section class="image-editor-group">
            <h4 class="image-editor-group-title">Crop</h4>
            <div class="image-editor-chips" id="image-editor-aspects">
              ${ASPECT_PRESETS.map(p => `<button class="preset-chip${p.key === 'free' ? ' active' : ''}" type="button" data-aspect="${p.key}">${p.label}</button>`).join('')}
            </div>
            <div class="image-editor-hint">Drag on the image to draw a crop. Once one exists, drag inside it to move it, or drag a handle to resize.</div>
            <button class="btn btn-ghost btn-sm" id="image-editor-crop-reset" type="button">Full Frame</button>
          </section>

          <section class="image-editor-group">
            <h4 class="image-editor-group-title">Rotate &amp; Flip</h4>
            <div class="image-editor-buttons">
              <button class="btn btn-secondary btn-sm" id="image-editor-rot-ccw" type="button">Rotate Left</button>
              <button class="btn btn-secondary btn-sm" id="image-editor-rot-cw" type="button">Rotate Right</button>
            </div>
            <div class="image-editor-buttons">
              <button class="btn btn-secondary btn-sm" id="image-editor-flip-h" type="button">Flip Horizontal</button>
              <button class="btn btn-secondary btn-sm" id="image-editor-flip-v" type="button">Flip Vertical</button>
            </div>
          </section>

          <section class="image-editor-group">
            <h4 class="image-editor-group-title">Resize</h4>
            <div class="image-editor-dims">
              <label class="image-editor-dim">
                <span>Width</span>
                <input class="input" type="number" id="image-editor-width" min="8" step="8" />
              </label>
              <label class="image-editor-dim">
                <span>Height</span>
                <input class="input" type="number" id="image-editor-height" min="8" step="8" />
              </label>
            </div>
            <label class="image-editor-check">
              <input type="checkbox" id="image-editor-lock-aspect" checked />
              <span>Lock aspect ratio</span>
            </label>
            <label class="image-editor-check">
              <input type="checkbox" id="image-editor-snap-eight" checked />
              <span>Snap to multiples of 8</span>
            </label>
            <button class="btn btn-ghost btn-sm" id="image-editor-resize-reset" type="button">Reset Size</button>
          </section>

          <section class="image-editor-group">
            <h4 class="image-editor-group-title">Adjust</h4>
            ${FILTER_SUPPORTED ? '' : '<div class="image-editor-hint">This browser does not support canvas filters, so adjustments are disabled.</div>'}
            <label class="image-editor-slider">
              <span>Brightness</span>
              <input class="slider" type="range" id="image-editor-brightness" min="0" max="2" step="0.01" value="1" ${FILTER_SUPPORTED ? '' : 'disabled'} />
              <output id="image-editor-brightness-value">1.00</output>
            </label>
            <label class="image-editor-slider">
              <span>Contrast</span>
              <input class="slider" type="range" id="image-editor-contrast" min="0" max="2" step="0.01" value="1" ${FILTER_SUPPORTED ? '' : 'disabled'} />
              <output id="image-editor-contrast-value">1.00</output>
            </label>
            <label class="image-editor-slider">
              <span>Saturation</span>
              <input class="slider" type="range" id="image-editor-saturation" min="0" max="2" step="0.01" value="1" ${FILTER_SUPPORTED ? '' : 'disabled'} />
              <output id="image-editor-saturation-value">1.00</output>
            </label>
          </section>
        </aside>
      </div>

      <footer class="image-editor-footer">
        <div class="image-editor-readout">
          <span id="image-editor-source-dims"></span>
          <span class="image-editor-arrow">&rarr;</span>
          <strong id="image-editor-out-dims"></strong>
        </div>
        <div class="image-editor-actions">
          <button class="btn btn-ghost btn-sm" id="image-editor-reset" type="button">Reset All</button>
          <button class="btn btn-secondary" id="image-editor-cancel" type="button">Cancel</button>
          <button class="btn btn-secondary" id="image-editor-use-input" type="button">Use as Input Image</button>
          <button class="btn btn-primary" id="image-editor-save" type="button">Save Copy</button>
        </div>
      </footer>
    </div>`;
}

function installEditorOverlay() {
    if (byId(ROOT_ID)) return;

    const overlay = document.createElement('div');
    overlay.id = ROOT_ID;
    overlay.className = 'image-editor-overlay hidden';
    overlay.setAttribute('aria-hidden', 'true');
    overlay.innerHTML = editorMarkup();
    document.body.appendChild(overlay);
}

export class ImageEditor {
    constructor(callbacks = {}) {
        this.callbacks = {
            onSaved: callbacks.onSaved || (() => { }),
            onUseAsInput: callbacks.onUseAsInput || null,
        };

        this.state = defaultState();
        this.sourceSrc = '';
        this.baseImage = null;
        this.oriented = document.createElement('canvas');
        this.orientedCtx = this.oriented.getContext('2d');
        this.viewScale = 1;
        this.drag = null;
        this.isOpen = false;
        this.supportsFilters = FILTER_SUPPORTED;
    }

    get overlay() {
        return byId(ROOT_ID);
    }

    /**
     * Open the editor on a gallery image.
     * @param {{src: string, name?: string, onSaved?: Function, onUseAsInput?: Function}} options
     */
    async open(options = {}) {
        const src = options?.src;
        if (!src) {
            toast('No image selected to edit', 'error');
            return;
        }

        installEditorOverlay();
        this.callbacks.onSaved = options.onSaved || this.callbacks.onSaved;
        this.callbacks.onUseAsInput = options.onUseAsInput || this.callbacks.onUseAsInput;

        this.state = defaultState();
        this.sourceSrc = src;
        this.baseImage = null;
        this.isOpen = true;

        const overlay = this.overlay;
        overlay.classList.remove('hidden');
        overlay.classList.add('active');
        overlay.setAttribute('aria-hidden', 'false');
        // LightboxManager's global Delete/Backspace handler must not fire behind the editor.
        document.body.classList.add('image-editor-open');

        this._bindEvents();
        this._syncControls();
        this._setLoading(true);
        this._renderEmpty();

        try {
            this.baseImage = await readFileAsImage(src);
        } catch (error) {
            this._setLoading(false);
            toast(error.message || 'Could not load the image for editing', 'error');
            this.close();
            return;
        }

        this._setLoading(false);
        this.rebuildOriented();
        this.state.crop = null;
        this._renderPreview();
        this._syncControls();
    }

    close() {
        if (!this.isOpen) return;
        this.isOpen = false;
        this.drag = null;

        const overlay = this.overlay;
        if (overlay) {
            overlay.classList.remove('active');
            overlay.setAttribute('aria-hidden', 'true');
            setTimeout(() => {
                if (!this.isOpen) overlay.classList.add('hidden');
            }, 180);
        }

        document.body.classList.remove('image-editor-open');
        this.baseImage = null;
    }

    isActive() {
        return this.isOpen;
    }

    // ── geometry ────────────────────────────────────────────────────────────

    get orientedSize() {
        return { width: this.oriented.width || 1, height: this.oriented.height || 1 };
    }

    /** Re-apply rotation and flips at full resolution. Crop is stored in this frame. */
    rebuildOriented() {
        const img = this.baseImage;
        if (!img) return;

        const quarter = ((this.state.rot % 4) + 4) % 4;
        const swap = quarter % 2 === 1;
        const width = swap ? img.naturalHeight : img.naturalWidth;
        const height = swap ? img.naturalWidth : img.naturalHeight;

        this.oriented.width = Math.max(1, width);
        this.oriented.height = Math.max(1, height);

        this.orientedCtx.save();
        this.orientedCtx.clearRect(0, 0, this.oriented.width, this.oriented.height);
        this.orientedCtx.translate(this.oriented.width / 2, this.oriented.height / 2);
        this.orientedCtx.rotate(quarter * (Math.PI / 2));
        const flipX = this.state.flipH ? -1 : 1;
        const flipY = this.state.flipV ? -1 : 1;
        this.orientedCtx.scale(flipX, flipY);
        this.orientedCtx.drawImage(img, -img.naturalWidth / 2, -img.naturalHeight / 2);
        this.orientedCtx.restore();
    }

    cropAspect() {
        return ASPECT_PRESETS.find(p => p.key === this.state.aspect)?.ratio || 0;
    }

    /**
     * Largest centred crop of the given ratio that fits the oriented frame, normalized 0..1.
     * Returns null for a free-form ratio, where there is nothing to seed.
     */
    _centeredCropForAspect(ratio) {
        if (!(ratio > 0)) return null;
        const { width, height } = this.orientedSize;
        let w = width;
        let h = w / ratio;
        if (h > height) {
            h = height;
            w = h * ratio;
        }
        return {
            x: (width - w) / 2 / width,
            y: (height - h) / 2 / height,
            w: w / width,
            h: h / height,
        };
    }

    /** Resolve the crop rectangle in oriented pixels, clamped and aspect-corrected. */
    resolveCrop() {
        const { width, height } = this.orientedSize;
        const crop = this.state.crop;
        if (!crop) return { x: 0, y: 0, width, height };

        let x = clamp(crop.x, 0, 1) * width;
        let y = clamp(crop.y, 0, 1) * height;
        // The 1px floor applies in *pixels*. Flooring the normalized value instead would
        // inflate every crop smaller than the full frame back up to the whole image.
        let w = Math.max(1, crop.w * width);
        let h = Math.max(1, crop.h * height);

        w = clamp(w, 0, width);
        h = clamp(h, 0, height);
        x = clamp(x, 0, Math.max(0, width - w));
        y = clamp(y, 0, Math.max(0, height - h));

        const ratio = this.cropAspect();
        if (ratio > 0) {
            if (w / h > ratio) w = h * ratio;
            else h = w / ratio;
            w = clamp(w, 1, width);
            h = clamp(w / ratio, 1, height);
            x = clamp(x, 0, Math.max(0, width - w));
            y = clamp(y, 0, Math.max(0, height - h));
        }

        return { x, y, width: Math.max(1, w), height: Math.max(1, h) };
    }

    /** Final output size in pixels, applying the explicit resize request. */
    outputSize() {
        const crop = this.resolveCrop();
        let width = this.state.targetW || crop.width;
        let height = this.state.targetH || crop.height;

        width = clamp(Math.round(width), 8, 8192);
        height = clamp(Math.round(height), 8, 8192);

        if (this.state.snapEight) {
            width = Math.max(8, roundTo(width, 8));
            height = Math.max(8, roundTo(height, 8));
        }
        return { width, height };
    }

    filterString() {
        if (!this.supportsFilters) return 'none';
        const { brightness, contrast, saturation } = this.state;
        if (brightness === 1 && contrast === 1 && saturation === 1) return 'none';
        return `brightness(${brightness}) contrast(${contrast}) saturate(${saturation})`;
    }

    // ── rendering ───────────────────────────────────────────────────────────

    _renderEmpty() {
        const canvas = byId('image-editor-canvas');
        if (!canvas) return;
        canvas.width = 1;
        canvas.height = 1;
    }

    _renderPreview() {
        const canvas = byId('image-editor-canvas');
        if (!canvas || !this.baseImage) return;

        const { width: fullW, height: fullH } = this.orientedSize;
        this.viewScale = Math.min(1, MAX_PREVIEW_EDGE / Math.max(fullW, fullH));
        const viewW = Math.max(1, Math.round(fullW * this.viewScale));
        const viewH = Math.max(1, Math.round(fullH * this.viewScale));
        if (canvas.width !== viewW) canvas.width = viewW;
        if (canvas.height !== viewH) canvas.height = viewH;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, viewW, viewH);
        ctx.imageSmoothingQuality = 'high';
        ctx.filter = this.filterString();
        // Always paint the whole oriented frame. The crop is an overlay on top of it, so
        // the excluded area stays visible while dragging; drawing only the crop here would
        // zoom into the selection and leave nothing to aim at.
        ctx.drawImage(this.oriented, 0, 0, viewW, viewH);
        ctx.filter = 'none';

        this._drawCropOverlay(ctx, viewW, viewH);
        this._updateReadout();
    }

    _drawCropOverlay(ctx, viewW, viewH) {
        if (!this.state.crop) return;

        const crop = this.resolveCrop();
        const s = this.viewScale;
        const x = crop.x * s;
        const y = crop.y * s;
        const w = crop.width * s;
        const h = crop.height * s;

        ctx.save();
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.beginPath();
        ctx.rect(0, 0, viewW, viewH);
        ctx.rect(x, y, w, h);
        ctx.fill('evenodd');

        ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);

        // Rule-of-thirds guides.
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 1; i < 3; i += 1) {
            ctx.moveTo(x + (w * i) / 3, y);
            ctx.lineTo(x + (w * i) / 3, y + h);
            ctx.moveTo(x, y + (h * i) / 3);
            ctx.lineTo(x + w, y + (h * i) / 3);
        }
        ctx.stroke();

        const size = 9;
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.75)';
        ctx.lineWidth = 1;
        for (const handle of HANDLES) {
            const point = this._handlePoint(handle, crop, s);
            ctx.beginPath();
            ctx.rect(point.x - size / 2, point.y - size / 2, size, size);
            ctx.fill();
            ctx.stroke();
        }
        ctx.restore();
    }

    _handlePoint(handle, crop, s) {
        const x = crop.x * s;
        const y = crop.y * s;
        const w = crop.width * s;
        const h = crop.height * s;
        const midX = x + w / 2;
        const midY = y + h / 2;
        const map = {
            nw: [x, y], n: [midX, y], ne: [x + w, y], e: [x + w, midY],
            se: [x + w, y + h], s: [midX, y + h], sw: [x, y + h], w: [x, midY],
        };
        const [px, py] = map[handle] || [midX, midY];
        return { x: px, y: py };
    }

    /** Render the edit at full resolution for saving or for the Studio input image. */
    async renderExport() {
        if (!this.baseImage) throw new Error('No image loaded');

        const crop = this.resolveCrop();
        const out = this.outputSize();
        const canvas = document.createElement('canvas');
        canvas.width = out.width;
        canvas.height = out.height;

        const ctx = canvas.getContext('2d');
        ctx.imageSmoothingQuality = 'high';
        ctx.filter = this.filterString();
        ctx.drawImage(
            this.oriented,
            crop.x, crop.y, crop.width, crop.height,
            0, 0, out.width, out.height,
        );
        ctx.filter = 'none';

        return canvasToBlob(canvas);
    }

    opsSummary() {
        const crop = this.resolveCrop();
        const out = this.outputSize();
        const summary = {};
        if (this.state.rot) summary.rotate = this.state.rot;
        if (this.state.flipH) summary.flip_h = true;
        if (this.state.flipV) summary.flip_v = true;
        if (Math.abs(crop.width - this.orientedSize.width) > 0.5 ||
            Math.abs(crop.height - this.orientedSize.height) > 0.5) {
            summary.crop = [
                Math.round(crop.x), Math.round(crop.y),
                Math.round(crop.width), Math.round(crop.height),
            ];
        }
        if (out.width !== Math.round(crop.width) || out.height !== Math.round(crop.height)) {
            summary.resized_to = [out.width, out.height];
        }
        if (this.state.brightness !== 1) summary.brightness = this.state.brightness;
        if (this.state.contrast !== 1) summary.contrast = this.state.contrast;
        if (this.state.saturation !== 1) summary.saturation = this.state.saturation;
        return summary;
    }

    // ── control sync ────────────────────────────────────────────────────────

    _setLoading(isLoading) {
        const el = byId('image-editor-loading');
        if (!el) return;
        el.classList.toggle('hidden', !isLoading);
        el.textContent = isLoading ? 'Loading image...' : '';
    }

    _updateReadout() {
        const { width, height } = this.orientedSize;
        const source = byId('image-editor-source-dims');
        const dest = byId('image-editor-out-dims');
        if (source) source.textContent = `${width} x ${height}`;
        if (dest) {
            const out = this.outputSize();
            dest.textContent = `${out.width} x ${out.height}`;
        }
    }

    _syncControls() {
        const name = this.sourceSrc.split('/').pop() || '';
        const sourceLabel = byId('image-editor-source');
        if (sourceLabel) sourceLabel.textContent = name;

        for (const button of document.querySelectorAll(`#${ROOT_ID} [data-aspect]`)) {
            button.classList.toggle('active', button.dataset.aspect === this.state.aspect);
        }

        const widthInput = byId('image-editor-width');
        const heightInput = byId('image-editor-height');
        if (widthInput && document.activeElement !== widthInput) {
            widthInput.value = String(this.outputSize().width);
        }
        if (heightInput && document.activeElement !== heightInput) {
            heightInput.value = String(this.outputSize().height);
        }

        for (const key of ['brightness', 'contrast', 'saturation']) {
            const slider = byId(`image-editor-${key}`);
            const output = byId(`image-editor-${key}-value`);
            if (slider && document.activeElement !== slider) slider.value = String(this.state[key]);
            if (output) output.textContent = Number(this.state[key]).toFixed(2);
        }

        const lock = byId('image-editor-lock-aspect');
        if (lock) lock.checked = this.state.lockAspect;
        const snap = byId('image-editor-snap-eight');
        if (snap) snap.checked = this.state.snapEight;

        const flipH = byId('image-editor-flip-h');
        if (flipH) flipH.classList.toggle('btn-primary', this.state.flipH);
        const flipV = byId('image-editor-flip-v');
        if (flipV) flipV.classList.toggle('btn-primary', this.state.flipV);

        this._updateReadout();
    }

    // ── interaction ─────────────────────────────────────────────────────────

    _bindEvents() {
        if (this._bound) return;
        this._bound = true;

        listen(byId('image-editor-close-x'), 'click', () => this.close());
        listen(byId('image-editor-cancel'), 'click', () => this.close());

        for (const button of document.querySelectorAll(`#${ROOT_ID} [data-aspect]`)) {
            listen(button, 'click', () => {
                this.state.aspect = button.dataset.aspect;
                if (this.state.crop) {
                    this._applyCropConstraint();
                } else {
                    // Picking a ratio on an uncropped image should visibly start a crop,
                    // otherwise the chip looks dead until you drag a rect yourself.
                    const seed = this._centeredCropForAspect(this.cropAspect());
                    if (seed) this.state.crop = seed;
                }
                this._renderPreview();
                this._syncControls();
            });
        }

        listen(byId('image-editor-crop-reset'), 'click', () => {
            this.state.crop = null;
            this.state.targetW = null;
            this.state.targetH = null;
            this._renderPreview();
            this._syncControls();
        });

        listen(byId('image-editor-rot-ccw'), 'click', () => this._rotate(-1));
        listen(byId('image-editor-rot-cw'), 'click', () => this._rotate(1));

        listen(byId('image-editor-flip-h'), 'click', () => {
            this.state.flipH = !this.state.flipH;
            this._afterOrientationChange();
        });
        listen(byId('image-editor-flip-v'), 'click', () => {
            this.state.flipV = !this.state.flipV;
            this._afterOrientationChange();
        });

        listen(byId('image-editor-resize-reset'), 'click', () => {
            this.state.targetW = null;
            this.state.targetH = null;
            this._renderPreview();
            this._syncControls();
        });

        listen(byId('image-editor-width'), 'input', (e) => this._onDimInput('width', e.target.value));
        listen(byId('image-editor-height'), 'input', (e) => this._onDimInput('height', e.target.value));

        listen(byId('image-editor-lock-aspect'), 'change', (e) => {
            this.state.lockAspect = e.target.checked;
        });
        listen(byId('image-editor-snap-eight'), 'change', (e) => {
            this.state.snapEight = e.target.checked;
            this._renderPreview();
            this._syncControls();
        });

        for (const key of ['brightness', 'contrast', 'saturation']) {
            listen(byId(`image-editor-${key}`), 'input', (e) => {
                this.state[key] = Number(e.target.value);
                this._renderPreview();
                this._syncControls();
            });
        }

        listen(byId('image-editor-reset'), 'click', () => this.reset());
        listen(byId('image-editor-save'), 'click', () => this.save());
        listen(byId('image-editor-use-input'), 'click', () => this.useAsInputImage());

        this._bindCanvasInteraction();

        this._onKeyDown = (e) => {
            if (!this.isOpen) return;
            if (e.key === 'Escape') {
                e.preventDefault();
                this.close();
            }
        };
        document.addEventListener('keydown', this._onKeyDown);

        this._onBackdropClick = (e) => {
            if (e.target === this.overlay) this.close();
        };
        listen(this.overlay, 'click', this._onBackdropClick);
    }

    _rotate(delta) {
        const prev = this.orientedSize;
        const prevCrop = this.state.crop ? { ...this.state.crop } : null;

        this.state.rot = ((this.state.rot + delta) % 4 + 4) % 4;
        this.rebuildOriented();

        // A rotation swaps the axes, so carry the crop across in normalized space.
        this.state.crop = prevCrop;
        this._rotateCropForOrientation(prev, this.orientedSize, delta);
        this._applyCropConstraint();
        this._renderPreview();
        this._syncControls();
    }

    _rotateCropForOrientation(prevSize, nextSize, delta) {
        const crop = this.state.crop;
        if (!crop) return;
        const turns = ((delta % 4) + 4) % 4;
        if (turns === 0) return;

        let rect = {
            x: crop.x * prevSize.width,
            y: crop.y * prevSize.height,
            w: crop.w * prevSize.width,
            h: crop.h * prevSize.height,
        };

        for (let i = 0; i < turns; i += 1) {
            rect = {
                x: prevSize.height - rect.y - rect.h,
                y: rect.x,
                w: rect.h,
                h: rect.w,
            };
        }

        this.state.crop = {
            x: rect.x / nextSize.width,
            y: rect.y / nextSize.height,
            w: rect.w / nextSize.width,
            h: rect.h / nextSize.height,
        };
    }

    _afterOrientationChange() {
        this.state.crop = null;
        this.rebuildOriented();
        this._renderPreview();
        this._syncControls();
    }

    _applyCropConstraint() {
        const crop = this.resolveCrop();
        const { width, height } = this.orientedSize;
        this.state.crop = {
            x: crop.x / width,
            y: crop.y / height,
            w: crop.width / width,
            h: crop.height / height,
        };
    }

    _onDimInput(which, rawValue) {
        const value = Number(rawValue);
        if (!Number.isFinite(value) || value <= 0) return;

        const crop = this.resolveCrop();
        const ratio = crop.width / Math.max(1, crop.height);
        const step = this.state.snapEight ? 8 : 1;
        let width = Math.max(8, roundTo(value, step));
        let height = Math.round(width / ratio);

        if (this.state.lockAspect) {
            if (which === 'width') this.state.targetH = Math.max(8, height);
            else {
                this.state.targetW = Math.max(8, height * ratio);
                width = this.state.targetW;
            }
        }

        if (which === 'width') this.state.targetW = width;
        else this.state.targetH = Math.max(8, height);

        this._renderPreview();
        this._syncControls();
    }

    reset() {
        this.state = defaultState();
        this.rebuildOriented();
        this._renderPreview();
        this._syncControls();
    }

    // ── crop dragging ───────────────────────────────────────────────────────

    _bindCanvasInteraction() {
        const canvas = byId('image-editor-canvas');
        if (!canvas) return;

        const toView = (e) => {
            const rect = canvas.getBoundingClientRect();
            const viewW = canvas.width || 1;
            const viewH = canvas.height || 1;
            return {
                x: ((e.clientX - rect.left) / Math.max(1, rect.width)) * viewW,
                y: ((e.clientY - rect.top) / Math.max(1, rect.height)) * viewH,
                scaleX: viewW / Math.max(1, rect.width),
                scaleY: viewH / Math.max(1, rect.height),
            };
        };

        const onDown = (e) => {
            if (!this.isOpen || !this.baseImage) return;
            const point = toView(e);
            const s = this.viewScale;
            // `point` is in backing pixels, so scale the on-screen handle target up to match.
            // Otherwise the grab area shrinks as the preview is downscaled.
            const tolerance = HANDLE_HIT_PX * point.scaleX;

            if (this.state.crop) {
                const crop = this.resolveCrop();
                let hit = null;
                for (const handle of HANDLES) {
                    const hp = this._handlePoint(handle, crop, s);
                    if (Math.abs(hp.x - point.x) <= tolerance && Math.abs(hp.y - point.y) <= tolerance) {
                        hit = handle;
                        break;
                    }
                }
                if (hit) {
                    this.drag = { mode: 'resize', handle: hit, start: point, crop: { ...this.state.crop } };
                } else if (
                    point.x >= crop.x * s && point.x <= (crop.x + crop.width) * s &&
                    point.y >= crop.y * s && point.y <= (crop.y + crop.height) * s
                ) {
                    this.drag = { mode: 'move', start: point, crop: { ...this.state.crop } };
                }
            }

            if (!this.drag) {
                this.state.crop = {
                    x: point.x / s / this.orientedSize.width,
                    y: point.y / s / this.orientedSize.height,
                    w: 0, h: 0,
                };
                this.drag = { mode: 'new', start: point };
            }

            canvas.setPointerCapture?.(e.pointerId);
            e.preventDefault();
            this._renderPreview();
        };

        const onMove = (e) => {
            if (!this.drag) return;
            const point = toView(e);
            const s = this.viewScale;
            const W = this.orientedSize.width;
            const H = this.orientedSize.height;

            if (this.drag.mode === 'new') {
                const x0 = Math.min(this.drag.start.x, point.x) / s;
                const y0 = Math.min(this.drag.start.y, point.y) / s;
                const x1 = Math.max(this.drag.start.x, point.x) / s;
                const y1 = Math.max(this.drag.start.y, point.y) / s;
                this.state.crop = {
                    x: clamp(x0 / W, 0, 1),
                    y: clamp(y0 / H, 0, 1),
                    w: clamp((x1 - x0) / W, 0, 1),
                    h: clamp((y1 - y0) / H, 0, 1),
                };
            } else if (this.drag.mode === 'move') {
                const dx = (point.x - this.drag.start.x) / s / W;
                const dy = (point.y - this.drag.start.y) / s / H;
                const crop = this.drag.crop;
                this.state.crop = {
                    x: clamp(crop.x + dx, 0, 1 - crop.w),
                    y: clamp(crop.y + dy, 0, 1 - crop.h),
                    w: crop.w,
                    h: crop.h,
                };
            } else if (this.drag.mode === 'resize') {
                this._resizeCrop(this.drag.handle, this.drag.crop, point, s);
            }

            this._renderPreview();
            e.preventDefault();
        };

        const onUp = (e) => {
            if (!this.drag) return;
            canvas.releasePointerCapture?.(e.pointerId);
            this.drag = null;

            const crop = this.state.crop;
            const oriented = this.orientedSize;
            const tooSmall = !crop ||
                crop.w * oriented.width < MIN_CROP_PX ||
                crop.h * oriented.height < MIN_CROP_PX;

            if (tooSmall) {
                this.state.crop = null;
            } else {
                this._applyCropConstraint();
            }
            this._renderPreview();
            this._syncControls();
        };

        listen(canvas, 'pointerdown', onDown);
        listen(canvas, 'pointermove', onMove);
        listen(canvas, 'pointerup', onUp);
        listen(canvas, 'pointercancel', onUp);
    }

    _resizeCrop(handle, startCrop, point, s) {
        const W = this.orientedSize.width;
        const H = this.orientedSize.height;

        let x0 = startCrop.x * W;
        let y0 = startCrop.y * H;
        let x1 = (startCrop.x + startCrop.w) * W;
        let y1 = (startCrop.y + startCrop.h) * H;

        const px = point.x / s;
        const py = point.y / s;

        if (handle.includes('w')) x0 = px;
        if (handle.includes('e')) x1 = px;
        if (handle.includes('n')) y0 = py;
        if (handle.includes('s')) y1 = py;

        if (x1 < x0) [x0, x1] = [x1, x0];
        if (y1 < y0) [y0, y1] = [y1, y0];

        const ratio = this.cropAspect();
        let w = x1 - x0;
        let h = y1 - y0;

        if (ratio > 0) {
            // Corner handles keep the anchor point and grow the other axis to match.
            if (handle.length === 2) {
                if (w / Math.max(1e-6, h) > ratio) w = h * ratio;
                else h = w / ratio;
            } else if (handle === 'e' || handle === 'w') {
                h = w / ratio;
            } else {
                w = h * ratio;
            }
        }

        if (handle.includes('w')) x0 = x1 - w;
        if (handle.includes('n')) y0 = y1 - h;

        this.state.crop = {
            x: clamp(x0 / W, 0, 1),
            y: clamp(y0 / H, 0, 1),
            w: clamp(w / W, 0, 1),
            h: clamp(h / H, 0, 1),
        };
    }

    // ── actions ─────────────────────────────────────────────────────────────

    async save() {
        if (!this.baseImage) return;
        const button = byId('image-editor-save');
        if (button) button.disabled = true;

        try {
            const blob = await this.renderExport();
            const file = new File([blob], 'edit.png', { type: 'image/png' });
            const result = await api.saveImageEdit(this.sourceSrc, file, this.opsSummary());
            const newPath = result?.path;

            toast('Saved edited copy to the gallery', 'success');
            this.close();
            if (newPath) this.callbacks.onSaved(newPath);
        } catch (error) {
            console.error('Image edit save failed:', error);
            toast(error.message || 'Failed to save the edited image', 'error');
        } finally {
            if (button) button.disabled = false;
        }
    }

    async useAsInputImage() {
        if (!this.baseImage) return;
        if (typeof this.callbacks.onUseAsInput !== 'function') {
            toast('Sending the image to the Studio is unavailable', 'error');
            return;
        }

        const button = byId('image-editor-use-input');
        if (button) button.disabled = true;

        try {
            const blob = await this.renderExport();
            const file = new File([blob], 'edited_input.png', { type: 'image/png' });
            this.close();
            await this.callbacks.onUseAsInput(file, { name: 'edited_input.png' });
        } catch (error) {
            console.error('Image edit input transfer failed:', error);
            toast(error.message || 'Failed to send the edited image to the Studio', 'error');
        } finally {
            if (button) button.disabled = false;
        }
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', installEditorOverlay, { once: true });
} else {
    installEditorOverlay();
}
