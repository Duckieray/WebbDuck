/**
 * MaskEditor Module
 * Handles mask drawing, erasing, and canvas management.
 */

import { byId, listen, show, hide, toast } from '../core/utils.js';
import { getState, setState } from '../core/state.js';

export class MaskEditor {
    constructor() {
        this.canvas = byId('mask_canvas');
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.overlay = byId('mask_overlay');
        this.wrapper = byId('mask_wrapper'); // The container div

        this.isDrawing = false;
        this.isErasing = false;
        this.lastX = 0;
        this.lastY = 0;

        // Configuration
        this.brushColor = 'rgba(255, 255, 255, 1)';
        this.eraseColor = 'rgba(0, 0, 0, 1)';

        if (this.canvas) {
            this.initEvents();
            this.initUI();
        }
    }

    /**
     * Initialize canvas events
     */
    initEvents() {
        // Mouse Events
        listen(this.canvas, 'mousedown', this.handleStart.bind(this));
        listen(this.canvas, 'mousemove', this.draw.bind(this));
        listen(window, 'mouseup', this.handleEnd.bind(this));

        // Touch Events
        listen(this.canvas, 'touchstart', this.handleStart.bind(this));
        listen(this.canvas, 'touchmove', this.draw.bind(this));
        listen(window, 'touchend', this.handleEnd.bind(this));
    }

    /**
     * Initialize UI controls
     */
    initUI() {
        // Cancel
        listen(byId('mask_cancel_btn'), 'click', () => this.close());

        // Erase Toggle
        listen(byId('mask_erase_btn'), 'click', () => this.toggleErase());

        // Clear
        listen(byId('mask_clear_btn'), 'click', () => this.clear());

        // Invert
        listen(byId('mask_invert_btn'), 'click', () => this.invert());

        // Save
        listen(byId('mask_save_btn'), 'click', () => this.save());

        // Brush Size Preview
        const brushSlider = byId('mask_brush_size');
        if (brushSlider) {
            listen(brushSlider, 'input', (e) => {
                const preview = byId('brushPreview');
                if (preview) {
                    preview.style.width = e.target.value + 'px';
                    preview.style.height = e.target.value + 'px';
                }
            });
        }

        // Blur Slider
        const blurSlider = byId('mask_blur');
        if (blurSlider) {
            listen(blurSlider, 'input', (e) => {
                const display = byId('blurVal');
                if (display) display.textContent = e.target.value;
            });
        }
    }

    /**
     * Open the mask editor with an image
     * @param {string} imageSrc - URL of the image to mask
     */
    open(imageSrc) {
        if (!this.canvas) return;

        // Set background image
        this.wrapper.style.backgroundImage = `url(${imageSrc})`;
        this.wrapper.style.backgroundSize = 'contain';
        this.wrapper.style.backgroundRepeat = 'no-repeat';
        this.wrapper.style.backgroundPosition = 'center';

        // Get image dimensions to set canvas size matching aspect ratio
        const img = new Image();
        img.onload = () => {
            // We use the wrapper's dimensions which are constrained by CSS (max-width/height)
            // But we want the canvas resolution to be high enough for quality.
            // Actually, to align perfectly, we should probably set canvas resolution to match image?
            // Or match the visual size?

            // Current app.js logic relied on scaling.
            // Best approach: Set canvas internal resolution to match displayed size or fixed high res.
            // Let's use the natural image size for best quality, then CSS scales it.

            this.canvas.width = img.width;
            this.canvas.height = img.height;

            // Ensure wrapper has correct aspect ratio
            // The wrapper size is controlled by flex/CSS.
            // But we removed flex:1.
            // We need to size the wrapper to match the image aspect ratio within the viewport.

            const aspect = img.width / img.height;
            const viewportW = window.innerWidth * 0.8;
            const viewportH = window.innerHeight * 0.7;

            let displayW = viewportW;
            let displayH = viewportW / aspect;

            if (displayH > viewportH) {
                displayH = viewportH;
                displayW = displayH * aspect;
            }

            this.wrapper.style.width = `${displayW}px`;
            this.wrapper.style.height = `${displayH}px`;

            show(this.overlay);
        };
        img.src = imageSrc;
    }

    close() {
        hide(this.overlay);
    }

    handleStart(e) {
        this.isDrawing = true;
        const { x, y } = this.getPos(e);
        this.lastX = x;
        this.lastY = y;
        this.draw(e);
    }

    handleEnd() {
        this.isDrawing = false;
    }

    getPos(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        let clientX = e.clientX;
        let clientY = e.clientY;

        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        }

        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    draw(e) {
        if (!this.isDrawing) return;
        if (e.preventDefault) e.preventDefault(); // Prevent scrolling

        const { x, y } = this.getPos(e);

        this.ctx.beginPath();
        this.ctx.moveTo(this.lastX, this.lastY);
        this.ctx.lineTo(x, y);

        if (this.isErasing) {
            this.ctx.globalCompositeOperation = 'destination-out';
            this.ctx.strokeStyle = this.eraseColor;
        } else {
            this.ctx.globalCompositeOperation = 'source-over';
            this.ctx.strokeStyle = this.brushColor;
        }

        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';

        // Calculate brush size relative to canvas
        const sizeInput = byId('mask_brush_size')?.value || 30;
        const rect = this.canvas.getBoundingClientRect();
        const scale = this.canvas.width / rect.width;
        this.ctx.lineWidth = sizeInput * scale;

        this.ctx.stroke();

        this.ctx.globalCompositeOperation = 'source-over';
        this.lastX = x;
        this.lastY = y;
    }

    toggleErase() {
        this.isErasing = !this.isErasing;
        const btn = byId('mask_erase_btn');
        if (this.isErasing) {
            btn.textContent = '✏️ Draw';
            btn.classList.add('btn-primary');
            btn.classList.remove('btn-secondary');
        } else {
            btn.textContent = '🧽 Erase';
            btn.classList.remove('btn-primary');
            btn.classList.add('btn-secondary');
        }
    }

    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    invert() {
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const data = imageData.data;
        for (let i = 0; i < data.length; i += 4) {
            data[i + 3] = 255 - data[i + 3];
            // Ensure white color
            data[i] = 255;
            data[i + 1] = 255;
            data[i + 2] = 255;
        }
        this.ctx.putImageData(imageData, 0, 0);
    }

    save() {
        this.canvas.toBlob((blob) => {
            window._maskBlob = blob;
            this.close();

            const editBtn = byId('edit-mask-btn');
            if (editBtn) editBtn.style.color = 'var(--green)';

            toast('Mask saved!', 'success');
            show(byId('inpaint-options'));
        });
    }
}
