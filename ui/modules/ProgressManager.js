/**
 * ProgressManager Module
 * Handles real-time generation progress visualization
 */

import { Events, on, emit } from '../core/events.js';
import { byId, show, hide, toggleClass } from '../core/utils.js';

export class ProgressManager {
    constructor() {
        this.el = byId('generation-progress');
        this.circle = byId('progress-circle');
        this.valueEl = byId('progress-value');
        this.stepEl = byId('progress-step');
        this.messageEl = byId('progress-message');
        this.statusBar = byId('status-indicator');
        this.statusText = byId('status-text');

        // SVG Circle properties
        this.radius = 40;
        this.circumference = 2 * Math.PI * this.radius;

        this.init();
    }

    init() {
        // Setup circle
        if (this.circle) {
            this.circle.style.strokeDasharray = `${this.circumference} ${this.circumference}`;
            this.circle.style.strokeDashoffset = this.circumference;
        }

        // Subscribe to status updates
        on(Events.STATUS_UPDATE, this.handleUpdate.bind(this));

        // Listen for manual cancel button
        const cancelBtn = byId('cancel-generation');
        if (cancelBtn) {
            cancelBtn.addEventListener('click', () => {
                emit(Events.GENERATION_CANCEL);
            });
        }
    }

    handleUpdate(data) {
        // Data format: { stage: "Denoising", progress: 0.45, vram: {...} }
        const { stage, progress, vram } = data;

        // Update main progress UI if active
        if (stage && stage !== 'Idle' && stage !== 'Error') {
            this.showProgress(stage, progress);
        } else {
            this.hideProgress();
        }

        // Update status bar (always visible)
        this.updateStatusBar(stage, vram);
    }

    showProgress(stage, progress) {
        if (!this.el) return;

        show(this.el);

        // Calculate percentages
        const pct = Math.min(Math.max(progress || 0, 0), 1);
        const offset = this.circumference - (pct * this.circumference);

        // Update Ring
        if (this.circle) {
            this.circle.style.strokeDashoffset = offset;
        }

        // Update Text
        if (this.valueEl) this.valueEl.textContent = `${Math.round(pct * 100)}%`;
        if (this.messageEl) this.messageEl.textContent = stage;

        // Try to estimate steps if possible? 
        // Backend only sends 0-1 float, so "Step X/Y" is unavailable unless we infer it or backend sends it.
        // For now, hide step counter or use it for something else.
        if (this.stepEl) {
            // If we had total steps, we could calc current step. 
            // Without it, just hide or show generic info.
            this.stepEl.style.display = 'none';
        }
    }

    hideProgress() {
        if (!this.el) return;

        // Add a small delay for "100%" to be seen
        if (this.el.style.display !== 'none') {
            // Only hide if it was showing
            setTimeout(() => {
                // Check if still idle (race condition protection)
                // For now just hide immediately to keep it simple.
                // A fade out animation in CSS would be better.
                hide(this.el);
            }, 500);
        } else {
            hide(this.el);
        }
    }

    updateStatusBar(stage, vram) {
        if (this.statusText) {
            this.statusText.textContent = stage === 'Idle' ? 'Ready' : stage;
        }

        if (this.statusBar) {
            // Colors based on state
            this.statusBar.className = 'status-indicator';
            if (stage === 'Idle') this.statusBar.classList.add('ready');
            else if (stage === 'Error') this.statusBar.classList.add('error');
            else this.statusBar.classList.add('busy');
        }

        // Optional: show VRAM usage in status text or separate element?
        // "Ready (VRAM: 4.2GB)"
        if (vram && this.statusText && stage === 'Idle') {
            this.statusText.textContent = `Ready (${vram.used.toFixed(1)} GB VRAM)`;
        } else if (vram && this.statusText) {
            this.statusText.textContent = `${stage} (${vram.used.toFixed(1)} GB)`;
        }
    }
}
