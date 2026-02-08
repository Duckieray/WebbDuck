/**
 * LoraManager Module
 * Handles loading, selecting, and managing LoRA models.
 */

import * as api from '../core/api.js';
import { byId, toast, listen } from '../core/utils.js';
import { getState, setState } from '../core/state.js';

export class LoraManager {
    constructor() {
        this.selectedLoras = new Map();
        this.availableLorasMap = new Map();

        // DOM Elements
        this.select = byId('lora-select');
        this.container = byId('lora-selected');

        this.init();
    }

    init() {
        if (this.select) {
            // Handle selection
            this.select.onchange = () => {
                if (this.select.value) {
                    this.addLora(this.select.value);
                    this.select.value = ''; // Reset to placeholder
                }
            };
        }
    }

    /**
     * Load available LoRAs for a specific base model
     * @param {string} modelName 
     */
    async loadForModel(modelName) {
        if (!this.select) return;

        try {
            const loras = await api.getLoras(modelName);

            // Clear cache
            this.availableLorasMap.clear();

            // Reset select
            this.select.innerHTML = '<option value="">➕ Add LoRA...</option>';

            loras.forEach(lora => {
                const name = typeof lora === 'string' ? lora : lora.name;

                // Store metadata
                if (typeof lora === 'string') {
                    this.availableLorasMap.set(name, { name, strength_default: 1.0 });
                } else {
                    this.availableLorasMap.set(name, lora);
                }

                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                this.select.appendChild(opt);
            });

            this.restoreFromState();

            // Re-validate selected LoRAs? 
            // If model changed, maybe clear selected LoRAs?
            // Usually yes, different base model = incompatible LoRAs.
            // But for now, we leave them or let user decide?
            // WebbDuck usually clears on model change?
            // The existing code didn't seem to clear explicitly in loadLoras, 
            // but loadLoras is called when base model changes?
            // Let's safe-guard by just keeping the current logic of loading available ones.

        } catch (error) {
            console.error('Failed to load LoRAs:', error);
            toast('Failed to load LoRAs', 'error');
        }
    }

    /**
     * Add a LoRA to the active set
     * @param {string} name 
     * @param {number|null} weight 
     */
    addLora(name, weight = null, options = {}) {
        const { persist = true, silent = false } = options;
        if (this.selectedLoras.has(name)) {
            if (!silent) {
                toast(`${name} already added`, 'info');
            }
            return;
        }

        // Use provided weight, or look up default, or fallback to 1.0
        if (weight === null) {
            const info = this.availableLorasMap.get(name);
            const defaultWeight = info?.weight ?? info?.strength_default;
            weight = defaultWeight !== undefined ? Number(defaultWeight) : 1.0;
        }

        this.selectedLoras.set(name, weight);
        this.renderCard(name, weight);
        if (persist) {
            this.persistSelection();
        }
    }

    /**
     * Remove a LoRA
     * @param {string} name 
     */
    removeLora(name) {
        this.selectedLoras.delete(name);
        const card = this.container.querySelector(`.lora-card[data-lora="${CSS.escape(name)}"]`);
        if (card) card.remove();
        this.persistSelection();
    }

    /**
     * Render a LoRA card in the selected container
     */
    renderCard(name, weight) {
        if (!this.container) return;

        const card = document.createElement('div');
        card.className = 'lora-card';
        card.dataset.lora = name;
        card.innerHTML = `
            <div class="lora-card-header">
                <span class="lora-card-name">${name}</span>
                <button class="btn btn-ghost btn-icon btn-sm lora-remove" title="Remove">✕</button>
            </div>
            <div class="lora-card-slider">
                <input type="range" class="slider lora-weight" min="0" max="2" step="0.05" value="${weight}" />
                <span class="lora-weight-val">${weight.toFixed(2)}</span>
            </div>
        `;

        // Weight slider handler
        const slider = card.querySelector('.lora-weight');
        const valDisplay = card.querySelector('.lora-weight-val');

        slider.oninput = () => {
            const val = parseFloat(slider.value);
            valDisplay.textContent = val.toFixed(2);
            this.selectedLoras.set(name, val);
            this.persistSelection();
        };

        // Remove handler
        card.querySelector('.lora-remove').onclick = () => {
            this.removeLora(name);
        };

        this.container.appendChild(card);
    }

    /**
     * Get list of selected LoRAs for generation
     * @returns {Array<{name: string, weight: number}>}
     */
    getSelected() {
        return Array.from(this.selectedLoras.entries()).map(([name, weight]) => ({ name, weight }));
    }

    /**
     * Clear all selected LoRAs
     */
    clear() {
        this.selectedLoras.clear();
        if (this.container) this.container.innerHTML = '';
        this.persistSelection();
    }

    persistSelection() {
        setState({ selectedLoras: this.getSelected() });
    }

    restoreFromState() {
        // Only auto-restore into an empty UI on fresh load/refresh.
        if (this.selectedLoras.size > 0 || (this.container && this.container.children.length > 0)) {
            return;
        }

        const saved = getState('selectedLoras');
        if (!Array.isArray(saved) || saved.length === 0) {
            return;
        }

        saved.forEach(entry => {
            const name = entry?.name;
            if (!name || !this.availableLorasMap.has(name)) {
                return;
            }

            const rawWeight = entry?.weight ?? entry?.strength;
            const parsedWeight = Number(rawWeight);
            const weight = Number.isFinite(parsedWeight) ? parsedWeight : null;
            this.addLora(name, weight, { persist: false, silent: true });
        });
    }
}
