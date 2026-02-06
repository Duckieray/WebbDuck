/**
 * LoraManager Module
 * Handles loading, selecting, and managing LoRA models.
 */

import * as api from '../core/api.js';
import { byId, toast, listen } from '../core/utils.js';

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
    addLora(name, weight = null) {
        if (this.selectedLoras.has(name)) {
            toast(`${name} already added`, 'info');
            return;
        }

        // Use provided weight, or look up default, or fallback to 1.0
        if (weight === null) {
            const info = this.availableLorasMap.get(name);
            weight = info && info.strength_default !== undefined ? info.strength_default : 1.0;
        }

        this.selectedLoras.set(name, weight);
        this.renderCard(name, weight);
    }

    /**
     * Remove a LoRA
     * @param {string} name 
     */
    removeLora(name) {
        this.selectedLoras.delete(name);
        const card = this.container.querySelector(`.lora-card[data-lora="${CSS.escape(name)}"]`);
        if (card) card.remove();
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
                <input type="range" class="slider lora-weight" min="0" max="2" step="0.1" value="${weight}" />
                <span class="lora-weight-val">${weight.toFixed(1)}</span>
            </div>
        `;

        // Weight slider handler
        const slider = card.querySelector('.lora-weight');
        const valDisplay = card.querySelector('.lora-weight-val');

        slider.oninput = () => {
            const val = parseFloat(slider.value);
            valDisplay.textContent = val.toFixed(1);
            this.selectedLoras.set(name, val);
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
    }
}
