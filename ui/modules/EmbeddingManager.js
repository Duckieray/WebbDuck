/**
 * EmbeddingManager Module
 * Handles loading, selecting, and managing textual inversion embeddings.
 */

import * as api from '../core/api.js';
import { byId, toast } from '../core/utils.js';
import { getState, setState } from '../core/state.js';

export class EmbeddingManager {
    constructor() {
        this.selectedEmbeddings = new Map();
        this.availableEmbeddingsMap = new Map();

        this.select = byId('embedding-select');
        this.container = byId('embedding-selected');

        this.init();
    }

    init() {
        if (!this.select) return;

        this.select.onchange = () => {
            if (!this.select.value) return;
            this.addEmbedding(this.select.value);
            this.select.value = '';
        };
    }

    async loadForModel(modelName) {
        if (!this.select) return;

        try {
            const embeddings = await api.getEmbeddings(modelName);
            this.availableEmbeddingsMap.clear();

            embeddings.forEach((embedding) => {
                const name = typeof embedding === 'string' ? embedding : embedding?.name;
                if (!name) return;

                if (typeof embedding === 'string') {
                    this.availableEmbeddingsMap.set(name, { name, token: name });
                    return;
                }

                this.availableEmbeddingsMap.set(name, {
                    name,
                    token: embedding.token || name,
                    description: embedding.description || '',
                });
            });

            this.restoreFromState();
            this.refreshSelectOptions();
        } catch (error) {
            console.error('Failed to load embeddings:', error);
            toast('Failed to load embeddings', 'error');
        }
    }

    refreshSelectOptions() {
        if (!this.select) return;
        const currentValue = this.select.value;
        this.select.innerHTML = '<option value="">+ Add Embedding...</option>';

        const names = Array.from(this.availableEmbeddingsMap.keys()).sort((a, b) => a.localeCompare(b));
        names.forEach((name) => {
            if (this.selectedEmbeddings.has(name)) return;
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            this.select.appendChild(opt);
        });

        if (currentValue && !this.selectedEmbeddings.has(currentValue) && this.availableEmbeddingsMap.has(currentValue)) {
            this.select.value = currentValue;
        }
    }

    addEmbedding(name, token = null, options = {}) {
        const { persist = true, silent = false } = options;
        if (this.selectedEmbeddings.has(name)) {
            if (!silent) {
                toast(`${name} already added`, 'info');
            }
            return;
        }

        const info = this.availableEmbeddingsMap.get(name);
        const resolvedToken = String(token || info?.token || name);

        this.selectedEmbeddings.set(name, resolvedToken);
        this.renderCard(name, resolvedToken);
        this.refreshSelectOptions();
        if (persist) {
            this.persistSelection();
        }
    }

    removeEmbedding(name) {
        this.selectedEmbeddings.delete(name);
        const card = this.container?.querySelector(`.embedding-card[data-embedding="${CSS.escape(name)}"]`);
        if (card) card.remove();
        this.refreshSelectOptions();
        this.persistSelection();
    }

    renderCard(name, token) {
        if (!this.container) return;
        const safeName = String(name).replace(/"/g, '&quot;');
        const safeToken = String(token).replace(/"/g, '&quot;');

        const card = document.createElement('div');
        card.className = 'lora-card embedding-card';
        card.dataset.embedding = name;
        card.innerHTML = `
            <div class="lora-card-header">
                <span class="lora-card-name">${safeName}</span>
                <button class="btn btn-ghost btn-icon btn-sm embedding-remove" title="Remove">✕</button>
            </div>
            <div class="embedding-token-row">
                <label class="form-label" for="">Token</label>
                <input type="text" class="input embedding-token" value="${safeToken}" />
            </div>
        `;

        const tokenInput = card.querySelector('.embedding-token');
        tokenInput?.addEventListener('input', () => {
            const nextToken = String(tokenInput.value || '').trim();
            this.selectedEmbeddings.set(name, nextToken || name);
            this.persistSelection();
        });

        card.querySelector('.embedding-remove')?.addEventListener('click', () => {
            this.removeEmbedding(name);
        });

        this.container.appendChild(card);
    }

    getSelected() {
        return Array.from(this.selectedEmbeddings.entries()).map(([name, token]) => ({ name, token }));
    }

    clear() {
        this.selectedEmbeddings.clear();
        if (this.container) this.container.innerHTML = '';
        this.refreshSelectOptions();
        this.persistSelection();
    }

    persistSelection() {
        setState({ selectedEmbeddings: this.getSelected() });
    }

    restoreFromState() {
        if (this.selectedEmbeddings.size > 0 || (this.container && this.container.children.length > 0)) {
            return;
        }

        const saved = getState('selectedEmbeddings');
        if (!Array.isArray(saved) || saved.length === 0) {
            return;
        }

        saved.forEach((entry) => {
            const name = entry?.name;
            if (!name || !this.availableEmbeddingsMap.has(name)) {
                return;
            }
            const token = String(entry?.token || this.availableEmbeddingsMap.get(name)?.token || name);
            this.addEmbedding(name, token, { persist: false, silent: true });
        });
    }
}
