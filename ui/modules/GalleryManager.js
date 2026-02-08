/**
 * GalleryManager Module
 * Handles fetching, rendering, and managing the gallery view.
 */

import * as api from '../core/api.js';
import { byId, listen, show, hide, toast } from '../core/utils.js';

export class GalleryManager {
    constructor() {
        this.data = [];
        this.page = 0;
        this.SESSIONS_PER_PAGE = 30;
        this.SEARCH_SESSIONS = 600;
        this.currentSearchTerm = '';
        this.searchData = null;

        // Bind methods
        this.load = this.load.bind(this);
        this.refreshLatest = this.refreshLatest.bind(this);
        this.render = this.render.bind(this);
        this.loadMore = this.loadMore.bind(this);
        this.handleDelete = this.handleDelete.bind(this);
    }

    init() {
        this.setupSearch();
        this.setupRefresh();
    }

    setupRefresh() {
        listen(byId('refresh-gallery'), 'click', () => this.load());
    }

    setupSearch() {
        const searchInput = byId('gallery-search');
        if (!searchInput) return;

        let timeout;
        listen(searchInput, 'input', (e) => {
            const val = e.target.value;
            clearTimeout(timeout);
            timeout = setTimeout(async () => {
                this.page = 0;
                if (val.trim()) {
                    await this.ensureSearchData();
                    this.render(val, this.searchData || this.data);
                    return;
                }

                this.render('');
            }, 300);
        });
    }

    async load() {
        const btn = byId('refresh-gallery');
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '↻ Loading...';
        }

        try {
            this.page = 0;
            this.data = []; // Clear existing
            this.hasMore = true;
            this.searchData = null;

            await this.fetchPage();

            toast('Gallery refreshed', 'success');
        } catch (error) {
            console.error('Failed to load gallery:', error);
            toast('Failed to load gallery', 'error');
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = 'Refresh';
            }
        }
    }

    async refreshLatest() {
        try {
            const data = await api.getGallery(0, this.SESSIONS_PER_PAGE);
            const items = Array.isArray(data) ? data : (data.sessions || []);

            this.data = items;
            this.page = 0;
            this.hasMore = items.length >= this.SESSIONS_PER_PAGE;
            this.searchData = null;

            const searchTerm = byId('gallery-search')?.value || '';
            this.render(searchTerm, searchTerm.trim() ? (this.searchData || this.data) : this.data);
        } catch (error) {
            console.error('Failed to refresh latest gallery items:', error);
        }
    }

    async fetchPage() {
        const start = this.page * this.SESSIONS_PER_PAGE;
        try {
            const data = await api.getGallery(start, this.SESSIONS_PER_PAGE);
            const items = Array.isArray(data) ? data : (data.sessions || []);

            if (items.length < this.SESSIONS_PER_PAGE) {
                this.hasMore = false;
            }

            if (this.page === 0) {
                this.data = items;
            } else {
                this.data = [...this.data, ...items];
            }
            this.searchData = null;

            const searchTerm = byId('gallery-search')?.value || '';
            this.render(searchTerm, searchTerm.trim() ? (this.searchData || this.data) : this.data);
            return items.length;
        } catch (e) {
            console.error('Fetch page error:', e);
            toast('Failed to load more images', 'error');
            return 0;
        }
    }

    async ensureSearchData() {
        if (this.searchData) return;
        try {
            const data = await api.getGallery(0, this.SEARCH_SESSIONS);
            const items = Array.isArray(data) ? data : (data.sessions || []);
            this.searchData = items;
        } catch (error) {
            console.error('Failed to load search dataset:', error);
            this.searchData = this.data;
        }
    }

    matchesSearch(session, term) {
        const meta = session?.meta || {};
        const loras = Array.isArray(meta.loras)
            ? meta.loras.map((l) => (typeof l === 'string' ? l : (l?.name || l?.model || ''))).join(' ')
            : '';
        const parts = [
            session?.prompt,
            meta?.prompt,
            meta?.negative,
            meta?.negative_prompt,
            meta?.base_model,
            meta?.model,
            meta?.scheduler,
            meta?.seed,
            meta?.width && meta?.height ? `${meta.width}x${meta.height}` : '',
            loras
        ];
        const haystack = parts.filter(Boolean).join(' ').toLowerCase();
        return haystack.includes(term);
    }

    render(filterText = '', sourceData = null) {
        this.currentSearchTerm = filterText.toLowerCase();
        const container = byId('gallery-sessions');
        const emptyState = byId('gallery-empty');
        const countEl = byId('gallery-count');

        let filteredData = sourceData || this.data;
        if (this.currentSearchTerm) {
            filteredData = filteredData.filter(session => this.matchesSearch(session, this.currentSearchTerm));
        }

        if (!filteredData.length) {
            hide(container);
            show(emptyState);
            countEl.textContent = '0 images';

            // Update empty message if searching
            if (this.currentSearchTerm && byId('gallery-empty')) {
                const emptyInner = byId('gallery-empty').querySelector('.gallery-empty-content');
                if (emptyInner) { // Only update if inner exists, or create it?
                    // Simplified: just update text if needed, or leave as is. 
                    // Actually, let's just make sure we show the empty state.
                }
            }
            return;
        }

        show(container);
        hide(emptyState);

        // Count total images
        const totalImages = filteredData.reduce((sum, session) => sum + (session.images?.length || 0), 0);
        countEl.textContent = `${totalImages} image${totalImages !== 1 ? 's' : ''}`;

        // Render sessions
        container.innerHTML = filteredData.map(session => this.renderSession(session)).join('');

        // Add "Load More" button if needed
        if (!this.currentSearchTerm && this.hasMore) {
            const loadMoreDiv = document.createElement('div');
            loadMoreDiv.className = 'gallery-load-more';
            loadMoreDiv.innerHTML = `
                <button class="btn btn-secondary" id="load-more-btn">
                    Load More
                </button>
            `;
            container.appendChild(loadMoreDiv);
            listen(loadMoreDiv.querySelector('#load-more-btn'), 'click', this.loadMore);
        }

        this.attachListeners(container);
    }

    async loadMore() {
        if (!this.hasMore) return;
        const btn = byId('load-more-btn');
        if (btn) {
            btn.innerText = 'Loading...';
            btn.disabled = true;
        }

        this.page++;
        await this.fetchPage();
    }

    renderSession(session) {
        const images = session.images || [];
        const variants = session.variants || {};
        const meta = session.meta || {};
        const prompt = meta.prompt || session.prompt || 'No prompt';
        const model = meta.base_model || session.model || 'Unknown';
        const seed = meta.seed || session.seed || '-';
        const timestamp = meta.timestamp || session.timestamp || Date.now() / 1000;

        const sessionJson = encodeURIComponent(JSON.stringify(session));

        return `
        <div class="session-group" data-json="${sessionJson}">
          <div class="session-header">
            <div class="session-header-left">
              <div class="session-expand">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <polyline points="6 9 12 15 18 9"></polyline>
                </svg>
              </div>
              <div class="session-info">
                <div class="session-prompt">${this.escapeHtml(prompt.substring(0, 100))}</div>
                <div class="session-meta">
                  <span class="session-meta-item">${model}</span>
                  <span class="session-meta-item">Seed: ${seed}</span>
                  <span class="session-meta-item">${this.timeAgo(timestamp)}</span>
                </div>
              </div>
            </div>
            <div class="session-header-right">
              <button class="btn-icon session-delete-btn" title="Delete entire run">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <polyline points="3 6 5 6 21 6"></polyline>
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                </svg>
              </button>
              <span class="session-image-count">${images.length} image${images.length !== 1 ? 's' : ''}</span>
            </div>
          </div>
          <div class="session-body">
            <div class="image-grid">
              ${images.map((img, i) => {
            const url = img.url || img;
            const filename = url.split('/').pop();
            const variantUrl = variants[filename];

            // Use thumbnail endpoint
            // Backend handles generation if missing
            const thumbUrl = `/thumbs/${url}`;
            const width = meta.width || 1024;
            const height = meta.height || 1024;

            return `
                <div class="image-item" data-src="${url}" data-index="${i}" data-width="${width}" data-height="${height}" ${variantUrl ? `data-variant="${variantUrl}"` : ''}>
                  <img src="${thumbUrl}" alt="Generated image" loading="lazy" />
                  ${variantUrl ? '<span class="hd-badge">HD</span>' : ''}
                </div>
              `;
        }).join('')}
            </div>
          </div>
        </div>
      `;
    }

    attachListeners(container) {
        // Expand/Collapse headers
        container.querySelectorAll('.session-header').forEach(header => {
            listen(header, 'click', (e) => {
                if (!e.target.closest('button')) {
                    header.closest('.session-group').classList.toggle('collapsed');
                }
            });
        });

        // Delete handlers
        container.querySelectorAll('.session-delete-btn').forEach(btn => {
            listen(btn, 'click', (e) => {
                e.stopPropagation();
                const group = btn.closest('.session-group');
                const img = group.querySelector('img');
                if (img && window.lightboxManager) {
                    window.lightboxManager.confirmRunDelete(img.src);
                }
            });
        });

        // Image Click (Lightbox)
        container.querySelectorAll('.image-item').forEach((item) => {
            listen(item, 'click', (e) => {
                if (e) e.stopPropagation();
                // Get all visible images for lightbox context
                const allImages = Array.from(container.querySelectorAll('.image-item img'));
                const clickedImg = item.querySelector('img');
                const realIndex = allImages.indexOf(clickedImg);
                if (window.lightboxManager) {
                    window.lightboxManager.open(allImages, realIndex);
                }
            });
        });
    }

    /**
     * Handle deletion request (called by LightboxManager)
     * Performs Optimistic UI updates
     */
    async handleDelete(src, type) {
        try {
            // Optimistic UI: Remove from grid instantly
            if (type === 'image') {
                const items = document.querySelectorAll('.image-item');
                let foundFn = false;

                items.forEach(item => {
                    const img = item.querySelector('img');
                    // Match loosely to handle relative/absolute paths
                    if ((item.dataset.src && src.includes(item.dataset.src)) ||
                        (img && img.src === src)) {

                        item.remove();
                        foundFn = true;

                        // Update group count / remove group
                        const group = item.closest('.session-group');
                        if (group) {
                            const remaining = group.querySelectorAll('.image-item');
                            if (remaining.length === 0) {
                                group.remove();
                            } else {
                                const countEl = group.querySelector('.session-image-count');
                                if (countEl) countEl.textContent = `${remaining.length} image${remaining.length !== 1 ? 's' : ''}`;
                            }
                        }
                    }
                });
            } else if (type === 'run') {
                const items = document.querySelectorAll('.image-item');
                for (let item of items) {
                    const img = item.querySelector('img');
                    if (img && img.src === src) {
                        const group = item.closest('.session-group');
                        if (group) group.remove();
                        break;
                    }
                }
            }

            toast(type === 'run' ? 'Run deleted' : 'Image deleted', 'success');

            // Background Request
            const formData = new FormData();
            formData.append('path', src);
            const endpoint = type === 'run' ? '/delete_run' : '/delete_image';

            fetch(endpoint, { method: 'POST', body: formData }).then(res => {
                if (!res.ok) {
                    console.error('Background delete failed');
                    toast('Failed to delete on server', 'error');
                }
            }).catch(e => console.error('Delete network error:', e));

            return true; // Return success immediately
        } catch (e) {
            console.error('Delete error:', e);
            return false;
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    timeAgo(timestamp) {
        const seconds = Math.floor((Date.now() - timestamp * 1000) / 1000);
        if (seconds < 60) return 'Just now';
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
        if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
        return `${Math.floor(seconds / 86400)}d ago`;
    }
}
