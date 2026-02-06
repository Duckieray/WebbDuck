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
        this.currentSearchTerm = '';

        // Bind methods
        this.load = this.load.bind(this);
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
            timeout = setTimeout(() => {
                this.page = 0;
                this.render(val);
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
            // Fetch gallery data (limit 50 by default from backend)
            const data = await api.getGallery();

            // API returns array directly, not {sessions: [...]}
            this.data = Array.isArray(data) ? data : (data.sessions || []);

            // Sort newest first
            this.data.sort((a, b) => {
                const tsA = (a.meta?.timestamp || a.timestamp || 0);
                const tsB = (b.meta?.timestamp || b.timestamp || 0);
                return tsB - tsA;
            });

            // Render with current search term
            const searchTerm = byId('gallery-search')?.value || '';
            this.render(searchTerm);

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

    render(filterText = '') {
        this.currentSearchTerm = filterText.toLowerCase();
        const container = byId('gallery-sessions');
        const emptyState = byId('gallery-empty');
        const countEl = byId('gallery-count');
        const emptyMsg = byId('gallery-empty-msg'); // Assuming inner container

        // Filter data
        let filteredData = this.data;
        if (this.currentSearchTerm) {
            filteredData = this.data.filter(session => {
                const prompt = (session.meta?.prompt || session.prompt || '').toLowerCase();
                return prompt.includes(this.currentSearchTerm);
            });
        }

        if (!filteredData.length) {
            hide(container);
            show(emptyState);
            countEl.textContent = '0 images';

            // Update empty message if searching
            if (this.currentSearchTerm && byId('gallery-empty')) {
                byId('gallery-empty').innerHTML = `
                    <div class="gallery-empty-content">
                        <h3>No matches found</h3>
                        <p>Try a different search term</p>
                    </div>
                `;
            }
            return;
        }

        show(container);
        hide(emptyState);

        // Count total images
        const totalImages = filteredData.reduce((sum, session) => sum + (session.images?.length || 0), 0);
        countEl.textContent = `${totalImages} image${totalImages !== 1 ? 's' : ''}`;

        // Render only current page of sessions
        const startIdx = 0;
        const endIdx = (this.page + 1) * this.SESSIONS_PER_PAGE;
        const sessionsToRender = filteredData.slice(startIdx, endIdx);

        container.innerHTML = sessionsToRender.map(session => this.renderSession(session)).join('');

        // Add "Load More" button if needed
        if (endIdx < filteredData.length) {
            const remaining = filteredData.length - endIdx;
            const loadMoreDiv = document.createElement('div');
            loadMoreDiv.className = 'gallery-load-more';
            loadMoreDiv.innerHTML = `
                <button class="btn btn-secondary" id="load-more-btn">
                    Load More (${remaining} more sessions)
                </button>
            `;
            container.appendChild(loadMoreDiv);
            listen(loadMoreDiv.querySelector('#load-more-btn'), 'click', this.loadMore);
        }

        this.attachListeners(container);
    }

    loadMore() {
        this.page++;
        this.render(this.currentSearchTerm);
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

            return `
                <div class="image-item" data-src="${url}" data-index="${i}" ${variantUrl ? `data-variant="${variantUrl}"` : ''}>
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
