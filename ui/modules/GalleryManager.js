/**
 * GalleryManager Module
 * Handles fetching, rendering, and managing the gallery view.
 */

import * as api from '../core/api.js';
import { byId, listen, show, hide, toast } from '../core/utils.js';

export class GalleryManager {
    constructor() {
        this.data = [];
        this.fullData = [];
        this.page = 0;
        this.SESSIONS_PER_PAGE = 30;
        this.currentSearchTerm = '';
        this.searchData = null;
        this.searchRequestId = 0;
        this.activeFilter = 'all';
        this.filterData = null;
        this.filterCache = new Map();

        // Bind methods
        this.load = this.load.bind(this);
        this.refreshLatest = this.refreshLatest.bind(this);
        this.render = this.render.bind(this);
        this.loadMore = this.loadMore.bind(this);
        this.handleDelete = this.handleDelete.bind(this);
        this.handleFavoriteToggle = this.handleFavoriteToggle.bind(this);
    }

    init() {
        this.setupSearch();
        this.setupRefresh();
        this.setupFilters();
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
                    const requestId = ++this.searchRequestId;
                    try {
                        const data = await api.searchGallery(val.trim(), 0, 2000);
                        if (requestId !== this.searchRequestId) return;
                        this.searchData = Array.isArray(data) ? data : (data.sessions || []);
                        if (!this.searchData.length) {
                            this.searchData = await this.searchLocalKeywordFallback(val.trim());
                            if (requestId !== this.searchRequestId) return;
                        }
                        this.hasMore = false;
                        this.render(val, this.searchData);
                    } catch (error) {
                        console.error('Search failed:', error);
                        toast('Search failed', 'error');
                    }
                    return;
                }

                this.searchRequestId++;
                this.render('');
            }, 300);
        });
    }

    setupFilters() {
        const chips = document.querySelectorAll('.gallery-filter-chip');
        chips.forEach((chip) => {
            listen(chip, 'click', async () => {
                const next = chip.dataset.filter || 'all';
                if (this.activeFilter === next) return;
                this.activeFilter = next;
                chips.forEach((c) => c.classList.toggle('active', c === chip));
                await this.ensureFilterData(next);
                const term = byId('gallery-search')?.value || '';
                this.render(term);
            });
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
            this.fullData = [];
            this.hasMore = true;
            this.searchData = null;
            this.filterData = null;
            this.filterCache.clear();

            await this.fetchPage();
            await this.ensureFilterData(this.activeFilter);

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
            this.fullData = items;
            this.page = 0;
            this.hasMore = items.length >= this.SESSIONS_PER_PAGE;
            this.searchData = null;
            this.filterData = null;
            this.filterCache.clear();

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
                this.fullData = items;
            } else {
                this.data = [...this.data, ...items];
                this.fullData = [...this.data];
            }
            this.searchData = null;
            this.filterData = null;
            this.filterCache.clear();

            const searchTerm = byId('gallery-search')?.value || '';
            this.render(searchTerm, searchTerm.trim() ? (this.searchData || this.data) : this.data);
            return items.length;
        } catch (e) {
            console.error('Fetch page error:', e);
            toast('Failed to load more images', 'error');
            return 0;
        }
    }

    matchesSearch(session, term) {
        const terms = this.tokenizeSearch(term);
        if (!terms.length) return true;

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
        return terms.every((t) => haystack.includes(t));
    }

    tokenizeSearch(term) {
        return String(term || '')
            .toLowerCase()
            .split(/\s+/)
            .map((t) => t.trim())
            .filter(Boolean);
    }

    async searchLocalKeywordFallback(term) {
        const pageSize = Math.max(50, this.SESSIONS_PER_PAGE || 30);
        const maxPages = 120; // Hard cap to avoid runaway loops.
        const all = [];
        let page = 0;

        while (page < maxPages) {
            const start = page * pageSize;
            let chunk = [];
            try {
                const data = await api.getGallery(start, pageSize);
                chunk = Array.isArray(data) ? data : (data.sessions || []);
            } catch (_) {
                break;
            }

            if (!chunk.length) break;
            all.push(...chunk);
            if (chunk.length < pageSize) break;
            page += 1;
        }

        if (!all.length) return [];
        return all.filter((session) => this.matchesSearch(session, term));
    }

    render(filterText = '', sourceData = null) {
        this.currentSearchTerm = filterText.toLowerCase();
        const container = byId('gallery-sessions');
        const emptyState = byId('gallery-empty');
        const countEl = byId('gallery-count');

        let baseData = sourceData || this.getActiveFilterData();
        if (sourceData && this.activeFilter !== 'all') {
            baseData = this.applyImageFilter(sourceData, this.activeFilter);
        }
        let filteredData = baseData;
        const shouldClientFilter = this.currentSearchTerm && sourceData == null;
        if (shouldClientFilter) {
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
        if (!this.currentSearchTerm && this.hasMore && this.activeFilter === 'all') {
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
        if (this.activeFilter !== 'all') {
            await this.ensureFilterData(this.activeFilter);
            this.render(byId('gallery-search')?.value || '');
        }
    }

    renderSession(session) {
        const images = session.images || [];
        const variants = session.variants || {};
        const favorites = session.favorites || {};
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
                <div class="image-item" data-src="${url}" data-index="${i}" data-width="${width}" data-height="${height}" data-favorite="${favorites[filename] ? '1' : '0'}" ${variantUrl ? `data-variant="${variantUrl}"` : ''}>
                  <img src="${thumbUrl}" alt="Generated image" loading="lazy" />
                  ${variantUrl ? '<span class="hd-badge">HD</span>' : ''}
                  ${favorites[filename] ? '<span class="favorite-badge">♥</span>' : ''}
                  <button class="image-favorite-btn ${favorites[filename] ? 'is-favorite' : ''}" type="button" title="${favorites[filename] ? 'Unfavorite' : 'Favorite'}" aria-label="${favorites[filename] ? 'Unfavorite image' : 'Favorite image'}">♥</button>
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

        container.querySelectorAll('.image-favorite-btn').forEach((btn) => {
            listen(btn, 'click', async (e) => {
                e.preventDefault();
                e.stopPropagation();
                const item = btn.closest('.image-item');
                if (!item) return;
                const src = item.dataset.src;
                if (!src) return;
                await this.handleFavoriteToggle(src, item.dataset.favorite === '1');
            });
        });
    }

    getActiveFilterData() {
        if (this.activeFilter === 'all') return this.data;
        return this.filterData || [];
    }

    async ensureFilterData(filter) {
        if (filter === 'all') {
            this.filterData = null;
            return;
        }
        if (this.filterCache.has(filter)) {
            this.filterData = this.filterCache.get(filter) || [];
            return;
        }
        const data = await api.filterGallery(filter, 0, 5000);
        const sessions = Array.isArray(data) ? data : (data.sessions || []);
        this.filterData = sessions;
        this.filterCache.set(filter, sessions);
    }

    async loadAllSessions() {
        const pageSize = Math.max(80, this.SESSIONS_PER_PAGE);
        const maxPages = 160;
        const all = [];
        for (let page = 0; page < maxPages; page++) {
            const start = page * pageSize;
            let chunk = [];
            try {
                const data = await api.getGallery(start, pageSize);
                chunk = Array.isArray(data) ? data : (data.sessions || []);
            } catch (_) {
                break;
            }
            if (!chunk.length) break;
            all.push(...chunk);
            if (chunk.length < pageSize) break;
        }
        this.fullData = all;
        return all;
    }

    applyImageFilter(sessions, filter) {
        const out = [];
        for (const session of (sessions || [])) {
            const images = session.images || [];
            const variants = session.variants || {};
            const favorites = session.favorites || {};
            const kept = [];
            for (const imgPath of images) {
                const filename = (imgPath || '').split('/').pop();
                if (!filename) continue;
                const hasHd = Boolean(variants[filename]);
                const isFav = Boolean(favorites[filename]);
                if (filter === 'hd' && !hasHd) continue;
                if (filter === 'favorites' && !isFav) continue;
                kept.push(imgPath);
            }
            if (!kept.length) continue;
            const nextSession = {
                ...session,
                images: kept,
                variants: Object.fromEntries(
                    Object.entries(variants).filter(([name]) => kept.some((p) => p.endsWith(`/${name}`)))
                ),
                favorites: Object.fromEntries(
                    Object.entries(favorites).filter(([name, v]) => v && kept.some((p) => p.endsWith(`/${name}`)))
                ),
            };
            out.push(nextSession);
        }
        return out;
    }

    async handleFavoriteToggle(src, currentlyFavorite) {
        try {
            const next = !currentlyFavorite;
            await api.setFavorite(src, next);
            this.updateFavoriteInMemory(src, next);
            this.filterCache.delete('favorites');
            this.render(byId('gallery-search')?.value || '');
            toast(next ? 'Added to favorites' : 'Removed from favorites', 'success');
            return true;
        } catch (e) {
            console.error('Favorite toggle error:', e);
            toast('Failed to update favorite', 'error');
            return false;
        }
    }

    updateFavoriteInMemory(src, favorite) {
        const mark = (sessions) => {
            (sessions || []).forEach((session) => {
                const images = session.images || [];
                const idx = images.findIndex((p) => src.includes(p));
                if (idx < 0) return;
                const file = images[idx].split('/').pop();
                if (!file) return;
                session.favorites = session.favorites || {};
                if (favorite) {
                    session.favorites[file] = true;
                } else {
                    delete session.favorites[file];
                }
            });
        };
        mark(this.data);
        mark(this.searchData);
        mark(this.fullData);
        mark(this.filterData);
    }

    /**
     * Handle deletion request (called by LightboxManager)
     * Performs Optimistic UI updates
     */
    async handleDelete(src, type) {
        try {
            this.removeImageInMemory(src, type);
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

    removeImageInMemory(src, type) {
        const prune = (sessions) => {
            if (!Array.isArray(sessions)) return sessions;
            const next = [];
            for (const s of sessions) {
                const allImages = Array.isArray(s.images) ? s.images : [];
                let kept = [];
                if (type === 'run') {
                    const runPrefixA = `outputs/${s.run}/`;
                    const runPrefixB = `/outputs/${s.run}/`;
                    if (src.includes(runPrefixA) || src.includes(runPrefixB)) {
                        continue;
                    }
                    kept = [...allImages];
                } else {
                    kept = allImages.filter((p) => !src.includes(p));
                }
                if (!kept.length) continue;

                const variants = { ...(s.variants || {}) };
                const favorites = { ...(s.favorites || {}) };
                Object.keys(variants).forEach((name) => {
                    if (!kept.some((p) => p.endsWith(`/${name}`))) delete variants[name];
                });
                Object.keys(favorites).forEach((name) => {
                    if (!kept.some((p) => p.endsWith(`/${name}`))) delete favorites[name];
                });
                next.push({ ...s, images: kept, variants, favorites });
            }
            return next;
        };
        this.data = prune(this.data);
        this.searchData = prune(this.searchData);
        this.fullData = prune(this.fullData);
        this.filterData = prune(this.filterData);
        this.filterCache.clear();
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
