/**
 * GalleryManager Module
 * Flat gallery rendering with infinite scroll and thumbnail size control.
 */

import * as api from '../core/api.js';
import { byId, listen, show, hide, toast } from '../core/utils.js';

const THUMB_SIZE_KEY = 'webbduck_gallery_thumb_size';

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
        this.hasMore = true;
        this.isLoadingPage = false;
        this.observer = null;
        this.thumbSize = this.loadThumbSize();

        this.load = this.load.bind(this);
        this.refreshLatest = this.refreshLatest.bind(this);
        this.render = this.render.bind(this);
        this.handleDelete = this.handleDelete.bind(this);
        this.handleFavoriteToggle = this.handleFavoriteToggle.bind(this);
    }

    init() {
        this.setupSearch();
        this.setupRefresh();
        this.setupFilters();
        this.setupThumbSizeControl();
        this.setupInfiniteScroll();
    }

    loadThumbSize() {
        const raw = Number(localStorage.getItem(THUMB_SIZE_KEY) || 240);
        if (!Number.isFinite(raw)) return 240;
        return Math.max(140, Math.min(420, Math.round(raw)));
    }

    saveThumbSize(size) {
        localStorage.setItem(THUMB_SIZE_KEY, String(size));
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
                this.searchData = null;
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

    setupThumbSizeControl() {
        const slider = byId('gallery-thumb-size');
        const valueEl = byId('gallery-thumb-size-value');
        if (!slider || !valueEl) return;
        slider.value = String(this.thumbSize);
        valueEl.textContent = `${this.thumbSize}px`;
        listen(slider, 'input', () => {
            const next = Math.max(140, Math.min(420, Number(slider.value || this.thumbSize)));
            this.thumbSize = Math.round(next);
            valueEl.textContent = `${this.thumbSize}px`;
            this.saveThumbSize(this.thumbSize);
            this.render(byId('gallery-search')?.value || '');
        });
    }

    setupInfiniteScroll() {
        const root = byId('gallery-content');
        if (!root) return;
        this.observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting) return;
                if (!this.shouldAutoLoadMore()) return;
                this.loadMoreAuto();
            });
        }, { root, threshold: 0.1 });
    }

    bindScrollSentinel() {
        if (!this.observer) return;
        this.observer.disconnect();
        const sentinel = byId('gallery-scroll-sentinel');
        if (sentinel) this.observer.observe(sentinel);
    }

    shouldAutoLoadMore() {
        const term = (byId('gallery-search')?.value || '').trim();
        return this.activeFilter === 'all' && !term && this.hasMore && !this.isLoadingPage;
    }

    async load() {
        const btn = byId('refresh-gallery');
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '↻ Loading...';
        }

        try {
            this.page = 0;
            this.data = [];
            this.fullData = [];
            this.hasMore = true;
            this.searchData = null;
            this.filterData = null;
            this.filterCache.clear();

            await this.fetchPage();
            await this.ensureFilterData(this.activeFilter);
            this.render(byId('gallery-search')?.value || '');
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
            this.render(byId('gallery-search')?.value || '');
        } catch (error) {
            console.error('Failed to refresh latest gallery items:', error);
        }
    }

    async fetchPage() {
        const start = this.page * this.SESSIONS_PER_PAGE;
        this.isLoadingPage = true;
        try {
            const data = await api.getGallery(start, this.SESSIONS_PER_PAGE);
            const items = Array.isArray(data) ? data : (data.sessions || []);

            if (items.length < this.SESSIONS_PER_PAGE) this.hasMore = false;
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
            return items.length;
        } catch (e) {
            console.error('Fetch page error:', e);
            toast('Failed to load more images', 'error');
            return 0;
        } finally {
            this.isLoadingPage = false;
        }
    }

    async loadMoreAuto() {
        if (!this.shouldAutoLoadMore()) return;
        this.page += 1;
        const loaded = await this.fetchPage();
        if (!loaded) this.hasMore = false;
        this.render(byId('gallery-search')?.value || '');
    }

    flattenSessions(sessions) {
        const flat = [];
        for (const session of (sessions || [])) {
            const images = session.images || [];
            const variants = session.variants || {};
            const favorites = session.favorites || {};
            const meta = session.meta || {};
            const prompt = meta.prompt || session.prompt || 'No prompt';
            const timestamp = Number(meta.timestamp || session.timestamp || Date.now() / 1000);

            images.forEach((imgPath, idx) => {
                const filename = (imgPath || '').split('/').pop() || '';
                flat.push({
                    src: imgPath,
                    width: Number(meta.width || 1024),
                    height: Number(meta.height || 1024),
                    variant: variants[filename] || null,
                    favorite: Boolean(favorites[filename]),
                    prompt,
                    timestamp,
                    run: session.run,
                    meta,
                    idx,
                });
            });
        }
        return flat;
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
        if (this.currentSearchTerm && sourceData == null) {
            filteredData = filteredData.filter((session) => this.matchesSearch(session, this.currentSearchTerm));
        }

        const flatImages = this.flattenSessions(filteredData);
        if (!flatImages.length) {
            hide(container);
            show(emptyState);
            countEl.textContent = '0 images';
            const emptyTitle = byId('gallery-empty-title');
            const emptyText = byId('gallery-empty-text');
            if (this.currentSearchTerm) {
                if (emptyTitle) emptyTitle.textContent = 'No matching images';
                if (emptyText) emptyText.textContent = 'Try fewer keywords or clear filters.';
            } else {
                if (emptyTitle) emptyTitle.textContent = 'No images yet';
                if (emptyText) emptyText.textContent = 'Generate images in Studio and they will appear here.';
            }
            return;
        }

        show(container);
        hide(emptyState);
        countEl.textContent = `${flatImages.length} image${flatImages.length !== 1 ? 's' : ''}`;

        const loadingTail = this.shouldAutoLoadMore()
            ? `<div class="gallery-tail">${this.isLoadingPage ? 'Loading more...' : 'Scroll for more'}</div>`
            : '';

        container.innerHTML = `
            <div class="gallery-grid" style="--thumb-size:${this.thumbSize}px">
                ${flatImages.map((item, i) => this.renderImageItem(item, i)).join('')}
            </div>
            ${loadingTail}
            <div id="gallery-scroll-sentinel" class="gallery-scroll-sentinel"></div>
        `;

        this.attachListeners(container);
        this.bindScrollSentinel();
    }

    renderImageItem(item, index) {
        const thumbUrl = `/thumbs/${item.src}`;
        const safeMeta = encodeURIComponent(JSON.stringify(item.meta || {}));
        const ratio = Math.max(0.5, Math.min(2.2, item.height / Math.max(1, item.width)));
        return `
            <div class="image-item flat-item" data-src="${item.src}" data-index="${index}" data-width="${item.width}" data-height="${item.height}" data-favorite="${item.favorite ? '1' : '0'}" data-meta="${safeMeta}" ${item.variant ? `data-variant="${item.variant}"` : ''}>
                <img src="${thumbUrl}" alt="${this.escapeHtml(item.prompt)}" loading="lazy" style="aspect-ratio:${1 / ratio};" />
                ${item.variant ? '<span class="hd-badge">HD</span>' : ''}
                ${item.favorite ? '<span class="favorite-badge">♥</span>' : ''}
                <button class="image-favorite-btn ${item.favorite ? 'is-favorite' : ''}" type="button" title="${item.favorite ? 'Unfavorite' : 'Favorite'}" aria-label="${item.favorite ? 'Unfavorite image' : 'Favorite image'}">♥</button>
            </div>
        `;
    }

    attachListeners(container) {
        container.querySelectorAll('.image-item').forEach((item) => {
            listen(item, 'click', (e) => {
                if (e) e.stopPropagation();
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
            loras,
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
        const maxPages = 120;
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
            out.push({
                ...session,
                images: kept,
                variants: Object.fromEntries(
                    Object.entries(variants).filter(([name]) => kept.some((p) => p.endsWith(`/${name}`))),
                ),
                favorites: Object.fromEntries(
                    Object.entries(favorites).filter(([name, v]) => v && kept.some((p) => p.endsWith(`/${name}`))),
                ),
            });
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
                if (favorite) session.favorites[file] = true;
                else delete session.favorites[file];
            });
        };
        mark(this.data);
        mark(this.searchData);
        mark(this.fullData);
        mark(this.filterData);
    }

    async handleDelete(src, type) {
        try {
            this.removeImageInMemory(src, type);
            this.render(byId('gallery-search')?.value || '');
            toast(type === 'run' ? 'Run deleted' : 'Image deleted', 'success');

            const formData = new FormData();
            formData.append('path', src);
            const endpoint = type === 'run' ? '/delete_run' : '/delete_image';
            fetch(endpoint, { method: 'POST', body: formData }).then((res) => {
                if (!res.ok) {
                    console.error('Background delete failed');
                    toast('Failed to delete on server', 'error');
                }
            }).catch((e) => console.error('Delete network error:', e));
            return true;
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
                    if (src.includes(runPrefixA) || src.includes(runPrefixB)) continue;
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
}
