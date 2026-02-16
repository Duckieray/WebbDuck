/**
 * GalleryManager Module
 * Flat gallery rendering with infinite scroll and thumbnail size control.
 */

import * as api from '../core/api.js';
import { byId, listen, show, hide, toast } from '../core/utils.js';

const THUMB_SIZE_KEY = 'webbduck_gallery_thumb_size';
const MOBILE_CONTROLS_KEY = 'webbduck_gallery_controls_collapsed';

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
        this.selectionMode = false;
        this.selectedPaths = new Set();
        this.mobileControlsCollapsed = this.loadMobileControlsCollapsed();

        this.load = this.load.bind(this);
        this.refreshLatest = this.refreshLatest.bind(this);
        this.render = this.render.bind(this);
        this.handleDelete = this.handleDelete.bind(this);
        this.handleFavoriteToggle = this.handleFavoriteToggle.bind(this);
        this.toggleSelectionMode = this.toggleSelectionMode.bind(this);
        this.updateSelectionUI = this.updateSelectionUI.bind(this);
        this.openBatchDeleteModal = this.openBatchDeleteModal.bind(this);
        this.closeBatchDeleteModal = this.closeBatchDeleteModal.bind(this);
        this.confirmBatchDelete = this.confirmBatchDelete.bind(this);
        this.toggleMobileControls = this.toggleMobileControls.bind(this);
        this.applyMobileControlsState = this.applyMobileControlsState.bind(this);
    }

    init() {
        this.setupSearch();
        this.setupRefresh();
        this.setupFilters();
        this.setupThumbSizeControl();
        this.setupInfiniteScroll();
        this.setupBatchDeleteControls();
        this.setupMobileControlsToggle();
    }

    loadThumbSize() {
        const raw = Number(localStorage.getItem(THUMB_SIZE_KEY) || 240);
        if (!Number.isFinite(raw)) return 240;
        return Math.max(140, Math.min(420, Math.round(raw)));
    }

    saveThumbSize(size) {
        localStorage.setItem(THUMB_SIZE_KEY, String(size));
    }

    loadMobileControlsCollapsed() {
        const saved = localStorage.getItem(MOBILE_CONTROLS_KEY);
        if (saved === '1') return true;
        if (saved === '0') return false;
        return window.matchMedia('(max-width: 860px)').matches;
    }

    saveMobileControlsCollapsed(value) {
        localStorage.setItem(MOBILE_CONTROLS_KEY, value ? '1' : '0');
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

    setupBatchDeleteControls() {
        const selectBtn = byId('gallery-select-toggle');
        const deleteBtn = byId('gallery-delete-selected');
        const modal = byId('gallery-batch-delete-modal');
        const cancelBtn = byId('gallery-batch-delete-cancel');
        const confirmBtn = byId('gallery-batch-delete-confirm');

        if (selectBtn) listen(selectBtn, 'click', () => this.toggleSelectionMode());
        if (deleteBtn) listen(deleteBtn, 'click', () => this.openBatchDeleteModal());
        if (cancelBtn) listen(cancelBtn, 'click', () => this.closeBatchDeleteModal());
        if (confirmBtn) listen(confirmBtn, 'click', () => this.confirmBatchDelete());
        if (modal) {
            listen(modal, 'click', (e) => {
                if (e.target === modal) this.closeBatchDeleteModal();
            });
        }

        this.updateSelectionUI();
    }

    setupMobileControlsToggle() {
        const btn = byId('gallery-controls-toggle');
        if (btn) listen(btn, 'click', this.toggleMobileControls);
        listen(window, 'resize', () => this.applyMobileControlsState());
        this.applyMobileControlsState();
    }

    toggleMobileControls() {
        this.mobileControlsCollapsed = !this.mobileControlsCollapsed;
        this.saveMobileControlsCollapsed(this.mobileControlsCollapsed);
        this.applyMobileControlsState();
    }

    applyMobileControlsState() {
        const layout = document.querySelector('#view-gallery .gallery-layout');
        const btn = byId('gallery-controls-toggle');
        const isMobile = window.matchMedia('(max-width: 860px)').matches;

        if (!layout) return;
        if (!isMobile) {
            layout.classList.remove('controls-collapsed');
            if (btn) btn.classList.add('hidden');
            return;
        }

        if (btn) {
            btn.classList.remove('hidden');
            btn.textContent = this.mobileControlsCollapsed ? 'Show Controls' : 'Hide Controls';
        }
        layout.classList.toggle('controls-collapsed', this.mobileControlsCollapsed);
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
            this.selectedPaths.clear();
            this.selectionMode = false;
            this.updateSelectionUI();

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
            this.selectedPaths.clear();
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

    toggleSelectionMode(forceValue = null) {
        if (typeof forceValue === 'boolean') this.selectionMode = forceValue;
        else this.selectionMode = !this.selectionMode;

        if (!this.selectionMode) {
            this.selectedPaths.clear();
            this.closeBatchDeleteModal();
        } else if (window.matchMedia('(max-width: 860px)').matches && this.mobileControlsCollapsed) {
            this.mobileControlsCollapsed = false;
            this.saveMobileControlsCollapsed(false);
            this.applyMobileControlsState();
        }
        this.updateSelectionUI();
        this.render(byId('gallery-search')?.value || '');
    }

    toggleImageSelection(src) {
        if (!src) return;
        if (this.selectedPaths.has(src)) this.selectedPaths.delete(src);
        else this.selectedPaths.add(src);
        this.updateSelectionUI();
        this.render(byId('gallery-search')?.value || '');
    }

    updateSelectionUI() {
        const selectBtn = byId('gallery-select-toggle');
        const deleteBtn = byId('gallery-delete-selected');
        const selectedCount = this.selectedPaths.size;

        if (selectBtn) {
            selectBtn.classList.toggle('is-active', this.selectionMode);
            selectBtn.textContent = this.selectionMode ? 'Cancel Select' : 'Select';
        }

        if (deleteBtn) {
            deleteBtn.classList.toggle('hidden', !this.selectionMode);
            deleteBtn.disabled = selectedCount === 0;
            deleteBtn.textContent = selectedCount > 0 ? `Delete (${selectedCount})` : 'Delete';
        }
    }

    openBatchDeleteModal() {
        if (!this.selectionMode || this.selectedPaths.size === 0) return;
        const modal = byId('gallery-batch-delete-modal');
        const msg = byId('gallery-batch-delete-message');
        if (!modal || !msg) return;
        const count = this.selectedPaths.size;
        msg.textContent = `Deleting ${count} image${count === 1 ? '' : 's'}, are you sure?`;
        modal.classList.remove('hidden');
        void modal.offsetWidth;
        setTimeout(() => modal.classList.add('active'), 10);
    }

    closeBatchDeleteModal() {
        const modal = byId('gallery-batch-delete-modal');
        if (!modal) return;
        modal.classList.remove('active');
        setTimeout(() => modal.classList.add('hidden'), 220);
    }

    async confirmBatchDelete() {
        if (!this.selectionMode || this.selectedPaths.size === 0) {
            this.closeBatchDeleteModal();
            return;
        }

        const paths = Array.from(this.selectedPaths);
        const confirmBtn = byId('gallery-batch-delete-confirm');
        const cancelBtn = byId('gallery-batch-delete-cancel');
        if (confirmBtn) confirmBtn.disabled = true;
        if (cancelBtn) cancelBtn.disabled = true;

        try {
            const result = await api.deleteImages(paths);
            const failedCount = Number(result?.failed_count || 0);
            const failedPaths = new Set(
                Array.isArray(result?.failed) ? result.failed.map((item) => item?.path).filter(Boolean) : [],
            );
            const removedPaths = paths.filter((path) => !failedPaths.has(path));
            const deletedCount = removedPaths.length;

            this.removeImagesInMemory(removedPaths);
            this.selectedPaths.clear();
            this.closeBatchDeleteModal();
            this.updateSelectionUI();
            this.render(byId('gallery-search')?.value || '');

            if (failedCount > 0) {
                toast(`Deleted ${deletedCount} image${deletedCount === 1 ? '' : 's'}, ${failedCount} failed`, 'error');
            } else {
                toast(`Deleted ${deletedCount} image${deletedCount === 1 ? '' : 's'}`, 'success');
            }
        } catch (error) {
            console.error('Batch delete failed:', error);
            toast('Batch delete failed', 'error');
        } finally {
            if (confirmBtn) confirmBtn.disabled = false;
            if (cancelBtn) cancelBtn.disabled = false;
        }
    }

    render(filterText = '', sourceData = null) {
        this.currentSearchTerm = filterText.toLowerCase();
        const container = byId('gallery-sessions');
        const emptyState = byId('gallery-empty');
        const countEl = byId('gallery-count');

        const hasExplicitSource = Array.isArray(sourceData);
        const hasSearchSource = !hasExplicitSource && Boolean(this.currentSearchTerm) && Array.isArray(this.searchData);

        let baseData = hasExplicitSource
            ? sourceData
            : (hasSearchSource ? this.searchData : this.getActiveFilterData());

        if ((hasExplicitSource || hasSearchSource) && this.activeFilter !== 'all') {
            baseData = this.applyImageFilter(baseData, this.activeFilter);
        }

        let filteredData = baseData;
        if (this.currentSearchTerm && !hasExplicitSource && !hasSearchSource) {
            filteredData = filteredData.filter((session) => this.matchesSearch(session, this.currentSearchTerm));
        }

        const flatImages = this.flattenSessions(filteredData);
        const available = new Set(flatImages.map((item) => item.src));
        for (const selected of Array.from(this.selectedPaths)) {
            if (!available.has(selected)) this.selectedPaths.delete(selected);
        }

        if (!flatImages.length) {
            hide(container);
            show(emptyState);
            countEl.textContent = '0 images';
            this.updateSelectionUI();
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
        container.classList.toggle('gallery-select-mode', this.selectionMode);

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
        this.applyMobileControlsState();
        this.updateSelectionUI();
    }

    renderImageItem(item, index) {
        const thumbUrl = `/thumbs/${item.src}`;
        const safeMeta = encodeURIComponent(JSON.stringify(item.meta || {}));
        const ratio = Math.max(0.5, Math.min(2.2, item.height / Math.max(1, item.width)));
        const selected = this.selectedPaths.has(item.src);
        const selectionBadge = this.selectionMode
            ? `<span class="selection-badge ${selected ? 'is-selected' : ''}">${selected ? '✓' : ''}</span>`
            : '';
        return `
            <div class="image-item flat-item ${selected ? 'is-selected' : ''}" data-src="${item.src}" data-index="${index}" data-width="${item.width}" data-height="${item.height}" data-favorite="${item.favorite ? '1' : '0'}" data-meta="${safeMeta}" ${item.variant ? `data-variant="${item.variant}"` : ''}>
                <img src="${thumbUrl}" alt="${this.escapeHtml(item.prompt)}" loading="lazy" style="aspect-ratio:${1 / ratio};" />
                ${item.variant ? '<span class="hd-badge">HD</span>' : ''}
                ${item.favorite ? '<span class="favorite-badge">♥</span>' : ''}
                ${selectionBadge}
                <button class="image-favorite-btn ${item.favorite ? 'is-favorite' : ''}" type="button" title="${item.favorite ? 'Unfavorite' : 'Favorite'}" aria-label="${item.favorite ? 'Unfavorite image' : 'Favorite image'}">♥</button>
            </div>
        `;
    }

    attachListeners(container) {
        container.querySelectorAll('.image-item').forEach((item) => {
            listen(item, 'click', (e) => {
                if (e) e.stopPropagation();
                const src = item.dataset.src;
                if (this.selectionMode) {
                    this.toggleImageSelection(src);
                    return;
                }
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
                if (this.selectionMode) return;
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
            if (type === 'run') await api.deleteRun(src);
            else await api.deleteImage(src);

            this.removeImageInMemory(src, type);
            this.selectedPaths.delete(src);
            this.updateSelectionUI();
            this.render(byId('gallery-search')?.value || '');
            toast(type === 'run' ? 'Run deleted' : 'Image deleted', 'success');
            return true;
        } catch (e) {
            console.error('Delete error:', e);
            toast('Failed to delete image', 'error');
            return false;
        }
    }

    sourceMatchesAny(path, targets) {
        if (!path) return false;
        if (targets.has(path)) return true;
        for (const target of targets) {
            if (path.includes(target) || target.includes(path)) return true;
        }
        return false;
    }

    removeImagesInMemory(paths) {
        const targets = new Set((paths || []).filter(Boolean));
        if (!targets.size) return;

        const prune = (sessions) => {
            if (!Array.isArray(sessions)) return sessions;
            const next = [];
            for (const s of sessions) {
                const allImages = Array.isArray(s.images) ? s.images : [];
                const kept = allImages.filter((p) => !this.sourceMatchesAny(p, targets));
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
