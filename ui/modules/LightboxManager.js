/**
 * LightboxManager Module
 * Handles PhotoSwipe integration and custom lightbox actions.
 */

import PhotoSwipeLightbox from '../lib/photoswipe-lightbox.esm.js';
import PhotoSwipe from '../lib/photoswipe.esm.js';
import { byId, listen } from '../core/utils.js';

export class LightboxManager {
    constructor(callbacks = {}) {
        this.callbacks = {
            onUpscale: callbacks.onUpscale || (() => { }),
            onInpaint: callbacks.onInpaint || (() => { }),
            onRegenerate: callbacks.onRegenerate || (() => { }),
            onDelete: callbacks.onDelete || (() => { })
        };

        this.currentPswpInstance = null;
        this.pendingDeleteRun = null;
        this.infoVisible = true;

        this.initActions();
    }

    initActions() {
        listen(byId('lightbox-info-toggle'), 'click', () => {
            const info = byId('lightbox-info');
            const btn = byId('lightbox-info-toggle');
            if (!info || !btn) return;

            this.infoVisible = !this.infoVisible;
            info.classList.toggle('hidden', !this.infoVisible);
            btn.textContent = this.infoVisible ? 'Hide Info' : 'Show Info';
        });

        listen(byId('lightbox-regen'), 'click', () => {
            if (!this.currentPswpInstance?.pswp) return;
            const curr = this.currentPswpInstance.pswp.currSlide.data;
            this.callbacks.onRegenerate(curr);
        });

        listen(byId('lightbox-upscale'), 'click', () => {
            if (!this.currentPswpInstance?.pswp) return;
            const curr = this.currentPswpInstance.pswp.currSlide.data;
            this.callbacks.onUpscale(curr.src, (upscaledUrl) => {
                curr.variant = upscaledUrl;
                curr.isShowingVariant = false;
                this.updateButtons(curr);
            });
        });

        listen(byId('lightbox-inpaint'), 'click', () => {
            if (!this.currentPswpInstance?.pswp) return;
            const curr = this.currentPswpInstance.pswp.currSlide.data;
            this.currentPswpInstance.pswp.close();
            this.callbacks.onInpaint(curr.src);
        });

        listen(byId('lightbox-view-hd'), 'click', () => {
            if (!this.currentPswpInstance?.pswp) return;
            const curr = this.currentPswpInstance.pswp.currSlide.data;
            this.toggleHD(curr);
        });

        listen(byId('lightbox-compare'), 'click', () => {
            if (!this.currentPswpInstance?.pswp) return;
            const curr = this.currentPswpInstance.pswp.currSlide.data;
            this.toggleCompare(curr);
        });

        listen(byId('lightbox-delete'), 'click', () => {
            const modal = byId('confirmation-modal');
            const imgBtn = byId('modal-delete-img');
            if (imgBtn) imgBtn.style.display = '';

            if (modal && modal.parentNode !== document.body) {
                document.body.appendChild(modal);
            }

            if (!modal) return;
            modal.classList.remove('hidden');
            void modal.offsetWidth;
            setTimeout(() => modal.classList.add('active'), 10);
        });

        listen(byId('modal-delete-img'), 'click', async () => {
            if (!this.currentPswpInstance?.pswp) return;
            const curr = this.currentPswpInstance.pswp.currSlide.data;
            const success = await this.callbacks.onDelete(curr.src, 'image');
            if (!success) return;

            this.currentPswpInstance.pswp.close();
            this.closeModal();
        });

        listen(byId('modal-delete-run'), 'click', async () => {
            if (this.pendingDeleteRun) {
                const { src, callback } = this.pendingDeleteRun;
                const success = await this.callbacks.onDelete(src, 'run');
                if (success && callback) callback();
                if (success) this.closeModal();
                this.pendingDeleteRun = null;
                return;
            }

            if (!this.currentPswpInstance?.pswp) return;
            const curr = this.currentPswpInstance.pswp.currSlide.data;
            const success = await this.callbacks.onDelete(curr.src, 'run');
            if (!success) return;

            this.currentPswpInstance.pswp.close();
            this.closeModal();
        });

        listen(byId('modal-cancel'), 'click', () => {
            this.closeModal();
            this.pendingDeleteRun = null;
        });
    }

    closeModal() {
        const modal = byId('confirmation-modal');
        if (!modal) return;

        modal.classList.remove('active');
        setTimeout(() => modal.classList.add('hidden'), 300);
    }

    confirmRunDelete(runSrc) {
        this.pendingDeleteRun = { src: runSrc };

        const modal = byId('confirmation-modal');
        const imgBtn = byId('modal-delete-img');
        if (imgBtn) imgBtn.style.display = 'none';

        if (modal && modal.parentNode !== document.body) {
            document.body.appendChild(modal);
        }

        if (!modal) return;
        modal.classList.remove('hidden');
        void modal.offsetWidth;
        setTimeout(() => modal.classList.add('active'), 10);
    }

    open(dataSource, startIndex = 0) {
        let items;

        if (dataSource.length > 0 && (dataSource[0].tagName || dataSource[0] instanceof Element)) {
            items = Array.from(dataSource).map(img => {
                const item = img.closest('.image-item');
                const sessionGroup = img.closest('.session-group');
                const sessionMeta = sessionGroup ? this.extractSessionMeta(sessionGroup) : {};
                const variantUrl = item?.dataset.variant || null;

                const datasetSrc = item?.dataset.src;
                const src = datasetSrc || img.src.replace('/thumbs/', '/').replace('_thumb', '');

                const width = item?.dataset.width ? parseInt(item.dataset.width, 10) : (img.naturalWidth || 1024);
                const height = item?.dataset.height ? parseInt(item.dataset.height, 10) : (img.naturalHeight || 1024);

                return {
                    src,
                    width,
                    height,
                    msrc: img.src,
                    alt: img.alt || '',
                    meta: sessionMeta,
                    originalSrc: src,
                    variant: variantUrl,
                    isShowingVariant: false
                };
            });
        } else {
            items = dataSource;
        }

        const lightbox = new PhotoSwipeLightbox({
            dataSource: items,
            pswpModule: PhotoSwipe,
            index: startIndex,
            bgOpacity: 0.95,
            showHideAnimationType: 'fade',
            closeOnVerticalDrag: true,
            padding: { top: 20, bottom: 280, left: 20, right: 20 },
            imageClickAction: false,
            tapAction: false,
            wheelToZoom: true
        });

        lightbox.on('openingAnimationStart', () => {
            const pswpEl = lightbox.pswp.element;
            const infoPanel = byId('lightbox-info');
            const toggleBtn = byId('lightbox-info-toggle');
            const comp = byId('lightbox-comparison');

            if (!pswpEl) return;

            if (infoPanel) {
                this.infoVisible = true;
                infoPanel.classList.remove('hidden');
                pswpEl.appendChild(infoPanel);
                listen(infoPanel, 'pointerdown', (e) => e.stopPropagation());
                listen(infoPanel, 'mousedown', (e) => e.stopPropagation());
                listen(infoPanel, 'click', (e) => e.stopPropagation());
            }

            if (toggleBtn) {
                pswpEl.appendChild(toggleBtn);
                toggleBtn.style.display = 'block';
                toggleBtn.textContent = 'Hide Info';
            }

            if (comp) {
                pswpEl.appendChild(comp);
            }
        });

        lightbox.on('destroy', () => {
            const infoPanel = byId('lightbox-info');
            const toggleBtn = byId('lightbox-info-toggle');
            const comp = byId('lightbox-comparison');
            const viewHdBtn = byId('lightbox-view-hd');
            const compareBtn = byId('lightbox-compare');

            if (infoPanel) {
                document.body.appendChild(infoPanel);
                infoPanel.classList.add('hidden');
            }
            if (toggleBtn) {
                document.body.appendChild(toggleBtn);
                toggleBtn.style.display = 'none';
                toggleBtn.textContent = 'Info';
            }
            if (comp) {
                document.body.appendChild(comp);
                comp.style.display = 'none';
            }

            if (viewHdBtn) {
                viewHdBtn.textContent = 'View HD';
                viewHdBtn.style.display = 'none';
            }
            if (compareBtn) {
                compareBtn.textContent = 'Compare';
                compareBtn.style.display = 'none';
            }

            this.currentPswpInstance = null;
        });

        lightbox.on('contentActivate', ({ content }) => {
            if (!content?.data) return;
            this.updateMeta(content.data.meta || content.data);
            this.updateButtons(content.data);
        });

        lightbox.on('change', () => {
            if (!lightbox.pswp) return;
            const curr = lightbox.pswp.currSlide.data;
            this.updateMeta(curr.meta);
            this.updateButtons(curr);
        });

        lightbox.init();
        lightbox.loadAndOpen(startIndex);
        this.currentPswpInstance = lightbox;
    }

    extractSessionMeta(sessionGroup) {
        const json = sessionGroup.dataset.json;
        if (!json) return {};

        try {
            const session = JSON.parse(decodeURIComponent(json));
            return session.meta || session;
        } catch (e) {
            console.warn(e);
            return {};
        }
    }

    updateMeta(meta) {
        if (!meta) return;

        const setText = (id, text) => {
            const el = byId(id);
            if (el) el.textContent = text || '--';
        };

        setText('lightbox-prompt', meta.prompt || 'No prompt');
        setText('lightbox-negative', meta.negative || meta.negative_prompt || 'None');
        setText('lightbox-model', meta.base_model || meta.model || 'Unknown');
        setText('lightbox-seed', meta.seed || '--');

        const settings = [];
        if (meta.steps) settings.push(`Steps: ${meta.steps}`);
        if (meta.cfg) settings.push(`CFG: ${meta.cfg}`);
        if (meta.scheduler) settings.push(meta.scheduler);
        if (meta.width && meta.height) settings.push(`${meta.width}x${meta.height}`);

        setText('lightbox-settings', settings.join(' | ') || '--');
    }

    updateButtons(curr) {
        const viewHdBtn = byId('lightbox-view-hd');
        const compareBtn = byId('lightbox-compare');
        if (viewHdBtn) viewHdBtn.style.display = curr.variant ? 'inline-flex' : 'none';
        if (compareBtn) compareBtn.style.display = curr.variant ? 'inline-flex' : 'none';
    }

    toggleHD(curr) {
        const btn = byId('lightbox-view-hd');
        if (!btn) return;

        if (curr.isShowingVariant) {
            curr.src = curr.originalSrc;
            curr.isShowingVariant = false;
            btn.textContent = 'View HD';
        } else {
            curr.src = curr.variant;
            curr.isShowingVariant = true;
            btn.textContent = 'Show Original';
        }

        this.currentPswpInstance.pswp.refreshSlideContent(this.currentPswpInstance.pswp.currSlide.index);
    }

    toggleCompare(curr) {
        const compContainer = byId('lightbox-comparison');
        const compOriginal = byId('comp-original');
        const compModified = byId('comp-modified');
        const compHandle = byId('comp-handle');
        const btn = byId('lightbox-compare');

        if (!compContainer || !compOriginal || !compModified || !compHandle || !btn) return;

        const isOpen = compContainer.style.display !== 'none';
        if (isOpen) {
            compContainer.style.display = 'none';
            btn.textContent = 'Compare';
            return;
        }

        compContainer.style.display = 'flex';
        compOriginal.src = curr.originalSrc;
        compModified.src = curr.variant;
        compHandle.style.left = '50%';
        compModified.style.clipPath = 'inset(0 0 0 50%)';
        btn.textContent = 'Close Compare';
        this.initCompareSlider(compContainer, compHandle, compModified);
    }

    initCompareSlider(container, handle, modified) {
        if (container.dataset.init) return;

        let isDragging = false;

        const setPos = (pct) => {
            const bounded = Math.max(0, Math.min(100, pct));
            handle.style.left = `${bounded}%`;
            modified.style.clipPath = `inset(0 0 0 ${bounded}%)`;
        };

        const onMove = (e) => {
            if (!isDragging) return;
            e.preventDefault();
            const rect = container.getBoundingClientRect();
            const x = (e.clientX || e.touches?.[0]?.clientX || 0) - rect.left;
            setPos((x / rect.width) * 100);
        };

        const onUp = () => {
            isDragging = false;
            window.removeEventListener('pointermove', onMove);
            window.removeEventListener('pointerup', onUp);
        };

        container.addEventListener('pointerdown', (e) => {
            isDragging = true;
            onMove(e);
            window.addEventListener('pointermove', onMove);
            window.addEventListener('pointerup', onUp);
        });

        container.dataset.init = 'true';
    }
}
