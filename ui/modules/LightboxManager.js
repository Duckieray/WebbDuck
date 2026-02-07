/**
 * LightboxManager Module
 * Handles PhotoSwipe integration and custom lightbox actions.
 */

import PhotoSwipeLightbox from '../lib/photoswipe-lightbox.esm.js';
import PhotoSwipe from '../lib/photoswipe.esm.js';
import { byId, listen, toggleClass, toast, show, hide } from '../core/utils.js';
import { setSeed } from '../core/state.js';

export class LightboxManager {
    constructor(callbacks = {}) {
        this.callbacks = {
            onUpscale: callbacks.onUpscale || (() => { }),
            onInpaint: callbacks.onInpaint || (() => { }),
            onRegenerate: callbacks.onRegenerate || (() => { }),
            onDelete: callbacks.onDelete || (() => { })
        };

        this.currentPswpInstance = null;
        this.initActions();
    }

    /**
     * Initialize lightbox UI actions (buttons, sliders)
     */
    initActions() {
        // Toggle info panel
        listen(byId('lightbox-info-toggle'), 'click', () => {
            const info = byId('lightbox-info');
            if (info) {
                info.style.display = info.style.display === 'none' ? 'block' : 'none';
            }
        });

        // Regenerate button
        listen(byId('lightbox-regen'), 'click', () => {
            if (!this.currentPswpInstance?.pswp) return;
            const curr = this.currentPswpInstance.pswp.currSlide.data;
            this.callbacks.onRegenerate(curr);
        });

        // Upscale button
        listen(byId('lightbox-upscale'), 'click', () => {
            if (!this.currentPswpInstance?.pswp) return;
            const curr = this.currentPswpInstance.pswp.currSlide.data;
            this.callbacks.onUpscale(curr.src, (upscaledUrl) => {
                curr.variant = upscaledUrl;
                this.updateButtons(curr);
            });
        });

        // Inpaint button
        listen(byId('lightbox-inpaint'), 'click', () => {
            if (!this.currentPswpInstance?.pswp) return;
            const curr = this.currentPswpInstance.pswp.currSlide.data;
            this.currentPswpInstance.pswp.close();
            this.callbacks.onInpaint(curr.src);
        });

        // View HD toggle
        listen(byId('lightbox-view-hd'), 'click', () => {
            if (!this.currentPswpInstance?.pswp) return;
            const curr = this.currentPswpInstance.pswp.currSlide.data;
            this.toggleHD(curr);
        });

        // Compare slider toggle
        listen(byId('lightbox-compare'), 'click', () => {
            if (!this.currentPswpInstance?.pswp) return;
            const curr = this.currentPswpInstance.pswp.currSlide.data;
            this.toggleCompare(curr);
        });

        // Delete button
        listen(byId('lightbox-delete'), 'click', () => {
            console.log('[Lightbox] Delete clicked');
            const modal = byId('confirmation-modal');

            // Ensure both buttons are visible (reset state)
            const imgBtn = byId('modal-delete-img');
            if (imgBtn) imgBtn.style.display = '';

            // Move modal to end of body to ensure it sits on top of everything (including lightbox)
            if (modal && modal.parentNode !== document.body) {
                document.body.appendChild(modal);
            }

            // Remove hidden first to display:flex
            modal.classList.remove('hidden');

            // Force repaint
            void modal.offsetWidth;

            // Add active for transition
            setTimeout(() => modal.classList.add('active'), 10);
        });

        // Modal confirm delete image
        listen(byId('modal-delete-img'), 'click', async () => {
            if (!this.currentPswpInstance?.pswp) return;
            const curr = this.currentPswpInstance.pswp.currSlide.data;
            const success = await this.callbacks.onDelete(curr.src, 'image');
            if (success) {
                this.currentPswpInstance.pswp.close();
                const modal = byId('confirmation-modal');
                modal.classList.remove('active');
                setTimeout(() => modal.classList.add('hidden'), 300);
            }
        });

        // Modal confirm delete run
        listen(byId('modal-delete-run'), 'click', async () => {
            // Check for external delete context
            if (this.pendingDeleteRun) {
                const { src, callback } = this.pendingDeleteRun;
                const success = await this.callbacks.onDelete(src, 'run');
                if (success) {
                    if (callback) callback();
                    this.closeModal();
                }
                this.pendingDeleteRun = null;
                return;
            }

            if (!this.currentPswpInstance?.pswp) return;
            const curr = this.currentPswpInstance.pswp.currSlide.data;
            const success = await this.callbacks.onDelete(curr.src, 'run');
            if (success) {
                this.currentPswpInstance.pswp.close();
                this.closeModal();
            }
        });

        // Modal cancel
        listen(byId('modal-cancel'), 'click', () => {
            this.closeModal();
            this.pendingDeleteRun = null;
        });
    }

    closeModal() {
        const modal = byId('confirmation-modal');
        modal.classList.remove('active');
        setTimeout(() => modal.classList.add('hidden'), 300);
    }

    /**
     * Trigger delete confirmation for a run from outside the lightbox
     */
    confirmRunDelete(runSrc) {
        this.pendingDeleteRun = { src: runSrc };

        const modal = byId('confirmation-modal');
        // Hide delete-image button since we are deleting a run
        const imgBtn = byId('modal-delete-img');
        if (imgBtn) imgBtn.style.display = 'none';

        if (modal && modal.parentNode !== document.body) {
            document.body.appendChild(modal);
        }
        modal.classList.remove('hidden');
        void modal.offsetWidth;
        setTimeout(() => modal.classList.add('active'), 10);

        // Restore button state on close? We should probably reset it in closeModal 
        // or just handle visibility dynamically. For simplicity, we toggle it back when opening normal lightbox delete.
    }

    /**
     * Open PhotoSwipe lightbox
     * @param {Array|NodeList} dataSource - Images
     * @param {number} startIndex 
     */
    open(dataSource, startIndex = 0) {
        let items;

        // Check if dataSource is DOM elements
        if (dataSource.length > 0 && (dataSource[0].tagName || dataSource[0] instanceof Element)) {
            items = Array.from(dataSource).map(img => {
                const item = img.closest('.image-item');
                const sessionGroup = img.closest('.session-group');
                const sessionMeta = sessionGroup ? this.extractSessionMeta(sessionGroup) : {};
                const variantUrl = item?.dataset.variant || null;

                const datasetSrc = item?.dataset.src;
                const src = datasetSrc || img.src.replace('/thumbs/', '/').replace('_thumb', '');

                const width = item?.dataset.width ? parseInt(item.dataset.width) : (img.naturalWidth || 1024);
                const height = item?.dataset.height ? parseInt(item.dataset.height) : (img.naturalHeight || 1024);

                return {
                    src: src,
                    width: width,
                    height: height,
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
            padding: { top: 20, bottom: 200, left: 20, right: 20 },
            imageClickAction: false,
            tapAction: false,
            wheelToZoom: true
        });

        // Inject custom UI
        lightbox.on('openingAnimationStart', () => {
            const pswpEl = lightbox.pswp.element;
            const infoPanel = byId('lightbox-info');
            const toggleBtn = byId('lightbox-info-toggle');
            const comp = byId('lightbox-comparison');

            if (pswpEl) {
                if (infoPanel) {
                    pswpEl.appendChild(infoPanel);
                    infoPanel.style.display = 'block';
                    infoPanel.classList.remove('hidden');
                    // Stop propagation
                    listen(infoPanel, 'pointerdown', (e) => e.stopPropagation());
                    listen(infoPanel, 'mousedown', (e) => e.stopPropagation());
                    listen(infoPanel, 'click', (e) => e.stopPropagation());
                }
                if (toggleBtn) {
                    pswpEl.appendChild(toggleBtn);
                    toggleBtn.style.display = 'block';
                }
                if (comp) {
                    pswpEl.appendChild(comp);
                }
            }
        });

        // Cleanup on destroy
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
            }
            if (comp) {
                document.body.appendChild(comp);
                comp.style.display = 'none';
            }

            if (viewHdBtn) {
                viewHdBtn.textContent = '👁️ View HD';
                viewHdBtn.style.display = 'none';
            }
            if (compareBtn) {
                compareBtn.textContent = '↔ Compare';
                compareBtn.style.display = 'none';
            }

            this.currentPswpInstance = null;
        });

        // Metadata handling
        lightbox.on('contentActivate', ({ content }) => {
            if (content && content.data) {
                try {
                    this.updateMeta(content.data.meta || content.data);
                    this.updateButtons(content.data);
                } catch (e) {
                    console.error('Failed to update lightbox meta:', e);
                }
            }
        });

        lightbox.on('change', () => {
            if (lightbox.pswp) {
                const curr = lightbox.pswp.currSlide.data;
                this.updateMeta(curr.meta);
                this.updateButtons(curr);
            }
        });

        lightbox.init();
        lightbox.loadAndOpen(startIndex);
        this.currentPswpInstance = lightbox;
    }

    extractSessionMeta(sessionGroup) {
        const json = sessionGroup.dataset.json;
        if (json) {
            try {
                const session = JSON.parse(decodeURIComponent(json));
                return session.meta || session;
            } catch (e) { console.warn(e); }
        }
        return {};
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
        if (meta.width && meta.height) settings.push(`${meta.width}×${meta.height}`);

        setText('lightbox-settings', settings.join(' · ') || '--');
    }

    updateButtons(curr) {
        const viewHdBtn = byId('lightbox-view-hd');
        const compareBtn = byId('lightbox-compare');
        if (viewHdBtn) viewHdBtn.style.display = curr.variant ? 'inline-flex' : 'none';
        if (compareBtn) compareBtn.style.display = curr.variant ? 'inline-flex' : 'none';
    }

    toggleHD(curr) {
        const btn = byId('lightbox-view-hd');
        if (curr.isShowingVariant) {
            curr.src = curr.originalSrc;
            curr.isShowingVariant = false;
            btn.textContent = '👁️ View HD';
        } else {
            curr.src = curr.variant;
            curr.isShowingVariant = true;
            btn.textContent = '↩️ Original';
        }
        this.currentPswpInstance.pswp.refreshSlideContent(this.currentPswpInstance.pswp.currSlide.index);
    }

    toggleCompare(curr) {
        const compContainer = byId('lightbox-comparison');
        const compOriginal = byId('comp-original');
        const compModified = byId('comp-modified');
        const compHandle = byId('comp-handle');
        const btn = byId('lightbox-compare');

        const isOpen = compContainer.style.display !== 'none';
        if (isOpen) {
            compContainer.style.display = 'none';
            btn.textContent = '↔ Compare';
        } else {
            compContainer.style.display = 'flex';
            compOriginal.src = curr.originalSrc;
            compModified.src = curr.variant;
            compHandle.style.left = '50%';
            byId('comp-modified').style.clipPath = 'inset(0 0 0 50%)';
            btn.textContent = '✕ Close';
            this.initCompareSlider(compContainer, compHandle, compModified);
        }
    }

    initCompareSlider(container, handle, modified) {
        // Simple slider logic attached to elements
        let isDragging = false;
        const setPos = (pct) => {
            pct = Math.max(0, Math.min(100, pct));
            handle.style.left = `${pct}%`;
            modified.style.clipPath = `inset(0 0 0 ${pct}%)`;
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

        // Remove old listeners? We can't easily without storing refs.
        // Assuming fresh init each time is okay or using cloneNode() trick.
        // For now, adding new listeners. Limitation: might stack if not careful.
        // But toggleCompare destroys UI on close? No, it just hides.
        // Better to check if already init.
        if (container.dataset.init) return;

        container.addEventListener('pointerdown', (e) => {
            isDragging = true;
            onMove(e);
            window.addEventListener('pointermove', onMove);
            window.addEventListener('pointerup', onUp);
        });
        container.dataset.init = 'true';
    }
}
