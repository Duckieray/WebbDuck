/**
 * LightboxManager Module
 * Handles PhotoSwipe integration and custom lightbox actions.
 */

import PhotoSwipeLightbox from '../lib/photoswipe-lightbox.esm.js';
import PhotoSwipe from '../lib/photoswipe.esm.js';
import { byId, listen, toast } from '../core/utils.js';

export class LightboxManager {
    constructor(callbacks = {}) {
        this.callbacks = {
            onUpscale: callbacks.onUpscale || (() => { }),
            onInpaint: callbacks.onInpaint || (() => { }),
            onRegenerate: callbacks.onRegenerate || (() => { }),
            onStageSettings: callbacks.onStageSettings || (() => { }),
            onDelete: callbacks.onDelete || (() => { }),
            onFavorite: callbacks.onFavorite || (async () => false),
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
            btn.classList.toggle('is-above-info', this.infoVisible);
            btn.classList.toggle('is-docked-bottom', !this.infoVisible);
        });

        listen(byId('lightbox-regen'), 'click', () => {
            if (!this.currentPswpInstance?.pswp) return;
            const curr = this.currentPswpInstance.pswp.currSlide.data;
            this.callbacks.onRegenerate(curr);
        });

        listen(byId('lightbox-stage'), 'click', () => {
            if (!this.currentPswpInstance?.pswp) return;
            const curr = this.currentPswpInstance.pswp.currSlide.data;
            this.callbacks.onStageSettings(curr);
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

        listen(byId('lightbox-download'), 'click', () => {
            if (!this.currentPswpInstance?.pswp) return;
            const curr = this.currentPswpInstance.pswp.currSlide.data;
            const src = (curr?.isShowingVariant && curr?.variant)
                ? curr.variant
                : (curr?.src || curr?.originalSrc);
            if (!src) return;

            const anchor = document.createElement('a');
            anchor.href = src;
            const suffix = (curr?.isShowingVariant && curr?.variant) ? '-upscaled' : '';
            anchor.download = `webbduck-${Date.now()}${suffix}.png`;
            document.body.appendChild(anchor);
            anchor.click();
            document.body.removeChild(anchor);
        });

        listen(byId('lightbox-favorite'), 'click', async () => {
            if (!this.currentPswpInstance?.pswp) return;
            const curr = this.currentPswpInstance.pswp.currSlide.data;
            const ok = await this.callbacks.onFavorite(curr.src, Boolean(curr.isFavorite));
            if (!ok) return;
            curr.isFavorite = !Boolean(curr.isFavorite);
            this.updateButtons(curr);
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
                    isShowingVariant: false,
                    isFavorite: item?.dataset.favorite === '1'
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
                toggleBtn.classList.add('is-above-info');
                toggleBtn.classList.remove('is-docked-bottom');
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
                toggleBtn.classList.remove('is-above-info', 'is-docked-bottom');
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

        const loras = Array.isArray(meta.loras) ? meta.loras : [];
        const loraText = loras.length
            ? loras.map((lora) => {
                if (typeof lora === 'string') return lora;
                const name = lora?.name || lora?.model || 'Unknown';
                const weight = lora?.weight ?? lora?.strength;
                return weight !== undefined ? `${name} (${weight})` : name;
            }).join(' | ')
            : 'None';
        setText('lightbox-loras', loraText);

        const inout = meta.inoutpaint || {};
        const parts = [];
        const mode = meta.mode || (inout.has_mask ? 'inpaint' : (inout.has_input_image ? 'img2img' : 'txt2img'));
        parts.push(`mode=${mode}`);

        if (inout.has_input_image) parts.push('input=yes');
        if (inout.has_mask) parts.push('mask=yes');

        if (inout.strength !== undefined && inout.strength !== null) {
            parts.push(`strength=${Number(inout.strength).toFixed(2)}`);
        }
        if (inout.inpainting_fill) parts.push(`fill=${inout.inpainting_fill}`);
        if (inout.mask_blur !== undefined && inout.mask_blur !== null) parts.push(`mask_blur=${inout.mask_blur}`);

        if (Array.isArray(inout.input_image_size) && inout.input_image_size.length === 2) {
            parts.push(`input=${inout.input_image_size[0]}x${inout.input_image_size[1]}`);
        }

        if (inout.smart_extend) {
            parts.push('smart_extend=on');
            if (inout.smart_extend_feather !== undefined && inout.smart_extend_feather !== null) {
                parts.push(`feather=${inout.smart_extend_feather}`);
            }
            if (inout.smart_extend_auto_step !== undefined && inout.smart_extend_auto_step !== null) {
                parts.push(`auto_step=${inout.smart_extend_auto_step ? 'on' : 'off'}`);
            }
            if (inout.smart_extend_step_growth !== undefined && inout.smart_extend_step_growth !== null) {
                parts.push(`step_growth=${Number(inout.smart_extend_step_growth).toFixed(2)}`);
            }
            if (inout.smart_extend_offset_x !== undefined && inout.smart_extend_offset_x !== null
                && inout.smart_extend_offset_y !== undefined && inout.smart_extend_offset_y !== null) {
                parts.push(`offset=(${inout.smart_extend_offset_x},${inout.smart_extend_offset_y})`);
            }
            if (inout.smart_extend_refine !== undefined && inout.smart_extend_refine !== null) {
                parts.push(`refine=${inout.smart_extend_refine ? 'on' : 'off'}`);
            }
            if (inout.smart_extend_refine_each_step !== undefined && inout.smart_extend_refine_each_step !== null) {
                parts.push(`refine_each=${inout.smart_extend_refine_each_step ? 'on' : 'off'}`);
            }
            if (inout.smart_extend_refine_width !== undefined && inout.smart_extend_refine_width !== null) {
                parts.push(`refine_w=${inout.smart_extend_refine_width}`);
            }
            if (inout.smart_extend_refine_strength !== undefined && inout.smart_extend_refine_strength !== null) {
                parts.push(`refine_s=${Number(inout.smart_extend_refine_strength).toFixed(2)}`);
            }
        }

        setText('lightbox-inoutpaint', parts.length ? parts.join(' | ') : 'None');
    }

    updateButtons(curr) {
        const viewHdBtn = byId('lightbox-view-hd');
        const compareBtn = byId('lightbox-compare');
        const favoriteBtn = byId('lightbox-favorite');
        if (viewHdBtn) viewHdBtn.style.display = curr.variant ? 'inline-flex' : 'none';
        if (compareBtn) compareBtn.style.display = curr.variant ? 'inline-flex' : 'none';
        if (favoriteBtn) {
            const canFavorite = typeof curr?.src === 'string' && curr.src.includes('/outputs/');
            favoriteBtn.style.display = canFavorite ? 'inline-flex' : 'none';
            if (!canFavorite) return;
            const isFav = Boolean(curr?.isFavorite);
            favoriteBtn.textContent = isFav ? 'Unfavorite' : 'Favorite';
            favoriteBtn.classList.toggle('btn-primary', isFav);
            favoriteBtn.classList.toggle('btn-secondary', !isFav);
        }
    }

    async toggleHD(curr) {
        const btn = byId('lightbox-view-hd');
        if (!btn) return;

        let nextSrc = null;
        if (curr.isShowingVariant) {
            nextSrc = curr.originalSrc;
            curr.src = nextSrc;
            curr.isShowingVariant = false;
            btn.textContent = 'View HD';
        } else {
            const variantSrc = this.withCacheBuster(curr.variant);
            const ready = await this.waitForImage(variantSrc, 6, 220);
            if (!ready) {
                toast('HD image is still being finalized. Try again in a moment.', 'warning');
                return;
            }
            nextSrc = variantSrc;
            curr.src = nextSrc;
            curr.isShowingVariant = true;
            btn.textContent = 'Show Original';
        }

        const pswp = this.currentPswpInstance?.pswp;
        if (!pswp) return;

        // Prefer in-place image src replacement to avoid flash/flicker.
        if (this.swapCurrentSlideImage(pswp, nextSrc)) {
            return;
        }

        const viewState = this.captureCurrentViewState();
        pswp.refreshSlideContent(pswp.currSlide.index);
        this.restoreCurrentViewState(viewState);
    }

    withCacheBuster(src) {
        if (!src || typeof src !== 'string') return src;
        const join = src.includes('?') ? '&' : '?';
        return `${src}${join}v=${Date.now()}`;
    }

    waitForImage(src, attempts = 4, delayMs = 250) {
        return new Promise((resolve) => {
            const tryLoad = (remaining) => {
                const probe = new Image();
                probe.onload = () => resolve(true);
                probe.onerror = () => {
                    if (remaining <= 1) {
                        resolve(false);
                        return;
                    }
                    setTimeout(() => tryLoad(remaining - 1), delayMs);
                };
                probe.src = this.withCacheBuster(src);
            };
            tryLoad(Math.max(1, attempts));
        });
    }

    toggleCompare(curr) {
        const compContainer = byId('lightbox-comparison');
        const compOverlay = byId('comp-overlay');
        const compOriginal = byId('comp-original');
        const compModified = byId('comp-modified');
        const compHandle = byId('comp-handle');
        const btn = byId('lightbox-compare');

        if (!compContainer || !compOverlay || !compOriginal || !compModified || !compHandle || !btn) return;

        const isOpen = compContainer.style.display !== 'none';
        if (isOpen) {
            compContainer.style.display = 'none';
            btn.textContent = 'Compare';
            return;
        }

        compContainer.style.display = 'flex';
        compOriginal.src = curr.originalSrc;
        compModified.src = this.withCacheBuster(curr.variant);
        compHandle.style.left = '50%';
        compOverlay.style.clipPath = 'inset(0 0 0 50%)';
        btn.textContent = 'Close Compare';
        this.initCompareSlider(compContainer, compHandle, compOverlay);
    }

    initCompareSlider(container, handle, overlay) {
        let isDragging = false;

        const setPos = (pct) => {
            const bounded = Math.max(0, Math.min(100, pct));
            handle.style.left = `${bounded}%`;
            overlay.style.clipPath = `inset(0 0 0 ${bounded}%)`;
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

        const previousDown = container._comparePointerDownHandler;
        if (previousDown) {
            container.removeEventListener('pointerdown', previousDown);
        }

        const onDown = (e) => {
            isDragging = true;
            onMove(e);
            window.addEventListener('pointermove', onMove);
            window.addEventListener('pointerup', onUp);
        };

        container.addEventListener('pointerdown', onDown);

        container._comparePointerDownHandler = onDown;
    }

    captureCurrentViewState() {
        const pswp = this.currentPswpInstance?.pswp;
        const slide = pswp?.currSlide;
        if (!slide) return null;

        return {
            zoom: slide.currZoomLevel || slide.zoomLevels?.initial || 1,
            panX: slide.pan?.x ?? 0,
            panY: slide.pan?.y ?? 0,
        };
    }

    restoreCurrentViewState(state) {
        if (!state) return;
        const pswp = this.currentPswpInstance?.pswp;
        if (!pswp) return;

        const apply = () => {
            const slide = pswp.currSlide;
            if (!slide) return false;

            try {
                const minZoom = slide.zoomLevels?.min ?? slide.zoomLevels?.fit ?? 1;
                const maxZoom = slide.zoomLevels?.max ?? state.zoom;
                const targetZoom = Math.max(minZoom, Math.min(maxZoom, state.zoom));

                if (typeof slide.zoomTo === 'function') {
                    slide.zoomTo(
                        targetZoom,
                        {
                            x: (pswp.viewportSize?.x || 0) / 2,
                            y: (pswp.viewportSize?.y || 0) / 2,
                        },
                        0,
                        false
                    );
                }

                if (typeof slide.panTo === 'function') {
                    slide.panTo(state.panX, state.panY);
                } else if (slide.pan) {
                    slide.pan.x = state.panX;
                    slide.pan.y = state.panY;
                    if (typeof slide.applyCurrentZoomPan === 'function') {
                        slide.applyCurrentZoomPan();
                    }
                }
                return true;
            } catch (_) {
                return false;
            }
        };

        [0, 40, 100, 180].forEach((delay) => {
            setTimeout(() => {
                apply();
            }, delay);
        });
    }

    swapCurrentSlideImage(pswp, src) {
        if (!src) return false;
        try {
            const slide = pswp.currSlide;
            const container = slide?.container || slide?.holderElement || null;
            const img = container?.querySelector?.('img.pswp__img, img');
            if (!img) return false;
            img.src = src;
            return true;
        } catch (_) {
            return false;
        }
    }
}
