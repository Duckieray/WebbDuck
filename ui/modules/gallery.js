import PhotoSwipe from '/ui/lib/photoswipe.esm.js';
import PhotoSwipeLightbox from '/ui/lib/photoswipe-lightbox.esm.js';

let isGalleryLoaded = false;
let maxGalleryTimestamp = 0;
let pswpInstance = null;

// PhotoSwipe Integration - No-op for compatibility with existing calls
export async function initPhotoSwipeLib() {
    return;
}

export async function openPhotoSwipe(index) {
    await initPhotoSwipeLib();
    if (!PhotoSwipeLightbox) return;

    const images = Array.from(document.querySelectorAll(".session-images img"));
    if (images.length === 0) return;

    const items = images.map(img => {
        return {
            src: img.src,
            w: img.naturalWidth || 1024,
            h: img.naturalHeight || 1024,
            msrc: img.src,
            element: img,
            element: img,
            meta: img._meta || {},
            variant: img._variant || null,
            originalSrc: img.src // Store original for toggling
        };
    });

    const options = {
        dataSource: items,
        pswpModule: PhotoSwipe,
        // index: index, // Passing to loadAndOpen
        bgOpacity: 0.95,
        showHideAnimationType: 'zoom',
        closeOnVerticalDrag: true,
        // Zoom settings
        zoom: false, // managed by PhotoSwipe
        wheelToZoom: true,
        secondaryZoomLevel: 2,
        maxZoomLevel: 4,
        padding: { top: 0, bottom: 0, left: 0, right: 0 }
    };

    const lightbox = new PhotoSwipeLightbox(options);

    // Custom UI Registration
    lightbox.on('uiRegister', function () {
        lightbox.pswp.ui.registerElement({
            name: 'custom-info',
            order: 9,
            isButton: false,
            appendTo: 'root',
            html: '',
            onInit: (el, pswp) => {
                const infoPanel = document.querySelector('.lightbox-info');
                const infoToggle = document.getElementById('lightbox-info-toggle');
                const compContainer = document.getElementById('lightbox-comparison');

                // Add class for flex layout
                el.classList.add('pswp-custom-grid');

                // Order: Toggle First (so it's above panel in Flex column with justify-end)
                if (infoToggle) {
                    infoToggle.style.display = 'flex';
                    el.appendChild(infoToggle);
                }

                if (infoPanel) {
                    infoPanel.style.display = 'block';
                    el.appendChild(infoPanel);
                }

                if (compContainer) {
                    el.appendChild(compContainer);
                }
            }
        });
    });

    // Wire up STATIC buttons (Upscale / Compare)
    const upscaleBtn = document.getElementById('lightbox-upscale');
    const compareBtn = document.getElementById('lightbox-compare');
    const compContainer = document.getElementById('lightbox-comparison');
    const compOriginal = document.getElementById('comp-original');
    const compModified = document.getElementById('comp-modified');
    const compHandle = document.getElementById('comp-handle');
    const compOverlay = document.getElementById('comp-overlay');

    // Slider Logic
    let isDragging = false;

    function setSliderPos(pct) {
        // Clamp 0-100
        pct = Math.max(0, Math.min(100, pct));

        // Move handle
        compHandle.style.left = `${pct}%`;

        // Clip the TOP image (Modified/Upscaled) to reveal Bottom (Original)
        // If we want Modified on Right and Original on Left:
        // inset(0 0 0 pct%) cuts off the left side.
        // Actually, usually "Before" is Left, "After" is Right.
        // So Bottom=Original, Top=Modified. 
        // We want to reveal Modified from the right side?
        // Or standard: Left side = Original, Right side = Modified.
        // If Top is Modified, we clip-path the LEFT side to reveal Original underneath.
        // inset(0 0 0 ${pct}%) -> Hides left pct%.

        // BUT, usually "After" (Upscaled) is the 'better' one, maybe we want that on left?
        // Standard "Compare" sliders usually have "Before" on Left.
        // So:
        // Bottom Layer: Upscaled (Full)
        // Top Layer: Original
        // Clip Top Layer (Original) from the RIGHT (inset(0 0 0 ${100-pct}%))? NO.
        // Clip Top Layer (Original) from the RIGHT: inset(0 ${100-pct}% 0 0).
        // Then Left side is Original, Right side is Upscaled (revealed from bottom).

        // Let's do: 
        // Bottom: Upscaled (comp-modified)
        // Top: Original (comp-original)
        // We clip the Original (Top) from the right side.

        compOverlay.style.width = '100%'; // Ensure full width for clip-path approach
        compOverlay.style.border = 'none'; // reset css

        // Swapping DOM order conceptually or just src?
        // Let's set sources explicitly in toggle.

        // CSS implementation: 
        // clipping the top element.
        // If top is Modified: clip left side => inset(0 0 0 pct%). Left shows bottom (Original).
        // Result: Left=Original, Right=Modified. Matches "Before -> After".
        compModified.style.clipPath = `inset(0 0 0 ${pct}%)`;
    }

    if (compContainer) {
        const onPointerMove = (e) => {
            if (!isDragging) return;
            e.preventDefault(); // Prevent scrolling on touch
            const rect = compContainer.getBoundingClientRect();
            const x = (e.clientX || e.touches[0].clientX) - rect.left;
            const pct = (x / rect.width) * 100;
            setSliderPos(pct);
        };

        const onPointerUp = () => {
            isDragging = false;
            window.removeEventListener('pointermove', onPointerMove);
            window.removeEventListener('touchmove', onPointerMove);
            window.removeEventListener('pointerup', onPointerUp);
            window.removeEventListener('touchend', onPointerUp);
        };

        const onPointerDown = (e) => {
            isDragging = true;
            // Capture initial
            onPointerMove(e);

            window.addEventListener('pointermove', onPointerMove);
            window.addEventListener('touchmove', onPointerMove, { passive: false });
            window.addEventListener('pointerup', onPointerUp);
            window.addEventListener('touchend', onPointerUp);
        };

        compContainer.addEventListener('pointerdown', onPointerDown);
        compContainer.addEventListener('touchstart', onPointerDown, { passive: false });
    }

    // Regenerate Click
    const regenBtn = document.getElementById('lightbox-regen');
    if (regenBtn) {
        regenBtn.onclick = () => {
            const curr = lightbox.pswp.currSlide.data;
            if (!curr.meta) {
                alert("No metadata found for this image.");
                return;
            }

            // No confirmation needed per user request
            // logical flow: Copy settings -> Switch Tab -> Click Generate

            // Helper to safe-set
            const setVal = (id, val) => {
                const el = document.getElementById(id);
                if (el) el.value = val;
            };

            const m = curr.meta;
            setVal('prompt', m.prompt || '');
            setVal('negative_prompt', m.negative_prompt || '');
            setVal('steps', m.steps || 30);
            setVal('cfg', m.cfg || 7.5);
            setVal('width', m.width || 1024);
            setVal('height', m.height || 1024);
            setVal('seed_input', m.seed || -1);

            // Model is trickier if it's a select and the option doesn't exist
            if (m.base_model) {
                const modelSel = document.getElementById('base_model');
                if (modelSel) {
                    const exists = Array.from(modelSel.options).some(o => o.value === m.base_model);
                    if (exists) {
                        modelSel.value = m.base_model;
                    } else {
                        console.warn("Model not found in list:", m.base_model);
                    }
                }
            }

            // set seed mode to random (per user request for variation)
            const sm = document.getElementById('seed_mode');
            if (sm) sm.value = 'random';

            /* 
            // Previous logic: Copy seed. 
            // NOW DISABLED: User wants random seed on regen.
            if (m.seed) {
                const sm = document.getElementById('seed_mode');
                if (sm) sm.value = 'manual';
                const si = document.getElementById('seed_input');
                if (si) { 
                    si.disabled = false;
                    si.value = m.seed; 
                }
            } 
            */

            // Close lightbox? User said "keep me in the gallery". 
            // If we stay in lightbox, they see the old image. 
            // If we close, they see the grid.
            // Let's NOT close the lightbox, just give feedback. 
            // They can close it if they want.

            // Visual Confirmation
            const originalText = regenBtn.innerText;
            regenBtn.innerText = "🚀 Started!";
            regenBtn.style.background = "var(--success, #238636)"; // Green feedback

            setTimeout(() => {
                regenBtn.innerText = originalText;
                regenBtn.style.background = "";
            }, 2000);

            // Trigger Generation (Background)
            // We don't switch tabs. We just click the button.
            // The button is in the DOM, just hidden.
            const genBtn = document.getElementById('genBtn');
            if (genBtn) {
                console.log("[Gallery] Triggering background generation...");
                genBtn.click();
            } else {
                alert("Error: Could not find Generate button.");
            }
        };
    }
    if (upscaleBtn) {
        upscaleBtn.onclick = () => {
            const curr = lightbox.pswp.currSlide.data;
            if (!confirm("Upscale this image (2x)?")) return;

            const formData = new FormData();
            formData.append("image", curr.msrc);
            formData.append("scale", 2);

            fetch("/upscale", { method: "POST", body: formData })
                .then(() => alert("Upscaling started... refresh gallery in a moment."))
                .catch(e => alert("Error: " + e));
        };
    }

    // View HD Click (Zoomable)
    const viewHdBtn = document.getElementById('lightbox-view-hd');
    if (viewHdBtn) {
        viewHdBtn.onclick = () => {
            const curr = lightbox.pswp.currSlide.data;
            if (!curr.variant) return;

            // Hide slider if open
            if (compContainer) {
                compContainer.style.display = 'none';
                // Show main image if it was hidden
                const mainImg = document.getElementById('lightbox-img');
                if (mainImg) mainImg.style.opacity = '1';
                // Reset compare button state
                if (compareBtn) {
                    compareBtn.innerText = "↔ Compare";
                    compareBtn.style.background = "";
                }
            }

            // Toggle Source
            if (curr.isShowingVariant) {
                curr.src = curr.originalSrc;
                curr.isShowingVariant = false;
                viewHdBtn.innerText = "👁️ View Upscaled";
                viewHdBtn.style.background = "";
            } else {
                curr.src = curr.variant;
                curr.isShowingVariant = true;
                viewHdBtn.innerText = "↩️ View Original";
                viewHdBtn.style.background = "var(--primary)";
            }

            // Refresh PhotoSwipe to load new src
            lightbox.pswp.refreshSlideContent(lightbox.pswp.currSlide.index);
        };
    }

    // Compare Click (Slider)
    if (compareBtn) {
        compareBtn.innerText = "↔ Compare"; // Reset label

        // Toggle Logic
        compareBtn.onclick = () => {
            const curr = lightbox.pswp.currSlide.data;
            if (!curr.variant) return;

            const isComparing = compContainer.style.display !== 'none';

            if (isComparing) {
                // Close Comparison
                compContainer.style.display = 'none';
                compareBtn.innerText = "↔ Compare";
                compareBtn.style.background = "";

                // Show main image again
                const mainImg = document.getElementById('lightbox-img');
                if (mainImg) mainImg.style.opacity = '1';

            } else {
                // Open Comparison
                compContainer.style.display = 'block';

                // Setup Images
                compOriginal.src = curr.originalSrc; // Bottom
                compModified.src = curr.variant;     // Top (clipped)

                // Hide main image so no bleed-through
                const mainImg = document.getElementById('lightbox-img');
                if (mainImg) mainImg.style.opacity = '0';

                // Reset slider pos
                setSliderPos(50);

                compareBtn.innerText = "✕ Close";
                compareBtn.style.background = "var(--primary)";
            }
        };
    }

    let lastIndex = -1;
    lightbox.on('change', () => {
        if (lightbox.pswp.currSlide.index === lastIndex) return;
        lastIndex = lightbox.pswp.currSlide.index;

        const curr = lightbox.pswp.currSlide.data;

        // Metadata Panel Updates
        if (curr.meta) {
            const setSafeText = (id, text) => {
                const el = document.getElementById(id);
                if (el) el.innerText = text;
            };

            setSafeText("lightbox-prompt", curr.meta.prompt || "No prompt");
            setSafeText("lightbox-negative", curr.meta.negative_prompt || "None");
            setSafeText("lightbox-model", curr.meta.base_model || "Unknown");
            setSafeText("lightbox-seed", curr.meta.seed || "--");

            const sets = [];
            if (curr.meta.steps) sets.push(`Steps: ${curr.meta.steps}`);
            if (curr.meta.cfg) sets.push(`CFG: ${curr.meta.cfg}`);
            if (curr.meta.scheduler) sets.push(curr.meta.scheduler);
            if (curr.meta.width) sets.push(`${curr.meta.width}x${curr.meta.height}`);

            setSafeText("lightbox-settings", sets.join(" · "));
        }

        const compContainer = document.getElementById('lightbox-comparison');
        if (compContainer) compContainer.style.display = 'none';

        // Button Visibility
        if (compareBtn) {
            compareBtn.style.display = curr.variant ? 'block' : 'none';
            compareBtn.innerText = "↔ Compare";
            compareBtn.style.background = "";
        }

        const viewHdBtn = document.getElementById('lightbox-view-hd');
        if (viewHdBtn) {
            viewHdBtn.style.display = curr.variant ? 'block' : 'none';
            viewHdBtn.innerText = "👁️ View Upscaled";
            viewHdBtn.style.background = "";
            curr.isShowingVariant = false; // Reset to original on clean slide load
        }

        // Hide upscale button if already upscaled? Optional. 
        // For now, keep it visible so they can upscale again (maybe 4x later).
    });

    lightbox.on('close', () => {
        const infoPanel = document.querySelector('.lightbox-info');
        const infoToggle = document.getElementById('lightbox-info-toggle');
        const compContainer = document.getElementById('lightbox-comparison');
        const safeZone = document.getElementById('lightbox-safe-zone') || document.body;

        if (infoPanel) {
            infoPanel.style.display = 'none';
            safeZone.appendChild(infoPanel);
        }
        if (infoToggle) {
            infoToggle.style.display = 'none'; // Hide it!
            safeZone.appendChild(infoToggle);
        }
        if (compContainer) {
            compContainer.style.display = 'none';
            safeZone.appendChild(compContainer);
        }

        // Restore main image
        const mainImg = document.getElementById('lightbox-img');
        if (mainImg) mainImg.style.opacity = '1';

        pswpInstance = null;
    });

    lightbox.init(); // Keep init for now to be safe with v5 types
    lightbox.loadAndOpen(index);
    pswpInstance = lightbox.pswp;
}

export function getPswpInstance() {
    return pswpInstance;
}

export async function loadGallery(incremental = false) {
    const galleryGrid = document.getElementById("full-gallery-grid");
    const refreshGalleryBtn = document.getElementById("refresh_gallery_btn");

    if (!galleryGrid) return;

    if (!incremental) {
        if (isGalleryLoaded && galleryGrid.children.length > 0) return;
        galleryGrid.innerHTML = '<div style="padding:20px; text-align:center; color: var(--muted);">Loading...</div>';
    } else {
        if (refreshGalleryBtn) refreshGalleryBtn.classList.add("spinning");
    }

    try {
        const url = incremental
            ? `/gallery?after=${maxGalleryTimestamp}`
            : `/gallery`;

        const r = await fetch(url);
        const data = await r.json();

        if (!incremental) {
            galleryGrid.innerHTML = "";
        }

        if (refreshGalleryBtn) refreshGalleryBtn.classList.remove("spinning");

        if (incremental && data.length === 0) {
            return;
        }

        if (data.length === 0 && !incremental) {
            galleryGrid.innerHTML = '<div style="padding:20px; text-align:center; color: var(--muted);">No history found</div>';
            isGalleryLoaded = true;
            return;
        }

        const frag = document.createDocumentFragment();

        data.forEach(session => {
            const ts = session.meta?.timestamp || 0;
            if (ts > maxGalleryTimestamp) maxGalleryTimestamp = ts;

            const el = document.createElement("div");
            el.className = "gallery-session";

            const model = session.meta?.base_model || "Unknown Model";
            const time = new Date(ts * 1000).toLocaleString();
            const variants = session.variants || {};

            el.innerHTML = `
         <div class="gallery-session-header">
           <span>${time}</span>
           <span>${model}</span>
         </div>
         <div class="session-images"></div>
       `;

            const imgContainer = el.querySelector(".session-images");

            session.images.forEach(imgUrl => {
                const img = document.createElement("img");
                img.src = imgUrl;
                img.loading = "lazy";
                img._meta = session.meta;
                img.style.cursor = "pointer";

                // 1. Identify Variant
                const variants = session.variants || {};
                const key = imgUrl.split("/").pop();
                if (variants[key]) {
                    img._variant = variants[key];
                }

                // 2. Define Click Handler FIRST
                const handleOpen = async (e) => {
                    e.stopPropagation();
                    const allImages = Array.from(document.querySelectorAll(".session-images img"));
                    const idx = allImages.indexOf(img);
                    if (idx !== -1) {
                        try {
                            await openPhotoSwipe(idx);
                        } catch (err) {
                            console.error("[Gallery] OpenPhotoSwipe failed:", err);
                        }
                    }
                };

                // 3. Attach standard listener
                img.addEventListener("click", handleOpen);

                // 4. Wrap if variant exists
                if (img._variant) {
                    const wrap = document.createElement("div");
                    wrap.className = "img-wrap";
                    wrap.style.position = "relative";
                    wrap.style.display = "inline-block";

                    wrap.appendChild(img);

                    const badge = document.createElement("span");
                    badge.innerText = "✨";
                    badge.className = "upscale-badge";
                    // Inline styles for reliability
                    badge.style.position = "absolute";
                    badge.style.top = "5px";
                    badge.style.right = "5px";
                    badge.style.fontSize = "16px";
                    badge.style.textShadow = "0 0 2px black";
                    badge.style.pointerEvents = "none";

                    wrap.appendChild(badge);
                    imgContainer.appendChild(wrap);
                } else {
                    imgContainer.appendChild(img);
                }
            });

            frag.appendChild(el);
        });

        if (incremental) {
            galleryGrid.prepend(frag);
        } else {
            galleryGrid.appendChild(frag);
        }

        isGalleryLoaded = true;

    } catch (e) {
        console.error(e);
        if (refreshGalleryBtn) refreshGalleryBtn.classList.remove("spinning");
        if (!incremental) {
            galleryGrid.innerHTML = '<div style="padding:20px; text-align:center; color: #ff7b72;">Failed to load gallery</div>';
        }
    }
}
