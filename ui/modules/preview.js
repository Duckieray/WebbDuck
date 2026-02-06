import { setCurrentMaskBlob } from './mask-editor.js';

let previewState = {
    original: null,
    upscaled: null,
    mode: "original",
    wipe: 50,
};

export function setPreviewMode(mode) {
    previewState.mode = mode;
}

export function setPreviewImages(original, upscaled = null) {
    previewState.original = original;
    previewState.upscaled = upscaled;
    previewState.mode = upscaled ? "compare" : "original";
}

export function getPreviewState() {
    return previewState;
}

export function renderPreview(selectedImageFile = null, runUpscaleCallback = null) {
    const el = document.getElementById("preview");
    const inputImagePreview = document.getElementById("input_image_preview");
    const editMaskBtn = document.getElementById("edit_mask_btn");

    // --- MASK EDITING MODE ---
    if (previewState.mode === "mask" && selectedImageFile) {
        el.innerHTML = `
      <div class="preview-mask-editor">
        <canvas id="preview_mask_canvas"></canvas>
      </div>

      <div class="preview-toolbar mask-toolbar-inline">
        <div class="mask-tools-inline">
          <label>Brush</label>
          <input type="range" id="preview_brush_size" min="5" max="100" value="30">
        </div>
        <div class="mask-tools-inline">
          <label>Mode</label>
          <select id="preview_inpainting_fill">
            <option value="replace">Replace Masked</option>
            <option value="keep">Keep Masked</option>
          </select>
        </div>
        <div class="mask-tools-inline">
          <label>Blur</label>
          <input type="range" id="preview_mask_blur" min="0" max="64" value="8" style="width:60px">
        </div>
        <button id="maskClearBtn">Clear</button>
        <button id="maskInvertBtn">Invert</button>
        <button id="maskSaveBtn" class="primary">Save Mask</button>
        <button id="maskCancelBtn">Cancel</button>
      </div>
    `;

        const canvas = document.getElementById("preview_mask_canvas");
        const ctx = canvas.getContext("2d");
        const brushSlider = document.getElementById("preview_brush_size");

        // Load image and size canvas to fill preview
        const img = new Image();
        img.onload = () => {
            const container = el.querySelector(".preview-mask-editor");
            const maxW = container.clientWidth;
            const maxH = container.clientHeight - 10; // Leave some padding

            let w = img.width;
            let h = img.height;
            const ratio = w / h;

            // Scale UP to fill the container (not down)
            if (ratio > maxW / maxH) {
                // Image is wider than container ratio
                w = maxW;
                h = w / ratio;
            } else {
                // Image is taller than container ratio
                h = maxH;
                w = h * ratio;
            }

            canvas.width = w;
            canvas.height = h;
            canvas.style.width = w + "px";
            canvas.style.height = h + "px";
            // Use the preview source from the hidden input preview
            canvas.style.backgroundImage = `url(${inputImagePreview.src})`;
            canvas.style.backgroundSize = "100% 100%";
            ctx.clearRect(0, 0, w, h);
        };
        img.src = inputImagePreview.src;

        // Drawing state
        let drawing = false;
        let lx = 0, ly = 0;

        function getPos(e) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const cx = e.touches ? e.touches[0].clientX : e.clientX;
            const cy = e.touches ? e.touches[0].clientY : e.clientY;
            return { x: (cx - rect.left) * scaleX, y: (cy - rect.top) * scaleY };
        }

        function drawLine(e) {
            if (!drawing) return;
            e.preventDefault();
            const { x, y } = getPos(e);
            ctx.beginPath();
            ctx.moveTo(lx, ly);
            ctx.lineTo(x, y);
            ctx.strokeStyle = "rgba(255,255,255,1)";
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            const rect = canvas.getBoundingClientRect();
            ctx.lineWidth = brushSlider.value * (canvas.width / rect.width);
            ctx.stroke();
            lx = x; ly = y;
        }

        canvas.onmousedown = (e) => { drawing = true; const p = getPos(e); lx = p.x; ly = p.y; drawLine(e); };
        canvas.onmousemove = drawLine;
        window.addEventListener("mouseup", () => drawing = false);
        canvas.ontouchstart = (e) => { drawing = true; const p = getPos(e); lx = p.x; ly = p.y; drawLine(e); };
        canvas.ontouchmove = drawLine;
        window.addEventListener("touchend", () => drawing = false);

        document.getElementById("maskClearBtn").onclick = () => ctx.clearRect(0, 0, canvas.width, canvas.height);

        document.getElementById("maskInvertBtn").onclick = () => {
            const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            for (let i = 0; i < imgData.data.length; i += 4) {
                imgData.data[i + 3] = 255 - imgData.data[i + 3];
                imgData.data[i] = imgData.data[i + 1] = imgData.data[i + 2] = 255;
            }
            ctx.putImageData(imgData, 0, 0);
        };

        document.getElementById("maskSaveBtn").onclick = () => {
            canvas.toBlob((blob) => {
                setCurrentMaskBlob(blob);
                // Store settings from inline controls
                window._maskInpaintingFill = document.getElementById("preview_inpainting_fill").value;
                window._maskBlur = document.getElementById("preview_mask_blur").value;
                editMaskBtn.style.color = "var(--green)";
                previewState.mode = "original";
                renderPreview(null, runUpscaleCallback);
            });
        };

        document.getElementById("maskCancelBtn").onclick = () => {
            previewState.mode = "original";
            renderPreview(null, runUpscaleCallback);
        };

        return;
    }

    // --- NORMAL PREVIEW MODES ---
    if (!previewState.original) {
        el.innerHTML = "<span>No image yet</span>";
        return;
    }

    let content = "";

    if (previewState.mode === "wipe" && previewState.upscaled) {
        content = `
      <div class="preview-wipe" id="wipeContainer" style="--wipe:${previewState.wipe}%">
        <div class="wipe-canvas">
          <img class="wipe-img base" src="${previewState.original}">
          <img class="wipe-img overlay" src="${previewState.upscaled}">
        </div>
        <div class="wipe-handle"></div>
      </div>
    `;
    }
    else if (previewState.mode === "compare" && previewState.upscaled) {
        content = `
      <div class="preview-compare">
        <img src="${previewState.original}">
        <img src="${previewState.upscaled}">
      </div>
    `;
    }
    else {
        const img =
            previewState.mode === "upscaled" && previewState.upscaled
                ? previewState.upscaled
                : previewState.original;

        content = `
      <div class="preview-inner">
        <img src="${img}">
      </div>
    `;
    }

    el.innerHTML = `
    ${content}

    <div class="preview-toolbar">
      <button data-mode="original">Original</button>
      <button data-mode="upscaled" ${!previewState.upscaled ? "disabled" : ""}>
        Upscaled
      </button>
      <button data-mode="compare" ${!previewState.upscaled ? "disabled" : ""}>
        Side-by-side
      </button>
      <button data-mode="wipe" ${!previewState.upscaled ? "disabled" : ""}>
        Wipe
      </button>
      <button id="upscaleBtn">🔼 Upscale ×2</button>
    </div>
  `;

    document.querySelectorAll(".preview-toolbar button[data-mode]")
        .forEach(btn => {
            btn.onclick = () => {
                previewState.mode = btn.dataset.mode;
                renderPreview(null, runUpscaleCallback);
            };
        });

    const container = document.getElementById("wipeContainer");

    if (container) {
        let dragging = false;

        const updateFromX = (x) => {
            const rect = container.getBoundingClientRect();
            let pct = ((x - rect.left) / rect.width) * 100;
            pct = Math.max(0, Math.min(100, pct));

            previewState.wipe = pct;
            container.style.setProperty("--wipe", pct + "%");
        };

        container.onmousedown = (e) => {
            e.preventDefault();
            dragging = true;
            updateFromX(e.clientX);
        };

        container.onmousemove = (e) => {
            if (dragging) updateFromX(e.clientX);
        };

        window.onmouseup = () => dragging = false;

        container.ontouchstart = (e) => {
            e.preventDefault();
            dragging = true;
            updateFromX(e.touches[0].clientX);
        };

        container.ontouchmove = (e) => {
            if (dragging) updateFromX(e.touches[0].clientX);
        };

        window.ontouchend = () => dragging = false;
    }

    if (runUpscaleCallback) {
        document.getElementById("upscaleBtn").onclick = runUpscaleCallback;
    }
}
