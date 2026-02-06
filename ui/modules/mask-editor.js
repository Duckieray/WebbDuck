import { renderPreview, setPreviewMode } from './preview.js';

let currentMaskBlob = null;
let isDrawing = false;
let lastX = 0;
let lastY = 0;

export function getCurrentMaskBlob() {
    return currentMaskBlob;
}

export function setCurrentMaskBlob(blob) {
    currentMaskBlob = blob;
}

// Drawing Logic
function getPos(e, canvas) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    // Standardize mouse/touch
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY
    };
}

function draw(e, ctx, brushSize) {
    if (!isDrawing) return;
    e.preventDefault(); // Prevent scrolling on touch

    const canvas = ctx.canvas;
    const { x, y } = getPos(e, canvas);

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.strokeStyle = "rgba(255, 255, 255, 1)";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    // Calculate line width relative to canvas scale
    const rect = canvas.getBoundingClientRect();
    const scale = canvas.width / rect.width;
    ctx.lineWidth = brushSize * scale;

    ctx.stroke();

    lastX = x;
    lastY = y;
}

export function initMaskEditor(selectedImageFile, getSelectedImageFile) {
    const maskOverlay = document.getElementById("mask_overlay");
    const maskCanvas = document.getElementById("mask_canvas");
    const maskCtx = maskCanvas.getContext("2d");
    const maskBrushSize = document.getElementById("mask_brush_size");
    const maskBrushPreview = document.getElementById("mask_brush_preview");
    const maskClearBtn = document.getElementById("mask_clear_btn");
    const maskInvertBtn = document.getElementById("mask_invert_btn");
    const maskCancelBtn = document.getElementById("mask_cancel_btn");
    const maskSaveBtn = document.getElementById("mask_save_btn");
    const editMaskBtn = document.getElementById("edit_mask_btn");

    function updateBrushPreview() {
        const size = maskBrushSize.value;
        maskBrushPreview.style.width = size + "px";
        maskBrushPreview.style.height = size + "px";
    }

    maskBrushSize.addEventListener("input", updateBrushPreview);
    updateBrushPreview();

    function openMaskEditor() {
        // Check if we have an image - handle both passed arg and getter
        const imgFile = typeof getSelectedImageFile === 'function' ? getSelectedImageFile() : selectedImageFile;
        if (!imgFile) return;

        // Use the main preview area as the mask editor
        setPreviewMode("mask");
        renderPreview(imgFile);
    }

    function closeMaskEditor() {
        maskOverlay.style.display = "none";
    }

    editMaskBtn.onclick = (e) => {
        e.stopPropagation();
        openMaskEditor();
    };

    maskCancelBtn.onclick = closeMaskEditor;

    maskClearBtn.onclick = () => {
        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    };

    maskInvertBtn.onclick = () => {
        const w = maskCanvas.width;
        const h = maskCanvas.height;
        const imageData = maskCtx.getImageData(0, 0, w, h);
        const data = imageData.data;

        for (let i = 0; i < data.length; i += 4) {
            // Invert Alpha: Transparent (0) <-> Opaque (255)
            const alpha = data[i + 3];
            data[i + 3] = 255 - alpha;

            // Ensure color is white
            data[i] = 255;
            data[i + 1] = 255;
            data[i + 2] = 255;
        }

        maskCtx.putImageData(imageData, 0, 0);
    };

    // Bind Events
    maskCanvas.addEventListener("mousedown", (e) => {
        isDrawing = true;
        const pos = getPos(e, maskCanvas);
        lastX = pos.x;
        lastY = pos.y;
        draw(e, maskCtx, maskBrushSize.value);
    });

    maskCanvas.addEventListener("mousemove", (e) => draw(e, maskCtx, maskBrushSize.value));
    window.addEventListener("mouseup", () => isDrawing = false);

    // Touch support
    maskCanvas.addEventListener("touchstart", (e) => {
        isDrawing = true;
        const pos = getPos(e, maskCanvas);
        lastX = pos.x;
        lastY = pos.y;
        draw(e, maskCtx, maskBrushSize.value);
    }, { passive: false });
    maskCanvas.addEventListener("touchmove", (e) => draw(e, maskCtx, maskBrushSize.value), { passive: false });
    window.addEventListener("touchend", () => isDrawing = false);

    maskSaveBtn.onclick = () => {
        maskCanvas.toBlob((blob) => {
            currentMaskBlob = blob;
            editMaskBtn.style.color = "var(--green)";
            closeMaskEditor();
        });
    };

    return { openMaskEditor, closeMaskEditor };
}
