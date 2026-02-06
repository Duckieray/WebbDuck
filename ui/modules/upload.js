import { saveState } from './state.js';
import { updateTokenCount } from './utils.js';

let selectedImageFile = null;
let captioningAvailable = false;

export function getSelectedImageFile() {
    return selectedImageFile;
}

export function setSelectedImageFile(file) {
    selectedImageFile = file;
}

export async function checkCaptionerAvailability() {
    try {
        const captionerData = await fetch("/captioners").then(r => r.json());
        captioningAvailable = captionerData.available;
        console.log("Captioning available:", captioningAvailable);
        return captioningAvailable;
    } catch (e) {
        console.log("Captioner check failed:", e);
        return false;
    }
}

export function initUploadHandling() {
    const imageUploadArea = document.getElementById("image-upload-area");
    const inputImageInput = document.getElementById("input_image");
    const inputImagePreview = document.getElementById("input_image_preview");
    const uploadPlaceholder = document.getElementById("upload_placeholder");
    const clearInputImageBtn = document.getElementById("clear_input_image");
    const editMaskBtn = document.getElementById("edit_mask_btn");
    const img2imgSettings = document.getElementById("img2img-settings");

    const baseModelLabel = document.getElementById("base-model-label");
    const secondPassModelSection = document.getElementById("second-pass-model-section");
    const secondPassSection = document.getElementById("second-pass-section");
    const captionControls = document.getElementById("caption-controls");

    function handleImageSelect(file) {
        if (!file || !file.type.startsWith("image/")) return;

        selectedImageFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            inputImagePreview.src = e.target.result;
            inputImagePreview.style.display = "block";
            uploadPlaceholder.style.display = "none";
            clearInputImageBtn.style.display = "block";
            img2imgSettings.style.display = "block";
            editMaskBtn.style.display = "flex";

            // Img2Img Mode
            secondPassModelSection.style.display = "none";
            secondPassSection.style.display = "none";
            baseModelLabel.textContent = "Model";

            if (captioningAvailable && captionControls) {
                captionControls.style.display = "block";
            }
        };
        reader.readAsDataURL(file);
    }

    imageUploadArea.onclick = (e) => {
        if (e.target !== clearInputImageBtn && e.target !== editMaskBtn) {
            inputImageInput.click();
        }
    };

    inputImageInput.onchange = (e) => {
        if (e.target.files && e.target.files[0]) {
            handleImageSelect(e.target.files[0]);
        }
    };

    imageUploadArea.ondragover = (e) => {
        e.preventDefault();
        imageUploadArea.style.borderColor = "var(--accent)";
    };

    imageUploadArea.ondragleave = (e) => {
        e.preventDefault();
        imageUploadArea.style.borderColor = "";
    };

    imageUploadArea.ondrop = (e) => {
        e.preventDefault();
        imageUploadArea.style.borderColor = "";
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleImageSelect(e.dataTransfer.files[0]);
        }
    };

    clearInputImageBtn.onclick = (e) => {
        e.stopPropagation();
        selectedImageFile = null;
        inputImageInput.value = "";
        inputImagePreview.src = "";
        inputImagePreview.style.display = "none";
        uploadPlaceholder.style.display = "block";
        clearInputImageBtn.style.display = "none";
        img2imgSettings.style.display = "none";
        editMaskBtn.style.display = "none";
        editMaskBtn.style.color = "";

        // Restore Text2Img Mode
        secondPassModelSection.style.display = "block";
        // Usually updateSecondPassVisibility is called here
        // app.js will handle listener for visibility

        baseModelLabel.textContent = "Base Model";

        if (captionControls) {
            captionControls.style.display = "none";
        }

        // Dispatch event so app can know to clear masks/state
        document.dispatchEvent(new CustomEvent('image-cleared'));
    };

    return { handleImageSelect };
}

export async function generateCaption() {
    if (!selectedImageFile || !captioningAvailable) return;

    const captionBtn = document.getElementById("caption_btn");
    const captionStyleSelect = document.getElementById("caption_style");
    const promptInput = document.getElementById("prompt");

    if (captionBtn) {
        captionBtn.disabled = true;
        captionBtn.textContent = "🔍 Captioning...";
    }

    try {
        const style = captionStyleSelect ? captionStyleSelect.value : "detailed";
        const fd = new FormData();
        fd.append("image", selectedImageFile);
        fd.append("style", style);

        const r = await fetch("/caption", { method: "POST", body: fd });
        const data = await r.json();

        if (data.error) {
            alert("Captioning failed: " + data.error);
            return;
        }

        promptInput.value = data.caption;
        saveState();
        updateTokenCount(promptInput, "prompt", document.getElementById("base_model").value);

    } catch (e) {
        console.error(e);
        alert("Captioning failed: " + e.message);
    } finally {
        if (captionBtn) {
            captionBtn.disabled = false;
            captionBtn.textContent = "🔍 Caption";
        }
    }
}
