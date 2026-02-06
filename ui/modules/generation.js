import { saveState, setSeed, lastSeed } from './state.js';
import { getSelectedImageFile, initUploadHandling } from './upload.js';
import { getCurrentMaskBlob } from './mask-editor.js';

export async function send(endpoint) {
    try {
        const d = new FormData();

        // Helper to get value fast
        const val = (id) => { const el = document.getElementById(id); return el ? el.value : ""; };
        const checked = (id) => { const el = document.getElementById(id); return el ? el.checked : false; };

        ["steps", "cfg", "width", "height", "num_images"].forEach(id => {
            d.append(id, val(id));
        });

        d.append("experimental_compress", checked("experimental_compress"));
        d.append("prompt", val("prompt"));
        d.append("prompt_2", val("prompt_2"));
        d.append("negative_prompt", val("negative_prompt"));
        d.append("base_model", val("base_model"));
        d.append("second_pass_model", val("second_pass_model"));
        d.append("second_pass_mode", val("second_pass_mode"));
        d.append("scheduler", val("scheduler"));
        d.append("strength", val("strength"));
        d.append("refinement_strength", val("refinement_strength"));

        const selectedImageFile = getSelectedImageFile();
        const currentMaskBlob = getCurrentMaskBlob();

        if (selectedImageFile) {
            d.append("image", selectedImageFile);
            d.set("second_pass_model", "None");

            if (currentMaskBlob) {
                d.append("mask", currentMaskBlob, "mask.png");
                const fill = window._maskInpaintingFill || val("inpainting_fill") || "replace";
                const blur = window._maskBlur || val("mask_blur") || "8";
                d.append("inpainting_fill", fill);
                d.append("mask_blur", blur);
            }
        }

        const seedMode = val("seed_mode");
        const seedIn = val("seed_input");

        if (seedMode === "reuse" && lastSeed !== null) {
            d.append("seed", lastSeed);
        }
        if (seedMode === "manual" && seedIn) {
            d.append("seed", seedIn);
        }

        const loras = [];
        document.querySelectorAll(".lora-card").forEach(card => {
            const enabled = card.querySelector("input[type=checkbox]").checked;
            if (!enabled) return;
            loras.push({
                name: card.dataset.lora,
                weight: parseFloat(card.querySelector("input[type=range]").value)
            });
        });
        d.append("loras", JSON.stringify(loras));

        const r = await fetch(endpoint, { method: "POST", body: d });

        if (!r.ok) {
            const text = await r.text();
            console.error("Server error:", r.status, text);
            alert(`Server Error ${r.status}: ${text.slice(0, 200)}`);
            return;
        }

        const data = await r.json();

        if (data.seed !== undefined) {
            setSeed(data.seed);
            saveState();
        }

        if (data?.type === "validation_error" || data?.error) {
            alert(data.error);
            throw new Error(data.error);
        }

        return data;

    } catch (e) {
        console.error("Generation failed:", e);
        alert(`Client Error: ${e.message}\n${e.stack || ""}`);
    }
}

export async function runUpscale(originalImageSrc, callback) {
    if (!originalImageSrc) return;

    const btn = document.getElementById("upscaleBtn");
    btn.disabled = true;
    btn.innerText = "🔼 Upscaling…";

    const fd = new FormData();
    fd.append("image", originalImageSrc); // Is this a URL or a file? 
    // If originalImageSrc is a URL (from server), backend needs to support downloading it or handling path
    // The original app.js logic suggests it passes the URL string and backend re-opens it OR it expects a blob?
    // Actually, 'upscale' endpoint in python likely expects 'image' as file or path
    // If we pass URL string as 'image' value in FormData, backend receives string.
    fd.append("scale", 2);

    try {
        const r = await fetch("/upscale", { method: "POST", body: fd });
        const data = await r.json();
        return data.image; // New image URL
    } catch (e) {
        alert("Upscale failed – check server logs");
        console.error(e);
    } finally {
        btn.disabled = false;
        btn.innerText = "🔼 Upscale ×2";
    }
}
