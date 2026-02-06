export const STORAGE_KEY = "webbduck_state";

export let lastSeed = null;

export function getSeed() {
    const seedInput = document.getElementById("seed_input");
    const v = seedInput.value;
    return v === "" ? null : Number(v);
}

export function setSeed(v) {
    const seedInput = document.getElementById("seed_input");
    seedInput.value = v ?? "";
    lastSeed = v;
}

export function saveState() {
    const state = {
        prompt: document.getElementById("prompt").value,
        prompt_2: document.getElementById("prompt_2").value,
        negative_prompt: document.getElementById("negative_prompt").value,
        base_model: document.getElementById("base_model").value,
        second_pass_model: document.getElementById("second_pass_model").value,
        second_pass_mode: document.getElementById("second_pass_mode").value,
        steps: document.getElementById("steps").value,
        cfg: document.getElementById("cfg").value,
        width: document.getElementById("width").value,
        height: document.getElementById("height").value,
        num_images: document.getElementById("num_images").value,
        seed: getSeed(),
        experimental_compress: document.getElementById("experimental_compress").checked,
        loras: Array.from(document.querySelectorAll(".lora-card")).map(card => ({
            name: card.dataset.lora,
            weight: parseFloat(card.querySelector("input[type=range]").value),
            enabled: card.querySelector("input[type=checkbox]").checked,
        })),
        scheduler: document.getElementById("scheduler").value,
        refinement_strength: document.getElementById("refinement_strength").value,
    };

    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

export function loadState() {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    try {
        return JSON.parse(raw);
    } catch {
        return null;
    }
}
