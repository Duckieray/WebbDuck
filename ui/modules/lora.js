import { saveState, loadState } from './state.js';

let loraDefaults = {};
let modelDefaults = {};

export function getModelDefaults() {
    return modelDefaults;
}

export function setModelDefaults(defaults) {
    modelDefaults = defaults;
}

export async function loadLoras() {
    const base_model = document.getElementById("base_model");
    const loraSelect = document.getElementById("lora-select");
    const loraSelected = document.getElementById("lora-selected");

    const baseModelVal = base_model.value;
    const loras = await fetch(
        `/models/${encodeURIComponent(baseModelVal)}/loras`
    ).then(r => r.json());

    loraDefaults = {};

    loraSelect.innerHTML =
        `<option value="" disabled selected>➕ Add LoRA…</option>`;

    loras.forEach(l => {
        loraDefaults[l.name] = l.weight ?? 1.0;

        const opt = document.createElement("option");
        opt.value = l.name;
        opt.textContent = l.name;
        loraSelect.appendChild(opt);
    });
}

export function addLoraCard(name, weight = 1.0, enabled = true) {
    const loraSelected = document.getElementById("lora-selected");
    const card = document.createElement("div");
    card.className = "lora-card";
    card.dataset.lora = name;

    card.innerHTML = `
    <div class="lora-card-header">
      <input type="checkbox" ${enabled ? "checked" : ""}>
      <div class="lora-name">${name}</div>
      <div class="lora-remove">✕</div>
    </div>

    <div class="lora-weight">
      <input
        type="range"
        min="0"
        max="2"
        step="0.05"
        value="${weight}"
      >
      <span>${Number(weight).toFixed(2)}</span>
    </div>
  `;

    const slider = card.querySelector("input[type=range]");
    const value = card.querySelector("span");

    slider.oninput = () => {
        value.textContent = Number(slider.value).toFixed(2);
        saveState();
    };

    card.querySelector("input[type=checkbox]").onchange = saveState;

    card.querySelector(".lora-remove").onclick = () => {
        card.remove();
        saveState();
    };

    loraSelected.appendChild(card);
}

export function initLoraEvents() {
    const loraSelect = document.getElementById("lora-select");

    loraSelect.onchange = () => {
        const name = loraSelect.value;
        if (!name) return;

        if (document.querySelector(`[data-lora="${name}"]`)) {
            loraSelect.value = "";
            return;
        }

        const saved = loadState()?.loras?.find(l => l.name === name);

        addLoraCard(
            name,
            saved?.weight ?? loraDefaults[name] ?? 1.0,
            saved?.enabled ?? true
        );
        loraSelect.value = "";
        saveState();
    };
}
