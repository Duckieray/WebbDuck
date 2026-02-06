export async function populateSelect(id, url, includeNone = true) {
    const data = await fetch(url).then(r => r.json());
    const el = document.getElementById(id);
    if (!el) return;

    el.innerHTML = includeNone ? `<option value="None">None</option>` : "";

    data.forEach(item => {
        if (typeof item === "string") {
            el.innerHTML += `<option value="${item}">${item}</option>`;
        } else {
            el.innerHTML += `
        <option value="${item.name}">
          ${item.name}${item.description ? " — " + item.description : ""}
        </option>`;
        }
    });

    if (id === "scheduler" && !includeNone) {
        if (el.options.length > 0) el.selectedIndex = 0;
    }
}

export async function updateTokenCount(el, which, baseModelVal) {
    if (!el) return;
    const fd = new FormData();
    fd.append("text", el.value);
    fd.append("base_model", baseModelVal);
    fd.append("which", which);

    const r = await fetch("/tokenize", { method: "POST", body: fd });
    const data = await r.json();

    el.dataset.tokens = data.tokens;
    el.dataset.over = data.over;

    el.style.outline = data.tokens > 77 ? "2px solid #ff7b72" : "";
}
