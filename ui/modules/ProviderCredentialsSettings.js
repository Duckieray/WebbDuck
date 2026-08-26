const PROVIDERS = [
    { id: 'huggingface', label: 'Hugging Face Token', placeholder: 'hf_…' },
    { id: 'civitai', label: 'Civitai Token', placeholder: 'Optional API token' },
];

async function request(url, options = {}) {
    const response = await fetch(url, {
        ...options,
        headers: {
            ...(options.body ? { 'Content-Type': 'application/json' } : {}),
            ...(options.headers || {}),
        },
    });
    if (!response.ok) {
        let message = `HTTP ${response.status}`;
        try {
            const payload = await response.json();
            message = payload?.detail || payload?.error || message;
        } catch {
            // Keep the status fallback.
        }
        throw new Error(message);
    }
    return response.json();
}

function statusText(status) {
    if (!status?.configured) return 'Not configured';
    return status.source === 'environment'
        ? 'Configured by environment override'
        : 'Configured in WebbDuck Settings';
}

function setProviderStatus(provider, status) {
    const node = document.getElementById(`provider-credential-status-${provider}`);
    if (node) node.textContent = statusText(status);
}

function renderStatus(payload) {
    const providers = payload?.providers || {};
    for (const provider of PROVIDERS) {
        setProviderStatus(provider.id, providers[provider.id]);
    }
}

async function refreshStatus() {
    try {
        renderStatus(await request('/settings/provider-credentials'));
    } catch (error) {
        for (const provider of PROVIDERS) {
            setProviderStatus(provider.id, { configured: false });
        }
        console.warn('Could not load provider credential status:', error);
    }
}

function providerRow(provider) {
    const group = document.createElement('div');
    group.className = 'form-group';

    const label = document.createElement('label');
    label.className = 'form-label';
    label.htmlFor = `provider-credential-${provider.id}`;
    label.textContent = provider.label;

    const row = document.createElement('div');
    row.className = 'remote-plugin-connect-row';

    const input = document.createElement('input');
    input.className = 'input';
    input.type = 'password';
    input.id = `provider-credential-${provider.id}`;
    input.placeholder = provider.placeholder;
    input.autocomplete = 'off';
    input.spellcheck = false;

    const save = document.createElement('button');
    save.className = 'btn btn-primary btn-sm';
    save.type = 'button';
    save.textContent = 'Save';

    const clear = document.createElement('button');
    clear.className = 'btn btn-secondary btn-sm';
    clear.type = 'button';
    clear.textContent = 'Clear';

    const status = document.createElement('span');
    status.className = 'form-hint';
    status.id = `provider-credential-status-${provider.id}`;
    status.textContent = 'Checking…';

    save.addEventListener('click', async () => {
        const token = input.value.trim();
        if (!token) {
            status.textContent = 'Enter a token to save, or use Clear.';
            return;
        }
        save.disabled = true;
        try {
            const payload = await request(`/settings/provider-credentials/${encodeURIComponent(provider.id)}`, {
                method: 'PUT',
                body: JSON.stringify({ token }),
            });
            input.value = '';
            renderStatus(payload);
        } catch (error) {
            status.textContent = `Save failed: ${error.message}`;
        } finally {
            save.disabled = false;
        }
    });

    clear.addEventListener('click', async () => {
        clear.disabled = true;
        try {
            const payload = await request(`/settings/provider-credentials/${encodeURIComponent(provider.id)}`, {
                method: 'DELETE',
            });
            input.value = '';
            renderStatus(payload);
        } catch (error) {
            status.textContent = `Clear failed: ${error.message}`;
        } finally {
            clear.disabled = false;
        }
    });

    row.append(input, save, clear);
    group.append(label, row, status);
    return group;
}

function installProviderCredentialSettings() {
    const modalBody = document.querySelector('#settings-modal .modal-body');
    if (!modalBody || document.getElementById('provider-credentials-settings')) return;

    const section = document.createElement('div');
    section.id = 'provider-credentials-settings';
    section.className = 'form-group';

    const heading = document.createElement('label');
    heading.className = 'form-label';
    heading.textContent = 'Model Provider Credentials';

    const hint = document.createElement('span');
    hint.className = 'form-hint';
    hint.textContent = 'Optional. Public models work without tokens; add credentials only for gated/private provider assets.';

    section.append(heading, hint);
    for (const provider of PROVIDERS) section.append(providerRow(provider));

    const remotePluginField = document.getElementById('remote-plugin-base')?.closest('.form-group');
    if (remotePluginField) modalBody.insertBefore(section, remotePluginField);
    else modalBody.append(section);

    document.getElementById('open-settings-modal')?.addEventListener('click', refreshStatus);
    refreshStatus();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', installProviderCredentialSettings, { once: true });
} else {
    installProviderCredentialSettings();
}
