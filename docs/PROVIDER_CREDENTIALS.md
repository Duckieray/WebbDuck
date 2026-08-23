# Provider Credentials

WebbDuck can optionally store credentials for model providers under **Settings → Model Provider Credentials**.

Supported providers today:

- Hugging Face
- Civitai

Credentials are optional. Public model discovery/downloads should continue to work anonymously when the provider permits it. Tokens are only needed for gated/private assets or provider operations that require authentication.

## Storage and precedence

Settings-managed credentials are stored locally in:

```text
~/.webbduck/provider_credentials.json
```

On POSIX systems WebbDuck writes the file with mode `0600` and its parent directory with mode `0700` when possible. The browser never receives saved token contents and tokens are not stored in `localStorage` or ordinary UI state.

Explicit environment variables take precedence over Settings-managed values:

```text
HF_TOKEN
HUGGING_FACE_HUB_TOKEN
CIVITAI_TOKEN
CIVITAI_API_TOKEN
CIVITAI_API_KEY
```

At startup WebbDuck exports Settings-managed credentials into canonical provider variables only when an explicit environment override is not already present. Local plugin/worker processes therefore inherit the same optional credentials.

Advanced deployments may override the credentials-file location with:

```text
WEBBDUCK_CREDENTIALS_FILE
```

## API contract

The Settings UI uses:

```text
GET    /settings/provider-credentials
PUT    /settings/provider-credentials/{provider}
DELETE /settings/provider-credentials/{provider}
```

GET/PUT/DELETE responses contain configuration status only. They never return a saved token value.
