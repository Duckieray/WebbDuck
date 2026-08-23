"""FastAPI surface for optional model-provider credentials.

The API never returns saved token contents.  Clients can only inspect provider
status and save/clear a token.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.provider_credentials import (
    clear_provider_token,
    provider_status,
    save_provider_token,
)


router = APIRouter()


class ProviderCredentialUpdate(BaseModel):
    token: str


@router.get("/settings/provider-credentials")
def get_provider_credentials_status():
    return provider_status()


@router.put("/settings/provider-credentials/{provider}")
def put_provider_credential(provider: str, payload: ProviderCredentialUpdate):
    try:
        save_provider_token(provider, payload.token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return provider_status()


@router.delete("/settings/provider-credentials/{provider}")
def delete_provider_credential(provider: str):
    try:
        clear_provider_token(provider)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return provider_status()
