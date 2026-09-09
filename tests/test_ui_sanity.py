import os
import re

import pytest
from playwright.sync_api import Page, expect

@pytest.mark.browser
def test_ui_loads_components(page: Page):
    """
    Verify that key UI components load correctly.
    This assumes the server is already running locally.
    """
    base_url = os.getenv("WEBBDUCK_TEST_BASE_URL", "http://127.0.0.1:8010")
    page.on("console", lambda msg: print(f"BROWSER CONSOLE: {msg.text}"))
    page.on("pageerror", lambda exc: print(f"BROWSER ERROR: {exc}"))
    page.on("response", lambda response: print(f"NETWORK: {response.status} {response.url}"))
    page.goto(base_url)
    
    # 1. Verify Title
    expect(page).to_have_title(re.compile("WebbDuck"))
    
    # 2. Verify Models Load
    model_select = page.locator("#base_model")
    expect(model_select).not_to_have_value("", timeout=15000) 
    options = model_select.locator("option")
    assert options.count() > 1
    
    # 3. Verify Schedulers Load
    scheduler_select = page.locator("#scheduler")
    expect(scheduler_select).not_to_have_value("", timeout=15000)
    
    # 4. Verify Gallery Loads (Empty state or Sessions)
    gallery_sessions = page.locator("#gallery-sessions")
    gallery_empty = page.locator("#gallery-empty")
    page.click(".nav-tab[data-view='gallery']")
    expect(page.locator("#view-gallery")).to_have_class("view active")
    expect(gallery_sessions).to_be_attached()
    expect(gallery_empty).to_be_attached()
    
    # 5. Verify Buttons are clickable (not disabled by default unless intended)
    btn_generate = page.locator("#btn-generate")
    expect(btn_generate).to_be_enabled()


@pytest.mark.browser
def test_krea_identity_provider_ui_adapts_to_selected_model(page: Page):
    """Krea owns one automatic identity provider and one reference anchor."""
    base_url = os.getenv("WEBBDUCK_TEST_BASE_URL", "http://127.0.0.1:8010")
    page.goto(base_url)
    model_select = page.locator("#base_model")
    expect(model_select).not_to_have_value("", timeout=15000)

    krea_value = None
    for i in range(model_select.locator("option").count()):
        value = model_select.locator("option").nth(i).get_attribute("value") or ""
        if "krea" in value.lower():
            krea_value = value
            break
    if not krea_value:
        pytest.skip("No krea-named model present in the catalog")

    model_select.select_option(krea_value)

    provider = page.locator("#ip-adapter-type")
    expect(provider).to_have_value("krea2_identity_edit", timeout=15000)
    expect(provider).to_have_attribute("type", "hidden")

    expect(page.locator("#ip-adapter-grounding-px")).not_to_have_class(re.compile("hidden"))
    expect(page.locator("#ip-adapter-lora-rank")).not_to_have_class(re.compile("hidden"))
    expect(page.locator("#ip-adapter-scale")).to_have_class(re.compile("hidden"))
    expect(page.locator("#ip-adapter-lora-scale")).to_have_class(re.compile("hidden"))

    # Krea accepts exactly one reference, so the file picker itself should not
    # offer multi-select either.
    expect(page.locator("#ip-adapter-refs-upload-input")).not_to_have_attribute("multiple", "")
