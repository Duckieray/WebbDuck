import pytest
from playwright.sync_api import Page, expect

@pytest.mark.browser
def test_ui_loads_components(page: Page):
    """
    Verify that key UI components load correctly.
    This assumes the server is already running at localhost:8000.
    """
    page.on("console", lambda msg: print(f"BROWSER CONSOLE: {msg.text}"))
    page.on("pageerror", lambda exc: print(f"BROWSER ERROR: {exc}"))
    page.on("response", lambda response: print(f"NETWORK: {response.status} {response.url}"))
    page.goto("http://127.0.0.1:8000")
    
    # 1. Verify Title
    expect(page).to_have_title("WebbDuck - AI Image Studio")
    
    # 2. Verify Models Load
    model_select = page.locator("#base_model")
    # Wait for options to populate (should vary from "Loading...")
    expect(model_select).not_to_have_value("", timeout=15000) 
    # Check that we have options other than the placeholder
    options = model_select.locator("option")
    expect(options).to_have_count_gt(1)
    
    # 3. Verify Schedulers Load
    scheduler_select = page.locator("#scheduler")
    expect(scheduler_select).not_to_have_value("", timeout=15000)
    
    # 4. Verify Gallery Loads (Empty state or Sessions)
    # Either gallery-sessions has children OR gallery-empty is visible
    gallery_sessions = page.locator("#gallery-sessions")
    gallery_empty = page.locator("#gallery-empty")
    
    # Switch to Gallery tab to ensure visibility
    page.click("button[data-view='gallery']")
    
    # Expect one of them to be visible
    expect(gallery_sessions.or_(gallery_empty)).to_be_visible()
    
    # 5. Verify Buttons are clickable (not disabled by default unless intended)
    btn_generate = page.locator("#btn-generate")
    expect(btn_generate).to_be_enabled()

