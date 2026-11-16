"""Open each app in a real browser, and click 'wake up' if it is asleep."""
import re
from playwright.sync_api import sync_playwright

APPS = ["https://noor-portfolio-clinical-trial-assistant.streamlit.app",
        "https://noor-portfolio-delivery-agent.streamlit.app"]

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    for url in APPS:
        page.goto(url, wait_until="domcontentloaded", timeout=90_000)
        page.wait_for_timeout(5_000)
        button = page.get_by_role("button", name=re.compile("get this app back up", re.I))
        if button.count():
            button.first.click()
            print("WOKE UP", url)
            page.wait_for_timeout(60_000)        # give it a minute to start
        else:
            print("AWAKE", url)
    browser.close()