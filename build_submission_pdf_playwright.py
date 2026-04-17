from __future__ import annotations

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright


ROOT = Path("/Users/skku_aws2_18/pre_project/say2_preproject")
HTML_PATH = ROOT / "docs" / "preproject_submission_report_20260413_source.html"
PDF_PATH = ROOT / "docs" / "preproject_submission_report_20260413.pdf"
SCREENSHOT_PATH = ROOT / "docs" / "preproject_submission_report_20260413_preview.png"


async def main() -> None:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            executable_path=str(
                ROOT.parent / ".playwright-browsers" / "chromium-1208" / "chrome-mac-arm64" / "Google Chrome for Testing.app" / "Contents" / "MacOS" / "Google Chrome for Testing"
            ),
        )
        page = await browser.new_page(viewport={"width": 1440, "height": 2200}, device_scale_factor=1.5)
        await page.goto(HTML_PATH.resolve().as_uri(), wait_until="load")
        await page.emulate_media(media="print")
        await page.evaluate(
            """async () => {
                const imgPromises = Array.from(document.images).map(img => {
                  if (img.complete) return Promise.resolve();
                  return new Promise(resolve => {
                    img.addEventListener('load', resolve, { once: true });
                    img.addEventListener('error', resolve, { once: true });
                  });
                });
                if (document.fonts) {
                  await document.fonts.ready;
                }
                await Promise.all(imgPromises);
                await new Promise(resolve => setTimeout(resolve, 600));
            }"""
        )
        await page.pdf(
            path=str(PDF_PATH),
            format="A4",
            print_background=True,
            prefer_css_page_size=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
        )
        await page.screenshot(path=str(SCREENSHOT_PATH), full_page=True)
        await browser.close()
        print(f"Saved {PDF_PATH}")
        print(f"Saved {SCREENSHOT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
