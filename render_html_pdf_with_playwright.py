from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright


ROOT = Path("/Users/skku_aws2_18/pre_project/say2_preproject")
DEFAULT_HTML = ROOT / "sample3_pipeline_presentation_20260416.html"
DEFAULT_PDF = ROOT / "sample3_pipeline_presentation_20260416.pdf"
CHROMIUM = (
    ROOT.parent
    / ".playwright-browsers"
    / "chromium-1208"
    / "chrome-mac-arm64"
    / "Google Chrome for Testing.app"
    / "Contents"
    / "MacOS"
    / "Google Chrome for Testing"
)


async def wait_until_ready(page) -> None:
    await page.evaluate(
        """async () => {
            const imagePromises = Array.from(document.images).map(img => {
              if (img.complete) return Promise.resolve();
              return new Promise(resolve => {
                img.addEventListener('load', resolve, { once: true });
                img.addEventListener('error', resolve, { once: true });
              });
            });
            if (document.fonts) {
              await document.fonts.ready;
            }
            await Promise.all(imagePromises);
            await new Promise(resolve => setTimeout(resolve, 700));
        }"""
    )


async def main() -> None:
    html_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_HTML
    pdf_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else DEFAULT_PDF

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            executable_path=str(CHROMIUM),
        )
        page = await browser.new_page(viewport={"width": 1440, "height": 2200}, device_scale_factor=1.5)
        await page.goto(html_path.as_uri(), wait_until="load")
        await page.emulate_media(media="print")
        await wait_until_ready(page)
        await page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            prefer_css_page_size=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
        )
        await browser.close()
        print(f"Saved {pdf_path}")


if __name__ == "__main__":
    asyncio.run(main())
