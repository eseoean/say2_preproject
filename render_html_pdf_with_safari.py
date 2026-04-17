from __future__ import annotations

import base64
import sys
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.print_page_options import PrintOptions
from selenium.webdriver.safari.options import Options as SafariOptions


ROOT = Path("/Users/skku_aws2_18/pre_project/say2_preproject")
HTML_PATH = ROOT / "docs" / "preproject_submission_report_20260413_source.html"
PDF_PATH = ROOT / "docs" / "preproject_submission_report_20260413_browser.pdf"


def wait_until_ready(driver: webdriver.Safari) -> None:
    driver.execute_async_script(
        """
        const done = arguments[arguments.length - 1];
        const imagePromises = Array.from(document.images).map((img) => {
          if (img.complete) return Promise.resolve();
          return new Promise((resolve) => {
            img.addEventListener('load', resolve, { once: true });
            img.addEventListener('error', resolve, { once: true });
          });
        });
        const fontPromise = document.fonts ? document.fonts.ready : Promise.resolve();
        Promise.all([fontPromise, Promise.all(imagePromises)]).then(() => {
          setTimeout(() => done(true), 700);
        });
        """
    )


def main() -> int:
    html_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HTML_PATH
    pdf_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else PDF_PATH

    options = SafariOptions()
    driver = webdriver.Safari(options=options)

    try:
        driver.set_window_size(1440, 2200)
        driver.get(html_path.as_uri())
        wait_until_ready(driver)

        print_options = PrintOptions()
        print_options.background = True
        print_options.shrink_to_fit = True
        print_options.page_width = 8.27
        print_options.page_height = 11.69
        print_options.margin_top = 0
        print_options.margin_bottom = 0
        print_options.margin_left = 0
        print_options.margin_right = 0

        pdf_base64 = driver.print_page(print_options)
        pdf_path.write_bytes(base64.b64decode(pdf_base64))
        print(f"Saved {pdf_path}")
        return 0
    finally:
        driver.quit()


if __name__ == "__main__":
    raise SystemExit(main())
