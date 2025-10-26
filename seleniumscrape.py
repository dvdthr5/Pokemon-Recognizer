import argparse
import json
import os
import sys
import time
import urllib.parse
from dataclasses import dataclass
from typing import Iterable, List, Optional

import requests
from PIL import Image
from io import BytesIO
from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    ElementNotInteractableException,
    NoSuchElementException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

# --- CONFIG / CONSTANTS ---
BASE_DIR = "data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
DEFAULT_TARGET_TRAIN = 240
DEFAULT_TARGET_TEST = 60

ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")
ALLOWED_MIME_KEYWORDS = ("image/jpeg", "image/jpg", "image/png")
CARD_KEYWORDS = ("card", "tcg", "trading", "promo", "full art", "gx", "ex", "vmax", "vstar", "trainer")
TRAIN_CARD_QUERIES = [
    "pokemon tcg card scan",
    "pokemon trading card holo",
    "pokemon tcg full art card",
    "pokemon tcg promo card",
]
TEST_CARD_QUERIES = [
    "pokemon tcg card illustration",
    "pokemon tcg reverse holo card",
    "pokemon tcg promo scan",
]
GOOGLE_IMAGE_URL = "https://www.google.com/search?tbm=isch&q="
DEFAULT_MIN_IMAGE_WIDTH = 150
DEFAULT_MIN_IMAGE_HEIGHT = 150
SEE_MORE_SELECTORS = [
    ".mye4qd",
]
SEE_MORE_CLICK_ATTEMPTS = 6
SCROLL_PASSES = 30
SCROLL_PAUSE_SECONDS = 0.8
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


@dataclass
class ImageCandidate:
    url: str
    alt_text: str
    width: Optional[int] = None
    height: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Pokémon TCG images for train/test datasets.")
    parser.add_argument(
        "names",
        nargs="*",
        help="Optional list of Pokémon names to scrape. Defaults to names in pokemon_list.txt.",
    )
    parser.add_argument("--train-target", type=int, default=DEFAULT_TARGET_TRAIN, help="Images per Pokémon for train.")
    parser.add_argument("--test-target", type=int, default=DEFAULT_TARGET_TEST, help="Images per Pokémon for test.")
    parser.add_argument(
        "--pokemon-list",
        default="pokemon_list.txt",
        help="Path to the text file that lists Pokémon names (one per line).",
    )
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Show the Chrome window while scraping (useful for debugging).",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=DEFAULT_MIN_IMAGE_WIDTH,
        help="Minimum width (pixels) a downloaded image must have to be saved.",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=DEFAULT_MIN_IMAGE_HEIGHT,
        help="Minimum height (pixels) a downloaded image must have to be saved.",
    )
    return parser.parse_args()


def load_pokemon_list(filename: str) -> List[str]:
    if not os.path.exists(filename):
        print(f"❌ Pokémon list file '{filename}' not found.")
        sys.exit(1)
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


def resolve_driver_path() -> Optional[str]:
    env_path = os.environ.get("CHROMEDRIVER") or os.environ.get("CHROME_DRIVER")
    candidate_paths = [
        env_path,
        os.path.join(os.getcwd(), "chromedriver-win64", "chromedriver.exe"),
        os.path.join(os.getcwd(), "chromedriver-mac-arm64", "chromedriver"),
        os.path.join(os.getcwd(), "chromedriver-mac-x64", "chromedriver"),
    ]
    for path in candidate_paths:
        if path and os.path.exists(path):
            return path
    return None


def create_webdriver(show_browser: bool = False) -> webdriver.Chrome:
    options = Options()
    if not show_browser:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--log-level=3")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1600,1200")

    driver_path = resolve_driver_path()
    try:
        if driver_path:
            service = Service(executable_path=driver_path)
            return webdriver.Chrome(service=service, options=options)
        return webdriver.Chrome(options=options)
    except WebDriverException as exc:
        print("❌ Unable to start ChromeDriver.")
        print(str(exc))
        sys.exit(1)


def ensure_base_dirs() -> None:
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)


def ensure_split_folder(split_dir: str, pokemon: str) -> str:
    folder = os.path.join(split_dir, pokemon)
    os.makedirs(folder, exist_ok=True)
    return folder


def count_image_files(folder: str) -> int:
    return len([f for f in os.listdir(folder) if f.lower().endswith(ALLOWED_EXTENSIONS)])


def build_query(pokemon: str, modifier: str) -> str:
    core = f"{pokemon} pokemon".strip()
    return f"{core} {modifier}".strip()


def scroll_results(driver: webdriver.Chrome, passes: int = SCROLL_PASSES) -> None:
    last_height = 0
    for _ in range(passes):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_SECONDS)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


def click_see_more_button(driver: webdriver.Chrome) -> bool:
    for selector in SEE_MORE_SELECTORS:
        try:
            button = driver.find_element(By.CSS_SELECTOR, selector)
        except NoSuchElementException:
            continue
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
        time.sleep(0.5)
        try:
            driver.execute_script("arguments[0].click();", button)
        except ElementClickInterceptedException:
            try:
                button.click()
            except Exception:
                continue
        except ElementNotInteractableException:
            continue
        time.sleep(1.2)
        return True
    return False


def expand_search_results(driver: webdriver.Chrome) -> None:
    scroll_results(driver)
    for _ in range(SEE_MORE_CLICK_ATTEMPTS):
        clicked = click_see_more_button(driver)
        if not clicked:
            break
        scroll_results(driver, passes=max(4, SCROLL_PASSES // 2))


def has_allowed_extension(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.lower()
    return any(path.endswith(ext) for ext in ALLOWED_EXTENSIONS)


def looks_like_card(url: str, alt_text: str, modifier: str) -> bool:
    haystack = f"{alt_text} {url} {modifier}".lower()
    return any(keyword in haystack for keyword in CARD_KEYWORDS)


def gather_candidates(driver: webdriver.Chrome, pokemon: str, modifier: str) -> List[ImageCandidate]:
    query = build_query(pokemon, modifier)
    search_url = f"{GOOGLE_IMAGE_URL}{urllib.parse.quote_plus(query)}"

    print(f"\n🔍 Searching: '{query}'")
    driver.get(search_url)

    # Scroll to load images
    for _ in range(SCROLL_PASSES):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_SECONDS)
        try:
            see_more = driver.find_element(By.CSS_SELECTOR, ".mye4qd")
            if see_more.is_displayed():
                driver.execute_script("arguments[0].click();", see_more)
                time.sleep(SCROLL_PAUSE_SECONDS)
        except NoSuchElementException:
            pass

    images = driver.find_elements(By.CSS_SELECTOR, ".rg_i")
    candidates: List[ImageCandidate] = []
    seen_urls = set()
    for img in images:
        src = img.get_attribute("src") or img.get_attribute("data-src")
        alt_text = img.get_attribute("alt") or ""
        if not src:
            continue
        if src in seen_urls:
            continue
        seen_urls.add(src)
        candidates.append(ImageCandidate(url=src, alt_text=alt_text))
    print(f"🖼️ Found {len(candidates)} candidates for '{pokemon}' ({modifier}).")
    return candidates


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode in ("RGB", "L"):
        return image.convert("RGB")
    return image.convert("RGB")


def fetch_image(
    session: requests.Session,
    url: str,
    folder: str,
    pokemon: str,
    index: int,
    min_width: int,
    min_height: int,
) -> Optional[str]:
    try:
        response = session.get(url, timeout=12)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if content_type:
            if not any(mime in content_type for mime in ALLOWED_MIME_KEYWORDS):
                return None
        else:
            if not has_allowed_extension(url):
                return None

        is_png = "png" in content_type if content_type else url.lower().endswith(".png")
        image_format = "PNG" if is_png else "JPEG"
        extension = ".png" if image_format == "PNG" else ".jpg"
        filename = f"{pokemon}_{index:03d}{extension}"
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            return None

        raw_image = Image.open(BytesIO(response.content))
        raw_image.verify()
        raw_image = Image.open(BytesIO(response.content))
        width, height = raw_image.size
        if width < min_width or height < min_height:
            return None

        if image_format == "JPEG":
            raw_image = ensure_rgb(raw_image)

        raw_image.save(filepath, image_format)
        return filename
    except Exception:
        return None


def download_images_for_modifier(
    driver: webdriver.Chrome,
    session: requests.Session,
    pokemon: str,
    folder: str,
    target_count: int,
    modifier: str,
    min_width: int,
    min_height: int,
) -> int:
    current_count = count_image_files(folder)
    if current_count >= target_count:
        return current_count

    candidates = gather_candidates(driver, pokemon, modifier)
    next_index = current_count + 1
    saved = 0
    for candidate in candidates:
        if current_count + saved >= target_count:
            break
        if candidate.width and candidate.width < min_width:
            continue
        if candidate.height and candidate.height < min_height:
            continue
        saved_name = fetch_image(
            session,
            candidate.url,
            folder,
            pokemon,
            next_index,
            min_width,
            min_height,
        )
        if saved_name:
            saved += 1
            next_index += 1
            print(f"✅ Saved: {saved_name}")
    total = current_count + saved
    print(f"✨ {pokemon} ({modifier}): {total}/{target_count} images in {folder}")
    return total


def fill_split(
    driver: webdriver.Chrome,
    session: requests.Session,
    pokemon: str,
    folder: str,
    target_count: int,
    min_width: int,
    min_height: int,
    modifiers: Iterable[str],
) -> None:
    for modifier in modifiers:
        total = download_images_for_modifier(
            driver,
            session,
            pokemon,
            folder,
            target_count,
            modifier,
            min_width,
            min_height,
        )
        if total >= target_count:
            break
    final_total = count_image_files(folder)
    if final_total < target_count:
        print(f"⚠️ {pokemon}: only {final_total}/{target_count} images in {folder}. Try rerunning later.")


def main() -> None:
    args = parse_args()
    ensure_base_dirs()
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)

    pokemons = [name.strip().lower() for name in args.names if name.strip()]
    if not pokemons:
        pokemons = load_pokemon_list(args.pokemon_list)
    print(f"Will process {len(pokemons)} Pokémon.")

    driver = create_webdriver(show_browser=args.headful)
    try:
        for pokemon in pokemons:
            print("\n========================")
            print(f"Processing: {pokemon}")

            train_folder = ensure_split_folder(TRAIN_DIR, pokemon)
            test_folder = ensure_split_folder(TEST_DIR, pokemon)

            fill_split(
                driver,
                session,
                pokemon,
                train_folder,
                args.train_target,
                args.min_width,
                args.min_height,
                TRAIN_CARD_QUERIES,
            )
            fill_split(
                driver,
                session,
                pokemon,
                test_folder,
                args.test_target,
                args.min_width,
                args.min_height,
                TEST_CARD_QUERIES,
            )

        print("\n✅ Finished scraping all requested Pokémon.")
    finally:
        driver.quit()
        session.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user.")
