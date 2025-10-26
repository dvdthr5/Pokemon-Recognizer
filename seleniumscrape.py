import argparse
import os
import sys
import time
import random
import urllib.parse
from typing import List, Optional
from io import BytesIO

import requests
from PIL import Image
from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    WebDriverException,
    ElementClickInterceptedException,
    ElementNotInteractableException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

# --- CONFIG / CONSTANTS ---
BASE_DIR = "data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN_TARGET = 240
TEST_TARGET = 60

ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")
CARD_KEYWORDS = (
    "card",
    "tcg",
    "trading",
    "promo",
    "full art",
    "holo",
    "gx",
    "ex",
    "vmax",
    "vstar",
    "trainer",
)
TRAIN_CARD_QUERIES = [
    "tcg card scan",
    "pokemon trading card full art",
    "pokemon promo card",
    "pokemon tcg holo",
]
TEST_CARD_QUERIES = [
    "pokemon tcg card artwork",
    "pokemon tcg illustration",
    "pokemon tcg reverse holo",
]
BING_IMAGE_SEARCH_URL = "https://www.bing.com/images/search?q="
SCROLL_PAUSE = 0.8
SCROLL_PASSES = 12
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Pokémon TCG card images for train/test datasets.")
    parser.add_argument(
        "names",
        nargs="*",
        help="Optional list of Pokémon names to scrape. Defaults to names in pokemon_list.txt.",
    )
    parser.add_argument("--train-target", type=int, default=TRAIN_TARGET, help="Images per Pokémon for train.")
    parser.add_argument("--test-target", type=int, default=TEST_TARGET, help="Images per Pokémon for test.")
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
    core = f"{pokemon} pokemon card".strip()
    return f"{core} {modifier}".strip()


def scroll_results(driver: webdriver.Chrome, passes: int = SCROLL_PASSES) -> None:
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(passes):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


def gather_candidates(driver: webdriver.Chrome, pokemon: str, modifier: str) -> List[str]:
    query = build_query(pokemon, modifier)
    search_url = f"{BING_IMAGE_SEARCH_URL}{urllib.parse.quote_plus(query)}"
    print(f"\n🔍 Searching for '{query}'")
    driver.get(search_url)
    scroll_results(driver)

    images = driver.find_elements(By.CSS_SELECTOR, "img.mimg")
    urls = set()
    for img in images:
        src = img.get_attribute("src")
        if src and src.startswith("http") and src.lower().endswith(ALLOWED_EXTENSIONS):
            urls.add(src)
    print(f"🖼️ Found {len(urls)} image candidates for '{pokemon}' ({modifier}).")
    return list(urls)


def fetch_and_save_image(
    session: requests.Session,
    url: str,
    folder: str,
    pokemon: str,
    index: int,
) -> Optional[str]:
    try:
        response = session.get(url, timeout=12)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content))
        image.verify()
        image = Image.open(BytesIO(response.content))

        if image.mode != "RGB":
            image = image.convert("RGB")

        filename = f"{pokemon}_{index:03d}.jpg"
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            return None

        image.save(filepath, "JPEG")
        return filename
    except Exception:
        return None


def download_images_for_split(
    driver: webdriver.Chrome,
    session: requests.Session,
    pokemon: str,
    folder: str,
    target_count: int,
    modifiers: List[str],
) -> None:
    print(f"\nStarting download for {pokemon} in folder {folder} aiming for {target_count} images.")
    for modifier in modifiers:
        current_count = count_image_files(folder)
        if current_count >= target_count:
            break
        urls = gather_candidates(driver, pokemon, modifier)
        random.shuffle(urls)
        for url in urls:
            current_count = count_image_files(folder)
            if current_count >= target_count:
                break
            saved_name = fetch_and_save_image(session, url, folder, pokemon, current_count + 1)
            if saved_name:
                print(f"✅ Saved: {saved_name}")
    final_count = count_image_files(folder)
    print(f"✨ {pokemon}: {final_count}/{target_count} images saved in {folder}")
    if final_count < target_count:
        print(f"⚠️ {pokemon}: only {final_count}/{target_count} images saved. Consider rerunning later.")


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

            download_images_for_split(
                driver,
                session,
                pokemon,
                train_folder,
                args.train_target,
                TRAIN_CARD_QUERIES,
            )

            download_images_for_split(
                driver,
                session,
                pokemon,
                test_folder,
                args.test_target,
                TEST_CARD_QUERIES,
            )

        print("\n✅ Finished scraping all requested Pokémon TCG card images.")
    finally:
        driver.quit()
        session.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user.")
