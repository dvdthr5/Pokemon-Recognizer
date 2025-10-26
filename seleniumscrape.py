import os
import sys
import time
import random
import requests
from PIL import Image
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import base64
import re

# --- CONFIG ---
BASE_DIR = "data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
TARGET_TRAIN = 240
TARGET_TEST = 60

CHROME_DRIVER_PATH = os.path.join(os.getcwd(), "chromedriver-win64", "chromedriver.exe")  # adjust as needed

# --- Load names ---
def load_pokemon_list(filename="pokemon_list.txt"):
    if not os.path.exists(filename):
        print(f"❌ Pokémon list file '{filename}' not found.")
        sys.exit(1)
    with open(filename, "r") as f:
        return [line.strip().lower() for line in f if line.strip()]

# --- Selenium setup ---
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--log-level=3")
service = Service(CHROME_DRIVER_PATH)
driver = webdriver.Chrome(service=service, options=options)

# Requests session with browser-like headers
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.bing.com/"
})

def download_images(pokemon, folder, target_count, search_modifier=""):
    os.makedirs(folder, exist_ok=True)
    existing = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    current_count = len(existing)
    remaining = target_count - current_count
    if remaining <= 0:
        print(f"⚠️ {pokemon}: already {current_count}/{target_count}, skipping.")
        return

    modifier = (" " + search_modifier.strip()) if search_modifier else ""
    query = f"{pokemon} pokemon{modifier}".strip()
    search_url = f"https://www.bing.com/images/search?q={query}"

    print(f"\n🔍 {pokemon}: need {remaining} more images. Searching '{query}'...")
    driver.get(search_url)

    # scroll and try to load more results
    for _ in range(18):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.2)
        try:
            more = driver.find_element(By.CSS_SELECTOR, "a#mmComponent_moreResults, a.btn_seemore")
            if more.is_displayed():
                more.click()
                time.sleep(1.0)
        except Exception:
            pass

    imgs = driver.find_elements(By.TAG_NAME, "img")
    urls_set = set()
    for img in imgs:
        for attr in ("src", "data-src", "data-lazy-src"):
            val = img.get_attribute(attr)
            if not val:
                continue
            urls_set.add(val)
    urls = list(urls_set)
    random.shuffle(urls)

    def _save_image_from_bytes(data: bytes, out_path: str) -> bool:
        try:
            img = Image.open(BytesIO(data))
            img = img.convert("RGB")
            # basic size filter to avoid tiny sprites and icons
            if img.width < 80 or img.height < 80:
                return False
            img.save(out_path, "JPEG")
            return True
        except Exception:
            return False

    count = current_count + 1
    for url in urls:
        if count > target_count:
            break
        try:
            # handle data URLs (base64)
            if url.startswith("data:image"):
                try:
                    match = re.match(r"data:image/[^;]+;base64,(.*)", url, re.IGNORECASE)
                    if not match:
                        continue
                    data = base64.b64decode(match.group(1))
                    fname = f"{pokemon}_{count:03d}.jpg"
                    fpath = os.path.join(folder, fname)
                    if os.path.exists(fpath):
                        continue
                    if _save_image_from_bytes(data, fpath):
                        print(f"✅ Saved: {fname} (data URI)")
                        count += 1
                    else:
                        print("⚠️ Skip image: too small or invalid (data URI)")
                    continue
                except Exception as e:
                    print(f"⚠️ Skip data URI: {type(e).__name__}")
                    continue

            # only http(s)
            if not (url.startswith("http://") or url.startswith("https://")):
                continue

            resp = session.get(url, timeout=12, stream=True)
            ctype = resp.headers.get("Content-Type", "").lower()
            if "image" not in ctype:
                # try to follow redirect once and fetch final content
                loc = resp.headers.get("Location")
                if loc and (loc.startswith("http://") or loc.startswith("https://")):
                    resp = session.get(loc, timeout=12, stream=True)
                    ctype = resp.headers.get("Content-Type", "").lower()
            if "image" not in ctype:
                raise ValueError(f"non-image content-type {ctype or 'unknown'}")
            data = resp.content

            fname = f"{pokemon}_{count:03d}.jpg"
            fpath = os.path.join(folder, fname)
            if os.path.exists(fpath):
                continue

            if _save_image_from_bytes(data, fpath):
                print(f"✅ Saved: {fname}")
                count += 1
            else:
                print("⚠️ Skip image: too small or invalid")
        except Exception as e:
            # keep going on errors but keep the log terse
            print(f"⚠️ Skip image: {type(e).__name__}")

    print(f"✨ Done with {pokemon}. Now {count-1}/{target_count} in {folder}")

def main():
    pokemons = load_pokemon_list()
    print(f"Will process {len(pokemons)} Pokémon from the list.")

    for pokemon in pokemons:
        print("\n========================")
        print("Processing:", pokemon)

        train_folder = os.path.join(TRAIN_DIR, pokemon)
        download_images(pokemon, train_folder, TARGET_TRAIN)

        test_folder = os.path.join(TEST_DIR, pokemon)
        download_images(pokemon, test_folder, TARGET_TEST, search_modifier="official artwork")

    driver.quit()
    print("\n✅ Done with all Pokémon.")

if __name__ == "__main__":
    main()