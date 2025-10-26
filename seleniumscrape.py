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
from concurrent.futures import ThreadPoolExecutor, as_completed

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
options.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2})
options.add_argument("--blink-settings=imagesEnabled=false")
service = Service(CHROME_DRIVER_PATH)
driver = webdriver.Chrome(service=service, options=options)

def fetch_image(url, folder, pokemon, count):
    try:
        resp = requests.get(url, timeout=10)
        img = Image.open(BytesIO(resp.content))
        img.verify()  # verify image integrity
        img = Image.open(BytesIO(resp.content)).convert("RGB")  # reopen after verify
        fname = f"{pokemon}_{count:03d}.jpg"
        fpath = os.path.join(folder, fname)
        if os.path.exists(fpath):
            return None
        img.save(fpath, "JPEG")
        return fname
    except Exception:
        return None

def download_images(pokemon, folder, target_count, search_modifier=""):
    if not os.path.exists(folder):
        print(f"⚠️ Folder '{folder}' does not exist for {pokemon}. Skipping downloads.")
        return 0

    existing = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    current_count = len(existing)
    remaining = target_count - current_count
    if remaining <= 0:
        print(f"⚠️ {pokemon} ({search_modifier}): already {current_count}/{target_count}, skipping.")
        return current_count

    query = f"{pokemon} pokemon {search_modifier}".strip()
    search_url = f"https://www.bing.com/images/search?q={query}"

    print(f"\n🔍 {pokemon} ({search_modifier}): need {remaining} more images. Searching '{query}'...")
    driver.get(search_url)

    # scroll
    for _ in range(12):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)

    imgs = driver.find_elements("tag name", "img")
    urls = [img.get_attribute("src") for img in imgs if img.get_attribute("src")]
    urls = list(set(urls))
    random.shuffle(urls)

    count = current_count + 1
    saved = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for url in urls:
            if count > target_count:
                break
            futures[executor.submit(fetch_image, url, folder, pokemon, count)] = count
            count += 1

        for future in as_completed(futures):
            fname = future.result()
            if fname:
                saved += 1
                print(f"✅ Saved: {fname}")

    total_now = current_count + saved
    print(f"✨ Done with {pokemon} ({search_modifier}). Now {total_now}/{target_count} in {folder}")
    return total_now

def main():
    pokemons = load_pokemon_list()
    print(f"Will process {len(pokemons)} Pokémon from the list.")

    for pokemon in pokemons:
        print("\n========================")
        print("Processing:", pokemon)

        train_folder = os.path.join(TRAIN_DIR, pokemon)
        half_target = TARGET_TRAIN // 2
        total_saved = 0
        for modifier in ["official artwork", "trading card"]:
            saved = download_images(pokemon, train_folder, half_target + total_saved, search_modifier=modifier)
            total_saved = saved

        print(f"🎯 Total images for {pokemon} in train folder: {total_saved}/{TARGET_TRAIN}")

        test_folder = os.path.join(TEST_DIR, pokemon)
        download_images(pokemon, test_folder, TARGET_TEST, search_modifier="art official illustration")

    driver.quit()
    print("\n✅ Done with all Pokémon.")

if __name__ == "__main__":
    main()
