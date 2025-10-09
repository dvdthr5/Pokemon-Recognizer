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

# --- CONFIG ---
BASE_DIR = "data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
TARGET_TRAIN = 100
TARGET_TEST = 10

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

    query = f"{pokemon} pokemon {search_modifier}"
    search_url = f"https://www.bing.com/images/search?q={query}"

    print(f"\n🔍 {pokemon}: need {remaining} more images. Searching '{query}'...")
    driver.get(search_url)

    # scroll
    for _ in range(12):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)

    imgs = driver.find_elements("tag name", "img")
    urls = [img.get_attribute("src") for img in imgs if img.get_attribute("src")]
    random.shuffle(urls)

    count = current_count + 1
    for url in urls:
        if count > target_count:
            break
        try:
            resp = requests.get(url, timeout=10)
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            fname = f"{pokemon}_{count:03d}.jpg"
            fpath = os.path.join(folder, fname)
            if os.path.exists(fpath):
                continue
            img.save(fpath, "JPEG")
            print(f"✅ Saved: {fname}")
            count += 1
        except Exception as e:
            print(f"⚠️ Skip image: {e}")

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
        download_images(pokemon, test_folder, TARGET_TEST, search_modifier="art official illustration")

    driver.quit()
    print("\n✅ Done with all Pokémon.")

if __name__ == "__main__":
    main()
