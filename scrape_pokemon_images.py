import os
import sys
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import urllib.parse
import random

# --- CONFIG ---
BASE_DIR = "data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

TARGET_TRAIN_IMAGES = 100
TARGET_TEST_IMAGES = 10
USER_AGENT = {"User-Agent": "Mozilla/5.0"}

# --- USAGE INFO ---
if len(sys.argv) < 2:
    print("Usage: python scrape_pokemon_images.py <pokemon1> <pokemon2> ...")
    print("Example: python scrape_pokemon_images.py pikachu eevee snorlax")
    sys.exit(0)

# --- CREATE BASE FOLDERS ---
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# --- FUNCTION: DOWNLOAD IMAGES ---
def download_images(pokemon, folder, target_count):
    """Scrape and download images until the folder has target_count images."""
    current_count = len([f for f in os.listdir(folder)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    remaining = target_count - current_count
    if remaining <= 0:
        print(f"⚠️ {pokemon.capitalize()} already has {current_count}/{target_count} images — skipping.")
        return

    print(f"\n🔍 Searching Bing Images for '{pokemon}' ({remaining} more images needed)...")

    # --- Build search URL ---
    query = urllib.parse.quote(pokemon + " pokemon")
    url = f"https://www.bing.com/images/search?q={query}&form=HDRSC2&first=1&tsc=ImageBasicHover"

    try:
        response = requests.get(url, headers=USER_AGENT, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print(f"❌ Failed to fetch page for {pokemon}: {e}")
        return

    # --- Extract image URLs ---
    image_elements = soup.find_all("img", {"class": "mimg"})
    image_urls = [img["src"] for img in image_elements if img.get("src")]

    if not image_urls:
        print(f"⚠️ No images found for {pokemon}.")
        return

    # Shuffle URLs to get a random mix
    random.shuffle(image_urls)

    print(f"🖼️ Found {len(image_urls)} candidate images for {pokemon}. Downloading up to {remaining}...")

    count = current_count + 1
    for img_url in image_urls:
        if count > target_count:
            break
        try:
            img_data = requests.get(img_url, headers=USER_AGENT, timeout=10)
            img = Image.open(BytesIO(img_data.content)).convert("RGB")

            filename = f"{pokemon}_{count:03d}.jpg"
            file_path = os.path.join(folder, filename)
            img.save(file_path, "JPEG")

            print(f"✅ Saved: {filename}")
            count += 1
        except Exception as e:
            print(f"⚠️ Skipped one image ({e})")

    print(f"✨ Finished: {pokemon} now has {count - 1}/{target_count} images.\n")


# --- MAIN LOOP ---
for pokemon in sys.argv[1:]:
    pokemon = pokemon.strip().lower()

    print(f"\n==============================")
    print(f"🔹 Processing Pokémon: {pokemon}")
    print(f"==============================")

    # --- Create folders if missing ---
    train_folder = os.path.join(TRAIN_DIR, pokemon)
    test_folder = os.path.join(TEST_DIR, pokemon)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # --- Fill training and testing sets ---
    download_images(pokemon, train_folder, TARGET_TRAIN_IMAGES)
    download_images(pokemon + " art", test_folder, TARGET_TEST_IMAGES)  # slight variation to get different images

print("\n✅ All requested Pokémon processed successfully!")
