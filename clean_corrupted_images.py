import os
from PIL import Image

DATA_DIR = "data"  # root folder containing 'train' and 'test'
removed = 0
checked = 0

for subset in ["train", "test"]:
    subset_path = os.path.join(DATA_DIR, subset)
    for root, _, files in os.walk(subset_path):
        for file in files:
            if not file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            path = os.path.join(root, file)
            checked += 1
            try:
                with Image.open(path) as img:
                    img.verify()  # check integrity
            except Exception:
                os.remove(path)
                removed += 1
                print(f"🗑️ Removed corrupted file: {path}")

print(f"✅ Scan complete. Checked {checked} files, removed {removed} corrupted or unreadable images.")