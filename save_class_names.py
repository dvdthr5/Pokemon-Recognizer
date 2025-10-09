import tensorflow as tf
import json
import os

DATA_DIR = "data/train"
OUTPUT_FILE = "class_names.json"

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=(180, 180),
    batch_size=32
)

class_names = train_ds.class_names

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

print(f"✅ Saved {len(class_names)} Pokémon class names to {OUTPUT_FILE}")
