import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import json
from config import MODEL_PATH, CLASS_NAMES_PATH, TEST_FOLDER, set_seed

set_seed()

# --- LOAD TRAINED MODEL ---
print(f"📦 Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# --- LOAD CLASS NAMES ---
if not os.path.exists(CLASS_NAMES_PATH):
    print("❌ Missing class_names.json — please run retrain_model.py first.")
    exit()

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

if not class_names:
    print("❌ No Pokémon names found in class_names.json!")
    exit()

print(f"✅ Loaded {len(class_names)} Pokémon names from {CLASS_NAMES_PATH}.\n")

# --- CHECK MODEL OUTPUT SIZE ---
model_classes = model.output_shape[-1]
if model_classes != len(class_names):
    print(f"⚠️ WARNING: Model expects {model_classes} classes, "
          f"but class_names.json lists {len(class_names)}.")
    print("👉 Make sure this matches your training dataset!")
    print()

# --- COLLECT ALL TEST IMAGES ---
image_files = [
    f for f in os.listdir(TEST_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not image_files:
    print(f"❌ No images found in '{TEST_FOLDER}' folder.")
    exit()

print(f"🎴 Evaluating {len(image_files)} Pokémon images from '{TEST_FOLDER}'...\n")

# --- LOOP THROUGH EACH IMAGE ---
for file_name in image_files:
    img_path = os.path.join(TEST_FOLDER, file_name)

    try:
        # --- LOAD AND PREPROCESS IMAGE ---
        img = image.load_img(img_path, target_size=(180, 180))  # match training size
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # --- MAKE PREDICTION ---
        prediction = model.predict(img_array, verbose=0)
        predicted_index = int(np.argmax(prediction))
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction))

        # --- PRINT RESULT ---
        print(f"{file_name:30} → {predicted_class:20} ({confidence * 100:.2f}% confidence)")

        # --- DISPLAY IMAGE ---
        plt.figure(figsize=(3, 3))
        plt.imshow(image.load_img(img_path))
        plt.title(f"{predicted_class.capitalize()} ({confidence * 100:.1f}%)")
        plt.axis("off")
        plt.show()

    except Exception as e:
        print(f"⚠️ Skipped {file_name} due to error: {e}")

print("\n✅ Finished evaluating all test images.")
