import os
import difflib

BASE_DIR = "data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

TARGET_TRAIN = 100
TARGET_TEST = 10

def count_images(folder):
    """Count .jpg/.jpeg/.png images in a folder."""
    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]) if os.path.exists(folder) else 0

# --- Collect Pokémon names from both train and test ---
train_pokemon = set(os.listdir(TRAIN_DIR)) if os.path.exists(TRAIN_DIR) else set()
test_pokemon = set(os.listdir(TEST_DIR)) if os.path.exists(TEST_DIR) else set()
all_pokemon = sorted(train_pokemon | test_pokemon)

# --- Try to detect near-duplicates (like bulbasuar vs bulbasaur) ---
normalized = {}
for name in all_pokemon:
    # if a similar name already exists, map it to the existing one
    match = difflib.get_close_matches(name, normalized.keys(), n=1, cutoff=0.88)
    if match:
        normalized[name] = match[0]
    else:
        normalized[name] = name

merged_names = sorted(set(normalized.values()))

# --- Print Summary ---
print("\n📊 DATASET SUMMARY\n")
print(f"{'Pokémon':15} {'Train':>7} {'Test':>7} {'Status':>10}")
print("-" * 45)

for pokemon in merged_names:
    # find any matching folder variants (handles typos)
    train_match = difflib.get_close_matches(pokemon, train_pokemon, n=1, cutoff=0.88)
    test_match = difflib.get_close_matches(pokemon, test_pokemon, n=1, cutoff=0.88)

    train_folder = os.path.join(TRAIN_DIR, train_match[0]) if train_match else None
    test_folder = os.path.join(TEST_DIR, test_match[0]) if test_match else None

    train_count = count_images(train_folder) if train_folder else 0
    test_count = count_images(test_folder) if test_folder else 0

    status = "✅ Complete" if train_count >= TARGET_TRAIN and test_count >= TARGET_TEST else "⚠️ Incomplete"
    print(f"{pokemon:15} {train_count:7} {test_count:7} {status:>10}")

print("\n✅ Summary complete!")
