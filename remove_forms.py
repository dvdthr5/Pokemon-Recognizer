import os
import re
import shutil

# Paths
BASE_DIR = os.path.dirname(__file__)
POKEMON_LIST = os.path.join(BASE_DIR, "pokemon_list.txt")
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")

# Patterns
FORM_PATTERN = re.compile(r"\s*\((?!alolan|galarian|hisuian).*?\)", re.IGNORECASE)

def get_base_name(name):
    """Remove disallowed forms but keep Alolan/Galarian/Hisuian."""
    cleaned = FORM_PATTERN.sub("", name).strip().lower()
    return cleaned

def main():
    print("🧹 Cleaning Pokémon list and removing unwanted form folders...")

    # Read Pokémon list
    with open(POKEMON_LIST, "r", encoding="utf-8") as f:
        all_names = [line.strip() for line in f if line.strip()]

    # Keep base and regional variants
    kept_names = sorted(set(get_base_name(name) for name in all_names))

    # Overwrite the cleaned list
    with open(POKEMON_LIST, "w", encoding="utf-8") as f:
        for name in kept_names:
            f.write(name + "\n")

    print(f"✅ Updated Pokémon list saved with {len(kept_names)} entries.")

    # Remove unwanted form folders
    for data_dir in [TRAIN_DIR, TEST_DIR]:
        if not os.path.exists(data_dir):
            continue

        removed_count = 0
        print(f"\n📁 Scanning {data_dir}...")
        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            # If folder name matches unwanted form pattern → delete
            if FORM_PATTERN.search(folder):
                shutil.rmtree(folder_path)
                removed_count += 1
                print(f"❌ Removed folder: {folder}")

        print(f"✅ Finished {data_dir}: {removed_count} form folders removed.")

    print("\n🎯 Cleaning complete. Regional variants preserved.")

if __name__ == "__main__":
    main()