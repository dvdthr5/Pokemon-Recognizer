import os
import requests
from bs4 import BeautifulSoup

URL = "https://pokemondb.net/pokedex/all"
OUTPUT_FILE = "pokemon_list.txt"

def fetch_pokemon_names():
    """Fetch all Pokémon names (including region variants) from PokémonDB."""
    resp = requests.get(URL)
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "pokedex"})

    names = []
    for row in table.tbody.find_all("tr"):
        name_cell = row.find("a", {"class": "ent-name"})
        variant_cell = row.find("small", {"class": "text-muted"})

        if not name_cell:
            continue

        base_name = name_cell.text.strip().lower()
        # If a variant exists, append it in parentheses, e.g. "meowth (alolan)"
        if variant_cell:
            variant = variant_cell.text.strip().lower()
            name = f"{base_name} ({variant})"
        else:
            name = base_name

        # Normalize names (e.g., remove symbols like ♀, ♂)
        name = (
            name.replace("♀", "female")
                .replace("♂", "male")
                .replace("’", "'")
                .replace("é", "e")
                .strip()
        )

        names.append(name)

    return names

def save_names(names, filename=OUTPUT_FILE):
    """Append only new Pokémon names to the list file."""
    existing = set()
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            existing = {line.strip() for line in f if line.strip()}

    with open(filename, "a", encoding="utf-8") as f:
        for name in names:
            if name not in existing:
                f.write(name + "\n")
                existing.add(name)

    print(f"✅ Updated {filename}: total {len(existing)} Pokémon listed.")

if __name__ == "__main__":
    print("Fetching Pokémon names...")
    try:
        names = fetch_pokemon_names()
        print(f"Fetched {len(names)} Pokémon from {URL}")
        save_names(names)
    except Exception as e:
        print(f"❌ Failed to fetch Pokémon names: {e}")
