import os
import re

base_dir = "data"  # root dataset folder

for dataset_type in ["train", "test"]:
    dataset_path = os.path.join(base_dir, dataset_type)
    
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue

        print(f"Renaming files in {class_path}...")

        # Get all current images
        files = [f for f in os.listdir(class_path) if os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png"]]

        # Find highest existing number so we can continue counting from there
        pattern = re.compile(rf"^{class_name}_(\d+)\.")
        existing_numbers = [
            int(match.group(1))
            for f in files
            if (match := pattern.match(f))
        ]
        start_index = max(existing_numbers, default=0) + 1

        counter = start_index
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()

            # Skip if already properly numbered
            if pattern.match(filename):
                continue

            new_name = f"{class_name}_{counter:03d}{ext}"
            src = os.path.join(class_path, filename)
            dst = os.path.join(class_path, new_name)

            # Make sure we don’t accidentally overwrite anything
            while os.path.exists(dst):
                counter += 1
                new_name = f"{class_name}_{counter:03d}{ext}"
                dst = os.path.join(class_path, new_name)

            os.rename(src, dst)
            print(f"Renamed {filename} → {new_name}")
            counter += 1

        print(f"✅ Done: {class_name}")
