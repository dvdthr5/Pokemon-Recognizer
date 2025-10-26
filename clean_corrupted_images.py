import os
from collections import Counter
from typing import Optional, Tuple

from PIL import Image, ImageStat, UnidentifiedImageError

print("🔍 Starting corruption scan...")

DATA_DIR = "data"  # root folder containing 'train' and 'test'
ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg")
MIN_FILE_SIZE_BYTES = 6 * 1024  # skip tiny placeholder files
MIN_DIMENSION = 64  # width/height must be at least this many pixels
MIN_STDDEV = 4.0  # grayscale stddev threshold to catch blank/solid images

removed = 0
checked = 0
reason_counts = Counter()


def should_remove(path: str) -> Tuple[bool, Optional[str]]:
    try:
        file_size = os.path.getsize(path)
    except OSError:
        return True, "unreadable file size"
    if file_size < MIN_FILE_SIZE_BYTES:
        return True, f"file too small ({file_size} bytes)"

    try:
        with Image.open(path) as img:
            img.verify()  # quick integrity check
    except (UnidentifiedImageError, OSError) as exc:
        return True, f"corrupted image ({exc})"

    # reopen after verify to inspect pixels
    with Image.open(path) as img:
        img = img.convert("RGBA")
        width, height = img.size
        if width < MIN_DIMENSION or height < MIN_DIMENSION:
            return True, f"dimensions too small ({width}x{height})"

        # drop alpha to inspect actual content
        rgb_img = img.convert("RGB")
        stat = ImageStat.Stat(rgb_img.convert("L"))
        if stat.stddev[0] < MIN_STDDEV:
            return True, f"low detail (stddev {stat.stddev[0]:.2f})"

        # detect fully transparent assets (invisible cards)
        if "A" in img.getbands():
            alpha = img.split()[-1]
            if ImageStat.Stat(alpha).sum[0] == 0:
                return True, "fully transparent image"

    return False, None


for subset in ["train", "test"]:
    subset_path = os.path.join(DATA_DIR, subset)
    print(f"📁 Scanning subset: {subset_path}")
    for root, _, files in os.walk(subset_path):
        for file in files:
            if not file.lower().endswith(ALLOWED_EXTENSIONS):
                continue
            path = os.path.join(root, file)
            checked += 1
            if checked % 500 == 0:
                print(f"Progress: {checked} files checked...")

            remove_file, reason = should_remove(path)
            if remove_file:
                try:
                    os.remove(path)
                    removed += 1
                    reason_counts[reason] += 1
                    print(f"🗑️ Removed {path} — {reason}")
                except OSError as exc:
                    print(f"⚠️ Failed to delete {path}: {exc}")

print(f"✅ Scan complete. Checked {checked} files, removed {removed} unusable images.")
if removed:
    print("🧾 Removal reasons:")
    for reason, count in reason_counts.most_common():
        print(f"   • {reason}: {count}")
print("🎉 Corruption scan finished successfully.")
