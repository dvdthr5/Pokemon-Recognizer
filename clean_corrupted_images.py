import argparse
import os
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple

from PIL import Image, ImageStat, UnidentifiedImageError

ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg")


@dataclass
class ScanSettings:
    data_dir: str
    min_file_size: int
    min_dimension: int
    low_variance_threshold: float
    dry_run: bool


def parse_args() -> ScanSettings:
    parser = argparse.ArgumentParser(description="Remove corrupted or blank Pokémon images.")
    parser.add_argument("--data-dir", default="data", help="Root folder containing 'train' and 'test'.")
    parser.add_argument("--min-file-size", type=int, default=0, help="Delete files smaller than this many bytes.")
    parser.add_argument("--min-dimension", type=int, default=1, help="Smallest acceptable width/height in pixels.")
    parser.add_argument(
        "--low-variance-threshold",
        type=float,
        default=0.0,
        help="Set >0 to delete nearly-blank images (grayscale stddev below threshold).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report problematic files without deleting.")
    args = parser.parse_args()
    return ScanSettings(
        data_dir=args.data_dir,
        min_file_size=args.min_file_size,
        min_dimension=args.min_dimension,
        low_variance_threshold=args.low_variance_threshold,
        dry_run=args.dry_run,
    )


def should_remove(path: str, settings: ScanSettings) -> Tuple[bool, Optional[str]]:
    try:
        file_size = os.path.getsize(path)
    except OSError:
        return True, "unreadable file size"
    if file_size < settings.min_file_size:
        return True, f"file too small ({file_size} bytes)"

    try:
        with Image.open(path) as img:
            img.verify()  # quick integrity check
    except (UnidentifiedImageError, OSError) as exc:
        return True, f"corrupted image ({exc})"

    with Image.open(path) as img:
        width, height = img.size
        if width < settings.min_dimension or height < settings.min_dimension:
            return True, f"dimensions too small ({width}x{height})"

        img = img.convert("RGBA")
        if "A" in img.getbands():
            alpha = img.split()[-1]
            if ImageStat.Stat(alpha).sum[0] == 0:
                return True, "fully transparent image"

        if settings.low_variance_threshold > 0:
            grayscale = img.convert("L")
            stat = ImageStat.Stat(grayscale)
            if stat.stddev[0] < settings.low_variance_threshold:
                return True, f"low detail (stddev {stat.stddev[0]:.2f})"

    return False, None


def main() -> None:
    settings = parse_args()
    print("🔍 Starting corruption scan...")
    removed = 0
    checked = 0
    reason_counts = Counter()

    for subset in ["train", "test"]:
        subset_path = os.path.join(settings.data_dir, subset)
        print(f"📁 Scanning subset: {subset_path}")
        for root, _, files in os.walk(subset_path):
            for file in files:
                if not file.lower().endswith(ALLOWED_EXTENSIONS):
                    continue
                path = os.path.join(root, file)
                checked += 1
                if checked % 500 == 0:
                    print(f"Progress: {checked} files checked...")

                remove_file, reason = should_remove(path, settings)
                if remove_file:
                    if settings.dry_run:
                        removed += 1
                        reason_counts[reason] += 1
                        print(f"📝 Would remove {path} — {reason}")
                        continue
                    try:
                        os.remove(path)
                        removed += 1
                        reason_counts[reason] += 1
                        print(f"🗑️ Removed {path} — {reason}")
                    except OSError as exc:
                        print(f"⚠️ Failed to delete {path}: {exc}")

    print(f"✅ Scan complete. Checked {checked} files, flagged {removed} unusable images.")
    if removed:
        print("🧾 Reasons:")
        for reason, count in reason_counts.most_common():
            print(f"   • {reason}: {count}")
    if settings.dry_run:
        print("ℹ️ Dry-run mode; rerun without --dry-run to delete the files above.")
    print("🎉 Corruption scan finished successfully.")


if __name__ == "__main__":
    main()
