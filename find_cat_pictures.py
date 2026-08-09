#!/usr/bin/env python3
"""
find_cat_pictures.py

Scans a folder (recursively) for image files, uses a pretrained image
classifier to detect photos of cats, and copies the matches into a
destination folder.

SETUP (run once):
    pip install torch torchvision pillow

USAGE:
    # Scan your whole home folder, copy cat pics to ~/CatPictures
    python find_cat_pictures.py

    # Scan a specific folder
    python find_cat_pictures.py --source "C:/Users/you/Pictures"

    # Choose a different destination
    python find_cat_pictures.py --dest "D:/CatPics"

    # Preview matches without copying anything
    python find_cat_pictures.py --dry-run

    # Be stricter/looser about what counts as a "cat photo"
    python find_cat_pictures.py --threshold 0.4

    # Combine options
    python find_cat_pictures.py --source "C:/Users/you/Pictures" --dest "D:/CatPics" --threshold 0.4 --dry-run

    # No flags at all -> interactive prompts walk you through every option
    python find_cat_pictures.py

NOTES:
    - The first run downloads a small (~14MB) pretrained model from
      PyTorch's servers, then it's cached locally for future runs.
    - This works best on real photos where a cat is a clear subject.
      It can miss cartoon/illustrated cats, or photos where the cat
      is small/partially hidden. See --threshold to tune sensitivity.
    - Original files are never modified or deleted -- only copied.
"""

import argparse
import shutil
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".heic"}

# ImageNet class indices that correspond to domestic cat breeds/types.
# (Full ImageNet has ~1000 classes; these are the "house cat" ones.)
CAT_CLASS_INDICES = {
    281: "tabby cat",
    282: "tiger cat",
    283: "Persian cat",
    284: "Siamese cat",
    285: "Egyptian cat",
}


def load_model():
    import torch
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess


def is_cat_photo(image_path, model, preprocess, threshold):
    import torch
    from PIL import Image, UnidentifiedImageError

    try:
        img = Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return False, 0.0

    try:
        batch = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            output = model(batch)
            probs = torch.nn.functional.softmax(output[0], dim=0)

        cat_prob = sum(probs[i].item() for i in CAT_CLASS_INDICES)
        return cat_prob >= threshold, cat_prob
    except Exception:
        return False, 0.0


def find_image_files(source_dir, dest_dir):
    dest_resolved = dest_dir.resolve()
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        if dest_resolved in path.resolve().parents:
            continue  # don't rescan our own output folder
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def unique_destination(dest_dir, filename):
    """Avoid overwriting files with the same name from different folders."""
    target = dest_dir / filename
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    counter = 1
    while True:
        candidate = dest_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def prompt_for_options():
    """Interactive walkthrough, used when the script is run with no flags."""
    print("No options given -- let's set them up. (Press Enter to accept the default.)\n")

    default_source = str(Path.home())
    source = input(f"Folder to scan [{default_source}]: ").strip() or default_source

    default_dest = str(Path.home() / "CatPictures")
    dest = input(f"Folder to copy matches into [{default_dest}]: ").strip() or default_dest

    default_threshold = "0.3"
    threshold_raw = input(
        f"Confidence threshold, 0-1, lower = more (looser) matches [{default_threshold}]: "
    ).strip() or default_threshold
    try:
        threshold = float(threshold_raw)
    except ValueError:
        print(f"  Couldn't parse '{threshold_raw}', using default {default_threshold}")
        threshold = float(default_threshold)

    dry_run_raw = input("Dry run first, i.e. preview matches without copying? [y/N]: ").strip().lower()
    dry_run = dry_run_raw in ("y", "yes")

    log_raw = input("Save a log file (CSV) of matches to the destination folder? [Y/n]: ").strip().lower()
    log = log_raw not in ("n", "no")

    print()
    return source, dest, threshold, dry_run, log


def main():
    parser = argparse.ArgumentParser(description="Find and copy cat pictures.")
    parser.add_argument(
        "--source", default=None,
        help="Folder to scan (default: your home folder)",
    )
    parser.add_argument(
        "--dest", default=None,
        help="Folder to copy cat pictures into (default: ~/CatPictures)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Confidence threshold 0-1 (default: 0.3). Lower = more matches, more false positives.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=None,
        help="List matches without copying files.",
    )
    parser.add_argument(
        "--log", dest="log", action="store_true", default=None,
        help="Write a CSV log of matches (path, confidence) into the destination folder.",
    )
    parser.add_argument(
        "--no-log", dest="log", action="store_false",
        help="Skip writing the CSV log.",
    )
    args = parser.parse_args()

    # If nothing at all was passed on the command line, walk the user through
    # every option interactively instead of silently using defaults.
    if len(sys.argv) == 1:
        source, dest, threshold, dry_run, log = prompt_for_options()
    else:
        source = args.source if args.source is not None else str(Path.home())
        dest = args.dest if args.dest is not None else str(Path.home() / "CatPictures")
        threshold = args.threshold if args.threshold is not None else 0.3
        dry_run = bool(args.dry_run)
        log = args.log if args.log is not None else True

    source_dir = Path(source).expanduser()
    dest_dir = Path(dest).expanduser()

    if not source_dir.exists():
        print(f"Source folder does not exist: {source_dir}")
        sys.exit(1)

    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning: {source_dir}")
    print(f"Destination: {dest_dir}")
    print(f"Threshold: {threshold}")
    print(f"Dry run: {dry_run}")
    print(f"Log file: {log}")
    print("Loading model (first run downloads ~14MB)...")
    model, preprocess = load_model()

    all_images = list(find_image_files(source_dir, dest_dir))
    print(f"Found {len(all_images)} image files. Classifying...\n")

    matches = []
    for i, img_path in enumerate(all_images, 1):
        is_cat, confidence = is_cat_photo(img_path, model, preprocess, threshold)
        if is_cat:
            matches.append((img_path, confidence))
            print(f"  [CAT {confidence:.0%}] {img_path}")

        if i % 200 == 0:
            print(f"  ...processed {i}/{len(all_images)}")

    print(f"\nDone. {len(matches)} cat picture(s) found out of {len(all_images)} images scanned.")

    if dry_run:
        print("(dry run, no files were copied)")
        return

    copied = 0
    log_rows = []
    for img_path, confidence in matches:
        try:
            target = unique_destination(dest_dir, img_path.name)
            shutil.copy2(img_path, target)
            copied += 1
            log_rows.append((str(img_path), f"{confidence:.3f}", str(target)))
        except Exception as e:
            print(f"  Failed to copy {img_path}: {e}")

    print(f"Copied {copied} file(s) to {dest_dir}")

    if log and log_rows:
        import csv

        log_path = dest_dir / "cat_pictures_log.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["original_path", "confidence", "copied_to"])
            writer.writerows(log_rows)
        print(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
