#!/usr/bin/env python3
"""
Build a balanced training dataset for AI vs real surgery image classifier.

Combines:
1. Real surgery images (from Reddit scraper)
2. Synthetic manipulated images (from generate_synthetic.py)
3. Academic datasets (FFHQ, SFHQ, GenImage — manual download required)

Outputs a train/val/test split with labels.

Usage:
    python build_dataset.py --data-dir ./data --output-dir ./dataset --split 0.7 0.15 0.15
"""

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path

from PIL import Image


def collect_images(directory, label, recursive=True):
    """Collect all images from a directory with a given label."""
    images = []
    exts = {".jpg", ".jpeg", ".png", ".webp"}

    if not os.path.exists(directory):
        return images

    if recursive:
        for root, dirs, files in os.walk(directory):
            for f in files:
                if Path(f).suffix.lower() in exts:
                    images.append({
                        "path": os.path.join(root, f),
                        "label": label,
                        "source": os.path.basename(directory),
                    })
    else:
        for f in os.listdir(directory):
            if Path(f).suffix.lower() in exts:
                images.append({
                    "path": os.path.join(directory, f),
                    "label": label,
                    "source": os.path.basename(directory),
                })

    return images


def validate_image(path, min_size=64):
    """Check that image is valid and meets minimum size."""
    try:
        img = Image.open(path)
        img.verify()
        img = Image.open(path)
        w, h = img.size
        return w >= min_size and h >= min_size
    except Exception:
        return False


def resize_and_save(src_path, dst_path, target_size=512):
    """Resize image to target size (maintaining aspect ratio) and save as JPEG."""
    try:
        img = Image.open(src_path).convert("RGB")
        w, h = img.size

        # Resize longest side to target_size
        if max(w, h) > target_size:
            scale = target_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        img.save(dst_path, "JPEG", quality=92)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Build balanced training dataset")
    parser.add_argument("--data-dir", "-d", default="./data", help="Root data directory")
    parser.add_argument("--output-dir", "-o", default="./dataset", help="Output dataset directory")
    parser.add_argument("--split", nargs=3, type=float, default=[0.7, 0.15, 0.15],
                        help="Train/val/test split ratios (default: 0.7 0.15 0.15)")
    parser.add_argument("--target-size", type=int, default=512, help="Max image dimension")
    parser.add_argument("--balance", action="store_true", default=True,
                        help="Balance classes to equal size (default: True)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Collect images from all sources
    print("Collecting images...")

    real_images = []
    fake_images = []

    # Real surgery images from Reddit
    real_dir = os.path.join(args.data_dir, "real")
    real_images += collect_images(real_dir, "real")
    print(f"  Real (Reddit): {len(real_images)}")

    # Academic real faces (FFHQ etc)
    academic_real_dir = os.path.join(args.data_dir, "academic", "real")
    academic_real = collect_images(academic_real_dir, "real")
    real_images += academic_real
    print(f"  Real (academic): {len(academic_real)}")

    # Synthetic manipulated
    synthetic_dir = os.path.join(args.data_dir, "synthetic")
    fake_images += collect_images(synthetic_dir, "manipulated")
    print(f"  Manipulated (synthetic): {len(fake_images)}")

    # Manipulated from Reddit (InstagramReality etc)
    manipulated_dir = os.path.join(args.data_dir, "manipulated")
    reddit_manip = collect_images(manipulated_dir, "manipulated")
    fake_images += reddit_manip
    print(f"  Manipulated (Reddit): {len(reddit_manip)}")

    # Academic fake faces (SFHQ, GenImage etc)
    academic_fake_dir = os.path.join(args.data_dir, "academic", "fake")
    academic_fake = collect_images(academic_fake_dir, "manipulated")
    fake_images += academic_fake
    print(f"  Manipulated (academic): {len(academic_fake)}")

    print(f"\nTotal: {len(real_images)} real, {len(fake_images)} manipulated")

    # Validate images
    print("\nValidating images...")
    real_valid = [img for img in real_images if validate_image(img["path"])]
    fake_valid = [img for img in fake_images if validate_image(img["path"])]
    print(f"  Valid: {len(real_valid)} real, {len(fake_valid)} manipulated")

    # Balance classes
    if args.balance:
        min_count = min(len(real_valid), len(fake_valid))
        if min_count == 0:
            print("ERROR: One class has 0 images. Cannot build dataset.")
            return

        random.shuffle(real_valid)
        random.shuffle(fake_valid)
        real_valid = real_valid[:min_count]
        fake_valid = fake_valid[:min_count]
        print(f"  Balanced to {min_count} per class ({min_count * 2} total)")

    # Combine and shuffle
    all_images = real_valid + fake_valid
    random.shuffle(all_images)

    # Split
    n = len(all_images)
    train_end = int(n * args.split[0])
    val_end = train_end + int(n * args.split[1])

    splits = {
        "train": all_images[:train_end],
        "val": all_images[train_end:val_end],
        "test": all_images[val_end:],
    }

    # Create output directory structure
    for split_name, split_images in splits.items():
        for label in ["real", "manipulated"]:
            os.makedirs(os.path.join(args.output_dir, split_name, label), exist_ok=True)

    # Copy and resize images
    print("\nBuilding dataset...")
    manifest = []

    for split_name, split_images in splits.items():
        count = {"real": 0, "manipulated": 0}
        for img in split_images:
            label = img["label"]
            idx = count[label]
            dst_name = f"{label}_{idx:05d}.jpg"
            dst_path = os.path.join(args.output_dir, split_name, label, dst_name)

            if resize_and_save(img["path"], dst_path, args.target_size):
                manifest.append({
                    "split": split_name,
                    "label": label,
                    "filename": dst_name,
                    "source": img["source"],
                    "original_path": img["path"],
                })
                count[label] += 1

        print(f"  {split_name}: {count['real']} real, {count['manipulated']} manipulated")

    # Save manifest
    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "label", "filename", "source", "original_path"])
        writer.writeheader()
        writer.writerows(manifest)

    # Save dataset stats
    stats = {
        "total_images": len(manifest),
        "splits": {name: len(imgs) for name, imgs in splits.items()},
        "sources": {},
    }
    for m in manifest:
        src = m["source"]
        stats["sources"][src] = stats["sources"].get(src, 0) + 1

    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset built at: {os.path.abspath(args.output_dir)}")
    print(f"Manifest: {manifest_path}")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
