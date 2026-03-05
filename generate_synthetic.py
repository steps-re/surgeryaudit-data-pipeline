#!/usr/bin/env python3
"""
Generate synthetic AI-manipulated surgery images for SurgeryAudit training.

Creates the "AI/manipulated" class by applying realistic edits to real images:
1. Stable Diffusion img2img (if SD available) — most realistic fakes
2. Programmatic face edits — smoothing, reshaping, color correction
3. GAN-style artifacts — simulate common AI generation tells

Usage:
    python generate_synthetic.py --input-dir ./data/real --output-dir ./data/synthetic --method all
    python generate_synthetic.py --input-dir ./data/real --output-dir ./data/synthetic --method programmatic
"""

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


# --- Manipulation functions ---

def apply_beauty_filter(img_cv, strength=0.7):
    """Simulate FaceTune/beauty filter: bilateral smoothing + sharpening."""
    # Bilateral filter preserves edges while smoothing skin
    smooth = cv2.bilateralFilter(img_cv, d=9, sigmaColor=75, sigmaSpace=75)
    smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=50, sigmaSpace=50)

    # Blend original and smoothed
    result = cv2.addWeighted(img_cv, 1.0 - strength, smooth, strength, 0)

    # Slight brightness/contrast boost (beauty filter style)
    result = cv2.convertScaleAbs(result, alpha=1.05, beta=8)

    return result


def apply_skin_smoothing_aggressive(img_cv):
    """Heavy skin smoothing — simulates AI enhancement apps."""
    # Detect skin regions via YCrCb
    ycrcb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Heavy blur on skin regions only
    blurred = cv2.GaussianBlur(img_cv, (21, 21), 0)
    blurred = cv2.bilateralFilter(blurred, 15, 80, 80)

    # Composite: smooth skin, sharp everything else
    mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
    result = (blurred * mask_3ch + img_cv * (1 - mask_3ch)).astype(np.uint8)

    return result


def apply_face_reshape(img_cv, intensity=0.15):
    """Subtle face reshaping — simulates nose/jaw/chin editing."""
    h, w = img_cv.shape[:2]

    # Create displacement field (simulate nose slimming + jaw narrowing)
    cx, cy = w // 2, int(h * 0.45)  # Approximate face center

    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    # Nose area: slight inward pull
    nose_y, nose_x = int(h * 0.5), w // 2
    dist_nose = np.sqrt((x_coords - nose_x)**2 + (y_coords - nose_y)**2)
    nose_mask = np.exp(-dist_nose**2 / (w * 0.08)**2)
    x_coords += (nose_x - x_coords) * nose_mask * intensity * 0.5

    # Jaw area: slight narrowing
    jaw_y = int(h * 0.7)
    dist_jaw = np.sqrt((x_coords - cx)**2 + (y_coords - jaw_y)**2)
    jaw_mask = np.exp(-dist_jaw**2 / (w * 0.15)**2)
    x_coords += (cx - x_coords) * jaw_mask * intensity * 0.3

    result = cv2.remap(img_cv, x_coords, y_coords, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)
    return result


def apply_eye_enhancement(img_cv):
    """Brighten and enlarge eyes — common in beauty AI."""
    h, w = img_cv.shape[:2]
    result = img_cv.copy()

    # Approximate eye regions (upper face, left and right of center)
    eye_regions = [
        (int(w * 0.3), int(h * 0.35), int(w * 0.15), int(h * 0.08)),
        (int(w * 0.55), int(h * 0.35), int(w * 0.15), int(h * 0.08)),
    ]

    for ex, ey, ew, eh in eye_regions:
        roi = result[ey:ey+eh, ex:ex+ew]
        if roi.size == 0:
            continue
        # Brighten
        roi = cv2.convertScaleAbs(roi, alpha=1.15, beta=15)
        # Slight sharpen
        kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        roi = cv2.filter2D(roi, -1, kernel * 0.3 + np.eye(3) * 0.7)
        result[ey:ey+eh, ex:ex+ew] = roi

    return result


def apply_teeth_whitening(img_cv):
    """Whiten teeth region — common beauty app feature."""
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

    # Approximate mouth region (lower center of face)
    h, w = img_cv.shape[:2]
    my, mx = int(h * 0.65), int(w * 0.5)
    mh, mw = int(h * 0.1), int(w * 0.2)

    roi_hsv = hsv[my:my+mh, mx-mw//2:mx+mw//2]
    if roi_hsv.size == 0:
        return img_cv

    # Desaturate and brighten (teeth whitening effect)
    roi_hsv[:, :, 1] = np.clip(roi_hsv[:, :, 1].astype(int) - 30, 0, 255).astype(np.uint8)
    roi_hsv[:, :, 2] = np.clip(roi_hsv[:, :, 2].astype(int) + 20, 0, 255).astype(np.uint8)
    hsv[my:my+mh, mx-mw//2:mx+mw//2] = roi_hsv

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_ai_upscale_artifacts(img_cv):
    """Simulate AI upscaling artifacts — downsample then upscale with slight hallucination."""
    h, w = img_cv.shape[:2]

    # Downsample to 50%
    small = cv2.resize(img_cv, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

    # Upscale back with cubic (simulates AI upscaler's smooth interpolation)
    upscaled = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    # Add slight sharpening (AI upscalers oversharpen)
    kernel = np.array([[-0.5,-1,-0.5],[-1,7,-1],[-0.5,-1,-0.5]]) / 2
    upscaled = cv2.filter2D(upscaled, -1, kernel)

    # Blend 70% upscaled, 30% original (partial effect)
    result = cv2.addWeighted(upscaled, 0.7, img_cv, 0.3, 0)
    return result


def apply_color_grading(img_cv):
    """Apply Instagram-style color grading that AI editors commonly use."""
    # Warm tint + lifted shadows + reduced highlights
    result = img_cv.astype(np.float32)

    # Warm tint (add to red/yellow, reduce blue)
    result[:, :, 2] = np.clip(result[:, :, 2] * 1.08 + 5, 0, 255)  # Red
    result[:, :, 1] = np.clip(result[:, :, 1] * 1.02, 0, 255)      # Green
    result[:, :, 0] = np.clip(result[:, :, 0] * 0.95, 0, 255)      # Blue

    # Lift shadows
    shadow_mask = (result.mean(axis=2, keepdims=True) < 80).astype(np.float32)
    result += shadow_mask * 15

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_composite_manipulation(img_cv):
    """Apply a random combination of 2-4 manipulations (most realistic)."""
    manipulations = [
        apply_beauty_filter,
        apply_skin_smoothing_aggressive,
        apply_face_reshape,
        apply_eye_enhancement,
        apply_teeth_whitening,
        apply_ai_upscale_artifacts,
        apply_color_grading,
    ]

    num_ops = random.randint(2, 4)
    selected = random.sample(manipulations, num_ops)

    result = img_cv.copy()
    applied = []
    for fn in selected:
        result = fn(result)
        applied.append(fn.__name__.replace("apply_", ""))

    return result, applied


def apply_double_jpeg(img_cv, quality_1=40, quality_2=92):
    """Double JPEG compression — tells of re-saved/re-uploaded images."""
    # First compression at low quality
    _, buf1 = cv2.imencode(".jpg", img_cv, [cv2.IMWRITE_JPEG_QUALITY, quality_1])
    img_1 = cv2.imdecode(buf1, cv2.IMREAD_COLOR)

    # Second compression at higher quality (simulates "save as")
    _, buf2 = cv2.imencode(".jpg", img_1, [cv2.IMWRITE_JPEG_QUALITY, quality_2])
    img_2 = cv2.imdecode(buf2, cv2.IMREAD_COLOR)

    return img_2


MANIPULATION_REGISTRY = {
    "beauty_filter": (apply_beauty_filter, "Beauty filter smoothing"),
    "skin_smoothing": (apply_skin_smoothing_aggressive, "Aggressive skin smoothing"),
    "face_reshape": (apply_face_reshape, "Subtle face reshaping"),
    "eye_enhancement": (apply_eye_enhancement, "Eye brightening/enlarging"),
    "teeth_whitening": (apply_teeth_whitening, "Teeth whitening"),
    "ai_upscale": (apply_ai_upscale_artifacts, "AI upscaling artifacts"),
    "color_grading": (apply_color_grading, "Instagram-style color grading"),
    "composite": (apply_composite_manipulation, "Random 2-4 manipulation combo"),
    "double_jpeg": (apply_double_jpeg, "Double JPEG compression"),
}


def process_image(input_path, output_dir, manipulation_type, manifest):
    """Apply manipulation to a single image and record metadata."""
    try:
        img = cv2.imread(input_path)
        if img is None:
            return False

        # Resize if too large (standardize for training)
        h, w = img.shape[:2]
        max_dim = 1024
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        fn, desc = MANIPULATION_REGISTRY[manipulation_type]

        if manipulation_type == "composite":
            result, applied_ops = fn(img)
            desc = f"Composite: {', '.join(applied_ops)}"
        else:
            result = fn(img)
            applied_ops = [manipulation_type]

        # Save
        basename = Path(input_path).stem
        out_name = f"{basename}_{manipulation_type}.jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 92])

        manifest.append({
            "original": os.path.basename(input_path),
            "manipulated": out_name,
            "manipulation_type": manipulation_type,
            "operations": applied_ops if isinstance(applied_ops, list) else [manipulation_type],
            "description": desc,
            "label": "manipulated",
        })

        return True
    except Exception as e:
        print(f"  Error processing {input_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic manipulated images")
    parser.add_argument("--input-dir", "-i", required=True, help="Directory of real images")
    parser.add_argument("--output-dir", "-o", default="./data/synthetic", help="Output directory")
    parser.add_argument("--method", "-m", default="all",
                        choices=list(MANIPULATION_REGISTRY.keys()) + ["all"],
                        help="Manipulation method (default: all)")
    parser.add_argument("--limit", "-l", type=int, default=0, help="Max images to process (0=all)")
    parser.add_argument("--manifests-per-type", type=int, default=0,
                        help="Max images per manipulation type (0=all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all input images
    input_images = []
    for root, dirs, files in os.walk(args.input_dir):
        for f in files:
            if Path(f).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                input_images.append(os.path.join(root, f))

    if not input_images:
        print(f"No images found in {args.input_dir}")
        sys.exit(1)

    if args.limit > 0:
        input_images = input_images[:args.limit]

    print(f"Found {len(input_images)} input images")

    # Determine manipulation types
    if args.method == "all":
        methods = list(MANIPULATION_REGISTRY.keys())
    else:
        methods = [args.method]

    manifest = []
    total = 0

    for method in methods:
        method_dir = os.path.join(args.output_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        images_for_method = input_images.copy()
        if args.manifests_per_type > 0:
            images_for_method = images_for_method[:args.manifests_per_type]

        print(f"\nApplying '{method}' to {len(images_for_method)} images...")
        count = 0
        for img_path in images_for_method:
            if process_image(img_path, method_dir, method, manifest):
                count += 1
        print(f"  Generated: {count}")
        total += count

    # Save manifest
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Also save as CSV
    csv_path = os.path.join(args.output_dir, "manifest.csv")
    if manifest:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
            writer.writeheader()
            writer.writerows(manifest)

    print(f"\nTotal generated: {total}")
    print(f"Manifest: {manifest_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
