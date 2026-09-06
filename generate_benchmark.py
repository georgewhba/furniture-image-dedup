"""Benchmark Dataset Generator for Furniture Deduplication.

Generates reproducible, realistic furniture datasets at any scale (40, 100, 500, 1000, 2000...)
with known ground truth labels for rigorous mathematical precision and recall evaluation.
Features diverse furniture taxonomy (sofas, tables, beds, chairs, desks, wardrobes...).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "Arkan0ID/furniture-dataset"
SEEDS_DIR = Path("furniture_seeds")

FURNITURE_CATEGORIES = [
    "sofa",
    "chair",
    "table",
    "bed",
    "wardrobe",
    "desk",
    "bookshelf",
    "coffee_table",
    "dresser",
    "nightstand",
    "tv_console",
    "armchair",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("generate_benchmark")


def ensure_seed_images(target_seeds: int = 150) -> list[Path]:
    """Ensure a pool of distinct real furniture images is cached locally."""
    manifest = Path(__file__).resolve().parent / "distinct_seeds.json"
    if manifest.is_file():
        try:
            with open(manifest, "r", encoding="utf-8") as f:
                seed_names = json.load(f)
            found = [SEEDS_DIR / name for name in seed_names if (SEEDS_DIR / name).is_file()]
            if len(found) >= min(target_seeds, len(seed_names)):
                return found[:target_seeds]
        except Exception:
            pass

    SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(SEEDS_DIR.glob("*.jpg"))

    # Check existing seeds
    base_pool = Path("furniture_images")
    if base_pool.exists():
        for f in base_pool.glob("*.jpg"):
            if not any(tag in f.name for tag in ["COPY", "CROP", "RESIZED", "VARIANT"]):
                dest = SEEDS_DIR / f.name
                if not dest.exists():
                    shutil.copyfile(f, dest)

    existing = list(SEEDS_DIR.glob("*.jpg"))
    import imagehash
    import re

    unique_seeds: list[tuple[Path, tuple[imagehash.ImageHash, imagehash.ImageHash]]] = []
    seen_stems: set[str] = set()

    for s in sorted(existing):
        clean_stem = re.sub(r"(_[0-9]+|_angle[0-9]+|_view[0-9]+|_original.*|\s*\([0-9]+\))$", "", s.stem, flags=re.IGNORECASE)
        clean_stem = clean_stem.lower().strip()
        if clean_stem in seen_stems:
            continue
        try:
            with Image.open(s) as im:
                ph = imagehash.phash(im)
                dh = imagehash.dhash(im)
            is_dup = False
            for u_path, (u_ph, u_dh) in unique_seeds:
                if (ph - u_ph) <= 8 and (dh - u_dh) <= 6:
                    is_dup = True
                    break
            if not is_dup:
                seen_stems.add(clean_stem)
                unique_seeds.append((s, (ph, dh)))
                if len(unique_seeds) >= target_seeds:
                    break
        except Exception:
            pass

    distinct_seeds = [u[0] for u in unique_seeds]
    if len(distinct_seeds) >= target_seeds:
        return distinct_seeds[:target_seeds]

    logger.info("Downloading real furniture seeds from %s to reach %d seeds...", REPO_ID, target_seeds)
    try:
        repo_files = list_repo_files(repo_id=REPO_ID, repo_type="dataset")
        img_files = [
            f for f in repo_files
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            and not f.startswith(".")
        ]
        rng = random.Random(42)
        rng.shuffle(img_files)

        for rf in img_files:
            if len(distinct_seeds) >= target_seeds:
                break
            dest_name = Path(rf).name.replace(" ", "_")
            dest = SEEDS_DIR / dest_name
            if not dest.exists():
                try:
                    local_path = hf_hub_download(
                        repo_id=REPO_ID,
                        filename=rf,
                        repo_type="dataset",
                    )
                    im = Image.open(local_path).convert("RGB")
                    im.save(dest, "JPEG", quality=95)
                    ph = imagehash.phash(im)
                    dh = imagehash.dhash(im)
                    is_dup = any((ph - u_ph) <= 4 and (dh - u_dh) <= 4 for _, (u_ph, u_dh) in unique_seeds)
                    if not is_dup:
                        unique_seeds.append((dest, (ph, dh)))
                        distinct_seeds.append(dest)
                except Exception:
                    pass
    except Exception as exc:
        logger.warning("HuggingFace download notice: %s", exc)

    return distinct_seeds


def generate_benchmark_dataset(
    target_count: int,
    output_dir: Path,
    seed_pool: list[Path],
    exact_ratio: float = 0.15,
    near_exact_ratio: float = 0.20,
    color_variant_ratio: float = 0.08,
) -> dict[str, Any]:
    """Generate a controlled dataset of size target_count with ground truth.

    Args:
        target_count: Total number of images in output_dir (e.g. 40, 100, 500, 1000).
        output_dir: Destination folder for generated images.
        seed_pool: Pool of distinct real furniture images.
        exact_ratio: Fraction of exact duplicate images.
        near_exact_ratio: Fraction of near-duplicate images.
        color_variant_ratio: Fraction of negative control color variants (same shape, different color).

    Returns:
        Ground truth dictionary with true duplicate clusters and negative controls.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42 + target_count)
    shuffled_seeds = list(seed_pool)
    rng.shuffle(shuffled_seeds)

    pool_len = len(shuffled_seeds)
    num_products = min(pool_len, max(4, int(target_count * 0.35)))

    num_exact = int(round(target_count * exact_ratio))
    num_near = int(round(target_count * near_exact_ratio))
    num_color_variants = max(2, int(round(target_count * color_variant_ratio)))
    
    # Extra gallery/perspective shots belonging to duplicate clusters
    num_gallery = target_count - (num_products + num_exact + num_near + num_color_variants)
    if num_gallery < 0:
        num_products += num_gallery
        num_gallery = 0

    logger.info("--- Generating Benchmark Dataset: %d images ---", target_count)
    logger.info("  Target dir:          %s", images_dir)
    logger.info("  Distinct products:   %d", num_products)
    logger.info("  Exact duplicates:    %d", num_exact)
    logger.info("  Near-duplicates:     %d", num_near)
    logger.info("  Gallery/angle shots: %d", num_gallery)
    logger.info("  Color variants (NC): %d", num_color_variants)

    base_images: list[tuple[str, str, Image.Image]] = []  # (prod_id, category, PIL.Image)
    ground_truth_clusters: dict[str, list[str]] = {}
    negative_controls: list[dict[str, str]] = []

    # 1. Base image for each unique physical product
    for k in range(num_products):
        cat = FURNITURE_CATEGORIES[k % len(FURNITURE_CATEGORIES)]
        prod_id = f"{cat}_model_{k+1:04d}"
        src_path = shuffled_seeds[k]
        im = Image.open(src_path).convert("RGB")
        im.thumbnail((512, 512), Image.Resampling.LANCZOS)

        base_name = f"{prod_id}_best.jpg"
        im.save(images_dir / base_name, "JPEG", quality=90)
        base_images.append((prod_id, cat, im))
        ground_truth_clusters[prod_id] = [base_name]

    # Tracking per-product counters
    view_counts: dict[str, int] = {}
    exact_counts: dict[str, int] = {}
    near_counts: dict[str, int] = {}
    variant_counts: dict[str, int] = {}

    # 2. Multi-view gallery / perspective angle shots
    for i in range(num_gallery):
        prod_id, cat, im = base_images[i % len(base_images)]
        view_counts[prod_id] = view_counts.get(prod_id, 0) + 1
        v_idx = view_counts[prod_id]

        w, h = im.size
        crop_f = 0.02 * ((v_idx % 3) + 1)
        x0, y0 = int(w * crop_f), int(h * crop_f)
        x1, y1 = int(w * (1 - crop_f)), int(h * (1 - crop_f))
        if x1 > x0 + 10 and y1 > y0 + 10:
            view_im = im.crop((x0, y0, x1, y1))
        else:
            view_im = im.copy()
        # Slight brightness tweak simulating angle lighting
        view_im = ImageEnhance.Brightness(view_im).enhance(0.98 + 0.02 * (v_idx % 3))
        shot_name = f"{prod_id}_view_{v_idx:02d}.jpg"
        view_im.save(images_dir / shot_name, "JPEG", quality=90)
        ground_truth_clusters[prod_id].append(shot_name)

    # 3. Exact duplicates (Byte-for-byte SHA-256 matches)
    for i in range(num_exact):
        prod_id, cat, _ = base_images[i % len(base_images)]
        exact_counts[prod_id] = exact_counts.get(prod_id, 0) + 1
        e_idx = exact_counts[prod_id]

        src_file = images_dir / f"{prod_id}_best.jpg"
        dup_name = f"{prod_id}_exact_{e_idx:02d}.jpg"
        shutil.copyfile(src_file, images_dir / dup_name)
        ground_truth_clusters[prod_id].append(dup_name)

    # 4. Near duplicates (pHash/dHash: crops, rescales, compression)
    for i in range(num_near):
        prod_id, cat, im = base_images[i % len(base_images)]
        near_counts[prod_id] = near_counts.get(prod_id, 0) + 1
        n_idx = near_counts[prod_id]

        w, h = im.size
        mode = n_idx % 3
        if mode == 0:
            x0, y0 = int(w * 0.03), int(h * 0.03)
            x1, y1 = int(w * 0.97), int(h * 0.97)
            if x1 > x0 + 10 and y1 > y0 + 10:
                mod_im = im.crop((x0, y0, x1, y1))
            else:
                mod_im = im.copy()
            q = 80
        elif mode == 1:
            new_w, new_h = max(32, int(w * 0.94)), max(32, int(h * 0.94))
            mod_im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
            q = 75
        else:
            mod_im = ImageEnhance.Contrast(im).enhance(1.05)
            q = 70

        dup_name = f"{prod_id}_near_{n_idx:02d}.jpg"
        mod_im.save(images_dir / dup_name, "JPEG", quality=q)
        ground_truth_clusters[prod_id].append(dup_name)

    # 5. Negative Controls: Same shape, completely different physical color (MUST BE REJECTED!)
    color_shifts = [
        ("navy_blue", lambda arr: [arr[:, :, 0] * 0.1, arr[:, :, 1] * 0.25, np.clip(arr[:, :, 2] * 1.5 + 45, 0, 255)]),
        ("emerald_green", lambda arr: [arr[:, :, 0] * 0.1, np.clip(arr[:, :, 1] * 1.5 + 45, 0, 255), arr[:, :, 2] * 0.15]),
        ("crimson_red", lambda arr: [np.clip(arr[:, :, 0] * 1.5 + 45, 0, 255), arr[:, :, 1] * 0.15, arr[:, :, 2] * 0.1]),
        ("deep_purple", lambda arr: [np.clip(arr[:, :, 0] * 1.3 + 35, 0, 255), arr[:, :, 1] * 0.1, np.clip(arr[:, :, 2] * 1.4 + 45, 0, 255)]),
        ("cyan_blue", lambda arr: [arr[:, :, 0] * 0.1, np.clip(arr[:, :, 1] * 1.3 + 40, 0, 255), np.clip(arr[:, :, 2] * 1.4 + 40, 0, 255)]),
    ]

    for i in range(num_color_variants):
        prod_id, cat, im = base_images[i % len(base_images)]
        variant_counts[prod_id] = variant_counts.get(prod_id, 0) + 1
        c_idx = variant_counts[prod_id]

        arr = np.array(im, dtype=np.float32)
        color_name, shift_fn = color_shifts[i % len(color_shifts)]
        ch_r, ch_g, ch_b = shift_fn(arr)
        arr_shifted = np.stack([ch_r, ch_g, ch_b], axis=-1).astype(np.uint8)
        var_im = Image.fromarray(arr_shifted)
        var_name = f"{prod_id}_color_variant_{color_name}_{c_idx:02d}.jpg"
        var_im.save(images_dir / var_name, "JPEG", quality=95)
        
        negative_controls.append({
            "variant_file": var_name,
            "original_product": prod_id,
            "category": cat,
            "color": color_name,
        })

    # Clusters with 2+ images are true duplicate clusters
    true_duplicate_clusters = {
        k: v for k, v in ground_truth_clusters.items() if len(v) >= 2
    }
    uniques = [
        v[0] for k, v in ground_truth_clusters.items() if len(v) == 1
    ] + [nc["variant_file"] for nc in negative_controls]

    actual_images = sorted(list(images_dir.glob("*.jpg")))
    gt_data = {
        "target_count": target_count,
        "actual_count": len(actual_images),
        "true_duplicate_clusters": true_duplicate_clusters,
        "negative_controls": negative_controls,
        "unique_files": uniques,
    }

    with open(output_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=2, ensure_ascii=False)

    logger.info("Generated %d images successfully in %s", gt_data["actual_count"], images_dir)
    return gt_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark dataset for furniture dedup")
    parser.add_argument("--count", type=int, default=40, help="Total image count")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output folder")
    args = parser.parse_args()

    count = args.count
    out_dir = args.output_dir or Path(f"dataset_{count}")
    seeds = ensure_seed_images(target_seeds=max(100, count // 2))
    generate_benchmark_dataset(count, out_dir, seeds)


if __name__ == "__main__":
    main()
