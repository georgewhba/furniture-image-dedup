"""Download and prepare a realistic furniture dataset for testing the deduplication pipeline."""

import os
import sys
import shutil
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download, HfApi
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

DEST_DIR = Path("furniture_images").resolve()
REPO_ID = "Arkan0ID/furniture-dataset"

def download_file(rel_path: str, target_name: str) -> Path | None:
    for attempt in range(4):
        try:
            cached_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=rel_path,
                repo_type="dataset",
            )
            out_file = DEST_DIR / target_name
            shutil.copyfile(cached_path, out_file)
            return out_file
        except Exception as exc:
            if attempt == 3:
                print(f"Failed to download {rel_path}: {exc}")
                return None
            time.sleep(1.5 * (attempt + 1))
    return None

def main():
    print(f"Preparing dataset in: {DEST_DIR}")
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch file list
    api = HfApi()
    all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    
    # Selected multi-shot products (same product photographed from different angles)
    # These test Stage 3 (CLIP embedding similarity) + Stage 4 (Color gate passes because same color)
    target_multi = [
        # Anish armchair (2 angles)
        ("classes/chair/Anish Upholstered Armchair_1.jpg", "chair_anish_angle1.jpg"),
        ("classes/chair/Anish Upholstered Armchair_3.jpg", "chair_anish_angle2.jpg"),
        # 101 sofa (2 angles)
        ("classes/sofa/101 Upholstered Sofa_1.jpg", "sofa_101_angle1.jpg"),
        ("classes/sofa/101 Upholstered Sofa_2.jpg", "sofa_101_angle2.jpg"),
        # Arlo leather chair (2 angles)
        ("classes/chair/Arlo Genuine Leather Mid Century Arm Chair_1.jpg", "chair_arlo_leather_angle1.jpg"),
        ("classes/chair/Arlo Genuine Leather Mid Century Arm Chair_3.jpg", "chair_arlo_leather_angle2.jpg"),
        # Astor armchair (2 angles)
        ("classes/chair/Astor Upholstered Armchair_4.jpg", "chair_astor_angle1.jpg"),
        ("classes/chair/Astor Upholstered Armchair_5.jpg", "chair_astor_angle2.jpg"),
        # Bantry leather armchair (2 angles)
        ("classes/chair/Bantry Vegan Leather Armchair_1.jpg", "chair_bantry_angle1.jpg"),
        ("classes/chair/Bantry Vegan Leather Armchair_6.jpg", "chair_bantry_angle2.jpg"),
    ]
    
    # Distinct furniture items (different products, non-duplicates)
    distinct_candidates = [
        f for f in all_files 
        if f.endswith(".jpg") and f.startswith("classes/") and not any(f == m[0] for m in target_multi)
    ]
    
    # Pick 25 diverse distinct furniture items across categories
    import random
    random.seed(42)
    sample_distinct = random.sample(distinct_candidates, min(25, len(distinct_candidates)))
    
    tasks = list(target_multi)
    for idx, f in enumerate(sample_distinct):
        cat = f.split("/")[1]
        name = Path(f).stem
        clean_name = f"{cat}_{idx+1}_{name[:20].strip().replace(' ', '_')}.jpg"
        tasks.append((f, clean_name))
        
    print(f"Downloading {len(tasks)} real furniture images from HuggingFace...")
    downloaded = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(download_file, rel, tgt): tgt for rel, tgt in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                downloaded.append(res)
                print(f"  Downloaded: {res.name}")
                
    print(f"Successfully downloaded {len(downloaded)} base images.")
    
    # Now create controlled duplicate test cases:
    # A) EXACT DUPLICATES (Stage 1: SHA-256 byte-identical)
    exact1 = DEST_DIR / "chair_anish_angle1.jpg"
    if exact1.exists():
        shutil.copyfile(exact1, DEST_DIR / "chair_anish_angle1_EXACT_COPY.jpg")
        print("  Created exact duplicate: chair_anish_angle1_EXACT_COPY.jpg")
        
    exact2 = DEST_DIR / "sofa_101_angle1.jpg"
    if exact2.exists():
        shutil.copyfile(exact2, DEST_DIR / "sofa_101_angle1_BACKUP_COPY.jpg")
        print("  Created exact duplicate: sofa_101_angle1_BACKUP_COPY.jpg")
        
    # B) NEAR-DUPLICATES (Stage 2: pHash + dHash - compressed, cropped, watermarked)
    if exact1.exists():
        im = Image.open(exact1)
        w, h = im.size
        # Crop 3% from borders and re-save at lower quality (simulating web re-upload)
        cropped = im.crop((int(w * 0.03), int(h * 0.03), int(w * 0.97), int(h * 0.97)))
        cropped.save(DEST_DIR / "chair_anish_angle1_CROP_RECOMPRESSED.jpg", quality=75)
        print("  Created near-duplicate (crop+recompress): chair_anish_angle1_CROP_RECOMPRESSED.jpg")

    if exact2.exists():
        im = Image.open(exact2)
        # Resize slightly (95%) and re-save at Q=70
        w, h = im.size
        resized = im.resize((int(w * 0.95), int(h * 0.95)), Image.Resampling.LANCZOS)
        resized.save(DEST_DIR / "sofa_101_RESIZED_RECOMPRESSED.jpg", quality=70)
        print("  Created near-duplicate (resize+recompress): sofa_101_RESIZED_RECOMPRESSED.jpg")

    # C) SAME-SHAPE-DIFFERENT-COLOR (MANDATORY NEGATIVE CONTROL for Stage 4 Color Gate)
    # Take an armchair, shift its hue drastically (e.g. rotate RGB channels)
    # The CLIP shape embedding will be very close (~0.88-0.95), but Stage 4 COLOR GATE MUST REJECT IT!
    if exact1.exists():
        im = Image.open(exact1).convert("RGB")
        arr = np.array(im, dtype=np.float32)
        # Apply a vibrant blue/cyan tint to turn beige/grey armchair into genuine blue variant
        arr[:, :, 0] *= 0.2  # kill red
        arr[:, :, 1] *= 0.5  # reduce green
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.3, 0, 255)  # boost blue
        diff_color = Image.fromarray(arr.astype(np.uint8))
        diff_color.save(DEST_DIR / "chair_anish_DIFFERENT_COLOR_VARIANT.jpg", quality=95)
        print("  Created color-variant (MUST be rejected by Stage 4 Color Gate): chair_anish_DIFFERENT_COLOR_VARIANT.jpg")

    total_files = list(DEST_DIR.glob("*.jpg"))
    print(f"\nDataset preparation complete! Total images in '{DEST_DIR.name}': {len(total_files)}")

if __name__ == "__main__":
    main()
