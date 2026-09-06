"""Build 5 strictly distinct, non-overlapping domain seed manifests from furniture_seeds."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("build_manifests")

SEEDS_DIR = Path("furniture_seeds")
OUTPUT_MANIFEST = Path(__file__).resolve().parent / "domain_manifests.json"


def main():
    if not SEEDS_DIR.exists():
        raise FileNotFoundError(f"Seeds directory {SEEDS_DIR} not found.")

    all_seeds = sorted(list(SEEDS_DIR.glob("*.jpg")))
    logger.info("Found %d total seed image files in %s", len(all_seeds), SEEDS_DIR)

    # 1. Group by clean model stem
    model_to_images: dict[str, list[Path]] = {}
    for s in all_seeds:
        cleaned = re.sub(r"(_[0-9]+|_angle[0-9]+|_view[0-9]+|_original.*|\s*\([0-9]+\))$", "", s.stem, flags=re.IGNORECASE)
        cleaned = cleaned.lower().strip()
        model_to_images.setdefault(cleaned, []).append(s)

    logger.info("Found %d unique model stems across all seeds", len(model_to_images))

    # 2. Assign each unique model to one of 5 distinct real-world domains
    domains: dict[str, list[dict[str, str]]] = {
        "D1_Living_Room_Sofas": [],
        "D2_Dining_and_Armchairs": [],
        "D3_Coffee_and_Accent_Tables": [],
        "D4_Couches_and_Loveseats": [],
        "D5_Modern_Suites_and_Studios": [],
    }

    assigned_models: set[str] = set()

    for m, paths in model_to_images.items():
        if m in assigned_models:
            continue
        
        # Primary seed image (choose first or lowest index)
        best_img = paths[0]
        item = {"model_id": m, "primary_file": best_img.name, "all_views": [p.name for p in paths]}

        if "sectional" in m or ("sofa" in m and len(domains["D1_Living_Room_Sofas"]) < 80):
            domains["D1_Living_Room_Sofas"].append(item)
            assigned_models.add(m)
        elif "arm" in m or "chair" in m or "recliner" in m:
            domains["D2_Dining_and_Armchairs"].append(item)
            assigned_models.add(m)
        elif "table" in m:
            domains["D3_Coffee_and_Accent_Tables"].append(item)
            assigned_models.add(m)
        elif "loveseat" in m or "couch" in m or ("sofa" in m and len(domains["D4_Couches_and_Loveseats"]) < 80):
            domains["D4_Couches_and_Loveseats"].append(item)
            assigned_models.add(m)
        else:
            domains["D5_Modern_Suites_and_Studios"].append(item)
            assigned_models.add(m)

    # Balance any remaining unassigned models into domains needing seeds
    for m, paths in model_to_images.items():
        if m not in assigned_models:
            item = {"model_id": m, "primary_file": paths[0].name, "all_views": [p.name for p in paths]}
            # Add to smallest domain
            smallest_domain = min(domains.keys(), key=lambda k: len(domains[k]))
            domains[smallest_domain].append(item)
            assigned_models.add(m)

    logger.info("Domain allocation summary:")
    for d_name, items in domains.items():
        logger.info("  - %-30s : %d unique models", d_name, len(items))

    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(domains, f, indent=2, ensure_ascii=False)

    logger.info("Domain manifests successfully written to %s", OUTPUT_MANIFEST)


if __name__ == "__main__":
    main()
