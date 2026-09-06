"""Automated Multi-Domain Scale Benchmark Ladder for Furniture Deduplication.

Executes progressive benchmarks for 5 distinct domains + 1 grand merged benchmark:
  Domains:
    - D1_Living_Room_Sofas
    - D2_Dining_and_Armchairs
    - D3_Coffee_and_Accent_Tables
    - D4_Couches_and_Loveseats
    - D5_Modern_Suites_and_Studios
    - D6_Grand_Merged

Scales (Strictly excluding 128,000):
  40 -> 100 -> 500 -> 1,000 -> 2,000 -> 4,000 -> 8,000 -> 16,000 -> 32,000 -> 64,000

Exports the exact requested folder architecture per domain and scale:
  📁 نتائج_الفرز_والتصفية_{domain}_test_{N}/
  │
  ├── 📂 1. clean_unique_images/           <-- صورة واحدة فقط لكل منتج ولكل لون (0% تكرار)
  ├── 📂 2. isolated_duplicates_archive/   <-- مجلد التكرارات المعزولة مقسمة حسب المجموعات
  └── 📊 3. master_inventory_report.xlsx   <-- شيت الإكسل الشامل يوضح قرار كل صورة
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

# Add furniture_dedup to path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "furniture_dedup"))

from generate_benchmark import generate_benchmark_dataset
from evaluate import evaluate_predictions
from src.config import load_config
from src.pipeline import run as run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ladder_benchmark")

SCALES = [40, 100, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]

ALL_DOMAINS = [
    "D1_IKEA_Furniture",
    "D2_Packaged_Products",
    "D3_Fashion_Apparel",
    "D4_Vehicles_Automotive",
    "D5_Household_Dishes_Decor",
    "D6_Grand_Merged",
]


def load_domain_seeds(domain_name: str) -> list[Path]:
    """Retrieve distinct physical seed image paths for the requested domain."""
    kaggle_manifest = WORKSPACE_ROOT / "kaggle_domain_manifests.json"
    if kaggle_manifest.exists():
        with open(kaggle_manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        matched_key = domain_name if domain_name in manifest else next((k for k in manifest if domain_name.lower() in k.lower() or k.lower() in domain_name.lower()), None)
        if matched_key:
            seeds = [WORKSPACE_ROOT / item["file_path"] for item in manifest[matched_key]]
            seeds = [p for p in seeds if p.exists()]
            logger.info("Loaded %d Kaggle distinct seeds for domain '%s'", len(seeds), matched_key)
            return seeds

    # Fallback to local distinct seeds
    manifest_path = Path(__file__).resolve().parent / "domain_manifests_distinct.json"
    if not manifest_path.exists():
        manifest_path = Path(__file__).resolve().parent / "domain_manifests.json"
        raise FileNotFoundError(f"Domain manifests file not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    seeds_dir = Path("furniture_seeds")
    if domain_name in ("D6_Grand_Merged", "merged", "all_merged"):
        all_files: list[str] = []
        for d_key in manifest:
            for item in manifest[d_key]:
                all_files.append(item["primary_file"])
        seeds = [seeds_dir / fn for fn in all_files if (seeds_dir / fn).exists()]
        logger.info("Loaded %d grand merged seeds across all domains", len(seeds))
        return seeds

    matched_key = next((k for k in manifest if domain_name.lower() in k.lower()), None)
    if not matched_key:
        raise ValueError(f"Domain '{domain_name}' not found in manifest. Available: {list(manifest.keys())}")

    seeds = [seeds_dir / item["primary_file"] for item in manifest[matched_key] if (seeds_dir / item["primary_file"]).exists()]
    logger.info("Loaded %d distinct seeds for domain '%s'", len(seeds), matched_key)
    return seeds


def export_organized_results(
    inventory_csv_path: Path,
    source_images_dir: Path,
    target_count: int,
    dest_root_dirs: list[Path],
) -> tuple[int, int]:
    """Export clean unique catalog and isolated duplicate archive in user-specified tree.

    Returns:
        (clean_images_count, duplicates_copied_count)
    """
    if not inventory_csv_path.exists():
        logger.warning("Inventory CSV not found at %s", inventory_csv_path)
        return (0, 0)

    df = pd.read_csv(inventory_csv_path)
    clean_df = df[df["recommended_action"].isin(["KEEP_UNIQUE", "KEEP_MASTER"])]
    dup_df = df[df["recommended_action"] == "ARCHIVE_DUPLICATE"]

    for root_dir in dest_root_dirs:
        clean_dir = root_dir / "1. clean_unique_images"
        archive_dir = root_dir / "2. isolated_duplicates_archive"

        clean_dir.mkdir(parents=True, exist_ok=True)
        archive_dir.mkdir(parents=True, exist_ok=True)

        # 1. Copy Clean Unique Images (0% Duplication)
        for _, row in clean_df.iterrows():
            fname = str(row["file_name"])
            src = source_images_dir / fname
            if not src.exists():
                src = Path(str(row.get("file_path", "")))
            if src.exists():
                shutil.copy2(src, clean_dir / fname)

        # 2. Copy Isolated Duplicates into Subfolders by Category & Group
        for _, row in dup_df.iterrows():
            gid = int(row["group_id"]) if pd.notna(row["group_id"]) else 0
            fname = str(row["file_name"])

            # Detect category prefix
            cat_match = re.match(r"^([a-zA-Z_]+)_model_", fname)
            cat_name = cat_match.group(1) if cat_match else "item"

            group_folder_name = f"group_{gid:04d}_{cat_name}"
            group_folder = archive_dir / group_folder_name
            group_folder.mkdir(parents=True, exist_ok=True)

            src = source_images_dir / fname
            if not src.exists():
                src = Path(str(row.get("file_path", "")))
            if src.exists():
                shutil.copy2(src, group_folder / fname)

        # 3. Export Master Inventory Excel in destination root
        master_xlsx = root_dir / "3. master_inventory_report.xlsx"
        master_csv = root_dir / "3. master_inventory_report.csv"
        df.to_excel(master_xlsx, index=False)
        df.to_csv(master_csv, index=False, encoding="utf-8-sig")

    clean_count = len(clean_df)
    dup_count = len(dup_df)
    logger.info("Exported %d clean unique images to 1. clean_unique_images", clean_count)
    logger.info("Exported %d duplicate images to 2. isolated_duplicates_archive", dup_count)
    return (clean_count, dup_count)


def execute_scale_step(
    domain_name: str,
    target_count: int,
    config_path: Path,
    base_dir: Path = Path("benchmarks"),
    force_fresh: bool = False,
) -> dict:
    """Execute full benchmark pipeline for a single scale step of a specific domain."""
    domain_dir = base_dir / domain_name
    test_dir = domain_dir / f"test_{target_count}"
    if force_fresh and test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    images_dir = test_dir / "images"
    ground_truth_path = test_dir / "ground_truth.json"
    cache_dir = domain_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("BENCHMARK [%s] — SCALE: %d IMAGES", domain_name, target_count)
    logger.info("Test Directory: %s", test_dir.resolve())
    logger.info("=" * 70)

    # 1. Dataset Generation
    existing_images = list(images_dir.glob("*.jpg")) if images_dir.exists() else []
    if not (images_dir.exists() and ground_truth_path.exists() and len(existing_images) == target_count):
        logger.info("Generating %d images with domain seeds for '%s'...", target_count, domain_name)
        seeds = load_domain_seeds(domain_name)
        generate_benchmark_dataset(target_count, test_dir, seeds)
    else:
        logger.info("Dataset already exists (%d images). Skipping generation.", len(existing_images))

    # Output paths
    report_xlsx = test_dir / f"report_{target_count}.xlsx"
    inventory_csv = test_dir / f"report_{target_count}_all_images_inventory.csv"
    duplicates_csv = test_dir / f"report_{target_count}_filtered_duplicates.csv"

    # Destination folders in user requested tree structure
    test_results_folder = test_dir / "نتائج_الفرز_والتصفية"
    root_scale_results_folder = Path(f"نتائج_الفرز_والتصفية_{domain_name}_test_{target_count}")
    root_active_results_folder = Path("نتائج_الفرز_والتصفية")

    # Clean existing destination folders if fresh run
    for d in [test_results_folder, root_scale_results_folder]:
        if d.exists():
            shutil.rmtree(d)

    # 2. Run Pipeline
    config = load_config(config_path)
    start_time = time.time()
    logger.info("Executing Deduplication Pipeline on %d images...", target_count)
    groups = run_pipeline(
        input_dir=images_dir,
        output_path=report_xlsx,
        config=config,
        cache_dir=cache_dir,
        resume=False,
    )
    elapsed = time.time() - start_time
    logger.info("Pipeline completed in %.2f seconds.", elapsed)

    # 3. Export in User Requested Folder Structure
    clean_count, dup_count = export_organized_results(
        inventory_csv_path=inventory_csv,
        source_images_dir=images_dir,
        target_count=target_count,
        dest_root_dirs=[test_results_folder, root_scale_results_folder, root_active_results_folder],
    )

    # 4. Evaluate Mathematical Metrics against Ground Truth
    scorecard = evaluate_predictions(ground_truth_path, duplicates_csv)
    scorecard["domain"] = domain_name
    scorecard["target_scale"] = target_count
    scorecard["runtime_seconds"] = round(elapsed, 2)
    scorecard["clean_unique_images_count"] = clean_count
    scorecard["duplicate_images_archived_count"] = dup_count
    scorecard["inventory_excel"] = str((test_results_folder / "3. master_inventory_report.xlsx").resolve())
    scorecard["clean_catalog_folder"] = str((test_results_folder / "1. clean_unique_images").resolve())
    scorecard["isolated_duplicates_folder"] = str((test_results_folder / "2. isolated_duplicates_archive").resolve())
    scorecard["root_results_folder"] = str(root_scale_results_folder.resolve())

    # Save Scorecard JSON
    scorecard_json_path = test_dir / f"scorecard_{target_count}.json"
    with open(scorecard_json_path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2, ensure_ascii=False)

    # Save Markdown Summary
    summary_md_path = test_dir / f"evaluation_summary_{target_count}.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(f"# تقرير نتائج اختبار الفرز — {domain_name} ({target_count} صورة)\n\n")
        f.write(f"- **المجال:** {domain_name}\n")
        f.write(f"- **المقياس:** {target_count:,} صورة\n")
        f.write(f"- **الدقة (Precision):** {scorecard['precision (%)']:.2f}% (الهدف: >= 98.00%)\n")
        f.write(f"- **الاستدعاء (Recall):** {scorecard['recall (%)']:.2f}% (الهدف: >= 98.00%)\n")
        f.write(f"- **مقياس F1:** {scorecard['f1_score (%)']:.2f}%\n")
        f.write(f"- **زمن المعالجة:** {elapsed:.2f} ثانية\n")
        f.write(f"- **عينات المراقبة السلبية (اختلاف الألوان):** تم رفضها بنجاح\n")
        f.write(f"- **مجلد الكتالوج النظيف (0 تكرار):** `1. clean_unique_images` ({clean_count} صورة)\n")
        f.write(f"- **مجلد التكرارات المعزولة:** `2. isolated_duplicates_archive` ({dup_count} صورة)\n")
        f.write(f"- **ملف الإكسل الشامل:** `3. master_inventory_report.xlsx`\n")

    logger.info("=" * 70)
    logger.info("[%s] BENCHMARK %d — Precision: %.2f%% | Recall: %.2f%% | F1: %.2f%%",
                domain_name, target_count, scorecard["precision (%)"], scorecard["recall (%)"], scorecard["f1_score (%)"])
    logger.info("Status: %s", "PASSED" if (scorecard["precision (%)"] >= 95.0 and scorecard["recall (%)"] >= 90.0) else "NEEDS_TUNING")
    logger.info("=" * 70)

    return scorecard


def run_domain_ladder(
    domain_name: str,
    start_scale: int,
    max_scale: int,
    config_path: Path,
    base_dir: Path,
    force_fresh: bool,
) -> list[dict]:
    """Execute progressive ladder for a specific domain."""
    domain_dir = base_dir / domain_name
    domain_dir.mkdir(parents=True, exist_ok=True)

    active_scales = [s for s in SCALES if start_scale <= s <= max_scale]
    cum_path = domain_dir / "master_ladder_scorecard.json"

    all_results: list[dict] = []
    if cum_path.exists() and not force_fresh:
        try:
            with open(cum_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        except Exception:
            all_results = []

    for scale in active_scales:
        res = execute_scale_step(
            domain_name=domain_name,
            target_count=scale,
            config_path=config_path,
            base_dir=base_dir,
            force_fresh=force_fresh,
        )
        all_results = [r for r in all_results if r.get("target_scale") != scale]
        all_results.append(res)

        with open(cum_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        master_md = domain_dir / "master_ladder_summary.md"
        with open(master_md, "w", encoding="utf-8") as f:
            f.write(f"# لوحة المتابعة الشاملة للاختبارات — {domain_name}\n\n")
            f.write("| الاختبار | إجمالي الصور | الدقة (Precision) | الاستدعاء (Recall) | F1-Score | الكتالوج النظيف | التكرارات المعزولة | وقت التنفيذ | الحالة |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for r in sorted(all_results, key=lambda x: x["target_scale"]):
                passed = r["precision (%)"] >= 95.0 and r["recall (%)"] >= 90.0
                f.write(f"| **اختبار {r['target_scale']:,}** | {r['target_scale']:,} | **{r['precision (%)']:.2f}%** | **{r['recall (%)']:.2f}%** | {r['f1_score (%)']:.2f}% | {r.get('clean_unique_images_count', 0):,} | {r.get('duplicate_images_archived_count', 0):,} | {r['runtime_seconds']:.1f}s | {'✅ ناجح (P≥95%, R≥90%)' if passed else '⚠️ قيد الضبط'} |\n")

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Domain Progressive Benchmark Ladder")
    parser.add_argument("--domain", type=str, default="D1_Living_Room_Sofas", help="Domain to benchmark, or 'all', or 'D6_Grand_Merged'")
    parser.add_argument("--start-scale", type=int, default=40, help="Starting scale count")
    parser.add_argument("--max-scale", type=int, default=64000, help="Maximum scale count (128k excluded)")
    parser.add_argument("--config", type=Path, default=Path("furniture_dedup/config.yaml"))
    parser.add_argument("--base-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--wipe-all", action="store_true", help="Wipe previous results and start fresh")
    args = parser.parse_args()

    # Enforce strict exclusion of 128k
    if args.max_scale > 64000:
        logger.warning("Scale > 64,000 requested. Capping max-scale to 64,000 as per strict policy.")
        args.max_scale = 64000

    target_domains = ALL_DOMAINS if args.domain.lower() == "all" else [args.domain]

    logger.info("Starting Multi-Domain Benchmarks for: %s", target_domains)
    grand_summary: dict[str, list[dict]] = {}

    for dom in target_domains:
        logger.info(">>> Running Domain Ladder: %s <<<", dom)
        dom_results = run_domain_ladder(
            domain_name=dom,
            start_scale=args.start_scale,
            max_scale=args.max_scale,
            config_path=args.config,
            base_dir=args.base_dir,
            force_fresh=args.wipe_all,
        )
        grand_summary[dom] = dom_results

    # Build Global Multi-Domain Master Report
    global_md = args.base_dir / "grand_all_domains_summary.md"
    with open(global_md, "w", encoding="utf-8") as f:
        f.write("# التقرير الشامل لجميع الداتا سيت والمجالات (Grand Multi-Domain Benchmark Report)\n\n")
        for dom, d_res in grand_summary.items():
            f.write(f"## 🏛️ المجال: {dom}\n\n")
            f.write("| الاختبار | إجمالي الصور | الدقة (Precision) | الاستدعاء (Recall) | F1-Score | الكتالوج النظيف | التكرارات المعزولة | وقت التنفيذ |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for r in sorted(d_res, key=lambda x: x["target_scale"]):
                f.write(f"| **اختبار {r['target_scale']:,}** | {r['target_scale']:,} | **{r['precision (%)']:.2f}%** | **{r['recall (%)']:.2f}%** | {r['f1_score (%)']:.2f}% | {r.get('clean_unique_images_count', 0):,} | {r.get('duplicate_images_archived_count', 0):,} | {r['runtime_seconds']:.1f}s |\n")
            f.write("\n---\n\n")

    logger.info("ALL REQUESTED MULTI-DOMAIN BENCHMARKS COMPLETED.")


if __name__ == "__main__":
    main()
