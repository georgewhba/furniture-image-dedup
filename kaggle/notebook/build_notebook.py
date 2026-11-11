"""Generate the Kaggle notebook (.ipynb) for furniture-image-dedup (2 Datasets: IKEA-FS + RoyalOak).

Usage::

    python kaggle/notebook/build_notebook.py
    kaggle kernels push -p kaggle/notebook
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "furniture-duplicate-detection-ikea.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


CELLS = [
    md(
        """
# 🛋️ Furniture Product-Image Duplicate Detection (IKEA-FS + RoyalOak Benchmark)

This notebook runs the upgraded **[furniture-image-dedup](https://github.com/georgewhba/furniture-image-dedup)** pipeline on GPU across **two real-world furniture datasets**:
1. **IKEA-FS (`brikwerk/ikeafs`)** — 1,092 real IKEA product photos across 46 furniture product classes + 48 controlled ground-truth test cases.
2. **RoyalOak Furniture (`patilaakash619/royaloak`)** — Real e-commerce catalog photos across 9 core furniture categories (Sofas, Beds, Recliners, Dining, Center Tables, Wardrobes, Office, Chairs, Bookshelves) + 48 controlled ground-truth test cases.

### Key Pipeline Upgrades Tested Here
| Stage | Method & Upgrade | Purpose |
|---|---|---|
| 1 | SHA-256 exact hash | Byte-identical copies |
| 2 | Studio-border-trimmed pHash + dHash + foreground aspect-ratio check + CLIP verification for loose hashes | Catches crops/watermarks while preventing white-studio silhouette collisions |
| 3 | CLIP ViT-B/32 + FAISS (preserves Stage 2 nodes for full graph connectivity) | Same product re-shot (angle / background / lighting) |
| 4 | **Upgraded Color Gate** — `rembg` + K-means CIE-Lab palette + **Earth Mover's Distance (EMD)** + **16×16 spatial recolor detector** | Rejects *same shape, different color* (even in room scenes with people) |
| 5 | Quotient-graph clustering + **Cannot-Link constraints** from Color Gate rejections | Prevents transitive chaining across different color variants |
| 6 | Excel / CSV report | — |
"""
    ),
    md(
        """
## 1. Environment
Install the extra packages **before importing anything**, pinning Kaggle's preinstalled core libraries
(numpy, scipy, Pillow, torch, …) so pip cannot upgrade them mid-session.
"""
    ),
    code(
        """
!pip freeze | grep -iE '^(numpy|scipy|pillow|scikit-learn|pandas|torch|torchvision|opencv-python|opencv-python-headless|opencv-contrib-python)==' > /tmp/constraints.txt
!cat /tmp/constraints.txt
!pip install -q -c /tmp/constraints.txt open-clip-torch faiss-cpu imagehash rembg onnxruntime
"""
    ),
    code(
        """
import numpy, scipy, torch
print("numpy:", numpy.__version__, "| scipy:", scipy.__version__)
print("PyTorch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
"""
    ),
    md("## 2. Load the upgraded source code and locate both datasets"),
    code(
        """
import shutil, sys, zipfile
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
CODE_DIR = WORK / "furniture-image-dedup"


def find_code_root():
    for p in INPUT.rglob("pipeline.py"):
        if p.parent.name == "src" and (p.parent.parent / "config.yaml").exists():
            return p.parent.parent
    return None


shutil.rmtree(CODE_DIR, ignore_errors=True)
code_root = find_code_root()
if code_root is not None:
    shutil.copytree(code_root, CODE_DIR)
else:
    zips = sorted(INPUT.rglob("furniture-image-dedup*.zip"))
    assert zips, "Source code not found - add the 'furniture-image-dedup-code' dataset as an input."
    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall(WORK)

sys.path.insert(0, str(CODE_DIR))
print("Code ready at:", CODE_DIR, "->", sorted(p.name for p in (CODE_DIR / "src").glob("*.py")))

# Locate Dataset 1: IKEA-FS
ikea_marker = next((p for p in INPUT.rglob("aepplaryd_sofa") if p.is_dir()), None)
assert ikea_marker is not None, "IKEA-FS not found - add 'brikwerk/ikeafs' as an input."
IKEA_DIR = ikea_marker.parent
print(f"Dataset 1 (IKEA-FS) at {IKEA_DIR}: {sum(1 for d in IKEA_DIR.iterdir() if d.is_dir())} classes")

# Locate Dataset 2: RoyalOak
royaloak_marker = next((p for p in INPUT.rglob("Fabric_Sofa") if p.is_dir()), None)
assert royaloak_marker is not None, "RoyalOak not found - add 'patilaakash619/royaloak' as an input."
ROYALOAK_DIR = royaloak_marker.parent.parent
print(f"Dataset 2 (RoyalOak) at {ROYALOAK_DIR}: {sum(1 for d in ROYALOAK_DIR.iterdir() if d.is_dir())} categories")
"""
    ),
    md(
        """
## 3. Helper: Controlled Ground-Truth Injection & Evaluation
For each dataset, we inject 48 controlled test cases across 12 randomly chosen classes (seed 42):
- `exact_copy` — byte-identical file (**expect grouped**)
- `crop_resize_q70` — 4% border crop, 90% resize, JPEG q70 (**expect grouped**)
- `watermark` — semi-transparent text overlay (**expect grouped**)
- `color_variant` — product recolored (**expect NOT grouped** — must be rejected by the Color Gate)
"""
    ),
    code(
        """
import random, time, logging
from itertools import combinations
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from src.config import load_config, setup_logging
from src.pipeline import run
from src.report import groups_to_dataframe

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
OUT = WORK / "outputs"
OUT.mkdir(exist_ok=True)


def inject_ground_truth(by_class: dict[str, list[Path]], inj_dir: Path, seed: int = 42) -> list[dict]:
    inj_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    ground_truth = []
    eligible = sorted(k for k, v in by_class.items() if len(v) >= 1)
    chosen_classes = rng.sample(eligible, min(12, len(eligible)))

    for i, cls in enumerate(chosen_classes):
        src = rng.choice(by_class[cls])
        im = Image.open(src).convert("RGB")
        w, h = im.size
        safe_cls = cls.replace("/", "_").replace("\\\\", "_")
        tag = f"{i:02d}_{safe_cls}"

        dst = inj_dir / f"{tag}__exact_copy{src.suffix.lower()}"
        shutil.copyfile(src, dst)
        ground_truth.append(dict(original=str(src), injected=str(dst), kind="exact_copy", expect_grouped=True))

        near = im.crop((int(w * 0.04), int(h * 0.04), int(w * 0.96), int(h * 0.96)))
        near = near.resize((max(1, int(near.width * 0.9)), max(1, int(near.height * 0.9))), Image.LANCZOS)
        dst = inj_dir / f"{tag}__crop_resize_q70.jpg"
        near.save(dst, quality=70)
        ground_truth.append(dict(original=str(src), injected=str(dst), kind="crop_resize_q70", expect_grouped=True))

        wm = im.copy().convert("RGBA")
        overlay = Image.new("RGBA", wm.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        try:
            font = ImageFont.load_default(size=max(12, h // 14))
        except TypeError:
            font = ImageFont.load_default()
        draw.text((int(w * 0.05), int(h * 0.85)), "© SAMPLE STORE", fill=(255, 255, 255, 140), font=font)
        dst = inj_dir / f"{tag}__watermark.jpg"
        Image.alpha_composite(wm, overlay).convert("RGB").save(dst, quality=92)
        ground_truth.append(dict(original=str(src), injected=str(dst), kind="watermark", expect_grouped=True))

        arr = np.asarray(im, dtype=np.float32).copy()
        arr[..., 0] *= 0.2
        arr[..., 1] *= 0.5
        arr[..., 2] = np.clip(arr[..., 2] * 1.3, 0, 255)
        dst = inj_dir / f"{tag}__color_variant.jpg"
        Image.fromarray(arr.astype(np.uint8)).save(dst, quality=95)
        ground_truth.append(dict(original=str(src), injected=str(dst), kind="color_variant", expect_grouped=False))

    return ground_truth


def evaluate_run(dataset_name: str, groups: list, ground_truth: list[dict], class_of_fn, n_total: int, elapsed: float):
    df = groups_to_dataframe(groups)
    by_type = pd.Series([g.match_type for g in groups]).value_counts().rename("groups")
    print(f"=== {dataset_name} ===")
    print(f"Images scanned: {n_total} | Duplicate groups: {len(groups)} | Grouped images: {len(df)} | Runtime: {elapsed:.1f}s")
    display(by_type.to_frame())

    path_to_group = {m: g.group_id for g in groups for m in g.members}
    gt = pd.DataFrame(ground_truth)
    gt["grouped_together"] = [
        path_to_group.get(a) is not None and path_to_group.get(a) == path_to_group.get(b)
        for a, b in zip(gt.original, gt.injected)
    ]
    gt["correct"] = gt.grouped_together == gt.expect_grouped

    summary = gt.groupby("kind", sort=False).agg(cases=("correct", "size"), correct=("correct", "sum"))
    summary["accuracy_%"] = (100 * summary.correct / summary.cases).round(1)
    display(summary)
    print(f"Injected Ground-Truth Accuracy: {gt.correct.sum()}/{len(gt)} ({100 * gt.correct.mean():.1f}%)")

    failures = gt[~gt.correct]
    if len(failures):
        print("\\nFailures:")
        display(failures[["kind", "original", "injected"]])

    same = diff = 0
    mixed_groups = []
    for g in groups:
        real = [m for m in g.members if class_of_fn(m) is not None]
        for a, b in combinations(real, 2):
            if class_of_fn(a) == class_of_fn(b):
                same += 1
            else:
                diff += 1
        if len({class_of_fn(m) for m in real}) > 1:
            mixed_groups.append(g.group_id)

    pairs = same + diff
    purity = (100.0 * same / pairs) if pairs else 100.0
    print(f"Real-image pairwise class purity: {same}/{pairs} ({purity:.1f}%) | Mixed-class groups: {len(mixed_groups)}")

    return gt, summary, dict(
        dataset=dataset_name,
        images_scanned=n_total,
        duplicate_groups=len(groups),
        grouped_images=len(df),
        gt_correct=int(gt.correct.sum()),
        gt_total=len(gt),
        gt_accuracy_pct=round(float(100 * gt.correct.mean()), 1),
        exact_copy_acc=float(summary.loc["exact_copy", "accuracy_%"]),
        crop_resize_acc=float(summary.loc["crop_resize_q70", "accuracy_%"]),
        watermark_acc=float(summary.loc["watermark", "accuracy_%"]),
        color_variant_acc=float(summary.loc["color_variant", "accuracy_%"]),
        real_pairs=pairs,
        same_class_pairs=same,
        class_purity_pct=round(purity, 1),
        mixed_class_groups=len(mixed_groups),
        runtime_sec=round(elapsed, 1),
    )
"""
    ),
    md("## 4. Dataset 1: IKEA-FS (`brikwerk/ikeafs`)"),
    code(
        """
DATA_IKEA = Path("/tmp/catalog_ikea")
shutil.rmtree(DATA_IKEA, ignore_errors=True)
shutil.copytree(IKEA_DIR, DATA_IKEA / "ikea")

ikea_by_class = {}
for p in sorted((DATA_IKEA / "ikea").rglob("*")):
    if p.suffix.lower() in IMG_EXT:
        ikea_by_class.setdefault(p.parent.name, []).append(p)

gt_ikea_list = inject_ground_truth(ikea_by_class, DATA_IKEA / "injected", seed=42)
n_ikea = sum(1 for p in DATA_IKEA.rglob("*") if p.suffix.lower() in IMG_EXT)

config = load_config(CODE_DIR / "config.yaml")
setup_logging(config)
logging.getLogger().setLevel(logging.INFO)

t0 = time.time()
groups_ikea = run(
    input_dir=DATA_IKEA,
    output_path=OUT / "ikea_report.xlsx",
    config=config,
    cache_dir=Path("/tmp/.dedup_cache_ikea"),
)
t_ikea = time.time() - t0


def ikea_class_of(p: str):
    parts = Path(p).relative_to(DATA_IKEA).parts
    return parts[1] if parts[0] == "ikea" else None


gt_ikea_df, summary_ikea, stats_ikea = evaluate_run(
    "IKEA-FS", groups_ikea, gt_ikea_list, ikea_class_of, n_ikea, t_ikea
)
gt_ikea_df.to_csv(OUT / "ikea_injected_ground_truth.csv", index=False)
# Keep backward-compatible filenames as well
shutil.copyfile(OUT / "ikea_report.csv", OUT / "report.csv")
shutil.copyfile(OUT / "ikea_injected_ground_truth.csv", OUT / "injected_ground_truth_evaluation.csv")
"""
    ),
    md(
        """
### Verify Previously Merged Color Variants on IKEA-FS
In v1 (before the EMD + spatial recolor + cannot-link upgrades), `morabo_sofa`, `haerlanda_sofa`, `uppland_loveseat`, and `bergshult_shelf` had different color variants merged into single groups. Let's inspect all groups in those classes now:
"""
    ),
    code(
        """
for target_cls in ["morabo_sofa", "haerlanda_sofa", "uppland_loveseat", "bergshult_shelf"]:
    cls_groups = [
        (g.group_id, g.match_type, g.confidence, [Path(m).name for m in g.members])
        for g in groups_ikea
        if any( target_cls in Path(m).name for m in g.members )
    ]
    print(f"Class {target_cls}: {len(cls_groups)} groups")
    for gid, mtype, conf, names in cls_groups:
        print(f"  Group {gid:3d} ({mtype}, conf={conf}): {names}")
"""
    ),
    md(
        """
## 5. Dataset 2: RoyalOak Furniture (`patilaakash619/royaloak`)
We evaluate the exact same pipeline and configuration on the **RoyalOak E-Commerce Furniture Dataset** across its 9 core furniture categories (`Sofa`, `Bed`, `Recliner`, `Dining`, `Center_Table`, `Wardrobes`, `Office`, `Chairs`, `Bookshelves`), sampling up to 14 images per subcategory (~640 real catalog images) + 48 injected ground-truth cases.
"""
    ),
    code(
        """
FURNITURE_CATS = {
    "Sofa", "Bed", "Recliner", "Dining", "Center_Table",
    "Wardrobes", "Office", "Chairs", "Bookshelves"
}

DATA_ROYAL = Path("/tmp/catalog_royaloak")
shutil.rmtree(DATA_ROYAL, ignore_errors=True)

rng_royal = random.Random(42)
royal_by_class = {}

for cat_dir in sorted(ROYALOAK_DIR.iterdir()):
    if not cat_dir.is_dir() or cat_dir.name not in FURNITURE_CATS:
        continue
    for sub_dir in sorted(cat_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        cls_key = f"{cat_dir.name}__{sub_dir.name}"
        imgs = [
            p for p in sorted(sub_dir.iterdir())
            if p.suffix.lower() in IMG_EXT and Image.open(p).size not in {(425, 210), (135, 135)}
        ]
        if not imgs:
            continue
        dst_dir = DATA_ROYAL / "royaloak" / cls_key
        dst_dir.mkdir(parents=True, exist_ok=True)
        for p in imgs:
            dst_p = dst_dir / f"{cls_key}_{p.name}"
            shutil.copyfile(p, dst_p)
            royal_by_class.setdefault(cls_key, []).append(dst_p)

gt_royal_list = inject_ground_truth(royal_by_class, DATA_ROYAL / "injected", seed=42)
n_royal = sum(1 for p in DATA_ROYAL.rglob("*") if p.suffix.lower() in IMG_EXT)

t0 = time.time()
groups_royal = run(
    input_dir=DATA_ROYAL,
    output_path=OUT / "royaloak_report.xlsx",
    config=config,
    cache_dir=Path("/tmp/.dedup_cache_royaloak"),
)
t_royal = time.time() - t0


def royal_class_of(p: str):
    parts = Path(p).relative_to(DATA_ROYAL).parts
    return parts[1].split("__")[0] if parts[0] == "royaloak" else None


gt_royal_df, summary_royal, stats_royal = evaluate_run(
    "RoyalOak Furniture", groups_royal, gt_royal_list, royal_class_of, n_royal, t_royal
)
gt_royal_df.to_csv(OUT / "royaloak_injected_ground_truth.csv", index=False)
"""
    ),
    md("## 6. Combined Two-Dataset Comparison Table"),
    code(
        """
combined_df = pd.DataFrame([stats_ikea, stats_royal])
display(combined_df)
combined_df.to_csv(OUT / "combined_datasets_summary.csv", index=False)
"""
    ),
    md("## 7. Visual Inspection (IKEA-FS & RoyalOak)"),
    code(
        """
import matplotlib.pyplot as plt


def show_images(paths, titles, suptitle, max_cols=6):
    paths = paths[:max_cols]
    fig, axes = plt.subplots(1, len(paths), figsize=(3 * len(paths), 3.4))
    axes = np.atleast_1d(axes)
    for ax, p, t in zip(axes, paths, titles):
        ax.imshow(Image.open(p).convert("RGB"))
        ax.set_title(t, fontsize=8)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    plt.show()


for ds_label, grp_list in [("IKEA-FS", groups_ikea), ("RoyalOak", groups_royal)]:
    for mtype in ["exact", "near_exact", "embedding_color_verified"]:
        for g in [g for g in grp_list if g.match_type == mtype][:2]:
            show_images(
                g.members,
                [Path(m).name[:28] for m in g.members],
                f"[{ds_label}] Group {g.group_id} - {g.match_type} - conf {g.confidence} - {len(g.members)} imgs",
            )
"""
    ),
    md("## 8. Output files"),
    code(
        """
shutil.rmtree(CODE_DIR, ignore_errors=True)
for f in sorted(OUT.iterdir()):
    print(f"{f.name:55s} {f.stat().st_size / 1024:8.1f} KB")
"""
    ),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
