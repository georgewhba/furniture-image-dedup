---
title: Furniture Duplicate Detection AI
emoji: 🪑
colorFrom: blue
colorTo: teal
sdk: docker
app_port: 7860
pinned: true
short_description: 6-stage CLIP + FAISS + CIE-Lab EMD duplicate furniture image detector
---

# Furniture Product-Image Duplicate Detection System

A production-grade computer-vision pipeline that detects duplicate product images in furniture e-commerce catalogs. It identifies exact byte copies, near-duplicates (cropped, resized, recompressed, watermarked), and cross-angle/re-shot product photos — guarded by a **mandatory color-verification gate (CIE-Lab + Earth Mover's Distance + spatial recolor detection)** to prevent false positives on same-shape, different-color SKUs.

**The tool never deletes, moves, or modifies any source image.** Its output is a color-coded audit report (`.xlsx` + `.csv` + master inventory) for human review.

---

## Audited Multi-Dataset Benchmarks (Tesla T4 GPU)

Reproducible on Kaggle: **[Interactive GPU Benchmark Notebook (v4)](https://www.kaggle.com/code/djnd27/furniture-duplicate-detection-ikea)**

| Metric / Scenario | IKEA-FS (Baseline v1) | IKEA-FS (Upgraded v4) | RoyalOak Furniture (Upgraded v4) |
|---|---|---|---|
| **Catalog + Evaluation Scope** | 1,141 images (1,093 IKEA + 48 GT) | 1,141 images (1,093 IKEA + 48 GT) | 671 images (623 clean products + 48 GT) |
| **Overall Ground-Truth Accuracy (48 pairs)** | 85.4% (41 / 48) | **95.8% (46 / 48)** | **100.0% (48 / 48)** |
| **Exact Byte Copies (`exact`)** | 12 / 12 (100.0%) | **12 / 12 (100.0%)** | **12 / 12 (100.0%)** |
| **4% Crop + Resize + JPEG q70** | 5 / 12 (41.7%) | **10 / 12 (83.3%)** | **12 / 12 (100.0%)** |
| **Watermarked Copies** | 12 / 12 (100.0%) | **12 / 12 (100.0%)** | **12 / 12 (100.0%)** |
| **Recolored Color-Variant Traps Blocked** | 12 / 12 GT (merged catalog colors) | **12 / 12 GT + 100% color split** | **12 / 12 GT + 98.6% category purity** |
| **Runtime (Tesla T4)** | 643s (~10.7 min) | 620.9s (~10.3 min) | 103.6s (~1.7 min) |

> **Data Audit Finding on `patilaakash619/royaloak`:** Raw scraped folders contained **678 website mega-menu banners** `(425×210)` repeated up to 74 times across categories; filtering banner dimensions isolated 623 true `(480×300)` furniture catalog photographs before evaluation.

---

## Table of Contents

- [Overview](#overview)
- [Interactive Web Application & Built-in Demo](#interactive-web-application--built-in-demo)
- [Deploying to Hugging Face Spaces (Recommended)](#deploying-to-hugging-face-spaces-recommended)
- [Running on Kaggle (free GPU)](#running-on-kaggle-free-gpu)
- [One-Click Runners](#one-click-runners)
- [Setup](#setup)
- [CLI Usage](#cli-usage)
- [Configuration](#configuration)
- [Architecture — The Six-Stage Pipeline](#architecture--the-six-stage-pipeline)
- [How Color Verification Works](#how-color-verification-works)
- [Output Schema](#output-schema)
- [Testing](#testing)
- [Incremental Re-runs](#incremental-re-runs)
- [Troubleshooting](#troubleshooting)

---

## Overview

### What it does
- Scans a folder of product images recursively (read-only)
- Groups images showing the **same physical product in the same color** across 3 tiers (`exact`, `near_exact`, `embedding_color_verified`)
- Enforces **cannot-link constraints** during quotient-graph clustering to stop transitive chaining between different colorways
- Outputs `.xlsx`, `.csv`, master all-images inventory, and `_telemetry.json` audit metrics

### What it doesn't do
- Never deletes, moves, or renames any image
- Never modifies source files
- Never merges two different-color SKUs — even when CLIP structural similarity exceeds 95%

---

## Interactive Web Application & Built-in Demo

The Streamlit web app (`streamlit_app.py`) supports **Bilingual English / Arabic RTL switching**, **1-click evaluation on a curated 13-image built-in catalog (`demo_catalog/`)**, live Color Gate rejection cards showing blocked false-positive pairs side-by-side, ZIP/Google Drive/Direct URL/Kaggle imports, and instant Excel downloads.

### Running locally
```bash
streamlit run streamlit_app.py
```
Or on Windows, double-click `تشغيل_واجهة_الويب_Streamlit.bat`.

## Deploying to Hugging Face Spaces (Recommended)

This repository includes a production `Dockerfile` configured for Hugging Face Spaces (`16 GB RAM`, `sdk: docker`, port `7860`) that pre-bakes OpenCLIP `ViT-B/32` and `rembg (u2netp)` weights during container build for zero cold-start wait:

1. Create a new Space on [huggingface.co/new-space](https://huggingface.co/new-space) selecting **Docker** SDK.
2. Push or link this repository (`main` branch). The YAML header at the top of `README.md` configures the Space automatically.
3. Alternatively, connect to [share.streamlit.io](https://share.streamlit.io) with `streamlit_app.py` (for smaller batches).

---

## Running on Kaggle (free GPU)

- **Notebook:** https://www.kaggle.com/code/djnd27/furniture-duplicate-detection-ikea — runs the full pipeline on the [IKEA-FS](https://www.kaggle.com/datasets/brikwerk/ikeafs) dataset (1,092 images, 46 products) plus injected ground-truth cases (exact copy, crop/recompress, watermark, recolored variant) and reports accuracy.
- **Source-code dataset:** https://www.kaggle.com/datasets/djnd27/furniture-image-dedup-code

To publish updates:
```bash
python kaggle/build_code_dataset.py
kaggle datasets version -p kaggle/dataset/build -m "update code"
python kaggle/notebook/build_notebook.py
kaggle kernels push -p kaggle/notebook
```

---

## One-Click Runners

For non-technical users and catalog reviewers:
- **`تشغيل_الأداة_بضغطة_واحدة.bat`**: Interactive Arabic CLI assistant with drag-and-drop folder support.
- **`تشغيل_واجهة_الويب_Streamlit.bat`**: One-click local web UI launcher in browser.

---

## Setup

### Prerequisites
- Python 3.10 or later
- ~2 GB disk space for model downloads (CLIP + rembg, cached after first run)

### Installation

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### First Run — Model Downloads

On the first run, two models are automatically downloaded and cached:

| Model | Size | Purpose |
|-------|------|---------|
| CLIP ViT-B/32 | ~600 MB | Image embeddings for semantic similarity |
| rembg (bria-rmbg) | ~170 MB | Background removal for color extraction |

These are downloaded once and cached locally. Subsequent runs use the cache.

---

## CLI Usage

### Basic run
```bash
python -m furniture_dedup run \
  --input /path/to/images \
  --output report.xlsx \
  --config config.yaml \
  --cache-dir .dedup_cache
```

### Quick test on a small sample
```bash
python -m furniture_dedup run \
  --input /path/to/images \
  --output test_report.xlsx \
  --config config.yaml \
  --sample-size 100 \
  --verbose
```

### Resume an interrupted run
```bash
python -m furniture_dedup run \
  --input /path/to/images \
  --output report.xlsx \
  --config config.yaml \
  --cache-dir .dedup_cache \
  --resume
```

### Full options
```
python -m furniture_dedup run --help

positional arguments: none

options:
  --input PATH          Folder containing images (searched recursively)
  --output PATH         Output report path (.xlsx and .csv generated)
  --config PATH         Path to config.yaml
  --cache-dir PATH      Cache directory (default: .dedup_cache)
  --sample-size N       Process only the first N images
  --resume              Use cache, process only new/changed files
  --verbose             Debug-level logging
```

---

## Configuration

Every tunable threshold lives in `config.yaml`. Nothing is hardcoded.

### Key Parameters

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `phash_threshold` | 8 | Max Hamming distance for perceptual match (plus crop4 hash). Lower = stricter. |
| `dhash_threshold` | 6 | Max dHash Hamming distance. |
| `similarity_threshold` | 0.925 | Min cosine similarity for CLIP embedding match. Higher = stricter. |
| `color_distance_threshold` | 16.5 | Max Lab-space EMD / spatial distance. Lower = stricter. **Most important guardrail.** |
| `use_background_removal` | true | Isolate product with `rembg (u2netp)` before color extraction. |
| `color_n_clusters` | 5 | K-means clusters for dominant color palette. |

### Tuning Guide

- **Too many false positives (different colors flagged)?** → Lower `color_distance_threshold` (e.g., 12–15).
- **Too many false negatives (same product missed)?** → Lower `similarity_threshold` (e.g., 0.88) or raise `phash_threshold` (e.g., 10).
- **Pipeline too slow?** → Increase `embedding_batch_size` (if GPU available), or disable background removal only on clean studio cutouts.

---

## Architecture — The Six-Stage Pipeline

```
Input Folder → Stage 1 → Stage 2       → Stage 3     → Stage 4        → Stage 5         → Stage 6
               SHA-256   Crop-Trimmed     CLIP +        Color Gate       Quotient-Graph    Report +
               Exact     pHash + dHash    FAISS         (EMD + Spatial)  + Cannot-Link     Inventory
```

Each stage applies strict precision guardrails while preserving full candidate graph connectivity:

### Stage 1 — Exact Hash (SHA-256)
Computes a cryptographic hash over raw file bytes. Two files with the same hash are byte-identical (confidence: 100%).

### Stage 2 — Crop-Aware Perceptual Hash (pHash + dHash)
Computes full-frame and 4%-border-cropped pHash/dHash plus trimmed-foreground aspect-ratio verification. Loose hash matches (`distance > 2`) are deferred to Stage 3/4 CLIP semantic verification to prevent white-backdrop silhouette collisions.

### Stage 3 — CLIP Embeddings + FAISS
Extracts semantic embeddings using OpenCLIP `ViT-B/32` (`laion2b_s34b_b79k`) and builds a FAISS inner-product index (`IndexFlatIP` / `IndexIVFFlat` with explicit `METRIC_INNER_PRODUCT`). Catches re-shot angles and lighting shifts — while Stage 4 guards against different colorways.

### Stage 4 — Mandatory Color Verification Gate
**Every candidate from Stage 2 (strict) and Stage 3 MUST pass this gate.** Uses `rembg (u2netp)` product isolation, CIE-Lab K-means palettes, **Earth Mover's Distance (Wasserstein-1 optimal transport via `scipy.optimize.linprog`)**, and a crop-safe `16×16` Lab spatial recolor check for lifestyle room scenes.

### Stage 5 — Quotient-Graph Clustering & Cannot-Link Constraints
Collapses exact equivalence classes into canonical nodes, enforces pairwise **cannot-link constraints** from Stage 2/4 Color Gate rejections to stop transitive chaining (`Navy ↔ Grey ↔ Green`), and verifies cluster density (`>= 0.70`) before assigning final group scores.

### Stage 6 — Report Generation
Outputs sorted `.xlsx` + `.csv` duplicate reports, master inventory of all scanned images, and `_telemetry.json` audit metrics.

---

## How Color Verification Works

This section explains the most important part of the system in plain language.

### The Problem

When you use AI to find similar images (Stage 3), it works by understanding the *shape and context* of objects. A beige sofa and a navy sofa look nearly identical to the AI — they have the same shape, proportions, cushions, legs, and style. The AI gives them a very high similarity score.

But they're **not duplicates**. They're different products (different SKUs, different catalog entries). Flagging them as duplicates would cause the catalog owner to delete one, losing a real product listing.

### The Solution

After the AI finds candidate pairs, every single one passes through a color verification gate:

1. **Background removal**: A lightweight AI model (`rembg`) removes the background (studio backdrop, room scene, etc.) from both images. This isolates just the furniture piece, so background colors don't contaminate the comparison.

2. **Color extraction**: The pixel colors of the isolated product are clustered into 5 dominant color groups using K-means. This produces a "color palette" — e.g., "60% beige, 20% brown, 15% cream, 5% dark brown" for a beige sofa.

3. **Color comparison in Lab space**: The two palettes are compared in CIE Lab color space using both Hungarian matching and **Earth Mover's Distance (Wasserstein-1 optimal transport via `scipy.optimize.linprog`)**, combined with a crop-safe **16×16 spatial recolor detector** to catch localized product recoloring inside identical lifestyle room scenes.

4. **Gate decision**: If the combined color distance exceeds the threshold (default: `16.5`, calibrated on IKEA-FS and RoyalOak), the match is **rejected outright**, regardless of how high the AI similarity score was, and logged as a **cannot-link constraint** so Stage 5 cannot bridge different colors through a third image.

### Why This Works

- **A beige sofa vs. a navy sofa** differ enormously in Lab space (distance > 60). The gate rejects them instantly.
- **A beige sofa in warm lighting vs. the same beige sofa in cool lighting** differ only slightly in Lab space (distance < 10). The gate passes them correctly.
- **Background removal** prevents a sofa-on-white-background vs. sofa-on-dark-background from being rejected due to background color differences.

---

## Output Schema

### Report Format: One Row Per Image

| Column | Type | Description |
|--------|------|-------------|
| `group_id` | int | Group identifier (1-based, sequential) |
| `file_path` | str | Absolute path to the image |
| `file_name` | str | Basename of the file |
| `confidence` | int | 0–100, higher = more certain |
| `match_type` | str | `exact`, `near_exact`, or `embedding_color_verified` |
| `group_size` | int | Number of images in this group |

### Match Types

| Type | Meaning | Confidence Range |
|------|---------|-----------------|
| `exact` | Byte-identical files | Always 100 |
| `near_exact` | Same image, cropped/compressed/watermarked | 75–95 |
| `embedding_color_verified` | Same product re-shot, color verified | 40–80 |

### Excel Features
- Conditional coloring: green (≥90), yellow (70–89), red (<70)
- Frozen header row
- Auto-width columns
- Summary sheet with aggregate statistics

---

## Testing

### Unit Tests
```bash
pytest tests/ -v --tb=short
```

Tests cover:
- `test_exact_hash.py`: Identical files produce same hash, different files don't, corrupted files don't crash
- `test_perceptual_hash.py`: Identical images → distance 0, compressed → small distance, different → large distance
- `test_color_verification.py`: **The most important tests** — same-color passes, different-color rejects (beige vs navy, red vs blue)
- `test_scoring.py`: Union-find correctness, transitive grouping, confidence aggregation

### Validation Harness
```bash
python validation/evaluate.py \
  --report-dir output/ \
  --ground-truth validation/ground_truth.csv
```

Reports:
- Overall precision, recall, F1
- **False-positive rate on "same_product_different_color" pairs** — the single most important metric

### Linting
```bash
ruff check src/ tests/
black --check src/ tests/
```

---

## Incremental Re-runs

The pipeline caches computed hashes, embeddings, and color palettes in a SQLite database (stored in `--cache-dir`). On re-runs with `--resume`:

1. Each file is identified by its path + content SHA-256
2. If the path+hash exists in cache, cached results are reused
3. Only new or modified files are reprocessed
4. The FAISS index is rebuilt (fast) using cached + new embeddings

**Killing the process mid-run is safe.** The cache is flushed every 500 images, so at most 500 images of work are lost. Use `--resume` to continue.

---

## Troubleshooting

### "CUDA out of memory"
Lower `embedding_batch_size` in `config.yaml` (e.g., from 32 to 8). Or switch to CPU by ensuring `faiss_use_gpu: false` and running without a CUDA-enabled PyTorch.

### "rembg download failed"
The background removal model downloads automatically. If behind a proxy, set `HTTP_PROXY` and `HTTPS_PROXY` environment variables. Or set `use_background_removal: false` in config (not recommended — reduces color verification accuracy).

### "No valid images found"
Check that `supported_extensions` in `config.yaml` includes your file types. The tool checks each file actually decodes as an image — a renamed text file with `.jpg` extension will be correctly skipped.

### "Pipeline is slow"
- Stage 1–2 are I/O-bound. SSD storage helps.
- Stage 3 is GPU-bound. A GPU dramatically speeds up embedding extraction.
- Stage 4 (color verification) is the bottleneck at scale due to background removal. Set `use_background_removal: false` for faster but less accurate color comparison.

### "Too many false positives"
Lower `color_distance_threshold` (e.g., 15–20). This makes the color gate stricter.

### "Too many false negatives"
Raise `color_distance_threshold` (e.g., 30–35) and/or lower `similarity_threshold` (e.g., 0.80).
