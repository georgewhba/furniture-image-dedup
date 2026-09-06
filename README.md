# Furniture Product-Image Duplicate Detection System

A production-grade tool that detects duplicate product images in a furniture e-commerce catalog. It finds exact copies, near-duplicates (cropped/recompressed/watermarked), and re-shot products — with a **mandatory color-verification gate** to prevent false positives from same-shape-different-color products.

**The tool never deletes, moves, or modifies any source image.** Its only output is a report (.xlsx + .csv) listing duplicate groups for human review.

---

## Table of Contents

- [Overview](#overview)
- [Streamlit Web Application](#streamlit-web-application)
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
- Scans a folder of product images (recursively)
- Groups images that show the **same physical product, in the same color**
- Assigns each group a confidence score (0–100) and match-type label
- Outputs an Excel + CSV report sorted by confidence for easy review

### What it doesn't do
- ❌ Never deletes, moves, or renames any image
- ❌ Never modifies source files
- ❌ Never needs write access to the image folder
- ❌ Never calls two images "duplicates" if they show different colors — even if the shape is identical

---

## Streamlit Web Application

The project includes an interactive web interface with Arabic RTL support, side-by-side visual inspection, and instant Excel report downloads.

### Running locally
```bash
streamlit run streamlit_app.py
```
Or simply double-click `تشغيل_واجهة_الويب_Streamlit.bat`.

### Deploying to Streamlit Cloud / Hugging Face Spaces
1. Push this repository to your GitHub account (Private or Public).
2. Connect to [share.streamlit.io](https://share.streamlit.io).
3. Select this repository and set main file path to `streamlit_app.py` or `app.py`.
4. Deploy! Users can upload `.zip` archives of product images or specify folders.

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
| `phash_threshold` | 10 | Max Hamming distance for perceptual match. Lower = stricter. |
| `similarity_threshold` | 0.85 | Min cosine similarity for embedding match. Higher = stricter. |
| `color_distance_threshold` | 25.0 | Max Lab-space color distance. Lower = stricter. **This is the most important threshold.** |
| `use_background_removal` | true | Isolate product before color extraction. Strongly recommended. |
| `color_n_clusters` | 5 | K-means clusters for dominant color palette. |

### Tuning Guide

- **Too many false positives (different colors flagged)?** → Lower `color_distance_threshold` (e.g., 15–20).
- **Too many false negatives (same product missed)?** → Lower `similarity_threshold` (e.g., 0.80) or raise `phash_threshold` (e.g., 12).
- **Pipeline too slow?** → Increase `embedding_batch_size` (if GPU available), decrease `faiss_nprobe`, or decrease `similarity_k`.

---

## Architecture — The Six-Stage Pipeline

```
Input Folder → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Stage 6
               Exact      Percep.   CLIP +    Color     Group +   Report
               Hash       Hash      FAISS     Gate ⚠️    Score
```

Each stage only processes images **not already matched** by earlier stages (funnel optimization).

### Stage 1 — Exact Hash (SHA-256)
Computes a cryptographic hash over raw file bytes. Two files with the same hash are byte-identical. Confidence: 100%. The cheapest and fastest check.

### Stage 2 — Perceptual Hash (pHash + dHash)
Computes perceptual hashes and compares by Hamming distance. Catches the same image after cropping, recompression, or watermarking. Both pHash *and* dHash must be within threshold (dual-hash check reduces false positives).

### Stage 3 — CLIP Embeddings + FAISS
Extracts a semantic embedding per image using a pretrained CLIP model. Builds a FAISS nearest-neighbor index for efficient similarity search. Catches "same product, different angle/background/lighting" — but also catches "same shape, different color," which is exactly why Stage 4 exists.

### Stage 4 — Color Verification Gate (MANDATORY)
**Every single candidate from Stage 3 must pass this gate.** No exceptions, no bypass. See [How Color Verification Works](#how-color-verification-works) for details.

### Stage 5 — Grouping + Confidence Scoring
Uses a union-find (disjoint-set) data structure to merge all confirmed pairwise matches into groups. Assigns each group the highest confidence and strongest match type from its constituent matches.

### Stage 6 — Report Generation
Outputs both `.xlsx` and `.csv` reports, sorted by confidence descending.

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

3. **Color comparison in Lab space**: The two palettes are compared in CIE Lab color space — a color system specifically designed so that "perceptual difference" corresponds to mathematical distance. Unlike RGB (where a warm-lit beige and cool-lit beige look very different), Lab handles lighting variation naturally. The optimal matching between palettes uses the Hungarian algorithm.

4. **Gate decision**: If the color distance exceeds the threshold (default: 25.0), the match is **rejected outright**, regardless of how high the AI similarity score was. The rejection is logged with both scores for auditability.

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
