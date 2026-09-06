"""Pipeline orchestrator — wires all six stages together.

This module owns the end-to-end flow:

1. **Discovery** — walk the input folder, validate each file as a real image
2. **Cache check** — skip already-processed files
3. **Stage 1** — Exact hash (SHA-256) → byte-identical groups
4. **Stage 2** — Perceptual hash (pHash + dHash) → crop/compress/watermark
5. **Stage 3** — CLIP embeddings + FAISS → re-shot candidates
6. **Stage 4** — Color verification gate (mandatory) → reject same-shape-different-color
7. **Stage 5** — Union-find grouping + confidence scoring
8. **Stage 6** — Report generation (.xlsx + .csv)

Progress is shown via ``tqdm`` bars.  The cache is flushed periodically
for crash resilience.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.cache import ImageCache
from src.color_verification import verify_color_match
from src.config import Config
from src.embeddings import EmbeddingExtractor, find_embedding_candidates
from src.exact_hash import (
    build_hash_map,
    compute_content_hash,
    compute_file_hash,
    find_exact_duplicates,
)
from src.perceptual_hash import HashPair, compute_hashes, find_near_duplicates
from src.report import generate_reports
from src.scoring import DuplicateGroup, PairMatch, build_groups, compute_embedding_confidence

logger = logging.getLogger(__name__)

# Flush cache every N images for crash resilience
_CACHE_FLUSH_INTERVAL = 500


# ---------------------------------------------------------------------------
# Image discovery & validation
# ---------------------------------------------------------------------------


def _discover_images(
    input_dir: Path,
    supported_extensions: tuple[str, ...],
    max_pixels: int,
    sample_size: int | None = None,
) -> list[Path]:
    """Walk the input directory and return validated image paths.

    Args:
        input_dir: Root directory to search recursively.
        supported_extensions: Allowed file extensions (lowercase, with dot).
        max_pixels: Maximum pixel count (decompression bomb guard).
        sample_size: If set, return only the first N valid images.

    Returns:
        List of validated image paths.
    """
    Image.MAX_IMAGE_PIXELS = max_pixels

    candidates = sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in supported_extensions
    )

    valid: list[Path] = []
    skipped = 0

    for path in tqdm(candidates, desc="Validating images", unit="file"):
        if sample_size is not None and len(valid) >= sample_size:
            break
        try:
            with Image.open(path) as img:
                img.verify()  # Checks file integrity without full decode
            valid.append(path)
        except Image.DecompressionBombError:
            logger.warning("Skipped (decompression bomb): %s", path)
            skipped += 1
        except Exception as exc:
            logger.warning("Skipped (invalid image): %s — %s", path, exc)
            skipped += 1

    logger.info(
        "Discovery: %d valid images, %d skipped, from %d candidates",
        len(valid),
        skipped,
        len(candidates),
    )
    return valid


def _safe_open_image(path: Path) -> Image.Image | None:
    """Open an image safely, returning None on failure."""
    try:
        img = Image.open(path)
        img.load()  # Force full decode
        return img
    except Exception as exc:
        logger.warning("Failed to open image: %s — %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(
    input_dir: Path,
    output_path: Path,
    config: Config,
    cache_dir: Path,
    sample_size: int | None = None,
    resume: bool = False,
    verbose: bool = False,
) -> list[DuplicateGroup]:
    """Execute the full six-stage deduplication pipeline.

    Args:
        input_dir: Path to the folder of images (read-only access).
        output_path: Path for the output report file.
        config: Validated pipeline configuration.
        cache_dir: Directory for the SQLite cache and FAISS index.
        sample_size: Process only the first N images (for quick tests).
        resume: If True, load cached results and process only new images.
        verbose: Enable debug-level logging.

    Returns:
        List of :class:`DuplicateGroup` instances.
    """
    t_start = time.time()

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Step 0: Discover and validate images
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 0: Image discovery")
    logger.info("=" * 60)

    image_paths = _discover_images(
        input_dir, config.supported_extensions, config.max_image_pixels, sample_size
    )

    if not image_paths:
        logger.warning("No valid images found in %s", input_dir)
        return []

    # ------------------------------------------------------------------
    # Step 1: Cache + Content hashes
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 1: Exact hash deduplication")
    logger.info("=" * 60)

    cache = ImageCache(cache_dir)

    # Compute content hashes and check cache
    file_content_hashes: dict[str, str] = {}
    new_files: list[str] = []
    cached_files: list[str] = []

    for path in tqdm(image_paths, desc="Computing content hashes", unit="file"):
        fpath = str(path)
        try:
            content_hash = compute_content_hash(path)
            file_content_hashes[fpath] = content_hash
            if resume and cache.has(fpath, content_hash):
                cached_files.append(fpath)
            else:
                new_files.append(fpath)
        except OSError as exc:
            logger.warning("Cannot read file: %s — %s", path, exc)

    logger.info("Cache: %d cached, %d new/changed", len(cached_files), len(new_files))

    # Compute exact hashes for new files
    file_exact_hashes: dict[str, str] = {}

    for fpath in cached_files:
        cached = cache.get(fpath, file_content_hashes[fpath])
        if cached and cached["exact_hash"]:
            file_exact_hashes[fpath] = cached["exact_hash"]

    for fpath in tqdm(new_files, desc="Computing exact hashes", unit="file"):
        try:
            h = compute_file_hash(Path(fpath), algorithm=config.hash_algorithm)
            file_exact_hashes[fpath] = h
            cache.put(fpath, file_content_hashes[fpath], exact_hash=h)
        except OSError as exc:
            logger.warning("Hash failed: %s — %s", fpath, exc)

    cache.flush()

    # Find exact duplicate groups
    hash_map = build_hash_map(file_exact_hashes)
    exact_groups = find_exact_duplicates(hash_map)

    # Track which files are redundant copies (canonical representative remains for downstream matching)
    redundant_files: set[str] = set()
    all_matches: list[PairMatch] = []

    for g in exact_groups:
        members = g["members"]
        # members[0] is canonical; members[1:] are redundant copies
        for p in members[1:]:
            redundant_files.add(p)
        # Create pairwise matches for scoring
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                all_matches.append(
                    PairMatch(
                        path_a=members[i],
                        path_b=members[j],
                        confidence=100,
                        match_type="exact",
                    )
                )

    # ------------------------------------------------------------------
    # Step 2: Perceptual hash
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 2: Perceptual hash deduplication")
    logger.info("=" * 60)

    ungrouped = [f for f in file_exact_hashes if f not in redundant_files]
    hash_pairs: dict[str, HashPair] = {}

    # Load cached hashes
    for fpath in ungrouped:
        if fpath in cached_files:
            cached = cache.get(fpath, file_content_hashes[fpath])
            if cached and cached["phash"] and cached["dhash"]:
                import imagehash

                hash_pairs[fpath] = HashPair(
                    phash=imagehash.hex_to_hash(cached["phash"]),
                    dhash=imagehash.hex_to_hash(cached["dhash"]),
                )

    # Compute for uncached
    to_compute = [f for f in ungrouped if f not in hash_pairs]
    count = 0
    for fpath in tqdm(to_compute, desc="Computing perceptual hashes", unit="file"):
        img = _safe_open_image(Path(fpath))
        if img is None:
            continue
        try:
            hp = compute_hashes(img)
            hash_pairs[fpath] = hp
            cache.update_field(
                fpath,
                file_content_hashes[fpath],
                phash=str(hp.phash),
                dhash=str(hp.dhash),
            )
            count += 1
            if count % _CACHE_FLUSH_INTERVAL == 0:
                cache.flush()
        except Exception as exc:
            logger.warning("Perceptual hash failed: %s — %s", fpath, exc)
        finally:
            img.close()

    cache.flush()

    # Find near-duplicate pairs
    phash_matches = find_near_duplicates(
        hash_pairs,
        phash_threshold=config.phash_threshold,
        dhash_threshold=config.dhash_threshold,
        confidence_base=config.confidence_phash_base,
        confidence_decay=config.confidence_phash_decay,
    )

    for m in phash_matches:
        a, b = m["pair"]
        img_a = _safe_open_image(Path(a))
        img_b = _safe_open_image(Path(b))
        if img_a is None or img_b is None:
            if img_a:
                img_a.close()
            if img_b:
                img_b.close()
            continue

        # Aspect ratio check (>15% diff => route to Stage 3)
        ar_a = img_a.width / img_a.height
        ar_b = img_b.width / img_b.height
        ar_diff = abs(ar_a - ar_b) / max(ar_a, ar_b)
        if ar_diff > 0.15:
            logger.info(
                "pHash pair %s ↔ %s aspect ratio diff %.1f%% > 15%% — skipping to Stage 3",
                Path(a).name,
                Path(b).name,
                ar_diff * 100,
            )
            img_a.close()
            img_b.close()
            continue

        # Color gate check on pHash matches (mandatory non-negotiable rule)
        color_res = verify_color_match(
            img_a,
            img_b,
            use_background_removal=config.use_background_removal,
            n_clusters=config.color_n_clusters,
            color_space=config.color_space,
            distance_threshold=config.color_distance_threshold,
            min_pixel_count=config.color_min_pixel_count,
            seed=config.random_seed,
            path_a=a,
            path_b=b,
        )
        img_a.close()
        img_b.close()

        if not color_res.passed:
            logger.info(
                "Color gate REJECTED pHash pair %s ↔ %s: distance=%.2f > threshold=%.2f — likely same shape, different color",
                Path(a).name,
                Path(b).name,
                color_res.distance,
                config.color_distance_threshold,
            )
            continue

        redundant_files.add(b)
        all_matches.append(
            PairMatch(
                path_a=a,
                path_b=b,
                confidence=m["confidence"],
                match_type="near_exact",
            )
        )

    # ------------------------------------------------------------------
    # Step 3: CLIP embeddings + FAISS
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 3: Embedding similarity search")
    logger.info("=" * 60)

    ungrouped_for_embedding = [f for f in file_exact_hashes if f not in redundant_files]

    if len(ungrouped_for_embedding) < 2:
        logger.info("Fewer than 2 ungrouped images — skipping embedding stage")
        embedding_candidates: list[dict] = []
    else:
        # Load or compute embeddings
        embeddings_map: dict[str, np.ndarray] = {}

        # Load cached embeddings
        for fpath in ungrouped_for_embedding:
            if fpath in cached_files:
                cached = cache.get(fpath, file_content_hashes[fpath])
                if cached and cached["embedding"] is not None:
                    embeddings_map[fpath] = cached["embedding"]

        # Compute missing embeddings
        to_embed = [f for f in ungrouped_for_embedding if f not in embeddings_map]

        if to_embed:
            logger.info("Extracting embeddings for %d new images", len(to_embed))
            extractor = EmbeddingExtractor(
                model_name=config.embedding_model,
                pretrained=config.embedding_pretrained,
            )

            # Process in batches to avoid loading all images at once
            batch_size = config.embedding_batch_size
            for batch_start in tqdm(
                range(0, len(to_embed), batch_size),
                desc="Extracting embeddings",
                unit="batch",
            ):
                batch_paths = to_embed[batch_start : batch_start + batch_size]
                batch_images: list[Image.Image] = []
                batch_valid_paths: list[str] = []

                for fpath in batch_paths:
                    img = _safe_open_image(Path(fpath))
                    if img is not None:
                        batch_images.append(img)
                        batch_valid_paths.append(fpath)

                if batch_images:
                    embs = extractor.extract_batch(batch_images, batch_size=len(batch_images))
                    for i, fpath in enumerate(batch_valid_paths):
                        embeddings_map[fpath] = embs[i]
                        cache.update_field(
                            fpath,
                            file_content_hashes[fpath],
                            embedding=embs[i],
                        )

                    # Close images
                    for img in batch_images:
                        img.close()

                if (batch_start // batch_size) % 10 == 0:
                    cache.flush()

            cache.flush()

        # Build ordered arrays for FAISS
        ordered_paths = [f for f in ungrouped_for_embedding if f in embeddings_map]
        if len(ordered_paths) >= 2:
            ordered_embeddings = np.vstack([embeddings_map[f] for f in ordered_paths])
            embedding_candidates = find_embedding_candidates(
                ordered_paths, ordered_embeddings, config
            )
        else:
            embedding_candidates = []

    # ------------------------------------------------------------------
    # Step 4: Color verification (MANDATORY — no exceptions)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 4: Color verification gate")
    logger.info("=" * 60)

    passed_candidates = 0
    rejected_candidates = 0

    for cand in tqdm(embedding_candidates, desc="Color verification", unit="pair"):
        path_a, path_b = cand["pair"]
        similarity = cand["similarity"]

        img_a = _safe_open_image(Path(path_a))
        img_b = _safe_open_image(Path(path_b))

        if img_a is None or img_b is None:
            logger.warning("Cannot open image for color check: %s ↔ %s", path_a, path_b)
            rejected_candidates += 1
            continue

        try:
            result = verify_color_match(
                img_a,
                img_b,
                use_background_removal=config.use_background_removal,
                n_clusters=config.color_n_clusters,
                color_space=config.color_space,
                distance_threshold=config.color_distance_threshold,
                min_pixel_count=config.color_min_pixel_count,
                seed=config.random_seed,
                embedding_similarity=similarity,
                path_a=path_a,
                path_b=path_b,
            )

            if result.passed:
                conf = compute_embedding_confidence(
                    similarity,
                    result.distance,
                    base=config.confidence_embedding_base,
                    distance_threshold=config.color_distance_threshold,
                )
                all_matches.append(
                    PairMatch(
                        path_a=path_a,
                        path_b=path_b,
                        confidence=conf,
                        match_type="embedding_color_verified",
                    )
                )
                passed_candidates += 1
            else:
                rejected_candidates += 1
        except Exception as exc:
            logger.warning("Color verification error: %s ↔ %s — %s", path_a, path_b, exc)
            rejected_candidates += 1
        finally:
            img_a.close()
            img_b.close()

    logger.info(
        "Stage 4 complete: %d passed, %d rejected",
        passed_candidates,
        rejected_candidates,
    )

    # ------------------------------------------------------------------
    # Step 5: Grouping + confidence scoring
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 5: Grouping and scoring")
    logger.info("=" * 60)

    groups = build_groups(all_matches)

    # ------------------------------------------------------------------
    # Step 6: Report generation
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 6: Report generation")
    logger.info("=" * 60)

    runtime = time.time() - t_start

    xlsx_path, csv_path, inv_xlsx_path, inv_csv_path = generate_reports(
        groups,
        output_path,
        total_images=len(image_paths),
        runtime_seconds=runtime,
        all_image_paths=[str(p) for p in image_paths],
        content_hashes=file_content_hashes,
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Total images scanned:         %d", len(image_paths))
    logger.info("Duplicate groups found:       %d", len(groups))
    logger.info(
        "Images in duplicate groups:   %d",
        sum(len(g.members) for g in groups),
    )
    logger.info(
        "Unique images (no dupes):     %d",
        len(image_paths) - sum(len(g.members) for g in groups),
    )
    logger.info("Runtime:                      %.1f seconds", runtime)
    logger.info("Duplicates Excel:             %s", xlsx_path)
    logger.info("Duplicates CSV:               %s", csv_path)
    logger.info("Master Inventory Excel:       %s", inv_xlsx_path)
    logger.info("Master Inventory CSV:         %s", inv_csv_path)

    cache.close()
    return groups
