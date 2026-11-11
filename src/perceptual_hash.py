"""Stage 2 — Perceptual hash near-duplicate detection.

Catches the same image after crop, recompression, or watermarking by
computing perceptual (pHash) and difference (dHash) hashes and comparing
them by Hamming distance.

Only runs on images **not** already grouped by Stage 1 (exact hash).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import imagehash
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def trim_uniform_background(image: Image.Image) -> Image.Image:
    """Crop uniform studio borders (e.g. white/neutral backdrop) around the product.

    If the four image borders are not uniform (such as in a full room scene),
    returns the original image unchanged.
    """
    rgb = np.array(image.convert("RGB"), dtype=np.int16)
    h, w, _ = rgb.shape
    if h < 32 or w < 32:
        return image
    border = np.concatenate([rgb[0, :], rgb[-1, :], rgb[:, 0], rgb[:, -1]], axis=0)
    bg_med = np.median(border, axis=0)
    border_diff = np.max(np.abs(border - bg_med), axis=1)
    if float(np.percentile(border_diff, 80)) > 18.0:
        return image
    fg_mask = np.max(np.abs(rgb - bg_med), axis=2) > 30
    ys, xs = np.where(fg_mask)
    if len(xs) < 50:
        return image
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    if (x1 - x0) < 16 or (y1 - y0) < 16:
        return image
    pad_x = max(2, int(w * 0.04))
    pad_y = max(2, int(h * 0.04))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)
    return image.crop((x0, y0, x1, y1))


@dataclass(frozen=True)
class HashPair:
    """Stores perceptual and difference hashes (plus 4% border-crop hashes) for an image."""

    phash: imagehash.ImageHash
    dhash: imagehash.ImageHash
    phash_crop4: imagehash.ImageHash | None = None
    dhash_crop4: imagehash.ImageHash | None = None


def compute_phash(image: Image.Image, hash_size: int = 8) -> imagehash.ImageHash:
    """Compute the perceptual hash (pHash) of an image.

    Args:
        image: A PIL Image (any mode — will be converted internally).
        hash_size: Size of the hash grid.  Default 8 → 64-bit hash.

    Returns:
        An :class:`imagehash.ImageHash` object.
    """
    return imagehash.phash(image, hash_size=hash_size)


def compute_dhash(image: Image.Image, hash_size: int = 8) -> imagehash.ImageHash:
    """Compute the difference hash (dHash) of an image.

    Args:
        image: A PIL Image.
        hash_size: Size of the hash grid.  Default 8 → 64-bit hash.

    Returns:
        An :class:`imagehash.ImageHash` object.
    """
    return imagehash.dhash(image, hash_size=hash_size)


def compute_hashes(image: Image.Image) -> HashPair:
    """Compute full-image and 4%-border-cropped pHash and dHash for an image.

    Args:
        image: A PIL Image.

    Returns:
        A :class:`HashPair` containing both full and crop4 hashes.
    """
    w, h = image.size
    if w >= 25 and h >= 25:
        c4 = image.crop((int(w * 0.04), int(h * 0.04), int(w * 0.96), int(h * 0.96)))
        pc4 = compute_phash(c4)
        dc4 = compute_dhash(c4)
    else:
        pc4 = None
        dc4 = None
    return HashPair(
        phash=compute_phash(image),
        dhash=compute_dhash(image),
        phash_crop4=pc4,
        dhash_crop4=dc4,
    )


def hamming_distance(
    hash_a: imagehash.ImageHash,
    hash_b: imagehash.ImageHash,
) -> int:
    """Compute the Hamming distance between two image hashes.

    Args:
        hash_a: First hash.
        hash_b: Second hash.

    Returns:
        Number of differing bits (0 = identical).
    """
    return hash_a - hash_b


def are_near_duplicates(
    pair_a: HashPair,
    pair_b: HashPair,
    phash_threshold: int,
    dhash_threshold: int,
) -> tuple[bool, int, int]:
    """Check whether two images are near-duplicates by perceptual hash.

    Both pHash *and* dHash distances must be within their respective
    thresholds for the pair to be considered a near-duplicate.  Also checks
    asymmetric 4% border-crop hashes at a strict <= 2 bit threshold.

    Args:
        pair_a: Hashes for the first image.
        pair_b: Hashes for the second image.
        phash_threshold: Maximum allowed pHash Hamming distance.
        dhash_threshold: Maximum allowed dHash Hamming distance.

    Returns:
        A tuple of ``(is_match, phash_distance, dhash_distance)``.
    """
    p_dist = int(hamming_distance(pair_a.phash, pair_b.phash))
    d_dist = int(hamming_distance(pair_a.dhash, pair_b.dhash))
    if pair_a.phash_crop4 is not None and pair_a.dhash_crop4 is not None:
        cp = int(hamming_distance(pair_a.phash_crop4, pair_b.phash))
        cd = int(hamming_distance(pair_a.dhash_crop4, pair_b.dhash))
        if cp <= 2 and cd <= 2 and (cp + cd) < (p_dist + d_dist):
            p_dist, d_dist = cp, cd
    if pair_b.phash_crop4 is not None and pair_b.dhash_crop4 is not None:
        cp = int(hamming_distance(pair_a.phash, pair_b.phash_crop4))
        cd = int(hamming_distance(pair_a.dhash, pair_b.dhash_crop4))
        if cp <= 2 and cd <= 2 and (cp + cd) < (p_dist + d_dist):
            p_dist, d_dist = cp, cd
    is_match = bool(p_dist <= phash_threshold and d_dist <= dhash_threshold)
    return is_match, p_dist, d_dist


def compute_confidence(
    phash_dist: int,
    dhash_dist: int,
    base: int = 95,
    decay: float = 2.0,
) -> int:
    """Compute confidence score for a perceptual-hash match.

    Confidence starts at *base* and decays with the average of the two
    hash distances.

    Args:
        phash_dist: Hamming distance for pHash.
        dhash_dist: Hamming distance for dHash.
        base: Starting confidence (default 95).
        decay: Confidence lost per unit of average distance.

    Returns:
        Confidence score clamped to ``[0, 100]``.
    """
    avg_dist = (phash_dist + dhash_dist) / 2.0
    score = base - avg_dist * decay
    return max(0, min(100, int(round(score))))


def find_near_duplicates(
    hash_pairs: dict[str, HashPair],
    phash_threshold: int,
    dhash_threshold: int,
    confidence_base: int = 95,
    confidence_decay: float = 2.0,
) -> list[dict]:
    """Find near-duplicate pairs among a set of image hashes.

    Performs pairwise comparison — acceptable at the scale this stage
    handles (images not already grouped by exact hash).

    Args:
        hash_pairs: Mapping of ``{file_path: HashPair}``.
        phash_threshold: Maximum pHash Hamming distance.
        dhash_threshold: Maximum dHash Hamming distance.
        confidence_base: Base confidence for scoring.
        confidence_decay: Decay factor for scoring.

    Returns:
        A list of match dicts, each containing:

        - ``pair``: tuple of two file paths
        - ``confidence``: computed confidence score
        - ``match_type``: ``"near_exact"``
        - ``phash_distance``: the pHash Hamming distance
        - ``dhash_distance``: the dHash Hamming distance
    """
    paths = list(hash_pairs.keys())
    matches: list[dict] = []
    n = len(paths)
    if n < 2:
        return matches

    try:
        import faiss
        import numpy as np

        p_bytes = []
        d_ints = []
        for p in paths:
            hp = hash_pairs[p]
            p_bytes.append(np.packbits(hp.phash.hash.flatten()))
            d_ints.append(int(str(hp.dhash), 16))

        xb = np.stack(p_bytes)
        index = faiss.IndexBinaryFlat(64)
        index.add(xb)
        # FAISS IndexBinaryFlat.range_search uses strict inequality (dist < radius).
        # To include distances <= phash_threshold, radius must be phash_threshold + 1.
        lims, D, I = index.range_search(xb, phash_threshold + 1)
        pair_best: dict[tuple[int, int], tuple[int, int]] = {}
        for i in range(n):
            for k in range(int(lims[i]), int(lims[i + 1])):
                j = int(I[k])
                if j > i:
                    p_dist = int(D[k])
                    d_dist = (d_ints[i] ^ d_ints[j]).bit_count()
                    if d_dist <= dhash_threshold:
                        pair_best[(i, j)] = (p_dist, d_dist)

        # Also check 4% border-cropped hashes against full hashes at strict <= 2 threshold
        crop_indices = []
        crop_p_bytes = []
        crop_d_ints = []
        for idx_i, p in enumerate(paths):
            hp = hash_pairs[p]
            if hp.phash_crop4 is not None and hp.dhash_crop4 is not None:
                crop_indices.append(idx_i)
                crop_p_bytes.append(np.packbits(hp.phash_crop4.hash.flatten()))
                crop_d_ints.append(int(str(hp.dhash_crop4), 16))
        if crop_p_bytes:
            xq_crop = np.stack(crop_p_bytes)
            c_lims, c_D, c_I = index.range_search(xq_crop, 3)
            for ci, i in enumerate(crop_indices):
                for k in range(int(c_lims[ci]), int(c_lims[ci + 1])):
                    j = int(c_I[k])
                    if i == j:
                        continue
                    p_dist = int(c_D[k])
                    d_dist = (crop_d_ints[ci] ^ d_ints[j]).bit_count()
                    if p_dist <= 2 and d_dist <= 2:
                        u, v = (i, j) if i < j else (j, i)
                        prev = pair_best.get((u, v))
                        if prev is None or (p_dist + d_dist) < (prev[0] + prev[1]):
                            pair_best[(u, v)] = (p_dist, d_dist)

        for (i, j), (p_dist, d_dist) in sorted(pair_best.items()):
            conf = compute_confidence(p_dist, d_dist, confidence_base, confidence_decay)
            matches.append(
                {
                    "pair": (paths[i], paths[j]),
                    "confidence": conf,
                    "match_type": "near_exact",
                    "phash_distance": p_dist,
                    "dhash_distance": d_dist,
                }
            )
    except Exception as exc:
        logger.warning("FAISS binary search failed, using pairwise fallback: %s", exc)
        for i in range(n):
            for j in range(i + 1, n):
                is_match, p_dist, d_dist = are_near_duplicates(
                    hash_pairs[paths[i]],
                    hash_pairs[paths[j]],
                    phash_threshold,
                    dhash_threshold,
                )
                if is_match:
                    conf = compute_confidence(p_dist, d_dist, confidence_base, confidence_decay)
                    matches.append(
                        {
                            "pair": (paths[i], paths[j]),
                            "confidence": conf,
                            "match_type": "near_exact",
                            "phash_distance": p_dist,
                            "dhash_distance": d_dist,
                        }
                    )

    logger.info(
        "Stage 2 complete: %d near-duplicate pairs found from %d images",
        len(matches),
        len(paths),
    )
    return matches
