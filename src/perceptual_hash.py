"""Stage 2 — Perceptual hash near-duplicate detection.

Catches the same image after crop, recompression, or watermarking by
computing perceptual (pHash) and difference (dHash) hashes and comparing
them by Hamming distance.

Only runs on images **not** already grouped by Stage 1 (exact hash).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import imagehash
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HashPair:
    """Stores both perceptual and difference hashes for a single image."""

    phash: imagehash.ImageHash
    dhash: imagehash.ImageHash


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
    """Compute both pHash and dHash for an image.

    Args:
        image: A PIL Image.

    Returns:
        A :class:`HashPair` containing both hashes.
    """
    return HashPair(
        phash=compute_phash(image),
        dhash=compute_dhash(image),
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
    thresholds for the pair to be considered a near-duplicate.  This
    dual-hash check reduces false positives compared to a single hash.

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
        for i in range(n):
            for k in range(int(lims[i]), int(lims[i + 1])):
                j = int(I[k])
                if j > i:
                    p_dist = int(D[k])
                    d_dist = (d_ints[i] ^ d_ints[j]).bit_count()
                    if d_dist <= dhash_threshold:
                        conf = compute_confidence(
                            p_dist, d_dist, confidence_base, confidence_decay
                        )
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
                    conf = compute_confidence(
                        p_dist, d_dist, confidence_base, confidence_decay
                    )
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

