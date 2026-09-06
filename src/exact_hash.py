"""Stage 1 — Exact (cryptographic) hash duplicate detection.

Computes a cryptographic hash over raw file bytes.  Two files sharing the
same digest are byte-identical — confidence 100 %, match type ``exact``.
This is the cheapest and fastest stage and runs first to shrink everything
downstream.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# Read files in 64 KiB chunks to avoid loading multi-GB images into memory.
_CHUNK_SIZE = 65_536


def compute_file_hash(path: Path, algorithm: str = "sha256") -> str:
    """Compute a cryptographic hash of a file's raw bytes.

    Args:
        path: Path to the file.
        algorithm: Hash algorithm name (``"sha256"`` or ``"md5"``).

    Returns:
        Hex digest string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If *algorithm* is not supported.
        OSError: If the file cannot be read.
    """
    if algorithm not in ("sha256", "md5"):
        raise ValueError(f"Unsupported hash algorithm: {algorithm!r}")

    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_content_hash(path: Path) -> str:
    """Compute a SHA-256 content hash for cache-keying purposes.

    This is always SHA-256 regardless of the user-configured hash algorithm,
    because the cache key must be stable across config changes.

    Args:
        path: Path to the file.

    Returns:
        SHA-256 hex digest.
    """
    return compute_file_hash(path, algorithm="sha256")


def find_exact_duplicates(
    hash_map: dict[str, list[str]],
) -> list[dict]:
    """Group files that share the same exact hash.

    Args:
        hash_map: Mapping of ``{hash_digest: [file_path, ...]}`` where
            each list has at least one entry.

    Returns:
        A list of group dicts, each containing:

        - ``members``: list of file paths in the group
        - ``confidence``: always ``100``
        - ``match_type``: always ``"exact"``

        Only groups with 2+ members are returned.
    """
    groups: list[dict] = []
    for digest, members in hash_map.items():
        if len(members) >= 2:
            groups.append(
                {
                    "members": sorted(members),
                    "confidence": 100,
                    "match_type": "exact",
                }
            )
            logger.debug(
                "Exact group (%s): %d members — %s",
                digest[:12],
                len(members),
                members[0],
            )

    logger.info(
        "Stage 1 complete: %d exact-duplicate groups found (%d images)",
        len(groups),
        sum(len(g["members"]) for g in groups),
    )
    return groups


def build_hash_map(
    file_hashes: dict[str, str],
) -> dict[str, list[str]]:
    """Invert a ``{file_path: hash}`` mapping to ``{hash: [file_paths]}``.

    Args:
        file_hashes: Mapping from file path to its hash digest.

    Returns:
        Inverted mapping grouped by hash.
    """
    result: dict[str, list[str]] = defaultdict(list)
    for fpath, digest in file_hashes.items():
        result[digest].append(fpath)
    return dict(result)
