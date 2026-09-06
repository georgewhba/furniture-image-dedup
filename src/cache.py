"""SQLite-backed incremental cache for computed image features.

Stores exact hashes, perceptual hashes, CLIP embeddings, and dominant color
palettes keyed by ``(file_path, content_hash)``.  On re-runs, only new or
changed files are reprocessed — the rest are loaded from cache.

The cache database is stored in the user-specified ``--cache-dir``, never in
the source image folder.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS image_cache (
    file_path       TEXT    NOT NULL,
    content_hash    TEXT    NOT NULL,
    exact_hash      TEXT,
    phash           TEXT,
    dhash           TEXT,
    embedding       BLOB,
    dominant_colors TEXT,
    processed_at    TEXT    DEFAULT (datetime('now')),
    PRIMARY KEY (file_path, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_content_hash
    ON image_cache (content_hash);

CREATE INDEX IF NOT EXISTS idx_exact_hash
    ON image_cache (exact_hash);
"""


class ImageCache:
    """Persistent cache for per-image computed features.

    Args:
        cache_dir: Directory where the SQLite database is stored.
            Created automatically if it does not exist.
    """

    DB_NAME = "dedup_cache.db"

    def __init__(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = cache_dir / self.DB_NAME
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("Cache opened at %s", self._db_path)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has(self, file_path: str, content_hash: str) -> bool:
        """Check whether a file with the given content hash is cached.

        Args:
            file_path: Absolute path to the image file.
            content_hash: SHA-256 hex digest of the file contents.

        Returns:
            ``True`` if the cache contains a complete entry.
        """
        row = self._conn.execute(
            "SELECT 1 FROM image_cache WHERE file_path = ? AND content_hash = ?",
            (file_path, content_hash),
        ).fetchone()
        return row is not None

    def get(self, file_path: str, content_hash: str) -> dict[str, Any] | None:
        """Retrieve cached features for a single image.

        Args:
            file_path: Absolute path to the image file.
            content_hash: SHA-256 hex digest of the file contents.

        Returns:
            A dict with keys ``exact_hash``, ``phash``, ``dhash``,
            ``embedding`` (numpy array or None), ``dominant_colors``
            (list of tuples or None).  Returns ``None`` if not cached.
        """
        row = self._conn.execute(
            "SELECT exact_hash, phash, dhash, embedding, dominant_colors "
            "FROM image_cache WHERE file_path = ? AND content_hash = ?",
            (file_path, content_hash),
        ).fetchone()

        if row is None:
            return None

        exact_hash, phash, dhash, emb_blob, colors_json = row
        embedding = np.frombuffer(emb_blob, dtype=np.float32) if emb_blob else None
        dominant_colors = json.loads(colors_json) if colors_json else None

        return {
            "exact_hash": exact_hash,
            "phash": phash,
            "dhash": dhash,
            "embedding": embedding,
            "dominant_colors": dominant_colors,
        }

    def get_all_embeddings(self) -> list[tuple[str, np.ndarray]]:
        """Retrieve all cached embeddings for FAISS index building.

        Returns:
            List of ``(file_path, embedding)`` tuples for every cached
            image that has an embedding.
        """
        cursor = self._conn.execute(
            "SELECT file_path, embedding FROM image_cache WHERE embedding IS NOT NULL"
        )
        results: list[tuple[str, np.ndarray]] = []
        for file_path, emb_blob in cursor:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            results.append((file_path, emb))
        return results

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def put(
        self,
        file_path: str,
        content_hash: str,
        *,
        exact_hash: str | None = None,
        phash: str | None = None,
        dhash: str | None = None,
        embedding: np.ndarray | None = None,
        dominant_colors: list[tuple[float, ...]] | None = None,
    ) -> None:
        """Insert or replace the cache entry for a single image.

        Args:
            file_path: Absolute path to the image file.
            content_hash: SHA-256 hex digest of the file contents.
            exact_hash: Hex digest of the exact-hash stage.
            phash: Hex string of the perceptual hash.
            dhash: Hex string of the difference hash.
            embedding: Normalized CLIP embedding vector.
            dominant_colors: List of ``(L, a, b, weight)`` tuples.
        """
        emb_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        colors_json = json.dumps(dominant_colors) if dominant_colors is not None else None

        self._conn.execute(
            "INSERT OR REPLACE INTO image_cache "
            "(file_path, content_hash, exact_hash, phash, dhash, embedding, dominant_colors) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (file_path, content_hash, exact_hash, phash, dhash, emb_blob, colors_json),
        )

    def update_field(
        self,
        file_path: str,
        content_hash: str,
        **fields: Any,
    ) -> None:
        """Update specific fields on an existing cache entry.

        Args:
            file_path: Absolute path to the image file.
            content_hash: SHA-256 hex digest of the file contents.
            **fields: Column name → new value pairs to update.

        Raises:
            ValueError: If an unknown field name is provided.
        """
        allowed = {"exact_hash", "phash", "dhash", "embedding", "dominant_colors"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"Unknown cache fields: {bad}")

        # Serialize special types
        if "embedding" in fields and fields["embedding"] is not None:
            fields["embedding"] = fields["embedding"].astype(np.float32).tobytes()
        if "dominant_colors" in fields and fields["dominant_colors"] is not None:
            fields["dominant_colors"] = json.dumps(fields["dominant_colors"])

        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [file_path, content_hash]
        self._conn.execute(
            f"UPDATE image_cache SET {set_clause} "  # noqa: S608
            "WHERE file_path = ? AND content_hash = ?",
            values,
        )

    def flush(self) -> None:
        """Commit any pending writes to disk.

        Call periodically during long runs for crash resilience.
        """
        self._conn.commit()

    def close(self) -> None:
        """Commit and close the database connection."""
        self._conn.commit()
        self._conn.close()
        logger.info("Cache closed")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of cached entries."""
        row = self._conn.execute("SELECT COUNT(*) FROM image_cache").fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ImageCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
