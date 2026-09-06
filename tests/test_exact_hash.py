"""Tests for exact_hash module."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.exact_hash import (
    build_hash_map,
    compute_content_hash,
    compute_file_hash,
    find_exact_duplicates,
)


class TestComputeFileHash:
    """Tests for compute_file_hash()."""

    def test_identical_files_same_hash(self, fixtures_dir: Path) -> None:
        """Two files with identical content produce the same hash."""
        h1 = compute_file_hash(fixtures_dir / "red_100x100.png")
        h2 = compute_file_hash(fixtures_dir / "red_100x100_copy.png")
        assert h1 == h2

    def test_different_files_different_hash(self, fixtures_dir: Path) -> None:
        """Files with different content produce different hashes."""
        h1 = compute_file_hash(fixtures_dir / "red_100x100.png")
        h2 = compute_file_hash(fixtures_dir / "blue_100x100.png")
        assert h1 != h2

    def test_sha256_and_md5_differ(self, fixtures_dir: Path) -> None:
        """SHA-256 and MD5 produce different digests for the same file."""
        sha = compute_file_hash(fixtures_dir / "red_100x100.png", algorithm="sha256")
        md5 = compute_file_hash(fixtures_dir / "red_100x100.png", algorithm="md5")
        assert sha != md5
        assert len(sha) == 64  # SHA-256 hex length
        assert len(md5) == 32  # MD5 hex length

    def test_unsupported_algorithm_raises(self, fixtures_dir: Path) -> None:
        """An unsupported algorithm raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported hash algorithm"):
            compute_file_hash(fixtures_dir / "red_100x100.png", algorithm="sha512")

    def test_nonexistent_file_raises(self, tmp_dir: Path) -> None:
        """A missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            compute_file_hash(tmp_dir / "does_not_exist.png")

    def test_corrupted_file_still_hashes(self, fixtures_dir: Path) -> None:
        """A corrupted file can still be hashed (hash is over raw bytes)."""
        h = compute_file_hash(fixtures_dir / "corrupted.jpg")
        assert isinstance(h, str) and len(h) == 64


class TestComputeContentHash:
    """Tests for compute_content_hash()."""

    def test_returns_sha256(self, fixtures_dir: Path) -> None:
        """Content hash is always SHA-256."""
        h = compute_content_hash(fixtures_dir / "red_100x100.png")
        assert len(h) == 64


class TestBuildHashMap:
    """Tests for build_hash_map()."""

    def test_inversion(self) -> None:
        """Inverts {file: hash} to {hash: [files]}."""
        file_hashes = {"a.png": "abc", "b.png": "abc", "c.png": "def"}
        result = build_hash_map(file_hashes)
        assert sorted(result["abc"]) == ["a.png", "b.png"]
        assert result["def"] == ["c.png"]


class TestFindExactDuplicates:
    """Tests for find_exact_duplicates()."""

    def test_finds_duplicates(self) -> None:
        """Groups files with the same hash."""
        hash_map = {"abc": ["a.png", "b.png"], "def": ["c.png"]}
        groups = find_exact_duplicates(hash_map)
        assert len(groups) == 1
        assert sorted(groups[0]["members"]) == ["a.png", "b.png"]
        assert groups[0]["confidence"] == 100
        assert groups[0]["match_type"] == "exact"

    def test_no_duplicates(self) -> None:
        """Returns empty when every file has a unique hash."""
        hash_map = {"a": ["1.png"], "b": ["2.png"]}
        groups = find_exact_duplicates(hash_map)
        assert groups == []

    def test_multiple_groups(self) -> None:
        """Handles multiple duplicate groups."""
        hash_map = {
            "h1": ["a.png", "b.png"],
            "h2": ["c.png", "d.png", "e.png"],
            "h3": ["f.png"],
        }
        groups = find_exact_duplicates(hash_map)
        assert len(groups) == 2
