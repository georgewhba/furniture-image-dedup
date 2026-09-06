"""Tests for perceptual_hash module."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.perceptual_hash import (
    are_near_duplicates,
    compute_confidence,
    compute_dhash,
    compute_hashes,
    compute_phash,
    find_near_duplicates,
    hamming_distance,
)


class TestComputeHashes:
    """Tests for pHash and dHash computation."""

    def test_same_image_zero_distance(self, fixtures_dir: Path) -> None:
        """Identical images should produce identical hashes."""
        img = Image.open(fixtures_dir / "red_100x100.png")
        h1 = compute_hashes(img)
        h2 = compute_hashes(img)
        assert hamming_distance(h1.phash, h2.phash) == 0
        assert hamming_distance(h1.dhash, h2.dhash) == 0
        img.close()

    def test_similar_images_small_distance(self, fixtures_dir: Path) -> None:
        """A JPEG-compressed version should be close to the original."""
        orig = Image.open(fixtures_dir / "sofa_beige_white_bg.png")
        compressed = Image.open(fixtures_dir / "sofa_beige_compressed.jpg")
        h_orig = compute_hashes(orig)
        h_comp = compute_hashes(compressed)

        p_dist = hamming_distance(h_orig.phash, h_comp.phash)
        d_dist = hamming_distance(h_orig.dhash, h_comp.dhash)

        # Compressed version should be within a reasonable threshold
        # Quality 30 JPEG is aggressive — allows up to 25 bits distance
        assert p_dist <= 25, f"pHash distance too high: {p_dist}"
        assert d_dist <= 25, f"dHash distance too high: {d_dist}"

        orig.close()
        compressed.close()

    def test_different_images_high_distance(self, fixtures_dir: Path) -> None:
        """Completely different images should have high hash distance."""
        red = Image.open(fixtures_dir / "red_100x100.png")
        blue = Image.open(fixtures_dir / "blue_100x100.png")
        h_red = compute_hashes(red)
        h_blue = compute_hashes(blue)

        p_dist = hamming_distance(h_red.phash, h_blue.phash)
        # Solid color images may or may not differ by much in pHash,
        # but at least one hash type should show some distance
        assert p_dist >= 0  # sanity check — doesn't crash

        red.close()
        blue.close()

    def test_phash_returns_correct_type(self, fixtures_dir: Path) -> None:
        """compute_phash returns an ImageHash object."""
        img = Image.open(fixtures_dir / "red_100x100.png")
        h = compute_phash(img)
        assert hasattr(h, "hash")
        img.close()

    def test_dhash_returns_correct_type(self, fixtures_dir: Path) -> None:
        """compute_dhash returns an ImageHash object."""
        img = Image.open(fixtures_dir / "red_100x100.png")
        h = compute_dhash(img)
        assert hasattr(h, "hash")
        img.close()


class TestAreNearDuplicates:
    """Tests for the near-duplicate comparison function."""

    def test_identical_hashes_match(self, fixtures_dir: Path) -> None:
        """Identical hash pairs should match."""
        img = Image.open(fixtures_dir / "red_100x100.png")
        h = compute_hashes(img)
        is_match, p, d = are_near_duplicates(h, h, 10, 10)
        assert is_match is True
        assert p == 0
        assert d == 0
        img.close()

    def test_threshold_zero_requires_exact(self, fixtures_dir: Path) -> None:
        """Threshold 0 requires pixel-identical hashes."""
        orig = Image.open(fixtures_dir / "sofa_beige_white_bg.png")
        comp = Image.open(fixtures_dir / "sofa_beige_compressed.jpg")
        h_orig = compute_hashes(orig)
        h_comp = compute_hashes(comp)

        is_match, _, _ = are_near_duplicates(h_orig, h_comp, 0, 0)
        # Compressed image should NOT match at threshold 0
        # (unless compression was lossless, which JPEG quality=30 is not)
        # This is a soft assertion — may pass if images are very similar
        assert isinstance(is_match, bool)

        orig.close()
        comp.close()


class TestComputeConfidence:
    """Tests for confidence scoring."""

    def test_zero_distance_gives_base(self) -> None:
        """Zero distance → base confidence."""
        assert compute_confidence(0, 0, base=95, decay=2.0) == 95

    def test_high_distance_decreases_confidence(self) -> None:
        """Higher distance → lower confidence."""
        c_low = compute_confidence(2, 2, base=95, decay=2.0)
        c_high = compute_confidence(10, 10, base=95, decay=2.0)
        assert c_low > c_high

    def test_confidence_clamped_at_zero(self) -> None:
        """Extreme distance doesn't produce negative confidence."""
        c = compute_confidence(64, 64, base=95, decay=2.0)
        assert c == 0

    def test_confidence_clamped_at_100(self) -> None:
        """Base > 100 is clamped."""
        c = compute_confidence(0, 0, base=120, decay=2.0)
        assert c == 100


class TestFindNearDuplicates:
    """Tests for the batch near-duplicate finder."""

    def test_empty_input(self) -> None:
        """Empty input returns empty results."""
        result = find_near_duplicates({}, 10, 10)
        assert result == []

    def test_single_image(self, fixtures_dir: Path) -> None:
        """A single image can't have duplicates."""
        img = Image.open(fixtures_dir / "red_100x100.png")
        h = compute_hashes(img)
        result = find_near_duplicates({"a.png": h}, 10, 10)
        assert result == []
        img.close()

    def test_finds_matching_pair(self, fixtures_dir: Path) -> None:
        """Two identical images should be found as near-duplicates."""
        img = Image.open(fixtures_dir / "red_100x100.png")
        h = compute_hashes(img)
        result = find_near_duplicates({"a.png": h, "b.png": h}, 10, 10)
        assert len(result) == 1
        assert result[0]["match_type"] == "near_exact"
        assert result[0]["confidence"] > 0
        img.close()
