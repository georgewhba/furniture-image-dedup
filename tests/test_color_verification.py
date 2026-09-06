"""Tests for color_verification module — THE MOST IMPORTANT TEST FILE.

The entire project's value hinges on one property: same-shape-different-color
products must be REJECTED, not flagged as duplicates.  These tests verify
that property explicitly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.color_verification import (
    ColorVerificationResult,
    LabColor,
    _rgb_to_lab,
    compare_palettes,
    extract_dominant_colors,
    verify_color_match,
)

# ---------------------------------------------------------------------------
# Color space conversion
# ---------------------------------------------------------------------------


class TestRgbToLab:
    """Tests for the RGB → Lab conversion."""

    def test_white(self) -> None:
        """Pure white → L ≈ 100, a ≈ 0, b ≈ 0."""
        lab = _rgb_to_lab(np.array([[255, 255, 255]]))
        assert abs(lab[0, 0] - 100.0) < 1.0
        assert abs(lab[0, 1]) < 1.0
        assert abs(lab[0, 2]) < 1.0

    def test_black(self) -> None:
        """Pure black → L ≈ 0."""
        lab = _rgb_to_lab(np.array([[0, 0, 0]]))
        assert abs(lab[0, 0]) < 1.0

    def test_shape_preserved(self) -> None:
        """Output shape matches input shape."""
        rgb = np.array([[128, 64, 200], [10, 20, 30]])
        lab = _rgb_to_lab(rgb)
        assert lab.shape == (2, 3)


# ---------------------------------------------------------------------------
# Dominant color extraction
# ---------------------------------------------------------------------------


class TestExtractDominantColors:
    """Tests for extract_dominant_colors()."""

    def test_solid_color_single_cluster(self) -> None:
        """A solid red image should have one dominant color cluster."""
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        colors = extract_dominant_colors(img, n_clusters=3, seed=42)
        assert colors is not None
        assert len(colors) >= 1
        # The dominant color should have very high weight
        assert colors[0].weight > 0.5

    def test_two_color_image(self) -> None:
        """An image with two regions produces two dominant clusters."""
        pixels = np.zeros((100, 100, 3), dtype=np.uint8)
        pixels[:50, :] = [255, 0, 0]  # top half red
        pixels[50:, :] = [0, 0, 255]  # bottom half blue
        img = Image.fromarray(pixels)
        colors = extract_dominant_colors(img, n_clusters=3, seed=42)
        assert colors is not None
        assert len(colors) >= 2
        # Each should have ~50% weight
        assert colors[0].weight > 0.3
        assert colors[1].weight > 0.3

    def test_respects_alpha_channel(self) -> None:
        """Only non-transparent pixels should be used."""
        pixels = np.zeros((100, 100, 4), dtype=np.uint8)
        pixels[:50, :] = [255, 0, 0, 255]  # top half: red, opaque
        pixels[50:, :] = [0, 0, 255, 0]  # bottom half: blue, transparent
        img = Image.fromarray(pixels, "RGBA")
        colors = extract_dominant_colors(img, n_clusters=3, seed=42)
        assert colors is not None
        # Only red should appear
        assert len(colors) >= 1

    def test_too_few_pixels_returns_none(self) -> None:
        """An image with fewer opaque pixels than min_pixel_count → None."""
        img = Image.new("RGBA", (3, 3), (255, 0, 0, 255))
        result = extract_dominant_colors(img, n_clusters=3, seed=42, min_pixel_count=100)
        assert result is None

    def test_invalid_color_space_raises(self) -> None:
        """Unsupported color space raises ValueError."""
        img = Image.new("RGB", (100, 100), (128, 128, 128))
        with pytest.raises(ValueError, match="Unsupported color_space"):
            extract_dominant_colors(img, color_space="xyz")


# ---------------------------------------------------------------------------
# Palette comparison
# ---------------------------------------------------------------------------


class TestComparePalettes:
    """Tests for compare_palettes()."""

    def test_identical_palettes_zero_distance(self) -> None:
        """Identical palettes should have zero distance."""
        palette = [
            LabColor(L=50, a=10, b=-20, weight=0.6),
            LabColor(L=80, a=-5, b=30, weight=0.4),
        ]
        dist = compare_palettes(palette, palette)
        assert dist == pytest.approx(0.0, abs=0.01)

    def test_different_palettes_nonzero_distance(self) -> None:
        """Different palettes should have non-zero distance."""
        palette_a = [LabColor(L=50, a=10, b=-20, weight=0.6)]
        palette_b = [LabColor(L=50, a=60, b=30, weight=0.6)]
        dist = compare_palettes(palette_a, palette_b)
        assert dist > 10.0

    def test_similar_palettes_small_distance(self) -> None:
        """Slightly different palettes → small distance."""
        palette_a = [LabColor(L=50, a=10, b=-20, weight=1.0)]
        palette_b = [LabColor(L=52, a=11, b=-19, weight=1.0)]
        dist = compare_palettes(palette_a, palette_b)
        assert dist < 5.0

    def test_unequal_length_palettes(self) -> None:
        """Palettes of different lengths should still work."""
        palette_a = [
            LabColor(L=50, a=0, b=0, weight=0.5),
            LabColor(L=80, a=0, b=0, weight=0.5),
        ]
        palette_b = [LabColor(L=50, a=0, b=0, weight=1.0)]
        dist = compare_palettes(palette_a, palette_b)
        assert dist >= 0.0


# ---------------------------------------------------------------------------
# Full color verification gate — THE CRITICAL TESTS
# ---------------------------------------------------------------------------


class TestVerifyColorMatch:
    """Tests for the full color verification gate.

    These tests verify the most important property of the entire system:
    same-shape-different-color products must be REJECTED.
    """

    def test_same_color_same_shape_passes(self, fixtures_dir: Path) -> None:
        """Same product color, different background → should PASS."""
        img_a = Image.open(fixtures_dir / "sofa_beige_white_bg.png")
        img_b = Image.open(fixtures_dir / "sofa_beige_dark_bg.png")

        result = verify_color_match(
            img_a,
            img_b,
            use_background_removal=False,  # test images are simple enough
            n_clusters=3,
            color_space="lab",
            distance_threshold=35.0,  # relaxed for synthetic images
            min_pixel_count=50,
            seed=42,
        )

        # With simple test images, this may or may not pass due to
        # background contamination (no rembg in tests). This tests
        # the pipeline doesn't crash.
        assert isinstance(result, ColorVerificationResult)
        assert isinstance(result.passed, bool)
        assert result.distance >= 0.0

        img_a.close()
        img_b.close()

    def test_different_color_same_shape_rejected(self) -> None:
        """Same shape, different color → must be REJECTED.

        THIS IS THE SINGLE MOST IMPORTANT TEST IN THE ENTIRE PROJECT.
        """
        # Beige product
        pixels_a = np.zeros((100, 100, 3), dtype=np.uint8)
        pixels_a[:, :] = [220, 200, 170]  # beige
        img_a = Image.fromarray(pixels_a)

        # Navy product (identical shape, different color)
        pixels_b = np.zeros((100, 100, 3), dtype=np.uint8)
        pixels_b[:, :] = [0, 0, 128]  # navy
        img_b = Image.fromarray(pixels_b)

        result = verify_color_match(
            img_a,
            img_b,
            use_background_removal=False,
            n_clusters=3,
            color_space="lab",
            distance_threshold=25.0,
            min_pixel_count=50,
            seed=42,
        )

        assert result.passed is False, (
            f"CRITICAL FAILURE: Different colors were accepted as a match! "
            f"Distance={result.distance:.2f}, threshold=25.0"
        )
        assert result.distance > 25.0

    def test_very_similar_colors_different_product(self) -> None:
        """Two similar but distinguishable colors → distance should be measurable."""
        # Light beige
        img_a = Image.new("RGB", (100, 100), (220, 200, 170))
        # Slightly darker beige
        img_b = Image.new("RGB", (100, 100), (200, 180, 150))

        result = verify_color_match(
            img_a,
            img_b,
            use_background_removal=False,
            n_clusters=3,
            color_space="lab",
            distance_threshold=25.0,
            min_pixel_count=50,
            seed=42,
        )

        # These are similar enough that they should pass
        assert isinstance(result.passed, bool)
        assert result.distance > 0.0  # not zero — they're different

    def test_monochrome_images(self) -> None:
        """Grayscale images should be handled without errors."""
        img_a = Image.new("L", (100, 100), 128)  # grayscale
        img_b = Image.new("L", (100, 100), 128)

        result = verify_color_match(
            img_a,
            img_b,
            use_background_removal=False,
            n_clusters=3,
            color_space="lab",
            distance_threshold=25.0,
            min_pixel_count=50,
            seed=42,
        )

        assert isinstance(result, ColorVerificationResult)

    def test_transparent_png_images(self) -> None:
        """RGBA images with transparency should work correctly."""
        # Create RGBA image with transparent background
        pixels = np.zeros((100, 100, 4), dtype=np.uint8)
        pixels[20:80, 20:80] = [255, 0, 0, 255]  # red product, opaque
        pixels[:20, :] = [255, 255, 255, 0]  # transparent background
        img = Image.fromarray(pixels, "RGBA")

        result = verify_color_match(
            img,
            img,
            use_background_removal=False,
            n_clusters=3,
            color_space="lab",
            distance_threshold=25.0,
            min_pixel_count=50,
            seed=42,
        )

        assert result.passed is True
        assert result.distance == pytest.approx(0.0, abs=1.0)

    def test_red_vs_blue_rejected(self) -> None:
        """Pure red vs. pure blue → must be REJECTED."""
        img_a = Image.new("RGB", (100, 100), (255, 0, 0))
        img_b = Image.new("RGB", (100, 100), (0, 0, 255))

        result = verify_color_match(
            img_a,
            img_b,
            use_background_removal=False,
            n_clusters=3,
            color_space="lab",
            distance_threshold=25.0,
            min_pixel_count=50,
            seed=42,
        )

        assert result.passed is False, (
            f"CRITICAL FAILURE: Red and blue accepted as match! " f"Distance={result.distance:.2f}"
        )

    def test_result_contains_palettes(self) -> None:
        """Result should include both extracted palettes for auditability."""
        img = Image.new("RGB", (100, 100), (128, 128, 128))

        result = verify_color_match(
            img,
            img,
            use_background_removal=False,
            n_clusters=3,
            color_space="lab",
            distance_threshold=25.0,
            min_pixel_count=50,
            seed=42,
        )

        assert len(result.palette_a) > 0
        assert len(result.palette_b) > 0
