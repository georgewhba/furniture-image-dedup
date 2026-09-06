"""Shared test fixtures for the furniture_dedup test suite.

Generates small synthetic test images programmatically so unit tests
don't require real product photos.  All fixtures are created as temporary
files that are cleaned up after the test session.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def tmp_dir():
    """Provide a session-scoped temporary directory, cleaned up on exit."""
    d = Path(tempfile.mkdtemp(prefix="dedup_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def fixtures_dir(tmp_dir: Path) -> Path:
    """Create a directory of synthetic test images."""
    d = tmp_dir / "fixtures"
    d.mkdir()

    # --- Solid color images ---
    _make_solid(d / "red_100x100.png", (255, 0, 0), (100, 100))
    _make_solid(d / "red_100x100_copy.png", (255, 0, 0), (100, 100))  # identical
    _make_solid(d / "blue_100x100.png", (0, 0, 255), (100, 100))
    _make_solid(d / "green_100x100.png", (0, 200, 0), (100, 100))
    _make_solid(d / "beige_100x100.png", (220, 200, 170), (100, 100))
    _make_solid(d / "navy_100x100.png", (0, 0, 128), (100, 100))

    # --- Furniture-like shapes (rectangle on background) ---
    _make_furniture(
        d / "sofa_beige_white_bg.png",
        product_color=(220, 200, 170),
        bg_color=(255, 255, 255),
    )
    _make_furniture(
        d / "sofa_beige_dark_bg.png",
        product_color=(220, 200, 170),
        bg_color=(40, 40, 40),
    )
    _make_furniture(
        d / "sofa_navy_white_bg.png",
        product_color=(0, 0, 128),
        bg_color=(255, 255, 255),
    )
    _make_furniture(
        d / "sofa_navy_dark_bg.png",
        product_color=(0, 0, 128),
        bg_color=(40, 40, 40),
    )

    # --- Crop variant ---
    _make_furniture(
        d / "sofa_beige_cropped.png",
        product_color=(220, 200, 170),
        bg_color=(255, 255, 255),
        size=(80, 80),
        product_rect=(10, 10, 70, 50),
    )

    # --- Compressed variant (JPEG) ---
    original = Image.open(d / "sofa_beige_white_bg.png")
    original.save(d / "sofa_beige_compressed.jpg", "JPEG", quality=30)
    original.close()

    # --- Corrupted file ---
    (d / "corrupted.jpg").write_bytes(b"this is not a JPEG file at all")

    # --- Non-image with image extension ---
    (d / "text_as_png.png").write_text("This is a text file pretending to be a PNG")

    # --- Tiny image (below min pixel count for color extraction) ---
    _make_solid(d / "tiny_5x5.png", (128, 128, 128), (5, 5))

    return d


@pytest.fixture(scope="session")
def config_path(tmp_dir: Path) -> Path:
    """Create a minimal test config.yaml."""
    cfg = tmp_dir / "config.yaml"
    cfg.write_text(
        """\
hash_algorithm: sha256
phash_threshold: 10
dhash_threshold: 10
embedding_model: ViT-B-32
embedding_pretrained: laion2b_s34b_b79k
embedding_batch_size: 4
embedding_dimension: 512
similarity_threshold: 0.85
faiss_nlist: 10
faiss_nprobe: 5
faiss_brute_force_threshold: 50
faiss_use_gpu: false
similarity_k: 5
use_background_removal: false
color_n_clusters: 3
color_space: lab
color_distance_threshold: 25.0
color_min_pixel_count: 50
confidence_exact: 100
confidence_phash_base: 95
confidence_phash_decay: 2.0
confidence_embedding_base: 80
max_image_pixels: 89478485
supported_extensions:
  - .jpg
  - .jpeg
  - .png
  - .webp
random_seed: 42
log_level: WARNING
""",
        encoding="utf-8",
    )
    return cfg


# ---------------------------------------------------------------------------
# Image generation helpers
# ---------------------------------------------------------------------------


def _make_solid(
    path: Path,
    color: tuple[int, int, int],
    size: tuple[int, int] = (100, 100),
) -> None:
    """Create a solid-color image."""
    img = Image.new("RGB", size, color)
    img.save(path)
    img.close()


def _make_furniture(
    path: Path,
    product_color: tuple[int, int, int],
    bg_color: tuple[int, int, int],
    size: tuple[int, int] = (100, 100),
    product_rect: tuple[int, int, int, int] | None = None,
) -> None:
    """Create a simple furniture-like image (colored rectangle on background).

    The 'product' is a filled rectangle in *product_color* placed over
    a *bg_color* background.  This is a minimal proxy for a real product
    photo that exercises the color extraction pipeline.
    """
    img = Image.new("RGB", size, bg_color)
    pixels = np.array(img)

    if product_rect is None:
        # Default: centered rectangle taking ~60% of the image
        h, w = size[1], size[0]
        x1 = w // 5
        y1 = h // 5
        x2 = w - w // 5
        y2 = h - h // 5
    else:
        x1, y1, x2, y2 = product_rect

    pixels[y1:y2, x1:x2] = product_color
    Image.fromarray(pixels).save(path)
