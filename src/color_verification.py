"""Stage 4 — Color verification (MANDATORY GATE).

This is the most critical module in the entire system.  Every candidate
pair produced by the embedding similarity stage (Stage 3) **must** pass
through this gate before being reported as a duplicate.  Visual embedding
similarity alone cannot distinguish "same product, different color" from
"same product, same color" — this module makes that distinction.

Pipeline per candidate pair
---------------------------
1. (Optional, recommended) Remove background with ``rembg`` to isolate
   the product from its background.
2. Convert pixel colors to CIE Lab space (perceptually uniform,
   illumination-robust — unlike RGB, which distorts under lighting shifts).
3. Run seeded K-means on the pixel colors to extract the dominant color
   palette as a list of ``(L, a, b, weight)`` tuples.
4. Match palettes between the two images using the Hungarian algorithm
   for optimal centroid assignment.
5. Compute weighted Euclidean distance in Lab space.
6. If the distance exceeds the configured threshold → **reject the match**
   and log both the embedding similarity score and the color distance
   for auditability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabColor:
    """A single color in CIE Lab space with a cluster weight."""

    L: float  # noqa: N815 — conventional name
    a: float
    b: float
    weight: float

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Return ``(L, a, b, weight)`` for serialization."""
        return (self.L, self.a, self.b, self.weight)

    @classmethod
    def from_tuple(cls, t: tuple[float, ...]) -> LabColor:
        """Construct from a ``(L, a, b, weight)`` tuple."""
        return cls(L=t[0], a=t[1], b=t[2], weight=t[3])


@dataclass(frozen=True)
class ColorVerificationResult:
    """Outcome of color verification on one candidate pair."""

    passed: bool
    distance: float
    palette_a: list[LabColor]
    palette_b: list[LabColor]
    reason: str = ""


# ---------------------------------------------------------------------------
# Background removal
# ---------------------------------------------------------------------------


_rembg_session = None


def _get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session

        _rembg_session = new_session("u2netp")
    return _rembg_session


def remove_background(image: Image.Image) -> Image.Image:
    """Remove the background from a product image using ``rembg``.

    Returns an RGBA image where transparent pixels correspond to the
    removed background.

    Args:
        image: Input PIL Image (any mode).

    Returns:
        RGBA image with the background made transparent.

    Raises:
        ImportError: If ``rembg`` is not installed.
    """
    from rembg import remove  # lazy import — heavy dependency

    session = _get_rembg_session()
    w, h = image.size
    if max(w, h) > 512:
        scale = 512.0 / max(w, h)
        proc_img = image.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    else:
        proc_img = image

    rgba = remove(proc_img, session=session)
    if not isinstance(rgba, Image.Image):
        # rembg can return bytes; convert back
        from io import BytesIO

        rgba = Image.open(BytesIO(rgba))
    return rgba.convert("RGBA")


# ---------------------------------------------------------------------------
# Color space conversion
# ---------------------------------------------------------------------------


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an (N, 3) array of sRGB [0..255] values to CIE Lab.

    Uses the D65 illuminant via the standard sRGB → XYZ → Lab path.

    Args:
        rgb: Shape ``(N, 3)`` uint8 or float array with values in [0, 255].

    Returns:
        Shape ``(N, 3)`` float64 array in Lab space.
    """
    # Normalize to [0, 1]
    rgb_norm = rgb.astype(np.float64) / 255.0

    # sRGB linearization
    mask = rgb_norm > 0.04045
    rgb_lin = np.where(mask, ((rgb_norm + 0.055) / 1.055) ** 2.4, rgb_norm / 12.92)

    # sRGB → XYZ (D65)
    # fmt: off
    srgb_to_xyz = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    # fmt: on
    xyz = rgb_lin @ srgb_to_xyz.T

    # XYZ → Lab (D65 reference white)
    ref = np.array([0.95047, 1.00000, 1.08883])
    xyz_norm = xyz / ref

    epsilon = 0.008856
    kappa = 903.3
    mask_lab = xyz_norm > epsilon
    f = np.where(mask_lab, np.cbrt(xyz_norm), (kappa * xyz_norm + 16.0) / 116.0)

    l_star = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b_ch = 200.0 * (f[:, 1] - f[:, 2])

    return np.column_stack([l_star, a, b_ch])


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert an (N, 3) array of sRGB [0..255] values to HSV.

    Args:
        rgb: Shape ``(N, 3)`` array with values in [0, 255].

    Returns:
        Shape ``(N, 3)`` array with H in [0, 360), S and V in [0, 1].
    """
    rgb_norm = rgb.astype(np.float64) / 255.0
    r, g, b = rgb_norm[:, 0], rgb_norm[:, 1], rgb_norm[:, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue
    h = np.zeros_like(delta)
    mask_r = (delta > 0) & (cmax == r)
    mask_g = (delta > 0) & (cmax == g)
    mask_b = (delta > 0) & (cmax == b)
    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60.0 * ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2)
    h[mask_b] = 60.0 * ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4)

    # Saturation
    s = np.where(cmax > 0, delta / cmax, 0.0)

    return np.column_stack([h, s, cmax])


# ---------------------------------------------------------------------------
# Dominant color extraction
# ---------------------------------------------------------------------------


def extract_dominant_colors(
    image: Image.Image,
    n_clusters: int = 5,
    seed: int = 42,
    color_space: str = "lab",
    min_pixel_count: int = 100,
) -> list[LabColor] | None:
    """Extract the dominant color palette from an image.

    If the image has an alpha channel, only non-transparent pixels
    (alpha > 128) are used — this allows background-removed images to
    yield only the product's colors.

    Args:
        image: PIL Image in any mode.
        n_clusters: Number of K-means clusters.
        seed: Random seed for reproducibility.
        color_space: ``"lab"`` or ``"hsv"``.
        min_pixel_count: Minimum opaque pixels required. Returns ``None``
            if the image has fewer.

    Returns:
        A list of :class:`LabColor` (or HSV-equivalent) sorted by weight
        descending.  Returns ``None`` if insufficient pixels are available.
    """
    img = image.convert("RGBA")
    pixels = np.array(img)

    # Filter to non-transparent pixels
    alpha = pixels[:, :, 3]
    opaque_mask = alpha > 128
    rgb_pixels = pixels[:, :, :3][opaque_mask]

    if len(rgb_pixels) < min_pixel_count:
        logger.debug(
            "Too few opaque pixels (%d < %d), skipping color extraction",
            len(rgb_pixels),
            min_pixel_count,
        )
        return None

    # Convert to target color space
    if color_space == "lab":
        converted = _rgb_to_lab(rgb_pixels)
    elif color_space == "hsv":
        converted = _rgb_to_hsv(rgb_pixels)
    else:
        raise ValueError(f"Unsupported color_space: {color_space!r}")

    # K-means clustering
    actual_k = min(n_clusters, len(converted))
    kmeans = KMeans(n_clusters=actual_k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(converted)
    centroids = kmeans.cluster_centers_

    # Compute cluster weights (fraction of total pixels)
    total = len(labels)
    colors: list[LabColor] = []
    for i in range(actual_k):
        count = int(np.sum(labels == i))
        weight = count / total
        c = centroids[i]
        colors.append(LabColor(L=float(c[0]), a=float(c[1]), b=float(c[2]), weight=weight))

    # Sort by weight descending (most dominant first)
    colors.sort(key=lambda c: c.weight, reverse=True)
    return colors


# ---------------------------------------------------------------------------
# Palette comparison
# ---------------------------------------------------------------------------


def _lab_euclidean(c1: LabColor, c2: LabColor) -> float:
    """Euclidean distance between two colors in Lab space."""
    return float(
        np.sqrt((c1.L - c2.L) ** 2 + (c1.a - c2.a) ** 2 + (c1.b - c2.b) ** 2)
    )


def compare_palettes(
    palette_a: list[LabColor],
    palette_b: list[LabColor],
) -> float:
    """Compare two color palettes using optimal centroid matching.

    Uses the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``)
    to find the optimal 1-to-1 matching between clusters, then computes a
    weight-averaged distance.

    Args:
        palette_a: Dominant palette of image A.
        palette_b: Dominant palette of image B.

    Returns:
        Weighted average Lab-space distance.  Lower = more similar.
    """
    n_a = len(palette_a)
    n_b = len(palette_b)
    n = max(n_a, n_b)

    # Build cost matrix — pad the shorter palette with high-cost dummies
    cost = np.full((n, n), fill_value=200.0)  # 200 is ~max Lab distance
    for i in range(n_a):
        for j in range(n_b):
            cost[i, j] = _lab_euclidean(palette_a[i], palette_b[j])

    row_idx, col_idx = linear_sum_assignment(cost)

    # Weight-averaged distance using the average of matched cluster weights
    total_weight = 0.0
    weighted_dist = 0.0
    for r, c in zip(row_idx, col_idx, strict=True):
        w_a = palette_a[r].weight if r < n_a else 0.0
        w_b = palette_b[c].weight if c < n_b else 0.0
        w = (w_a + w_b) / 2.0
        weighted_dist += cost[r, c] * w
        total_weight += w

    if total_weight > 0:
        return weighted_dist / total_weight
    return float(cost[row_idx, col_idx].mean())


# ---------------------------------------------------------------------------
# Public API — the mandatory gate
# ---------------------------------------------------------------------------


_image_palette_cache: dict[tuple, list[LabColor] | None] = {}


def _get_cached_palette(
    image: Image.Image,
    path: str,
    use_background_removal: bool,
    n_clusters: int,
    seed: int,
    color_space: str,
    min_pixel_count: int,
) -> list[LabColor] | None:
    cache_key = (path, use_background_removal, n_clusters, color_space) if path else None
    if cache_key and cache_key in _image_palette_cache:
        return _image_palette_cache[cache_key]

    img = image
    if use_background_removal:
        try:
            bg_img = remove_background(img)
            # Check if background removal left enough opaque pixels
            alpha = np.array(bg_img.convert("RGBA"))[:, :, 3]
            if np.sum(alpha > 128) >= min_pixel_count:
                img = bg_img
            else:
                logger.debug("Background removal left too few pixels for %s, falling back to original", path)
        except Exception as exc:
            logger.warning("Background removal failed for %s: %s", path, exc)

    palette = extract_dominant_colors(
        img,
        n_clusters=n_clusters,
        seed=seed,
        color_space=color_space,
        min_pixel_count=min_pixel_count,
    )
    if palette is None and use_background_removal:
        # Fallback to extracting from original image without background removal
        palette = extract_dominant_colors(
            image.convert("RGB"),
            n_clusters=n_clusters,
            seed=seed,
            color_space=color_space,
            min_pixel_count=10,
        )

    if cache_key:
        _image_palette_cache[cache_key] = palette
    return palette


def verify_color_match(
    image_a: Image.Image,
    image_b: Image.Image,
    *,
    use_background_removal: bool = True,
    n_clusters: int = 5,
    color_space: str = "lab",
    distance_threshold: float = 25.0,
    min_pixel_count: int = 100,
    seed: int = 42,
    embedding_similarity: float | None = None,
    path_a: str = "",
    path_b: str = "",
) -> ColorVerificationResult:
    """Run the full color-verification gate on a candidate pair.

    This function is the mandatory gate for every Stage 3 candidate.
    It must be called on every embedding-based match — no exceptions.

    Args:
        image_a: First image.
        image_b: Second image.
        use_background_removal: Whether to isolate the product first.
        n_clusters: K-means clusters for palette extraction.
        color_space: ``"lab"`` or ``"hsv"``.
        distance_threshold: Max distance for a color match.
        min_pixel_count: Min opaque pixels after background removal.
        seed: Random seed for K-means.
        embedding_similarity: Embedding cosine similarity (for logging).
        path_a: File path of image A (for logging).
        path_b: File path of image B (for logging).

    Returns:
        A :class:`ColorVerificationResult` indicating pass/fail, the
        computed distance, and both palettes.
    """
    palette_a = _get_cached_palette(
        image_a,
        path_a,
        use_background_removal,
        n_clusters,
        seed,
        color_space,
        min_pixel_count,
    )
    palette_b = _get_cached_palette(
        image_b,
        path_b,
        use_background_removal,
        n_clusters,
        seed,
        color_space,
        min_pixel_count,
    )

    # Edge case: not enough pixels
    if palette_a is None or palette_b is None:
        reason = (
            f"Insufficient opaque pixels (A={'none' if palette_a is None else 'ok'}, "
            f"B={'none' if palette_b is None else 'ok'})"
        )
        logger.info(
            "Color gate REJECTED %s ↔ %s: %s (emb_sim=%.3f)",
            path_a,
            path_b,
            reason,
            embedding_similarity or 0.0,
        )
        return ColorVerificationResult(
            passed=False,
            distance=float("inf"),
            palette_a=palette_a or [],
            palette_b=palette_b or [],
            reason=reason,
        )

    # Step 4–5: Compare palettes
    distance = float(compare_palettes(palette_a, palette_b))

    # Step 6: Gate decision
    passed = bool(distance <= distance_threshold)

    if passed:
        logger.debug(
            "Color gate PASSED %s ↔ %s: distance=%.2f (threshold=%.2f, emb_sim=%.3f)",
            path_a,
            path_b,
            distance,
            distance_threshold,
            embedding_similarity or 0.0,
        )
    else:
        logger.info(
            "Color gate REJECTED %s ↔ %s: distance=%.2f > threshold=%.2f "
            "(emb_sim=%.3f) — likely same shape, different color",
            path_a,
            path_b,
            distance,
            distance_threshold,
            embedding_similarity or 0.0,
        )

    return ColorVerificationResult(
        passed=bool(passed),
        distance=float(distance),
        palette_a=palette_a,
        palette_b=palette_b,
        reason=(
            ""
            if passed
            else f"Color distance {distance:.2f} > threshold {distance_threshold:.2f}"
        ),
    )
