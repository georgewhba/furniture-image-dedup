"""Configuration loading and validation.

Reads ``config.yaml``, validates every field, and returns a frozen dataclass
so the rest of the pipeline can rely on typed, validated values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    """Immutable, validated pipeline configuration.

    All thresholds and parameters are loaded from ``config.yaml``.
    Default values here match the defaults documented in that file.
    """

    # Stage 1 — Exact hash
    hash_algorithm: str = "sha256"

    # Stage 2 — Perceptual hash
    phash_threshold: int = 10
    dhash_threshold: int = 10

    # Stage 3 — Embeddings
    embedding_model: str = "ViT-B-32"
    embedding_pretrained: str = "laion2b_s34b_b79k"
    embedding_batch_size: int = 32
    embedding_dimension: int = 512
    similarity_threshold: float = 0.85
    faiss_nlist: int = 100
    faiss_nprobe: int = 10
    faiss_use_gpu: bool = False
    faiss_brute_force_threshold: int = 1000
    similarity_k: int = 10

    # Stage 4 — Color verification
    use_background_removal: bool = True
    color_n_clusters: int = 5
    color_space: str = "lab"
    color_distance_threshold: float = 25.0
    color_min_pixel_count: int = 100

    # Stage 5 — Confidence
    confidence_exact: int = 100
    confidence_phash_base: int = 95
    confidence_phash_decay: float = 2.0
    confidence_embedding_base: int = 80

    # Safety
    max_image_pixels: int = 89_478_485
    supported_extensions: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
    )

    # Reproducibility
    random_seed: int = 42

    # Logging
    log_level: str = "INFO"
    log_file: str | None = None


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

_VALID_HASH_ALGORITHMS = {"sha256", "md5"}
_VALID_COLOR_SPACES = {"lab", "hsv"}
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _validate(cfg: Config) -> list[str]:
    """Return a list of validation error messages (empty = valid)."""
    errors: list[str] = []

    if cfg.hash_algorithm not in _VALID_HASH_ALGORITHMS:
        errors.append(
            f"hash_algorithm must be one of {_VALID_HASH_ALGORITHMS}, "
            f"got '{cfg.hash_algorithm}'"
        )

    if not 0 <= cfg.phash_threshold <= 64:
        errors.append(f"phash_threshold must be 0–64, got {cfg.phash_threshold}")

    if not 0 <= cfg.dhash_threshold <= 64:
        errors.append(f"dhash_threshold must be 0–64, got {cfg.dhash_threshold}")

    if cfg.embedding_batch_size < 1:
        errors.append(
            f"embedding_batch_size must be ≥ 1, got {cfg.embedding_batch_size}"
        )

    if not 0.0 <= cfg.similarity_threshold <= 1.0:
        errors.append(
            f"similarity_threshold must be 0.0–1.0, got {cfg.similarity_threshold}"
        )

    if cfg.faiss_nlist < 1:
        errors.append(f"faiss_nlist must be ≥ 1, got {cfg.faiss_nlist}")

    if cfg.faiss_nprobe < 1:
        errors.append(f"faiss_nprobe must be ≥ 1, got {cfg.faiss_nprobe}")

    if cfg.color_n_clusters < 2:
        errors.append(f"color_n_clusters must be ≥ 2, got {cfg.color_n_clusters}")

    if cfg.color_space not in _VALID_COLOR_SPACES:
        errors.append(
            f"color_space must be one of {_VALID_COLOR_SPACES}, "
            f"got '{cfg.color_space}'"
        )

    if cfg.color_distance_threshold < 0:
        errors.append(
            f"color_distance_threshold must be ≥ 0, "
            f"got {cfg.color_distance_threshold}"
        )

    if cfg.max_image_pixels < 1:
        errors.append(f"max_image_pixels must be ≥ 1, got {cfg.max_image_pixels}")

    if cfg.log_level.upper() not in _VALID_LOG_LEVELS:
        errors.append(
            f"log_level must be one of {_VALID_LOG_LEVELS}, got '{cfg.log_level}'"
        )

    return errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(path: Path) -> Config:
    """Load and validate configuration from a YAML file.

    Args:
        path: Path to the ``config.yaml`` file.

    Returns:
        A validated, frozen :class:`Config` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If any configuration value is invalid.
        yaml.YAMLError: If the YAML is malformed.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    # Normalise supported_extensions from list → tuple
    exts = raw.get("supported_extensions")
    if isinstance(exts, list):
        raw["supported_extensions"] = tuple(
            e if e.startswith(".") else f".{e}" for e in exts
        )

    # Build the dataclass, ignoring unknown keys gracefully
    known_fields = {f.name for f in Config.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in known_fields}

    unknown = set(raw) - known_fields
    if unknown:
        logger.warning("Unknown config keys ignored: %s", unknown)

    cfg = Config(**filtered)

    errors = _validate(cfg)
    if errors:
        raise ValueError(
            "Invalid configuration:\n" + "\n".join(f"  • {e}" for e in errors)
        )

    logger.info("Configuration loaded from %s", path)
    return cfg


def setup_logging(cfg: Config) -> None:
    """Configure the root logger based on the loaded config.

    Args:
        cfg: Validated pipeline configuration.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if cfg.log_file:
        handlers.append(logging.FileHandler(cfg.log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
