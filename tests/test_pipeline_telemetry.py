"""Lightweight integration tests for pipeline telemetry and Color Gate tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.config import Config
from src.pipeline import run as run_pipeline


def _draw_chair(path: Path, body_rgb: tuple[int, int, int], accent_dot: bool = False) -> None:
    """Create a synthetic chair silhouette on white background."""
    img = Image.new("RGB", (128, 128), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Seat + backrest + legs silhouette
    draw.rectangle([28, 24, 100, 66], fill=body_rgb)
    draw.rectangle([24, 66, 104, 88], fill=body_rgb)
    draw.rectangle([28, 88, 38, 114], fill=(60, 40, 20))
    draw.rectangle([90, 88, 100, 114], fill=(60, 40, 20))
    if accent_dot:
        draw.rectangle([2, 2, 6, 6], fill=(240, 240, 240))
    img.save(path, "PNG")


def test_pipeline_records_color_gate_telemetry(tmp_path: Path, monkeypatch: Any) -> None:
    """Verify telemetry JSON captures exact matches and Color Gate rejections."""
    input_dir = tmp_path / "catalog"
    input_dir.mkdir()
    _draw_chair(input_dir / "chair_navy.png", (20, 45, 130))
    # Byte-identical duplicate
    (input_dir / "chair_navy_copy.png").write_bytes((input_dir / "chair_navy.png").read_bytes())
    # Same silhouette, Crimson Red (near-zero pHash diff, huge Lab Delta-E)
    _draw_chair(input_dir / "chair_crimson.png", (185, 30, 35), accent_dot=True)

    # Stub EmbeddingExtractor so unit test requires zero network/weights
    class _DummyExtractor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def extract_batch(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
            vecs = np.tile(np.ones((1, 512), dtype=np.float32) / np.sqrt(512.0), (len(images), 1))
            return vecs

    monkeypatch.setattr("src.pipeline.EmbeddingExtractor", _DummyExtractor)

    cfg = Config(
        use_background_removal=False,
        color_distance_threshold=16.5,
        similarity_threshold=0.90,
        phash_threshold=8,
        dhash_threshold=6,
        faiss_brute_force_threshold=5000,
    )
    out_xlsx = tmp_path / "out" / "report.xlsx"
    telemetry: dict[str, Any] = {}
    groups = run_pipeline(
        input_dir=input_dir,
        output_path=out_xlsx,
        config=cfg,
        cache_dir=tmp_path / "cache",
        telemetry_out=telemetry,
        progress_lang="en",
    )

    # Navy + exact copy merge into Group 1; Crimson is blocked by Color Gate
    assert len(groups) == 1
    names = {Path(m).name for m in groups[0].members}
    assert names == {"chair_navy.png", "chair_navy_copy.png"}

    assert telemetry["total_images"] == 3
    assert telemetry["stage1_exact_groups"] == 1
    assert len(telemetry["rejected_pairs"]) >= 1

    telemetry_file = out_xlsx.with_name(f"{out_xlsx.stem}_telemetry.json")
    assert telemetry_file.is_file()
    saved = json.loads(telemetry_file.read_text(encoding="utf-8"))
    assert saved["groups_found"] == 1
    assert len(saved["rejected_pairs"]) >= 1
