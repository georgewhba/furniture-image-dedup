"""Package the project source code as a Kaggle Dataset.

Creates ``kaggle/dataset/build/`` containing:
  - ``furniture-image-dedup.zip`` (src/, config.yaml, requirements.txt, ...)
  - ``dataset-metadata.json``

Usage::

    python kaggle/build_code_dataset.py
    # first upload:
    kaggle datasets create -p kaggle/dataset/build
    # later updates:
    kaggle datasets version -p kaggle/dataset/build -m "update code"
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "kaggle" / "dataset" / "build"
METADATA_SRC = ROOT / "kaggle" / "dataset" / "dataset-metadata.json"

INCLUDE = [
    "src",
    "config.yaml",
    "requirements.txt",
    "pyproject.toml",
    "README.md",
    "__init__.py",
    "__main__.py",
]


def main() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)

    zip_path = BUILD / "furniture-image-dedup.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in INCLUDE:
            p = ROOT / item
            files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
            for f in files:
                if "__pycache__" in f.parts or f.suffix == ".pyc":
                    continue
                zf.write(f, Path("furniture-image-dedup") / f.relative_to(ROOT))

    shutil.copyfile(METADATA_SRC, BUILD / "dataset-metadata.json")
    meta = json.loads(METADATA_SRC.read_text(encoding="utf-8"))
    print(f"Built {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB) for dataset {meta['id']}")


if __name__ == "__main__":
    main()

