"""Allow running as ``python -m furniture_dedup`` from the parent directory."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so the ``src`` package is importable.
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
