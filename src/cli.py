"""Command-line interface for the furniture deduplication pipeline.

Usage::

    python -m furniture_dedup run \\
        --input /path/to/images \\
        --output report.xlsx \\
        --config config.yaml \\
        --cache-dir .dedup_cache \\
        [--sample-size N] \\
        [--resume] \\
        [--verbose]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import load_config, setup_logging
from src.pipeline import run


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="furniture_dedup",
        description=(
            "Furniture Product-Image Duplicate Detection System.\n\n"
            "Detects exact copies, near-duplicates (crop/recompress/watermark),\n"
            "and re-shot products — with a mandatory color-verification gate\n"
            "to prevent false positives from same-shape-different-color products.\n\n"
            "The tool NEVER deletes, moves, or modifies source images.\n"
            "Output is a report (.xlsx + .csv) for human review."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run subcommand ---
    run_parser = subparsers.add_parser(
        "run",
        help="Run the full deduplication pipeline",
        description=(
            "Execute the six-stage pipeline:\n"
            "  1. Exact hash (SHA-256)\n"
            "  2. Perceptual hash (pHash + dHash)\n"
            "  3. CLIP embedding similarity (FAISS)\n"
            "  4. Color verification gate (mandatory)\n"
            "  5. Grouping + confidence scoring\n"
            "  6. Report generation (.xlsx + .csv)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    run_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the folder containing images. Searched recursively.",
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Path for the output report file. Both .xlsx and .csv are "
            "generated regardless of the extension you specify."
        ),
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the config.yaml file with all tunable thresholds.",
    )
    run_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".dedup_cache"),
        help=(
            "Directory for the SQLite cache and FAISS index. "
            "Enables incremental re-runs. Default: .dedup_cache"
        ),
    )
    run_parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Process only the first N valid images (for quick testing).",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted run. Loads cached hashes/embeddings "
            "and processes only new or changed files."
        ),
    )
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging for detailed pipeline output.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "run":
        return _cmd_run(args)

    parser.print_help()
    return 1


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute the ``run`` subcommand."""
    # Validate input directory
    if not args.input.is_dir():
        print(f"Error: Input directory does not exist: {args.input}", file=sys.stderr)
        return 1

    # Load and validate config
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    setup_logging(config)

    # Run the pipeline
    try:
        groups = run(
            input_dir=args.input,
            output_path=args.output,
            config=config,
            cache_dir=args.cache_dir,
            sample_size=args.sample_size,
            resume=args.resume,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user. Cache has been saved — use --resume to continue.")
        return 130
    except Exception as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    print(f"\nDone. {len(groups)} duplicate groups found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
