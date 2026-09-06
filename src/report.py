"""Stage 6 — Report generation (Excel and CSV).

Produces a sorted, human-reviewable report of all duplicate groups.
Format: one row per image, grouped by ``group_id``, so the catalog owner
can filter/sort in Excel without any special tooling.

Columns
-------
- ``group_id``: Integer group identifier (1-based)
- ``file_path``: Absolute path to the image
- ``file_name``: Basename for quick scanning
- ``confidence``: 0–100, highest-confidence groups first
- ``match_type``: ``exact`` | ``near_exact`` | ``embedding_color_verified``
- ``group_size``: Number of images in this group

The Excel workbook includes:
- A "Duplicates" sheet with the main data, conditionally colored
- A "Summary" sheet with aggregate statistics
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from src.scoring import DuplicateGroup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color bands for confidence in Excel
# ---------------------------------------------------------------------------

_FILL_HIGH = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")  # green
_FILL_MED = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")  # yellow
_FILL_LOW = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")  # red


def _confidence_fill(confidence: int) -> PatternFill:
    """Return an Excel fill color for a confidence score."""
    if confidence >= 90:
        return _FILL_HIGH
    if confidence >= 70:
        return _FILL_MED
    return _FILL_LOW


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------


def groups_to_dataframe(groups: list[DuplicateGroup]) -> pd.DataFrame:
    """Convert duplicate groups to a flat DataFrame (one row per image).

    Args:
        groups: List of confirmed duplicate groups.

    Returns:
        A :class:`pandas.DataFrame` sorted by confidence descending,
        then group_id.
    """
    rows: list[dict[str, Any]] = []
    for g in groups:
        for member in g.members:
            rows.append(
                {
                    "group_id": g.group_id,
                    "file_path": member,
                    "file_name": Path(member).name,
                    "confidence": g.confidence,
                    "match_type": g.match_type,
                    "group_size": len(g.members),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "group_id",
                "file_path",
                "file_name",
                "confidence",
                "match_type",
                "group_size",
            ]
        )
    return df


def build_summary(
    groups: list[DuplicateGroup],
    total_images: int,
    runtime_seconds: float,
) -> pd.DataFrame:
    """Build a summary statistics DataFrame.

    Args:
        groups: Duplicate groups.
        total_images: Total images scanned.
        runtime_seconds: Total pipeline runtime.

    Returns:
        A two-column DataFrame with metric names and values.
    """
    n_groups = len(groups)
    n_images_in_groups = sum(len(g.members) for g in groups)
    by_type: dict[str, int] = {}
    for g in groups:
        by_type[g.match_type] = by_type.get(g.match_type, 0) + 1

    rows = [
        {"metric": "Total images scanned", "value": str(total_images)},
        {"metric": "Duplicate groups found", "value": str(n_groups)},
        {"metric": "Images in duplicate groups", "value": str(n_images_in_groups)},
        {
            "metric": "Unique images (no duplicates)",
            "value": str(total_images - n_images_in_groups),
        },
        {"metric": "Total runtime (seconds)", "value": f"{runtime_seconds:.1f}"},
    ]

    for match_type in ["exact", "near_exact", "embedding_color_verified"]:
        count = by_type.get(match_type, 0)
        rows.append({"metric": f"Groups — {match_type}", "value": str(count)})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write the duplicates DataFrame to a CSV file.

    Args:
        df: The duplicates DataFrame (one row per image).
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("CSV report written to %s (%d rows)", path, len(df))


def write_excel(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    path: Path,
) -> None:
    """Write the duplicates report to an Excel workbook.

    Includes:
    - "Duplicates" sheet with conditional coloring by confidence
    - "Summary" sheet with aggregate statistics
    - Frozen header row, auto-width columns

    Args:
        df: The duplicates DataFrame.
        summary_df: The summary statistics DataFrame.
        path: Output file path (``.xlsx``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # --- Duplicates sheet ---
        df.to_excel(writer, sheet_name="Duplicates", index=False)
        ws = writer.sheets["Duplicates"]

        # Freeze header row
        ws.freeze_panes = "A2"

        # Header styling
        header_font = Font(bold=True)
        for col_idx in range(1, len(df.columns) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")

        # Conditional fill on confidence column
        conf_col_idx = (
            list(df.columns).index("confidence") + 1
            if "confidence" in df.columns
            else None
        )
        if conf_col_idx is not None:
            for row_idx in range(2, len(df) + 2):
                cell = ws.cell(row=row_idx, column=conf_col_idx)
                try:
                    val = int(cell.value) if cell.value is not None else 0
                    cell.fill = _confidence_fill(val)
                except (ValueError, TypeError):
                    pass

        # Auto-width columns
        for col_idx in range(1, len(df.columns) + 1):
            col_letter = get_column_letter(col_idx)
            max_len = max(
                len(str(ws.cell(row=r, column=col_idx).value or ""))
                for r in range(1, min(len(df) + 2, 200))
            )
            ws.column_dimensions[col_letter].width = min(max_len + 4, 80)

        # --- Summary sheet ---
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        ws_sum = writer.sheets["Summary"]
        ws_sum.freeze_panes = "A2"
        for col_idx in range(1, len(summary_df.columns) + 1):
            cell = ws_sum.cell(row=1, column=col_idx)
            cell.font = header_font
        for col_idx in range(1, len(summary_df.columns) + 1):
            col_letter = get_column_letter(col_idx)
            max_len = max(
                len(str(ws_sum.cell(row=r, column=col_idx).value or ""))
                for r in range(1, len(summary_df) + 2)
            )
            ws_sum.column_dimensions[col_letter].width = min(max_len + 4, 60)

    logger.info("Excel report written to %s (%d rows)", path, len(df))


def build_inventory_dataframe(
    all_image_paths: list[str],
    groups: list[DuplicateGroup],
    content_hashes: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a comprehensive inventory DataFrame of ALL scanned images.

    Categorizes every image into:
    - CANONICAL_MASTER: Primary/representative shot in a duplicate group (Action: KEEP_MASTER)
    - DUPLICATE: Secondary duplicate copy (Action: ARCHIVE_DUPLICATE)
    - UNIQUE: Distinct product image with no duplicates (Action: KEEP_UNIQUE)

    Args:
        all_image_paths: List of all valid image paths scanned.
        groups: List of confirmed duplicate groups.
        content_hashes: Optional mapping of file_path to content SHA-256 hash.

    Returns:
        DataFrame with full inventory and classification for every image.
    """
    # Map file path to group info
    group_map: dict[str, tuple[int, int, str, int, str, str]] = {}
    for g in groups:
        for idx, member in enumerate(g.members):
            status = "CANONICAL_MASTER" if idx == 0 else "DUPLICATE"
            action = "KEEP_MASTER" if idx == 0 else "ARCHIVE_DUPLICATE"
            group_map[member] = (
                g.group_id,
                g.confidence,
                g.match_type,
                len(g.members),
                status,
                action,
            )

    rows: list[dict[str, Any]] = []
    for fpath in all_image_paths:
        p = Path(fpath)
        file_size_kb = round(p.stat().st_size / 1024, 1) if p.exists() else 0.0
        sha256 = content_hashes.get(fpath, "") if content_hashes else ""

        if fpath in group_map:
            gid, conf, mtype, gsize, status, action = group_map[fpath]
            rows.append(
                {
                    "file_name": p.name,
                    "status": status,
                    "recommended_action": action,
                    "group_id": gid,
                    "group_size": gsize,
                    "confidence": conf,
                    "match_type": mtype,
                    "file_size_kb": file_size_kb,
                    "sha256_hash": sha256,
                    "file_path": fpath,
                }
            )
        else:
            rows.append(
                {
                    "file_name": p.name,
                    "status": "UNIQUE",
                    "recommended_action": "KEEP_UNIQUE",
                    "group_id": "",
                    "group_size": 1,
                    "confidence": "",
                    "match_type": "none",
                    "file_size_kb": file_size_kb,
                    "sha256_hash": sha256,
                    "file_path": fpath,
                }
            )

    # Sort: DUPLICATE first, then CANONICAL_MASTER, then UNIQUE; and by group_id
    status_order = {"DUPLICATE": 0, "CANONICAL_MASTER": 1, "UNIQUE": 2}
    rows.sort(
        key=lambda r: (
            status_order.get(r["status"], 3),
            r["group_id"] if isinstance(r["group_id"], int) else 999999,
            r["file_name"],
        )
    )
    return pd.DataFrame(rows)


def write_inventory_excel(df: pd.DataFrame, path: Path) -> None:
    """Write the full image inventory to a formatted Excel file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    header_fill = PatternFill(
        start_color="1F497D", end_color="1F497D", fill_type="solid"
    )
    header_font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
    data_font = Font(name="Segoe UI", size=9)

    fill_dup = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")  # soft red
    fill_master = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")  # soft green
    fill_unique = PatternFill(start_color="DDEBF7", end_color="DDEBF7", fill_type="solid")  # soft blue

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All_Images_Inventory", index=False)
        ws = writer.sheets["All_Images_Inventory"]
        ws.freeze_panes = "A2"

        # Headers
        for col_idx in range(1, len(df.columns) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # Color-code rows by status
        status_col = list(df.columns).index("status") + 1 if "status" in df.columns else 2
        for row_idx in range(2, len(df) + 2):
            status_val = str(ws.cell(row=row_idx, column=status_col).value or "")
            row_fill = None
            if status_val == "DUPLICATE":
                row_fill = fill_dup
            elif status_val == "CANONICAL_MASTER":
                row_fill = fill_master
            elif status_val == "UNIQUE":
                row_fill = fill_unique

            for col_idx in range(1, len(df.columns) + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.font = data_font
                if row_fill:
                    cell.fill = row_fill

        # Auto-width
        for col_idx in range(1, len(df.columns) + 1):
            col_letter = get_column_letter(col_idx)
            max_len = max(
                len(str(ws.cell(row=r, column=col_idx).value or ""))
                for r in range(1, min(len(df) + 2, 200))
            )
            ws.column_dimensions[col_letter].width = min(max_len + 4, 80)

    logger.info("Inventory Excel report written to %s (%d rows)", path, len(df))


def generate_reports(
    groups: list[DuplicateGroup],
    output_path: Path,
    total_images: int,
    runtime_seconds: float,
    all_image_paths: list[str] | None = None,
    content_hashes: dict[str, str] | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Generate both Excel and CSV reports for duplicates AND all-images inventory.

    Produces:
    1. Primary / Filtered Duplicates: {base}_filtered_duplicates.xlsx & .csv
       (also writes {base}.xlsx & .csv for compatibility)
    2. All Images Master Inventory: {base}_all_images_inventory.xlsx & .csv

    Args:
        groups: Confirmed duplicate groups.
        output_path: Base output path.
        total_images: Total images scanned.
        runtime_seconds: Total pipeline runtime in seconds.
        all_image_paths: List of all scanned image paths.
        content_hashes: Mapping of file_path to SHA-256 content hash.

    Returns:
        Tuple of (duplicates_xlsx, duplicates_csv, inventory_xlsx, inventory_csv).
    """
    base = output_path.with_suffix("")
    
    # 1. Duplicates reports
    dup_xlsx = base.parent / f"{base.name}_filtered_duplicates.xlsx"
    dup_csv = base.parent / f"{base.name}_filtered_duplicates.csv"
    compat_xlsx = base.with_suffix(".xlsx")
    compat_csv = base.with_suffix(".csv")

    df_dup = groups_to_dataframe(groups)
    summary_df = build_summary(groups, total_images, runtime_seconds)

    write_csv(df_dup, dup_csv)
    write_excel(df_dup, summary_df, dup_xlsx)

    if dup_csv != compat_csv:
        write_csv(df_dup, compat_csv)
    if dup_xlsx != compat_xlsx:
        write_excel(df_dup, summary_df, compat_xlsx)

    # 2. Inventory reports (ALL images)
    inv_xlsx = base.parent / f"{base.name}_all_images_inventory.xlsx"
    inv_csv = base.parent / f"{base.name}_all_images_inventory.csv"

    all_paths = all_image_paths if all_image_paths is not None else [
        m for g in groups for m in g.members
    ]
    df_inv = build_inventory_dataframe(all_paths, groups, content_hashes)

    write_csv(df_inv, inv_csv)
    write_inventory_excel(df_inv, inv_xlsx)

    return dup_xlsx, dup_csv, inv_xlsx, inv_csv
