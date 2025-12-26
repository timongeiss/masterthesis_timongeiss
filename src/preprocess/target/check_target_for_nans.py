"""
Scan export CSV files for missing values (NaN/empty) and report
when (timestamp if available), where (column), and which value is missing.

Usage:
  python check_target_for_nans.py                # scans default dir data/processed/physical_input
  python check_target_for_nans.py --dir <dir>    # specify directory
  python check_target_for_nans.py --glob "<dir>/pv_weather_*.csv"
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path
from typing import Iterable, List, Tuple


MISSING_TOKENS = {"nan", "na", "null", "none"}


def is_missing(value: str) -> Tuple[bool, str]:
    """Return (True, reason) if the cell is considered missing.

    Missing if empty after strip, or case-insensitive token in MISSING_TOKENS.
    """
    raw = value
    if raw is None:
        return True, "None"
    s = raw.strip()
    if s == "":
        return True, "empty"
    if s.lower() in MISSING_TOKENS:
        return True, s
    return False, ""


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
DEFAULT_DIR = PROJECT_ROOT / "data" / "processed" / "physical_input"


def iter_csv_files(args) -> Iterable[str]:
    if args.glob:
        for path in glob.glob(args.glob):
            if os.path.isfile(path) and path.lower().endswith(".csv"):
                yield path
        return
    # Default: all CSVs in the given directory
    directory = args.dir
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and name.lower().endswith(".csv"):
            yield path


def preferred_time_field(header: List[str]) -> str | None:
    candidates = [
        "valid_time",
        "timestamp",
        "time",
        "datetime",
        "date",
    ]
    lower = {h.lower(): h for h in header}
    for key in candidates:
        if key in lower:
            return lower[key]
    return None


def check_file(path: str) -> Tuple[int, int]:
    missing_count = 0
    row_count = 0
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                print(f"{path}: empty file")
                return 0, 0
            time_field = preferred_time_field(header)
            # Map index to column name for performance
            columns = list(header)
            for row_idx, row in enumerate(reader, start=2):  # header at line 1
                row_count += 1
                # Defensive: pad row to header length
                if len(row) < len(columns):
                    row.extend([""] * (len(columns) - len(row)))
                timestamp = None
                if time_field is not None:
                    try:
                        ts_index = columns.index(time_field)
                        timestamp = row[ts_index]
                    except Exception:
                        timestamp = None
                for col_idx, (col, cell) in enumerate(zip(columns, row), start=1):
                    miss, reason = is_missing(cell)
                    if miss:
                        missing_count += 1
                        when = timestamp if timestamp is not None else "n/a"
                        # Show compact origin info: file, line, column name
                        # Also show a short representation of the raw value
                        raw_preview = cell if cell.strip() != "" else ""  # empty shows as empty
                        print(
                            f"{path}:{row_idx}: col='{col}': when='{when}': missing='{reason}' raw='{raw_preview}'"
                        )
    except FileNotFoundError:
        print(f"File not found: {path}")
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return missing_count, row_count


def main():
    parser = argparse.ArgumentParser(description="Scan export CSVs for NaNs/empty cells")
    parser.add_argument(
        "--dir",
        default=str(DEFAULT_DIR),
        help=f"Directory containing CSV files (default: {DEFAULT_DIR})",
    )
    parser.add_argument(
        "--glob",
        default=None,
        help="Optional glob pattern overriding --dir, e.g. 'exports/*.csv'",
    )
    args = parser.parse_args()

    if args.glob is None and not os.path.isdir(args.dir):
        print(f"Directory not found: {args.dir}")
        return

    total_files = 0
    total_rows = 0
    total_missing = 0

    for path in iter_csv_files(args):
        total_files += 1
        missing, rows = check_file(path)
        total_missing += missing
        total_rows += rows

    print("\nSummary:")
    print(f"  files scanned : {total_files}")
    print(f"  rows checked  : {total_rows}")
    print(f"  missing cells : {total_missing}")


if __name__ == "__main__":
    main()
