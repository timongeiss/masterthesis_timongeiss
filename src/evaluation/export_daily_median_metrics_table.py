from __future__ import annotations

import argparse
from functools import reduce
from pathlib import Path
from typing import Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "data" / "results"
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"

PHYSICAL_INPUT_DIR = RESULTS_ROOT / "physical_output"
DEFAULT_OUTPUT = ARTIFACTS_DIR / "daily_median_metrics.xlsx"
DEFAULT_START_DATE = "2025-07-07"

PHYS_TRUE_CANDS = ["Mittelwertleistung [W]", "mittelwertleistung [w]", "mean_power", "avg_power"]
DIRECT_CANDS = ["aswdir_s"]
DIFFUSE_CANDS = ["aswdifd_s"]
TEMP_CANDS = ["t_2m", "t2m"]
SOLAR_ELEV_CANDS = ["solar_elevation_deg", "solar_elevation", "sol_elev_deg"]

METRIC_SPECS = [
    ("physical", ARTIFACTS_DIR / "physical_model_testday_metrics.csv"),
    ("source", ARTIFACTS_DIR / "source_model_testday_metrics.csv"),
    ("target", ARTIFACTS_DIR / "target_model_testday_metrics.csv"),
    ("transfer", ARTIFACTS_DIR / "transfer_model_testday_metrics.csv"),
]


def _find_col(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _list_physical_exports(directory: Path) -> list[Path]:
    csvs = sorted(directory.rglob("*.csv"))
    exports = [fp for fp in csvs if fp.name.endswith("_with_pv_power.csv")]
    if not exports:
        raise FileNotFoundError(f"No '*_with_pv_power.csv' found in '{directory}'.")
    return exports


def _load_physical_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "valid_time" not in df.columns:
        raise ValueError(f"Missing 'valid_time' in {path}.")

    real_col = _find_col(df, PHYS_TRUE_CANDS)
    direct_col = _find_col(df, DIRECT_CANDS)
    diffuse_col = _find_col(df, DIFFUSE_CANDS)
    temp_col = _find_col(df, TEMP_CANDS)
    solar_col = _find_col(df, SOLAR_ELEV_CANDS)

    missing = []
    if real_col is None:
        missing.append("real power")
    if direct_col is None:
        missing.append("direct irradiance")
    if diffuse_col is None:
        missing.append("diffuse irradiance")
    if temp_col is None:
        missing.append("temperature")
    if solar_col is None:
        missing.append("solar elevation")
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(missing)}.")

    out = pd.DataFrame(
        {
            "valid_time": pd.to_datetime(df["valid_time"], errors="coerce"),
            "real_power_w": _to_float(df[real_col]),
            "direct_wm2": _to_float(df[direct_col]),
            "diffuse_wm2": _to_float(df[diffuse_col]),
            "temp_k": _to_float(df[temp_col]),
            "solar_elevation_deg": _to_float(df[solar_col]),
        }
    )
    out = out.dropna(subset=["valid_time", "solar_elevation_deg"])
    out = out[out["solar_elevation_deg"] > 0.0].copy()
    out["date"] = out["valid_time"].dt.normalize()
    return out[["date", "real_power_w", "direct_wm2", "diffuse_wm2", "temp_k"]]


def build_daily_physical_medians(start_date: pd.Timestamp) -> pd.DataFrame:
    exports = _list_physical_exports(PHYSICAL_INPUT_DIR)
    frames = [_load_physical_rows(fp) for fp in exports]
    if not frames:
        raise RuntimeError("No physical rows loaded.")

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows = all_rows[all_rows["date"] >= start_date].copy()
    if all_rows.empty:
        raise RuntimeError(f"No physical daylight rows on or after {start_date.date()}.")

    daily = (
        all_rows.groupby("date", as_index=False)
        .agg(
            median_real_power_w=("real_power_w", "median"),
            median_direct_wm2=("direct_wm2", "median"),
            median_diffuse_wm2=("diffuse_wm2", "median"),
            median_temp_k=("temp_k", "median"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    return daily


def _load_model_metrics(path: Path, prefix: str, start_date: pd.Timestamp) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {path}")

    df = pd.read_csv(path)
    if "day" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"day": "date"})

    required = {"date", "group", "nMAE", "nRMSE", "count"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise KeyError(f"Missing column(s) in {path.name}: {missing}")

    df = df[df["group"] == "all"].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    df = df[df["date"] >= start_date].copy()
    df["nMAE"] = _to_float(df["nMAE"])
    df["nRMSE"] = _to_float(df["nRMSE"])
    df["count"] = _to_float(df["count"])
    df = df.dropna(subset=["nMAE", "nRMSE", "count"])
    df["count"] = df["count"].astype(int)

    out = (
        df[["date", "nMAE", "nRMSE", "count"]]
        .rename(
            columns={
                "nMAE": f"{prefix}_nMAE",
                "nRMSE": f"{prefix}_nRMSE",
                "count": f"{prefix}_count",
            }
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    if out["date"].duplicated().any():
        raise ValueError(f"Duplicate 'all' rows per date found in {path.name}.")
    return out


def _intersect_dates(frames: Sequence[pd.DataFrame]) -> set[pd.Timestamp]:
    common: set[pd.Timestamp] | None = None
    for frame in frames:
        dates = set(frame["date"].tolist())
        common = dates if common is None else (common & dates)
    return common or set()


def build_table(start_date: pd.Timestamp) -> pd.DataFrame:
    medians = build_daily_physical_medians(start_date)
    metric_frames = [_load_model_metrics(path, prefix, start_date) for prefix, path in METRIC_SPECS]

    common_dates = _intersect_dates([medians, *metric_frames])
    if not common_dates:
        raise RuntimeError("No common dates across physical medians and model metrics.")

    medians = medians[medians["date"].isin(common_dates)].copy()
    metric_frames = [df[df["date"].isin(common_dates)].copy() for df in metric_frames]

    merged = reduce(lambda left, right: left.merge(right, on="date", how="inner"), [medians, *metric_frames])
    merged = merged.sort_values("date").reset_index(drop=True)
    merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
    return merged


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a daily table with physical medians (daylight only) and per-model testday metrics "
            "(group == all)."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output .xlsx path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help=f"Only include dates on/after this day in YYYY-MM-DD format (default: {DEFAULT_START_DATE}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    start_date = pd.Timestamp(args.start_date).normalize()
    output_path = args.output if args.output.is_absolute() else (PROJECT_ROOT / args.output)

    table = build_table(start_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_excel(output_path, index=False)

    print(
        f"Saved daily table: {output_path} "
        f"(rows={len(table)}, range={table['date'].iloc[0]}..{table['date'].iloc[-1]})."
    )


if __name__ == "__main__":
    main()
