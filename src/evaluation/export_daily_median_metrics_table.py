from __future__ import annotations

import argparse
from functools import reduce
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "data" / "results"
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"
REPORTS_OVERALL_DIR = PROJECT_ROOT / "reports" / "overall"

PHYSICAL_INPUT_DIR = RESULTS_ROOT / "physical_output"
DEFAULT_OUTPUT = ARTIFACTS_DIR / "daily_median_metrics.xlsx"
DEFAULT_HEATMAP_OUTPUT = REPORTS_OVERALL_DIR / "daily_median_metrics_heatmap.png"
DEFAULT_START_DATE = "2025-07-07"
HEATMAP_START_DATE = pd.Timestamp("2025-07-07").normalize()
HEATMAP_END_DATE = pd.Timestamp("2025-10-05").normalize()  # day 91

MODEL_ORDER = ["physical", "source", "target", "transfer"]

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


def _normalize_from_values(values: np.ndarray) -> Normalize:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    vmin = float(finite.min())
    vmax = float(finite.max())
    if np.isclose(vmin, vmax):
        delta = max(1e-9, abs(vmin) * 0.05)
        return Normalize(vmin=vmin - delta, vmax=vmax + delta)
    return Normalize(vmin=vmin, vmax=vmax)


def _scale_to_unit(values: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=float)
    if vmin is None:
        vmin = float(finite.min())
    if vmax is None:
        vmax = float(finite.max())
    if np.isclose(vmin, vmax):
        return np.full(values.shape, 0.5, dtype=float)
    return np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)


def _extrema_row_indices(values: np.ndarray) -> set[int]:
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return set()
    finite_rows = np.where(finite_mask)[0]
    finite_vals = values[finite_mask]
    min_row = int(finite_rows[int(np.argmin(finite_vals))])
    max_row = int(finite_rows[int(np.argmax(finite_vals))])
    return {min_row, max_row}


def export_daily_metrics_heatmap(table: pd.DataFrame, output_path: Path) -> None:
    df = table.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).copy()
    df["day_num"] = (df["date"] - HEATMAP_START_DATE).dt.days + 1

    df = df[(df["date"] >= HEATMAP_START_DATE) & (df["date"] <= HEATMAP_END_DATE)].copy()
    df = df[(df["day_num"] >= 1) & (df["day_num"] <= 91)].copy()
    if len(df) != 91:
        raise RuntimeError(f"Expected exactly 91 rows for heatmap export, got {len(df)}.")

    nmae_cols = [f"{model_key}_nMAE" for model_key in MODEL_ORDER]
    nrmse_cols = [f"{model_key}_nRMSE" for model_key in MODEL_ORDER]
    required_cols = ["median_real_power_w", "median_direct_wm2", "median_diffuse_wm2", "day_num", *nmae_cols, *nrmse_cols]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing heatmap column(s): {missing}")

    for col in [*nmae_cols, *nrmse_cols, "median_real_power_w", "median_direct_wm2", "median_diffuse_wm2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["median_real_power_w", "median_direct_wm2", "median_diffuse_wm2", *nmae_cols, *nrmse_cols]).copy()
    if len(df) != 91:
        raise RuntimeError(f"Expected 91 complete rows after numeric cleanup, got {len(df)}.")

    df = df.sort_values("median_direct_wm2", ascending=True).reset_index(drop=True)

    nmae_values = df[nmae_cols].to_numpy(dtype=float)
    nrmse_values = df[nrmse_cols].to_numpy(dtype=float)
    day_labels = [str(int(v)) for v in df["day_num"].tolist()]
    feature_specs = [
        ("median_real_power_w", "Power", "#6495ed"),
        ("median_direct_wm2", "Direct", "#ffd900"),
        ("median_diffuse_wm2", "Diffuse", "#f70eff"),
    ]
    power_values = df["median_real_power_w"].to_numpy(dtype=float)
    direct_values = df["median_direct_wm2"].to_numpy(dtype=float)
    diffuse_values = df["median_diffuse_wm2"].to_numpy(dtype=float)
    direct_diffuse_stack = np.concatenate([direct_values, diffuse_values])
    dd_finite = direct_diffuse_stack[np.isfinite(direct_diffuse_stack)]
    if dd_finite.size == 0:
        dd_vmin, dd_vmax = 0.0, 1.0
    else:
        dd_vmin, dd_vmax = float(dd_finite.min()), float(dd_finite.max())
    feature_levels = {
        "median_real_power_w": _scale_to_unit(power_values),
        "median_direct_wm2": _scale_to_unit(direct_values, dd_vmin, dd_vmax),
        "median_diffuse_wm2": _scale_to_unit(diffuse_values, dd_vmin, dd_vmax),
    }
    feature_extrema = {col: _extrema_row_indices(df[col].to_numpy(dtype=float)) for col, _, _ in feature_specs}

    cmap_metric = LinearSegmentedColormap.from_list("metric_rwg", ["#1f9e5a", "#ffffff", "#b00020"])
    norm_nmae = _normalize_from_values(nmae_values)
    norm_nrmse = _normalize_from_values(nrmse_values)
    cmap_metric_transparent = cmap_metric.with_extremes(bad=(1.0, 1.0, 1.0, 0.0))

    n_rows = len(df)
    n_models = len(MODEL_ORDER)
    metric_block = np.full((n_rows, n_models * 2), np.nan, dtype=float)
    for i in range(n_models):
        metric_block[:, i * 2] = nmae_values[:, i]
        metric_block[:, i * 2 + 1] = nrmse_values[:, i]
    nmae_block = metric_block.copy()
    nrmse_block = metric_block.copy()
    nmae_block[:, 1::2] = np.nan
    nrmse_block[:, 0::2] = np.nan

    fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=300)  # DIN A4 portrait
    ax.set_facecolor("white")

    ax.imshow(
        nmae_block,
        cmap=cmap_metric_transparent,
        norm=norm_nmae,
        interpolation="nearest",
        aspect="auto",
        origin="upper",
        extent=(3.5, 11.5, n_rows - 0.5, -0.5),
    )
    ax.imshow(
        nrmse_block,
        cmap=cmap_metric_transparent,
        norm=norm_nrmse,
        interpolation="nearest",
        aspect="auto",
        origin="upper",
        extent=(3.5, 11.5, n_rows - 0.5, -0.5),
    )

    ax.axvspan(2.5, 3.5, color="white", zorder=2)

    for col_idx, (feature_col, _feature_label, feature_color) in enumerate(feature_specs):
        levels = feature_levels[feature_col]
        raw_vals = df[feature_col].to_numpy(dtype=float)
        extrema_rows = feature_extrema[feature_col]
        for row_idx, level in enumerate(levels):
            y0 = row_idx - 0.36
            x0 = col_idx - 0.44
            height = 0.72
            width = 0.88
            fill_width = width * float(level)
            if fill_width > 0.0:
                ax.add_patch(
                    Rectangle(
                        (x0, y0),
                        fill_width,
                        height,
                        facecolor=feature_color,
                        edgecolor="none",
                        alpha=0.95,
                        zorder=4,
                    )
                )
            value = raw_vals[row_idx]
            if np.isfinite(value) and row_idx in extrema_rows:
                ax.text(
                    x0 + width - 0.03,
                    row_idx,
                    f"{value:.0f}",
                    ha="right",
                    va="center",
                    fontsize=6.2,
                    color="#111111",
                    zorder=6,
                )

    for sep_x in (5.5, 7.5, 9.5):
        ax.axvline(sep_x, color="#222222", linewidth=0.9, zorder=5)

    for model_idx in range(n_models):
        nmae_col = nmae_values[:, model_idx]
        nrmse_col = nrmse_values[:, model_idx]
        for row_idx in _extrema_row_indices(nmae_col):
            nmae_val = nmae_col[row_idx]
            if np.isfinite(nmae_val):
                ax.text(
                    4 + model_idx * 2,
                    row_idx,
                    f"{nmae_val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color="#111111",
                    zorder=6,
                )
        for row_idx in _extrema_row_indices(nrmse_col):
            nrmse_val = nrmse_col[row_idx]
            if np.isfinite(nrmse_val):
                ax.text(
                    5 + model_idx * 2,
                    row_idx,
                    f"{nrmse_val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color="#111111",
                    zorder=6,
                )

    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(
        [
            "Power\n[W]",
            "Direct\n[W/m2]",
            "Diffuse\n[W/m2]",
            "",
            "nMAE",
            "nRMSE",
            "nMAE",
            "nRMSE",
            "nMAE",
            "nRMSE",
            "nMAE",
            "nRMSE",
        ]
    )
    ax.tick_params(axis="x", labelrotation=0, labelsize=8, labeltop=True, labelbottom=False, top=False, bottom=False, pad=6)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(day_labels, fontsize=7.6)
    for tick_label, day_label in zip(ax.get_yticklabels(), day_labels):
        day_num = int(day_label)
        if 62 <= day_num <= 91:
            tick_label.set_color("#ff00aa")
    ax.set_ylabel("Day number (sorted by median direct irradiance)", fontsize=11.5)

    ax.set_xticks(np.arange(-0.5, 12.0, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, float(n_rows), 1.0), minor=True)
    ax.grid(which="minor", color="#e7e7e7", linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    model_labels = ["Physical", "Source", "Target", "Transfer"]
    for i, model_label in enumerate(model_labels):
        ax.text(4.5 + i * 2.0, -3.2, model_label, ha="center", va="bottom", fontsize=11, fontweight="bold", clip_on=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.22, right=0.96, top=0.93, bottom=0.08)
    fig.savefig(output_path)
    plt.close(fig)


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
    export_daily_metrics_heatmap(table, DEFAULT_HEATMAP_OUTPUT)

    print(
        f"Saved daily table: {output_path} "
        f"(rows={len(table)}, range={table['date'].iloc[0]}..{table['date'].iloc[-1]})."
    )
    print(f"Saved daily metrics heatmap: {DEFAULT_HEATMAP_OUTPUT}")


if __name__ == "__main__":
    main()
