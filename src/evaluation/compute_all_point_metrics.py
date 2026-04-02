"""Compute point-level nMAE/nRMSE from model result CSVs in data/results.

Output:
- data/artifacts/physical_model_all_metrics.csv
- data/artifacts/source_model_all_metrics.csv
- data/artifacts/target_model_all_metrics.csv
- data/artifacts/transfer_model_all_metrics.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
RESULTS_ROOT = PROJECT_ROOT / "data" / "results"
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"

INSTALLED_CAPACITY_W = 9720.0  # 9.72 kWp
MAX_DAY_COUNT = 91
ROLLING_RETRAIN_DAYS = 61


@dataclass(frozen=True)
class ModelSpec:
    key: str
    input_dir: Path
    output_csv: str
    y_true_col: str
    y_pred_col: str
    uses_decimal_comma: bool
    pred_already_normalized: bool


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        key="physical",
        input_dir=RESULTS_ROOT / "physical_output",
        output_csv="physical_model_all_metrics.csv",
        y_true_col="mittelwertleistung [w]",
        y_pred_col="pv_power_W",
        uses_decimal_comma=True,
        pred_already_normalized=False,
    ),
    ModelSpec(
        key="source",
        input_dir=RESULTS_ROOT / "source_output",
        output_csv="source_model_all_metrics.csv",
        y_true_col="y_true",
        y_pred_col="y_pred",
        uses_decimal_comma=False,
        pred_already_normalized=True,
    ),
    ModelSpec(
        key="target",
        input_dir=RESULTS_ROOT / "target_output",
        output_csv="target_model_all_metrics.csv",
        y_true_col="y_true",
        y_pred_col="y_pred",
        uses_decimal_comma=False,
        pred_already_normalized=True,
    ),
    ModelSpec(
        key="transfer",
        input_dir=RESULTS_ROOT / "transfer_output",
        output_csv="transfer_model_all_metrics.csv",
        y_true_col="y_true",
        y_pred_col="y_pred",
        uses_decimal_comma=False,
        pred_already_normalized=True,
    ),
]


def list_csvs(directory: Path) -> List[Path]:
    return sorted(directory.rglob("*.csv"))


def to_numeric(series: pd.Series, use_decimal_comma: bool) -> pd.Series:
    if use_decimal_comma:
        return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def load_model_points(spec: ModelSpec) -> pd.DataFrame:
    files = list_csvs(spec.input_dir)
    if spec.key == "physical":
        # Keep only official physical result exports; ignore debug CSVs.
        files = [fp for fp in files if fp.name.endswith("_with_pv_power.csv")]
    if not files:
        raise FileNotFoundError(f"No CSV files found in '{spec.input_dir}'.")

    frames: List[pd.DataFrame] = []
    skipped = 0
    required = {"valid_time", "solar_elevation_deg", spec.y_true_col, spec.y_pred_col}

    for fp in files:
        try:
            df = pd.read_csv(fp, parse_dates=["valid_time"])
        except Exception:
            skipped += 1
            continue
        if not required.issubset(df.columns):
            skipped += 1
            continue

        sub = df[["valid_time", "solar_elevation_deg", spec.y_true_col, spec.y_pred_col]].copy()
        sub["solar_elevation_deg"] = to_numeric(sub["solar_elevation_deg"], spec.uses_decimal_comma)
        sub["y_true_raw"] = to_numeric(sub[spec.y_true_col], spec.uses_decimal_comma)
        sub["y_pred_raw"] = to_numeric(sub[spec.y_pred_col], spec.uses_decimal_comma)

        sub = sub.dropna(subset=["valid_time", "solar_elevation_deg", "y_true_raw", "y_pred_raw"])
        sub = sub[sub["solar_elevation_deg"] >= 0].copy()
        if sub.empty:
            continue

        abs_err = (sub["y_pred_raw"] - sub["y_true_raw"]).abs()
        if spec.pred_already_normalized:
            err_norm = abs_err
        else:
            err_norm = abs_err / INSTALLED_CAPACITY_W

        sub["nMAE"] = err_norm
        sub["nRMSE"] = np.sqrt(err_norm**2)
        sub["date"] = sub["valid_time"].dt.normalize()
        sub["source_file"] = fp.relative_to(RESULTS_ROOT).as_posix()
        sub["model"] = spec.key

        sub["y_true"] = sub["y_true_raw"]
        sub["y_pred"] = sub["y_pred_raw"]
        frames.append(
            sub[
                [
                    "model",
                    "date",
                    "valid_time",
                    "source_file",
                    "solar_elevation_deg",
                    "y_true",
                    "y_pred",
                    "nMAE",
                    "nRMSE",
                ]
            ]
        )

    if not frames:
        raise RuntimeError(f"No valid daylight rows found for model '{spec.key}' in '{spec.input_dir}'.")

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "valid_time"]).reset_index(drop=True)

    if skipped > 0:
        print(f"[raw-{spec.key}] skipped files: {skipped}")
    print(
        f"[raw-{spec.key}] rows={len(out)} unique_days={out['date'].nunique()} "
        f"range={out['date'].min().date()}..{out['date'].max().date()}"
    )
    return out


def align_to_mlp_window(frames_by_model: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Restrict all models to the common source/target/transfer day axis."""
    ref_keys = ["source", "target", "transfer"]
    missing = [k for k in ref_keys if k not in frames_by_model]
    if missing:
        raise KeyError(f"Missing reference model(s) for window alignment: {missing}")

    ref_common = None
    for key in ref_keys:
        model_dates = set(frames_by_model[key]["date"].unique())
        ref_common = model_dates if ref_common is None else (ref_common & model_dates)

    if not ref_common:
        raise RuntimeError("No common dates found across source/target/transfer for alignment.")

    ordered_ref = sorted(ref_common)
    ref_start = ordered_ref[0]
    ref_end = ordered_ref[-1]
    print(f"[align] reference window = {ref_start.date()}..{ref_end.date()} (days={len(ordered_ref)})")

    aligned: Dict[str, pd.DataFrame] = {}
    for key, df in frames_by_model.items():
        sub = df[df["date"].isin(ref_common)].copy()
        sub = sub.sort_values(["date", "valid_time"]).reset_index(drop=True)
        aligned[key] = sub
        print(
            f"[aligned-{key}] rows={len(sub)} unique_days={sub['date'].nunique()} "
            f"range={sub['date'].min().date()}..{sub['date'].max().date()}"
        )
    return aligned


def apply_common_day_axis(frames_by_model: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    common_dates = None
    for key, df in frames_by_model.items():
        model_dates = set(df["date"].unique())
        common_dates = model_dates if common_dates is None else (common_dates & model_dates)
        if not common_dates:
            raise RuntimeError(f"No common dates remain after intersecting model '{key}'.")

    ordered_common = sorted(common_dates)
    if len(ordered_common) < MAX_DAY_COUNT:
        raise RuntimeError(
            f"Need at least {MAX_DAY_COUNT} common days across all models, got {len(ordered_common)}."
        )

    selected = ordered_common[:MAX_DAY_COUNT]
    day_map = {day: idx + 1 for idx, day in enumerate(selected)}
    print(
        "[days] using d1..d91 = "
        f"{selected[0].date()}..{selected[-1].date()} (common_days={len(ordered_common)})"
    )

    out: Dict[str, pd.DataFrame] = {}
    for key, df in frames_by_model.items():
        sub = df[df["date"].isin(selected)].copy()
        sub["day_idx"] = sub["date"].map(day_map).astype(int)
        sub["phase"] = np.where(
            sub["day_idx"] <= ROLLING_RETRAIN_DAYS,
            "rolling_retrains",
            "frozen_deployment",
        )
        out[key] = sub
    return out


def validate(df: pd.DataFrame, model_key: str) -> None:
    if not ((df["day_idx"] >= 1) & (df["day_idx"] <= MAX_DAY_COUNT)).all():
        raise AssertionError(f"[{model_key}] day_idx contains values outside 1..{MAX_DAY_COUNT}.")
    if not (df["solar_elevation_deg"] >= 0).all():
        raise AssertionError(f"[{model_key}] found rows with solar_elevation_deg < 0.")
    if not ((df["nMAE"] >= 0).all() and (df["nRMSE"] >= 0).all()):
        raise AssertionError(f"[{model_key}] found negative metric values.")
    if not np.allclose(df["nMAE"].to_numpy(), df["nRMSE"].to_numpy(), equal_nan=True):
        raise AssertionError(f"[{model_key}] nMAE and nRMSE differ (expected equal for point-level definition).")


def write_outputs(frames_by_model: Dict[str, pd.DataFrame]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    spec_map = {s.key: s for s in MODEL_SPECS}

    for key, df in frames_by_model.items():
        validate(df, key)
        out_path = ARTIFACTS_DIR / spec_map[key].output_csv
        export = df[
            [
                "date",
                "day_idx",
                "phase",
                "valid_time",
                "source_file",
                "y_true",
                "y_pred",
                "nMAE",
                "nRMSE",
            ]
        ].copy()
        export["date"] = pd.to_datetime(export["date"]).dt.strftime("%Y-%m-%d")
        export.to_csv(out_path, index=False)
        print(
            f"[write] {out_path.name}: rows={len(export)} "
            f"days={export['day_idx'].nunique()} "
            f"phase_counts={export['phase'].value_counts().to_dict()}"
        )


def main() -> None:
    frames_by_model: Dict[str, pd.DataFrame] = {}
    for spec in MODEL_SPECS:
        frames_by_model[spec.key] = load_model_points(spec)

    aligned = align_to_mlp_window(frames_by_model)
    sliced = apply_common_day_axis(aligned)
    write_outputs(sliced)
    print("Done. Point-level metric CSVs written to data/artifacts.")


if __name__ == "__main__":
    main()
