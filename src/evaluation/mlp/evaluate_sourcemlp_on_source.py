"""Evaluate the frozen Source MLP on its held-out source-domain blocks.

This script performs checkpoint inference only. It does not train a model,
modify source splits, or write any model artifacts. The source test split
contains one continuous held-out block per plant; daily daylight-only metrics
are therefore computed per plant before aggregating across plants.

Outputs:
    - data/artifacts/source_model_source_domain/daily_metrics.csv
    - data/artifacts/source_model_source_domain/summary.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]

SOURCE_TEST_DIR = PROJECT_ROOT / "data" / "processed" / "mlp_input" / "test"
METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "source_label_space" / "metadata.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "artifacts" / "source_mlp_checkpoint.pt"
CONFIG_SOURCE = PROJECT_ROOT / "configs" / "config_source_mlp.yaml"
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"
OUTPUT_DIR = ARTIFACTS_DIR / "source_model_source_domain"
DAILY_METRICS_PATH = OUTPUT_DIR / "daily_metrics.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
TARGET_REFERENCE_PATH = ARTIFACTS_DIR / "source_model_testday_metrics.csv"

TARGET_REFERENCE_START = pd.Timestamp("2025-07-07")
TARGET_REFERENCE_END = pd.Timestamp("2025-10-05")
EXPECTED_SOURCE_PLANT_COUNT = 172
DAYLIGHT_THRESHOLD_DEG = 0.0
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.mlp import mlp_model as mlp


CFG_SOURCE = mlp.load_mlp_config(CONFIG_SOURCE)
INPUT_DIM = mlp.compute_input_dim(CFG_SOURCE["feature_cols"], CFG_SOURCE["meta_dim"])


@dataclass(frozen=True)
class SourceTestWindow:
    """One source-test prediction window with its evaluation metadata."""

    run_id: str
    features: np.ndarray
    targets: np.ndarray
    valid_times: np.ndarray
    solar_elevation_deg: np.ndarray


def _infer_hidden_sizes(state: dict[str, torch.Tensor]) -> list[int]:
    """Reconstruct hidden-layer sizes from a saved Source-MLP state dict."""
    weight_keys = [key for key in state if key.endswith("weight")]
    return [int(state[key].shape[0]) for key in weight_keys[:-1]]


def load_checkpoint_model(
    checkpoint_path: Path = CHECKPOINT_PATH,
    device: torch.device = DEVICE,
) -> nn.Module:
    """Load the existing checkpoint for inference without any training step."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Source MLP checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint content in {checkpoint_path}.")

    hidden_sizes = _infer_hidden_sizes(state)
    model = mlp.WindowMLP(hidden_sizes, INPUT_DIM).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def collect_source_windows_with_metadata(
    data: pd.DataFrame,
    meta_tuple: tuple,
    config: dict,
) -> list[SourceTestWindow]:
    """Mirror the source collector while retaining metadata for evaluation.

    ``mlp.collect_windows_source`` only returns features and labels. This local
    implementation deliberately mirrors its eligibility checks and uses the
    same ``mlp.make_feature_block`` helper, but additionally keeps the first
    191 timestamps and solar-elevation values used by that feature block.
    """
    feature_cols = config["feature_cols"]
    target_col = config["target_col"]
    grouped = {
        run_id: group.sort_values("valid_time")
        for run_id, group in data.groupby("run_id")
    }
    windows: list[SourceTestWindow] = []

    for run_id, group in sorted(grouped.items(), key=lambda item: item[0]):
        if len(group) != mlp.EXPECTED_STEPS + 1:
            continue
        if group[feature_cols + [target_col]].isna().any().any():
            continue

        window_start = group["valid_time"].min()
        start_time = window_start - pd.Timedelta(minutes=15)
        candidates = data.loc[data["valid_time"] == start_time, target_col]
        if candidates.empty:
            continue

        normalized_group = group.reset_index(drop=True)
        features, targets = mlp.make_feature_block(
            normalized_group,
            meta_tuple,
            float(candidates.iloc[0]),
            feature_cols,
            target_col,
        )
        evaluation_steps = normalized_group.iloc[: mlp.EXPECTED_STEPS]
        windows.append(
            SourceTestWindow(
                run_id=str(run_id),
                features=features,
                targets=targets,
                valid_times=evaluation_steps["valid_time"].to_numpy(dtype="datetime64[ns]"),
                solar_elevation_deg=evaluation_steps["solar_elevation_deg"].to_numpy(dtype=float),
            )
        )
    return windows


def validate_local_windows(
    windows: list[SourceTestWindow],
    data: pd.DataFrame,
    meta_tuple: tuple,
    config: dict,
    *,
    plant_id: str,
) -> None:
    """Assert exact agreement with the unchanged shared source collector."""
    shared_samples = mlp.collect_windows_source(data, meta_tuple, config)
    if len(windows) != len(shared_samples):
        raise AssertionError(
            f"{plant_id}: local collector returned {len(windows)} windows, but "
            f"mlp.collect_windows_source returned {len(shared_samples)}."
        )

    for index, (window, (features, targets)) in enumerate(zip(windows, shared_samples)):
        if not np.array_equal(window.features, features):
            raise AssertionError(f"{plant_id}: feature mismatch in source window {index}.")
        if not np.array_equal(window.targets, targets):
            raise AssertionError(f"{plant_id}: target mismatch in source window {index}.")


def predict_windows(model: nn.Module, windows: list[SourceTestWindow]) -> np.ndarray:
    """Run batched checkpoint inference for one plant's valid source windows."""
    if not windows:
        raise ValueError("Cannot predict an empty source-window collection.")

    features = np.stack([window.features for window in windows]).astype(np.float32, copy=False)
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), BATCH_SIZE):
            batch = torch.from_numpy(features[start : start + BATCH_SIZE]).to(DEVICE)
            predictions.append(model(batch).cpu().numpy())

    output = np.concatenate(predictions, axis=0)
    target_shape = (len(windows), mlp.EXPECTED_STEPS)
    if output.shape != target_shape:
        raise ValueError(
            f"Checkpoint output shape {output.shape} does not match expected {target_shape}."
        )
    if not np.isfinite(output).all():
        raise ValueError("Checkpoint inference produced non-finite source predictions.")
    return output


def accumulate_daily_errors(
    daily_sums: dict[tuple[str, pd.Timestamp], dict[str, float | int]],
    *,
    plant_id: str,
    windows: list[SourceTestWindow],
    predictions: np.ndarray,
) -> None:
    """Accumulate daylight absolute and squared errors by plant and day."""
    for window, prediction in zip(windows, predictions):
        if prediction.shape != window.targets.shape:
            raise ValueError(
                f"{plant_id}/{window.run_id}: prediction and target shapes differ: "
                f"{prediction.shape} vs {window.targets.shape}."
            )
        daylight = window.solar_elevation_deg >= DAYLIGHT_THRESHOLD_DEG
        if not daylight.any():
            continue

        errors = prediction.astype(float) - window.targets.astype(float)
        dates = pd.DatetimeIndex(window.valid_times).normalize()
        for date in dates[daylight].unique():
            day_mask = daylight & (dates == date)
            day_errors = errors[day_mask]
            key = (plant_id, pd.Timestamp(date))
            sums = daily_sums.setdefault(
                key,
                {
                    "absolute_error_sum": 0.0,
                    "squared_error_sum": 0.0,
                    "count": 0,
                    "daylight_forecast_window_count": 0,
                },
            )
            sums["absolute_error_sum"] = float(sums["absolute_error_sum"]) + float(
                np.abs(day_errors).sum()
            )
            sums["squared_error_sum"] = float(sums["squared_error_sum"]) + float(
                np.square(day_errors).sum()
            )
            sums["count"] = int(sums["count"]) + int(day_errors.size)
            sums["daylight_forecast_window_count"] = int(
                sums["daylight_forecast_window_count"]
            ) + 1


def build_daily_metrics(
    daily_sums: dict[tuple[str, pd.Timestamp], dict[str, float | int]],
    expected_plant_ids: set[str],
) -> pd.DataFrame:
    """Convert accumulated source errors into daily nMAE/nRMSE rows."""
    rows: list[dict[str, object]] = []
    for (plant_id, date), sums in sorted(daily_sums.items()):
        count = int(sums["count"])
        if count <= 0:
            raise ValueError(f"{plant_id}/{date.date()}: no daylight points accumulated.")
        rows.append(
            {
                "anlage_id": plant_id,
                "date": date.strftime("%Y-%m-%d"),
                "group": "all",
                "nMAE": float(sums["absolute_error_sum"]) / count,
                "nRMSE": float(np.sqrt(float(sums["squared_error_sum"]) / count)),
                "count": count,
                "daylight_forecast_window_count": int(
                    sums["daylight_forecast_window_count"]
                ),
            }
        )

    daily_metrics = pd.DataFrame(rows)
    if daily_metrics.empty:
        raise RuntimeError("No daylight source-domain daily metrics were generated.")
    actual_plant_ids = set(daily_metrics["anlage_id"])
    if actual_plant_ids != expected_plant_ids:
        missing = sorted(expected_plant_ids - actual_plant_ids)
        unexpected = sorted(actual_plant_ids - expected_plant_ids)
        raise AssertionError(
            "Daily source-domain metrics do not cover exactly the expected plants: "
            f"missing={missing}, unexpected={unexpected}."
        )
    if daily_metrics[["nMAE", "nRMSE"]].isna().any().any():
        raise ValueError("Source-domain daily metrics contain invalid values.")
    return daily_metrics.sort_values(["anlage_id", "date"]).reset_index(drop=True)


def _as_numeric(series: pd.Series) -> pd.Series:
    """Parse numeric values from dot- or decimal-comma CSV exports."""
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def load_target_domain_reference(
    path: Path = TARGET_REFERENCE_PATH,
) -> dict[str, object]:
    """Load the same 91-day technical Source-on-Target reference as the plot."""
    if not path.exists():
        raise FileNotFoundError(f"Target-domain reference metrics not found: {path}")
    reference = pd.read_csv(path)
    required_columns = {"date", "group", "nMAE", "nRMSE", "count"}
    missing = required_columns - set(reference.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    reference = reference[reference["group"] == "all"].copy()
    reference["date"] = pd.to_datetime(reference["date"], errors="coerce").dt.normalize()
    reference["nMAE"] = _as_numeric(reference["nMAE"])
    reference["nRMSE"] = _as_numeric(reference["nRMSE"])
    reference["count"] = _as_numeric(reference["count"])
    reference = reference[
        (reference["date"] >= TARGET_REFERENCE_START)
        & (reference["date"] <= TARGET_REFERENCE_END)
    ].sort_values("date")

    expected_dates = pd.date_range(TARGET_REFERENCE_START, TARGET_REFERENCE_END, freq="D")
    actual_dates = pd.DatetimeIndex(reference["date"])
    if (
        len(reference) != len(expected_dates)
        or actual_dates.has_duplicates
        or not actual_dates.equals(expected_dates)
    ):
        raise ValueError(
            f"{path} must provide one all-group row per day from "
            f"{TARGET_REFERENCE_START.date()} through {TARGET_REFERENCE_END.date()}."
        )
    if reference[["nMAE", "nRMSE", "count"]].isna().any().any():
        raise ValueError(f"{path} contains invalid values in the 91-day reference window.")

    return {
        "evaluation_domain": "target",
        "aggregation_level": "domain_reference",
        "anlage_id": pd.NA,
        "aggregation_method": "median_of_91_daily_metrics",
        "evaluation_start": TARGET_REFERENCE_START.strftime("%Y-%m-%d"),
        "evaluation_end": TARGET_REFERENCE_END.strftime("%Y-%m-%d"),
        "plant_count": 1,
        "daily_metric_count": len(reference),
        "daylight_point_count": int(reference["count"].sum()),
        "daylight_forecast_window_count": pd.NA,
        "nMAE": float(reference["nMAE"].median()),
        "nRMSE": float(reference["nRMSE"].median()),
    }


def build_summary(
    daily_metrics: pd.DataFrame,
    target_reference: dict[str, object],
) -> pd.DataFrame:
    """Build plant, source-domain, and target-reference summary rows."""
    plant_rows: list[dict[str, object]] = []
    for plant_id, metrics in daily_metrics.groupby("anlage_id", sort=True):
        dates = pd.to_datetime(metrics["date"])
        plant_rows.append(
            {
                "evaluation_domain": "source",
                "aggregation_level": "plant",
                "anlage_id": plant_id,
                "aggregation_method": "median_of_daily_metrics",
                "evaluation_start": dates.min().strftime("%Y-%m-%d"),
                "evaluation_end": dates.max().strftime("%Y-%m-%d"),
                "plant_count": 1,
                "daily_metric_count": len(metrics),
                "daylight_point_count": int(metrics["count"].sum()),
                "daylight_forecast_window_count": int(
                    metrics["daylight_forecast_window_count"].sum()
                ),
                "nMAE": float(metrics["nMAE"].median()),
                "nRMSE": float(metrics["nRMSE"].median()),
            }
        )

    plant_summary = pd.DataFrame(plant_rows)
    if len(plant_summary) != EXPECTED_SOURCE_PLANT_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_SOURCE_PLANT_COUNT} plant summaries, got {len(plant_summary)}."
        )

    all_dates = pd.to_datetime(daily_metrics["date"])
    source_domain_row = {
        "evaluation_domain": "source",
        "aggregation_level": "domain",
        "anlage_id": pd.NA,
        "aggregation_method": "median_of_172_plant_daily_medians",
        "evaluation_start": all_dates.min().strftime("%Y-%m-%d"),
        "evaluation_end": all_dates.max().strftime("%Y-%m-%d"),
        "plant_count": len(plant_summary),
        "daily_metric_count": len(daily_metrics),
        "daylight_point_count": int(daily_metrics["count"].sum()),
        "daylight_forecast_window_count": int(
            daily_metrics["daylight_forecast_window_count"].sum()
        ),
        "nMAE": float(plant_summary["nMAE"].median()),
        "nRMSE": float(plant_summary["nRMSE"].median()),
    }
    summary = pd.concat(
        [plant_summary, pd.DataFrame([source_domain_row, target_reference])],
        ignore_index=True,
    )
    summary["nMAE_percent"] = summary["nMAE"] * 100.0
    summary["nRMSE_percent"] = summary["nRMSE"] * 100.0
    return summary


def evaluate_source_domain() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run frozen-checkpoint source-domain inference and assemble both reports."""
    if not SOURCE_TEST_DIR.exists():
        raise FileNotFoundError(f"Source test directory not found: {SOURCE_TEST_DIR}")

    test_paths = sorted(SOURCE_TEST_DIR.glob("*_windows_norm.csv"))
    test_plant_ids = {path.stem.removesuffix("_windows_norm") for path in test_paths}
    if len(test_plant_ids) != EXPECTED_SOURCE_PLANT_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_SOURCE_PLANT_COUNT} source test plants, found {len(test_plant_ids)}."
        )

    meta_map = mlp.load_metadata_source(METADATA_PATH, 1.0)
    missing_metadata = sorted(test_plant_ids - set(meta_map))
    if missing_metadata:
        raise ValueError(f"Missing source metadata for test plants: {missing_metadata}")

    model = load_checkpoint_model()
    daily_sums: dict[tuple[str, pd.Timestamp], dict[str, float | int]] = {}
    for index, path in enumerate(test_paths, start=1):
        plant_id = path.stem.removesuffix("_windows_norm")
        data = pd.read_csv(path, parse_dates=["valid_time"])
        if data.empty:
            raise ValueError(f"{plant_id}: source test file is empty: {path}")

        windows = collect_source_windows_with_metadata(data, meta_map[plant_id], CFG_SOURCE)
        validate_local_windows(windows, data, meta_map[plant_id], CFG_SOURCE, plant_id=plant_id)
        predictions = predict_windows(model, windows)
        accumulate_daily_errors(
            daily_sums,
            plant_id=plant_id,
            windows=windows,
            predictions=predictions,
        )
        print(
            f"[{index}/{len(test_paths)}] {plant_id}: "
            f"validated and evaluated {len(windows)} source windows.",
            flush=True,
        )

    daily_metrics = build_daily_metrics(daily_sums, test_plant_ids)
    summary = build_summary(daily_metrics, load_target_domain_reference())
    return daily_metrics, summary


def export_reports(
    daily_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Write source-domain detail and comparison summaries in one artifact folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = output_dir / DAILY_METRICS_PATH.name
    summary_path = output_dir / SUMMARY_PATH.name
    daily_metrics.to_csv(daily_path, index=False)
    summary.to_csv(summary_path, index=False)
    return daily_path, summary_path


def main() -> None:
    """Evaluate the stored Source checkpoint and export source-domain metrics."""
    mlp.seed_everything()
    print(f"Evaluating frozen Source MLP checkpoint on {DEVICE}: {CHECKPOINT_PATH}")
    daily_metrics, summary = evaluate_source_domain()
    daily_path, summary_path = export_reports(daily_metrics, summary)

    source_row = summary[
        (summary["evaluation_domain"] == "source")
        & (summary["aggregation_level"] == "domain")
    ].iloc[0]
    target_row = summary[
        (summary["evaluation_domain"] == "target")
        & (summary["aggregation_level"] == "domain_reference")
    ].iloc[0]
    print(
        "Source on source domain "
        f"(median of {int(source_row['plant_count'])} plant daily medians): "
        f"nMAE={source_row['nMAE_percent']:.3f}%, "
        f"nRMSE={source_row['nRMSE_percent']:.3f}%"
    )
    print(
        "Source on target domain "
        f"(91-day technical reference): nMAE={target_row['nMAE_percent']:.3f}%, "
        f"nRMSE={target_row['nRMSE_percent']:.3f}%"
    )
    print(f"Saved daily source-domain metrics: {daily_path}")
    print(f"Saved source-domain summary: {summary_path}")


if __name__ == "__main__":
    main()
