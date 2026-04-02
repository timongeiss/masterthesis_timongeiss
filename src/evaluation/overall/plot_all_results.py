"""Create scatter overviews from point-level all-metrics CSVs.

Generates figures in reports/overall:
- overall_all_results_scatter_kw.png
- overall_all_results_scatter_kw_retrain61.png
- overall_all_results_scatter_kw_frozen30.png
- overall_all_results_boxplots.png
- overall_all_results_histograms.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "artifacts"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "overall"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config_overall.yaml"
P_INSTALLED_KW = 9.72
PHYSICAL_ALL_METRICS_CSV = "physical_model_all_metrics.csv"


def load_config() -> Tuple[List[List[str]], Dict[str, str]]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "models_all" not in cfg:
        raise KeyError("config_overall.yaml requires 'models_all' for plot_all_results.py")
    return cfg["models_all"], cfg["model_colors"]


MODELS_ALL, MODEL_COLORS = load_config()


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def load_metrics(csv_name: str) -> pd.DataFrame:
    path = DATA_DIR / csv_name
    df = pd.read_csv(path, parse_dates=["valid_time"], low_memory=False)
    required = {"y_true", "y_pred", "phase", "nMAE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_name} is missing required columns: {sorted(missing)}")

    for col in ("y_true", "y_pred", "nMAE"):
        df[col] = _to_float(df[col])
    return df


def prepare_data() -> Dict[str, Dict[str, object]]:
    data: Dict[str, Dict[str, object]] = {}
    for csv_name, label, color_key in MODELS_ALL:
        df = load_metrics(csv_name)
        if csv_name == PHYSICAL_ALL_METRICS_CSV:
            # Physical outputs are in W -> convert to kW.
            df["y_true_plot"] = df["y_true"] / 1000.0
            df["y_pred_plot"] = df["y_pred"] / 1000.0
        else:
            # MLP outputs are normalized to installed capacity -> convert to kW.
            df["y_true_plot"] = df["y_true"] * P_INSTALLED_KW
            df["y_pred_plot"] = df["y_pred"] * P_INSTALLED_KW

        data[csv_name] = {
            "df": df,
            "label": label,
            "color": MODEL_COLORS[color_key],
        }
    return data


def plot_all_models_scatter(
    data: Dict[str, Dict[str, object]],
    save_path: Path,
    phase_filter: str | None = None,
    title_suffix: str = "all days",
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes_flat = np.array(axes).reshape(-1)
    axis_max = P_INSTALLED_KW

    for idx, (ax, (csv_name, _, _)) in enumerate(zip(axes_flat, MODELS_ALL)):
        entry = data[csv_name]
        df = entry["df"]
        if phase_filter is not None:
            df = df[df["phase"] == phase_filter]
        subset = df.dropna(subset=["y_true_plot", "y_pred_plot"])

        ax.scatter(
            subset["y_true_plot"],
            subset["y_pred_plot"],
            s=6,
            alpha=0.22,
            color=entry["color"],
            edgecolors="none",
        )
        ax.plot([0, axis_max], [0, axis_max], color="#333333", linestyle="--", linewidth=1.0)
        ax.set_title(entry["label"])
        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.95, f"N={len(subset):,}", transform=ax.transAxes, ha="left", va="top", fontsize=8)

        if idx % 2 == 0:
            ax.set_ylabel("Prediction [kW]")
        if idx // 2 == 1:
            ax.set_xlabel("Truth [kW]")

    fig.suptitle(f"Prediction vs Truth by model ({title_suffix})")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_point_error_boxplots(data: Dict[str, Dict[str, object]], save_path: Path) -> None:
    """Create boxplots from point-level nMAE values (all/61d/30d)."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True, sharey=True)
    phase_specs = [
        (None, "All data"),
        ("rolling_retrains", "Rolling retrains (61 days)"),
        ("frozen_deployment", "Frozen deployment (30 days)"),
    ]

    for ax, (phase_filter, title) in zip(axes, phase_specs):
        labels = []
        box_data = []
        colors = []

        for csv_name, _, _ in MODELS_ALL:
            entry = data[csv_name]
            df = entry["df"]
            if phase_filter is not None:
                df = df[df["phase"] == phase_filter]
            # Guard against tiny negative render artifacts: enforce non-negative metric domain.
            vals = df["nMAE"].clip(lower=0.0).dropna()

            labels.append(entry["label"])
            box_data.append(vals)
            colors.append(entry["color"])

        bp = ax.boxplot(box_data, patch_artist=True, tick_labels=labels, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        for line in bp["whiskers"] + bp["caps"]:
            line.set_clip_on(True)
            line.set_solid_capstyle("butt")
            line.set_linewidth(1.0)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.4)

        ax.set_title(title)
        ax.set_ylabel("nMAE")
        ax.set_ylim(0.0, 0.4)
        ax.margins(y=0.0)
        ax.axhline(0.0, color="#222222", linewidth=1.0, zorder=5)
        ax.grid(True, axis="y", alpha=0.3)

    axes[-1].tick_params(axis="x", labelrotation=20)
    axes[-1].set_xlabel("Model")
    fig.suptitle("Point-level error boxplots by period")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_point_error_histograms(data: Dict[str, Dict[str, object]], save_path: Path) -> None:
    """Create histogram view from point-level nMAE values (all/61d/30d)."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True, sharey=True)
    phase_specs = [
        (None, "All data"),
        ("rolling_retrains", "Rolling retrains (61 days)"),
        ("frozen_deployment", "Frozen deployment (30 days)"),
    ]
    bins = np.linspace(0.0, 0.4, 81)

    for ax, (phase_filter, title) in zip(axes, phase_specs):
        for csv_name, _, _ in MODELS_ALL:
            entry = data[csv_name]
            df = entry["df"]
            if phase_filter is not None:
                df = df[df["phase"] == phase_filter]
            vals = df["nMAE"].clip(lower=0.0).dropna()

            ax.hist(
                vals,
                bins=bins,
                histtype="step",
                linewidth=1.4,
                density=True,
                color=entry["color"],
                label=entry["label"],
            )

        ax.set_title(title)
        ax.set_ylabel("Density")
        ax.set_xlim(0.0, 0.4)
        ax.grid(True, axis="both", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("nMAE")
    fig.suptitle("Point-level error histograms by period")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare_data()

    plot_all_models_scatter(
        data,
        OUTPUT_DIR / "overall_all_results_scatter_kw.png",
        phase_filter=None,
        title_suffix="all days",
    )
    plot_all_models_scatter(
        data,
        OUTPUT_DIR / "overall_all_results_scatter_kw_retrain61.png",
        phase_filter="rolling_retrains",
        title_suffix="rolling retrains (61 days)",
    )
    plot_all_models_scatter(
        data,
        OUTPUT_DIR / "overall_all_results_scatter_kw_frozen30.png",
        phase_filter="frozen_deployment",
        title_suffix="frozen deployment (30 days)",
    )
    plot_point_error_boxplots(
        data,
        OUTPUT_DIR / "overall_all_results_boxplots.png",
    )
    plot_point_error_histograms(
        data,
        OUTPUT_DIR / "overall_all_results_histograms.png",
    )


if __name__ == "__main__":
    main()
