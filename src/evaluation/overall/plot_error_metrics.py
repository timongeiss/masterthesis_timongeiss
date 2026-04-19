"""Generates overall metric plots and saves them as png files."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import pandas as pd
import yaml


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "artifacts"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "overall"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config_overall.yaml"
CHRONO_START_DATE = pd.Timestamp("2025-07-07")
CHRONO_END_DATE = pd.Timestamp("2025-10-05")  # day 91


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["models"], cfg["transfer_csv"], cfg["model_colors"]


MODELS, TRANSFER_CSV, MODEL_COLORS = load_config()



# ------------------------
# Helper
# ------------------------


def load_metrics(csv_name: str) -> pd.DataFrame:
    """Load one csv, keep group=='all', normalize columns and sorting."""
    
    csv_path = DATA_DIR / csv_name
    df = pd.read_csv(csv_path)

    # Normalize column names (pysical anderer spaltenname)
    if "day" in df.columns:
        df = df.rename(columns={"day": "date"})

    # Convert decimal comma/dot to float
    for col in ("nMAE", "nRMSE"):
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")

    df["date"] = pd.to_datetime(df["date"])
    df = df[df["group"] == "all"]
    return df[["date", "nMAE", "nRMSE"]].sort_values("date")


def prepare_data():
    """Load all model data."""
    
    data = {}

    for csv_name, label, color_key in MODELS:
        df = load_metrics(csv_name)
        data[csv_name] = {"df": df, "label": label, "color": MODEL_COLORS[color_key]}

    return data


def order_days_by_metric(data: dict, metric: str):
    """Sort days by the metric of the Transfer Learning model (ascending)."""
    
    transfer_df = data[TRANSFER_CSV]["df"]
    ordered_dates = transfer_df.sort_values(metric)["date"].tolist()    # Transfer‑Tage aufsteigend nach der gewählten Metrik sortiert
    day_rank = {date: idx + 1 for idx, date in enumerate(ordered_dates)}    # Dict: jeder Datumswert bekommt seine Rangnummer (1, 2, 3, …)

    # Original Tagnummern in zeitlicher Reihenfolge (wie in CSV)
    original_dates = sorted(transfer_df["date"].unique())
    original_day_num = {date: idx + 1 for idx, date in enumerate(original_dates)}   #erstellt (chronologische Reihenfolge)

    return ordered_dates, day_rank, original_day_num

def get_chrono_day_numbers(data: dict):
    """Original Tagnummern in zeitlicher Reihenfolge (aus Transfer-Daten)."""
    
    transfer_df = data[TRANSFER_CSV]["df"]
    dates = sorted(transfer_df["date"].unique())
    day_num = {date: idx + 1 for idx, date in enumerate(dates)}
    return dates, day_num



# ------------------------
# Different Plots
# ------------------------

def plot_metric(metric: str, data: dict, save_path: Path) -> None:
    """Plot der die Absoluten abweichungen vom TL Modell darstellt"""
    
    ordered_dates, day_rank, original_day_num = order_days_by_metric(data, metric)

    fig, ax = plt.subplots(figsize=(10, 7))

    for csv_name, _, _ in MODELS:
        entry = data[csv_name]
        df = entry["df"].copy()
        df["day_num"] = df["date"].map(day_rank)
        df = df.sort_values("day_num")

        line_style = "-" if csv_name == TRANSFER_CSV else ""
        line_width = 0.8 if csv_name == TRANSFER_CSV else 0
        ax.plot(
            df["day_num"],
            df[metric],
            linestyle=line_style,
            linewidth=line_width,
            marker="o",
            markersize=4,
            label=entry["label"],
            color=entry["color"],
        )

    ax.set_ylabel(metric)
    ax.set_xlabel("Testday number")
    ax.set_ylim(0.0, 0.3)
    num_days = len(ordered_dates)
    ax.set_xticks(range(1, num_days + 1))
    ax.set_xticklabels([original_day_num[d] for d in ordered_dates])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    ax.set_title(f"{metric} per day")

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)



def plot_differences_single_model(model_csv: str, data: dict, save_path: Path) -> None:
    """Plot Abstaende vs. Transfer fuer ein Modell (nMAE und nRMSE in einem Bild)."""
    
    dates, day_num = get_chrono_day_numbers(data)
    transfer_df = data[TRANSFER_CSV]["df"].copy()
    transfer_map_nmae = dict(zip(transfer_df["date"], transfer_df["nMAE"]))
    transfer_map_nrmse = dict(zip(transfer_df["date"], transfer_df["nRMSE"]))

    entry = data[model_csv]
    df = entry["df"].copy()
    df["day_num"] = df["date"].map(day_num)
    df = df.sort_values("day_num")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    metrics = [("nMAE", "Delta nMAE"), ("nRMSE", "Delta nRMSE")]

    for ax, (metric, ylabel) in zip(axes, metrics):
        transfer_map_metric = transfer_map_nmae if metric == "nMAE" else transfer_map_nrmse
        df["diff"] = df.apply(lambda row: row[metric] - transfer_map_metric.get(row["date"], float("nan")), axis=1)
        ax.plot(
            df["day_num"],
            df["diff"],
            linestyle="",
            marker="o",
            markersize=4,
            color=entry["color"],
        )
        ax.axhline(0, color="blue", linewidth=1)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)

    num_days = len(dates)
    axes[-1].set_xticks(range(1, num_days + 1))
    axes[-1].set_xlabel("Testday number")
    fig.suptitle(f"Win/ Loss to the transfer model: {entry['label']}")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_metrics_over_time_all_models(data: dict, save_path: Path) -> None:
    """Plot both metrics in one DIN A4 figure with 8 stacked subplots in one column."""

    ordered_model_csvs = [
        "physical_model_testday_metrics.csv",
        "source_model_testday_metrics.csv",
        "target_model_testday_metrics.csv",
        "transfer_model_testday_metrics.csv",
    ]
    model_order = [csv_name for csv_name in ordered_model_csvs if csv_name in data]
    tick_days = [1, 14, 28, 42, 56, 70, 84, 91]
    tick_labels = [(CHRONO_START_DATE + pd.Timedelta(days=d - 1)).strftime("%d.%m.") for d in tick_days]
    subplot_specs = [(csv_name, "nMAE") for csv_name in model_order] + [(csv_name, "nRMSE") for csv_name in model_order]

    fig, axes = plt.subplots(
        len(subplot_specs),
        1,
        figsize=(8.27, 11.69),  # DIN A4 portrait
        dpi=300,
        sharex=True,
    )
    axes_flat = np.atleast_1d(axes)

    highlight_models = {
        "target_model_testday_metrics.csv",
        "transfer_model_testday_metrics.csv",
    }

    def plot_segment_median(
        ax, x_seg: np.ndarray, y_seg: np.ndarray, color: str, linestyle: str
    ) -> tuple[float, float, float] | None:
        valid = np.isfinite(y_seg)
        if valid.sum() == 0:
            return None
        xv = x_seg[valid]
        yv = y_seg[valid]
        median_val = float(np.nanmedian(yv))
        xmin = float(np.min(xv))
        xmax = float(np.max(xv))
        ax.hlines(
            y=median_val,
            xmin=xmin,
            xmax=xmax,
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            alpha=0.9,
        )
        return median_val, xmin, xmax

    filtered_by_model: dict[str, pd.DataFrame] = {}
    for csv_name in model_order:
        entry = data[csv_name]
        df = entry["df"].copy().sort_values("date")
        df = df.drop_duplicates(subset=["date"], keep="last")
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df[(df["date"] >= CHRONO_START_DATE) & (df["date"] <= CHRONO_END_DATE)].copy()
        df["day_num"] = (df["date"] - CHRONO_START_DATE).dt.days + 1
        df = df[(df["day_num"] >= 1) & (df["day_num"] <= 91)].copy()
        filtered_by_model[csv_name] = df.sort_values("day_num").drop_duplicates(subset=["day_num"], keep="last")

    metric_limits: dict[str, tuple[float, float]] = {}
    for metric_key in ("nMAE", "nRMSE"):
        vals = []
        for csv_name in model_order:
            arr = pd.to_numeric(filtered_by_model[csv_name][metric_key], errors="coerce").to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                vals.append(arr)
        if not vals:
            metric_limits[metric_key] = (0.0, 1.0)
            continue
        all_vals = np.concatenate(vals)
        vmin = float(np.nanmin(all_vals))
        vmax = float(np.nanmax(all_vals))
        if np.isclose(vmin, vmax):
            delta = max(1e-6, abs(vmin) * 0.05, 0.005)
            metric_limits[metric_key] = (vmin - delta, vmax + delta)
        else:
            metric_limits[metric_key] = (vmin, vmax)

    for plot_idx, (csv_name, metric_key) in enumerate(subplot_specs):
        ax = axes_flat[plot_idx]
        entry = data[csv_name]
        df = filtered_by_model[csv_name]

        x = df["day_num"].to_numpy(dtype=float)
        y = pd.to_numeric(df[metric_key], errors="coerce").to_numpy(dtype=float)
        if x.size == 0:
            continue

        first_mask = x <= 61
        last_mask = x >= 62

        y_min, y_max = metric_limits[metric_key]
        if csv_name in highlight_models:
            ax.axvspan(61.5, 91.5, color="lightgray", alpha=0.3, zorder=0)

        ax.plot(
            x,
            y,
            linestyle="-",
            linewidth=1,
            marker="o",
            markersize=1,
            color="#8B8B8B",
        )
        # Median lines over the fixed day intervals 1-61 and 62-91.
        seg_first = plot_segment_median(ax, x[first_mask], y[first_mask], color=entry["color"], linestyle=":")
        seg_last = plot_segment_median(ax, x[last_mask], y[last_mask], color=entry["color"], linestyle=":")

        ax.set_xlim(0.5, 91.5)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(tick_days)
        ax.set_xticklabels(tick_labels)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=8.8)
        ax.tick_params(axis="x", labelsize=8.8)

        y_span = max(1e-9, y_max - y_min)
        y_off = y_span * 0.015
        if seg_first is not None:
            med, xmin, xmax = seg_first
            ax.text(
                (xmin + xmax) / 2.0,
                med + y_off,
                f"{med:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.0,
                color=entry["color"],
            )
        if seg_last is not None:
            med, xmin, xmax = seg_last
            ax.text(
                (xmin + xmax) / 2.0,
                med + y_off,
                f"{med:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.0,
                color=entry["color"],
            )

        ax.set_ylabel(metric_key, fontsize=10.0)
        ax.text(
            0.01,
            0.95,
            entry["label"],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            color="#222222",
        )

        if plot_idx < len(subplot_specs) - 1:
            ax.tick_params(axis="x", which="both", labelbottom=False)

        if csv_name in highlight_models:
            txt_style = dict(transform=ax.get_xaxis_transform(), va="top", fontsize=7.6, color="#444444")
            ax.text(
                60.8,
                0.98,
                "n = 61 rolling retrains",
                ha="right",
                **txt_style,
            )
            ax.text(
                62.2,
                0.98,
                "n = 30 frozen deployment days",
                ha="left",
                **txt_style,
            )


    fig.tight_layout(rect=(0, 0, 1, 0.98))
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_boxplots(data: dict, save_path: Path) -> None:
    """Create 2 boxplot subplots with 8 boxes each (4 models x 2 periods)."""

    ordered_model_csvs = [
        "physical_model_testday_metrics.csv",
        "source_model_testday_metrics.csv",
        "target_model_testday_metrics.csv",
        "transfer_model_testday_metrics.csv",
    ]
    model_order = [csv_name for csv_name in ordered_model_csvs if csv_name in data]

    fig, axes = plt.subplots(2, 1, figsize=(8.27, 6.69), sharex=False, sharey=True)
    subplot_specs = [("nMAE", "nMAE"), ("nRMSE", "nRMSE")]

    def compact_model_label(label: str) -> str:
        compact = label.replace("Model", "").replace("Learning", "")
        return " ".join(compact.split())

    for ax, (metric, title) in zip(axes, subplot_specs):
        labels_61 = []
        box_data_61 = []
        colors_61 = []
        labels_30 = []
        box_data_30 = []
        colors_30 = []

        for csv_name in model_order:
            entry = data[csv_name]
            df = entry["df"].copy().sort_values("date").drop_duplicates(subset=["date"], keep="last")
            n_days = len(df)
            first_days = min(61, n_days)
            last_days = min(30, n_days)

            values_first = df.iloc[:first_days][metric].dropna()
            values_last = df.iloc[-last_days:][metric].dropna()

            short_label = compact_model_label(entry["label"])
            labels_61.append(short_label)
            box_data_61.append(values_first)
            colors_61.append(entry["color"])
            labels_30.append(short_label)
            box_data_30.append(values_last)
            colors_30.append(entry["color"])

        labels = labels_61 + labels_30
        box_data = box_data_61 + box_data_30
        colors = colors_61 + colors_30

        median_values = []
        for values in box_data:
            if len(values) == 0:
                median_values.append(float("nan"))
            else:
                median_values.append(float(np.nanmedian(values)))

        tick_labels_with_median = [
            f"{label}\n{median_val:.3f}" if np.isfinite(median_val) else f"{label}\n-"
            for label, median_val in zip(labels, median_values)
        ]

        bp = ax.boxplot(box_data, patch_artist=True, tick_labels=tick_labels_with_median)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

        split_index = len(model_order)
        if split_index > 0:
            highlight_30_models = {
                "target_model_testday_metrics.csv",
                "transfer_model_testday_metrics.csv",
            }
            highlight_positions = [
                split_index + model_order.index(csv_name) + 1
                for csv_name in highlight_30_models
                if csv_name in model_order
            ]
            if highlight_positions:
                span_start = min(highlight_positions) - 0.5
                span_end = max(highlight_positions) + 0.5
                ax.axvspan(span_start, span_end, color="lightgray", alpha=0.25, zorder=0)
                ax.text(
                    (span_start + span_end) / 2,
                    0.96,
                    "frozen deployment days",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=10,
                    color="#555555",
                )

            ax.axvline(split_index + 0.5, color="#666666", linewidth=1.0, linestyle="--", alpha=0.9)


        ax.set_title(
            f"{title}: retraining days 1 to 61 vs. deployment days 62 to 91",
            fontsize=14,
            pad=12,
        )
        ax.set_ylabel(metric, fontsize=13)
        ax.set_ylim(0.0, 0.25)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=0, labelsize=12, pad=8)
        ax.tick_params(axis="y", labelsize=12)

    axes[-1].set_xlabel("Model", fontsize=13, labelpad=10)
    fig.tight_layout(pad=1.8)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


# ------------------------
# MAIN
# ------------------------

def main() -> None:
    data = prepare_data()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # PNG Testtage Absolut
    plot_metric("nMAE", data, OUTPUT_DIR / "overall_metrics_per_day_nmae.png")
    plot_metric("nRMSE", data, OUTPUT_DIR / "overall_metrics_per_day_nrmse.png")
    
    # PNG Testtage Win/Loss
    plot_differences_single_model(
        "physical_model_testday_metrics.csv", data, OUTPUT_DIR / "overall_metrics_diff_physical.png"
    )
    plot_differences_single_model(
        "source_model_testday_metrics.csv", data, OUTPUT_DIR / "overall_metrics_diff_source.png"
    )
    plot_differences_single_model(
        "target_model_testday_metrics.csv", data, OUTPUT_DIR / "overall_metrics_diff_target.png"
    )

    # PNG Boxplots (split by period: first 61 and last 30 days)
    plot_boxplots(data, OUTPUT_DIR / "overall_metrics_boxplots.png")

    # PNG Chronological combined (8 subplots = 4 models x 2 metrics) up to day 91
    plot_metrics_over_time_all_models(data, OUTPUT_DIR / "overall_metrics_chrono_combined.png")


if __name__ == "__main__":
    main()
