"""Generates overall metric plots and saves them as png files."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "artifacts"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "overall"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config_overall.yaml"


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


def plot_metric_over_time_all_models(metric: str, data: dict, save_path: Path) -> None:
    """Plot one metric over all test days with one subplot per model."""

    ordered_model_csvs = [
        "physical_model_testday_metrics.csv",
        "source_model_testday_metrics.csv",
        "target_model_testday_metrics.csv",
        "transfer_model_testday_metrics.csv",
    ]
    model_order = [csv_name for csv_name in ordered_model_csvs if csv_name in data]

    n_models = len(model_order)
    fig, axes = plt.subplots(n_models, 1, figsize=(14, 12), sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes)

    highlight_models = {
        "target_model_testday_metrics.csv",
        "transfer_model_testday_metrics.csv",
    }

    y_min, y_max = 0.0, 0.3

    def plot_segment_mean(
        ax,
        x_seg: np.ndarray,
        y_seg: np.ndarray,
        color: str,
        linestyle: str,
        label_x: float,
        label_ha: str,
    ) -> None:
        valid = np.isfinite(y_seg)
        if valid.sum() == 0:
            return
        mean_val = float(np.nanmean(y_seg[valid]))
        ax.hlines(
            y=mean_val,
            xmin=float(x_seg[0]),
            xmax=float(x_seg[-1]),
            color=color,
            linestyle=linestyle,
            linewidth=1.4,
            alpha=0.95,
        )
        if mean_val > y_max - 0.02:
            y_label = mean_val - 0.006
            va = "top"
        else:
            y_label = mean_val + 0.006
            va = "bottom"
        ax.text(
            label_x,
            y_label,
            f"{mean_val:.3f}",
            ha=label_ha,
            va=va,
            fontsize=7,
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.15),
        )

    for idx, (ax, csv_name) in enumerate(zip(axes_flat, model_order)):
        entry = data[csv_name]
        df = entry["df"].copy().sort_values("date")
        df = df.drop_duplicates(subset=["date"], keep="last")

        n_days = len(df)
        x = np.arange(1, n_days + 1, dtype=float)
        y = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
        x_labels = df["date"].dt.strftime("%m-%d")

        highlight_days = min(30, n_days)
        highlight_start = n_days - highlight_days + 1
        first_days = min(61, n_days)
        last_days = min(30, n_days)

        if csv_name in highlight_models:
            ax.axvspan(highlight_start - 0.5, n_days + 0.5, color="lightgray", alpha=0.3, zorder=0)
        ax.plot(
            x,
            y,
            linestyle="-",
            linewidth=1.0,
            marker="o",
            markersize=3,
            color=entry["color"],
        )
        first_label_x = float((x[0] + x[first_days - 1]) / 2) if first_days > 0 else 1.0
        last_label_x = float((x[-last_days] + x[-1]) / 2) if last_days > 0 else float(n_days)
        plot_segment_mean(
            ax,
            x[:first_days],
            y[:first_days],
            color="#555555",
            linestyle="--",
            label_x=first_label_x,
            label_ha="center",
        )
        plot_segment_mean(
            ax,
            x[-last_days:],
            y[-last_days:],
            color="#222222",
            linestyle="-.",
            label_x=last_label_x,
            label_ha="center",
        )

        if csv_name in highlight_models and n_days > 0:
            transition_x = highlight_start - 0.5
            y_text = 0.98
            txt_style = dict(transform=ax.get_xaxis_transform(), va="top", fontsize=7, color="#444444")
            ax.text(
                transition_x - 0.6,
                y_text,
                f"n = {first_days} rolling retrains",
                ha="right",
                **txt_style,
            )
            ax.text(
                transition_x + 0.6,
                y_text,
                f"n = {last_days} frozen deployment days",
                ha="left",
                **txt_style,
            )

        ax.set_title(entry["label"])
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0.5, n_days + 0.5)
        ax.set_xticks(x)
        if idx == n_models - 1:
            ax.set_xticklabels(x_labels, rotation=90, ha="center", fontsize=7)
        else:
            ax.tick_params(axis="x", which="both", labelbottom=False)
        ax.grid(True, axis="y", alpha=0.3)

    for ax in axes_flat:
        ax.set_ylabel(metric)
    axes_flat[-1].set_xlabel("Testday (date)")

    fig.suptitle(f"{metric} over all test days per model")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
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

    # PNG Chronological per metric (4 subplots = 1 per model)
    plot_metric_over_time_all_models("nMAE", data, OUTPUT_DIR / "overall_metrics_chrono_nmae.png")
    plot_metric_over_time_all_models("nRMSE", data, OUTPUT_DIR / "overall_metrics_chrono_nrmse.png")


if __name__ == "__main__":
    main()
