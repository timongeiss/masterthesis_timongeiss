"""Generates three different plot types, in total five png"""

from pathlib import Path
import matplotlib.pyplot as plt
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


def plot_boxplots(ax, metric: str, data: dict) -> None:
    
    labels = []
    box_data = []
    colors = []

    for csv_name, _, _ in MODELS:
        entry = data[csv_name]
        labels.append(entry["label"])
        box_data.append(entry["df"][metric].dropna())
        colors.append(entry["color"])

    bp = ax.boxplot(box_data, patch_artist=True, tick_labels=labels)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(f"{metric} Boxplot over all testdays for each model")


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

    # PNG Boxplots
    fig_box, box_axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    plot_boxplots(box_axes[0], "nMAE", data)
    plot_boxplots(box_axes[1], "nRMSE", data)
    box_axes[1].set_xlabel("Models")

    fig_box.tight_layout()
    plt.savefig(OUTPUT_DIR / "overall_metrics_boxplots.png", dpi=150)
    plt.close(fig_box)


if __name__ == "__main__":
    main()
