from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = PROJECT_ROOT / "data" / "artifacts"
DEFAULT_CSVS = [
    DEFAULT_DIR / "physical_model_testday_metrics.csv",
    DEFAULT_DIR / "transfer_model_testday_metrics.csv",
    DEFAULT_DIR / "target_model_testday_metrics.csv",
    DEFAULT_DIR / "source_model_testday_metrics.csv",
]
EXPORT_DIR = PROJECT_ROOT / "reports" / "overall"


def add_testday_index(df: pd.DataFrame) -> pd.DataFrame:
    """Map dates to consecutive integers (testday = 1..N)."""
    out = df.copy()
    if "date" not in out.columns and "day" in out.columns:
        out = out.rename(columns={"day": "date"})
    if "date" not in out.columns:
        raise KeyError("Missing 'date' column in metrics CSV.")
    days_sorted = sorted(out["date"].unique())
    mapping = {d: i + 1 for i, d in enumerate(days_sorted)}
    out["testday"] = out["date"].map(mapping)
    return out


def plot_testday_boxplots(csv_path: Path, out_dir: Path) -> Path:
    df = pd.read_csv(csv_path)

    # normalize column names and numeric fields
    if "day" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"day": "date"})
    for col in ("nMAE", "nRMSE"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")

    df = add_testday_index(df)

    id_cols = ["date", "group", "testday"]
    df_long = df.melt(id_vars=id_cols, value_vars=["nMAE", "nRMSE"], var_name="Metric", value_name="Value")
    df_long["GroupMetric"] = df_long["group"] + " " + df_long["Metric"]

    order = [
        "all nMAE",
        "intra_day nMAE",
        "day_ahead nMAE",
        "all nRMSE",
        "intra_day nRMSE",
        "day_ahead nRMSE",
    ]

    palette_points = {
        "all nMAE": "#ff7f0e",
        "intra_day nMAE": "#2ca02c",
        "day_ahead nMAE": "#d62728",
        "all nRMSE": "#ff7f0e",
        "intra_day nRMSE": "#2ca02c",
        "day_ahead nRMSE": "#d62728",
    }

    sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.linewidth": 0.4})
    plt.rcParams.update(
        {
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "grid.color": "gray",
            "grid.alpha": 0.35,
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )

    fig = plt.figure(figsize=(10, 5), dpi=150)
    ax = plt.gca()

    sns.boxplot(
        data=df_long,
        x="GroupMetric",
        y="Value",
        order=order,
        width=0.6,
        showcaps=True,
        boxprops={"facecolor": "#4c72b0", "alpha": 0.35, "edgecolor": "black", "linewidth": 0.8},
        medianprops={"color": "black", "linewidth": 0.8},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        flierprops={"marker": "", "alpha": 0},
        ax=ax,
    )

    rng = np.random.default_rng(42)
    for i, key in enumerate(order):
        sub = df_long[df_long["GroupMetric"] == key]
        if sub.empty:
            continue

        jitter = (rng.random(len(sub)) - 0.5) * 0.35
        x_vals = np.full(len(sub), i) + jitter
        y_vals = sub["Value"].to_numpy()
        nums = sub["testday"].to_numpy()

        ax.scatter(
            x_vals,
            y_vals,
            color=palette_points[key],
            s=14,
            alpha=0.6,
            zorder=3,
            linewidths=0.3,
            edgecolors="black",
        )

        for x, y, n in zip(x_vals, y_vals, nums):
            ax.text(x, y, str(int(n)), ha="center", va="center", fontsize=7, color="black", zorder=4)

    ax.axvline(2.5, color="black", linestyle="--", alpha=0.3, linewidth=0.8)

    ax.set_title("Group-wise comparison of nMAE and nRMSE", fontsize=11)
    ax.set_xlabel("Forecast group and metric")
    ax.set_ylabel("Normalized error value")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.35)

    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{csv_path.stem}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Create test-day boxplots from a metrics CSV (writes PNG to reports/overall)."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        help="Optional: single CSV path. If omitted, all default testday metrics CSVs are processed.",
    )
    parser.add_argument("--show", action="store_true", help="Show plot interactively (after saving).")
    args = parser.parse_args(argv)

    targets = []
    if args.csv:
        targets = [Path(args.csv)]
    else:
        targets = DEFAULT_CSVS

    for csv_path in targets:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        out_path = plot_testday_boxplots(csv_path, EXPORT_DIR)
        print(f"Saved plot: {out_path}")

        if args.show:
            import matplotlib.image as mpimg

            img = mpimg.imread(out_path)
            plt.figure(figsize=(10, 5), dpi=150)
            plt.imshow(img)
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    main()
