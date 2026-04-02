from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = PROJECT_ROOT / "data" / "artifacts"
DEFAULT_CSVS = [
    DEFAULT_DIR / "physical_model_testday_metrics.csv",
    DEFAULT_DIR / "transfer_model_testday_metrics.csv",
    DEFAULT_DIR / "target_model_testday_metrics.csv",
    DEFAULT_DIR / "source_model_testday_metrics.csv",
]
EXPORT_DIR = PROJECT_ROOT / "reports" / "overall"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config_overall.yaml"


def load_overall_color_map() -> dict[str, str]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["model_colors"]


MODEL_COLORS = load_overall_color_map()


def _beeswarm_x_positions(
    y_vals: np.ndarray,
    center_x: float,
    x_half_width: float,
    min_y_sep: float,
    min_x_sep: float,
) -> np.ndarray:
    """Assign collision-aware x-positions for labels/points within one category."""
    n = len(y_vals)
    if n == 0:
        return np.array([], dtype=float)

    order = np.argsort(y_vals, kind="mergesort")
    x_out = np.full(n, center_x, dtype=float)
    placed: list[tuple[float, float]] = []

    max_steps = max(1, int(x_half_width / min_x_sep))
    offsets = [0.0]
    for step in range(1, max_steps + 1):
        d = step * min_x_sep
        offsets.extend([d, -d])

    for idx in order:
        y = float(y_vals[idx])
        chosen_x = center_x
        found = False
        for off in offsets:
            if abs(off) > x_half_width:
                continue
            candidate_x = center_x + off
            collides = False
            for px, py in placed:
                if abs(y - py) < min_y_sep and abs(candidate_x - px) < min_x_sep:
                    collides = True
                    break
            if not collides:
                chosen_x = candidate_x
                found = True
                break
        if not found:
            # Fallback in very dense areas
            chosen_x = center_x + np.clip((len(placed) % 5 - 2) * 0.01, -x_half_width, x_half_width)

        x_out[idx] = chosen_x
        placed.append((chosen_x, y))

    return x_out


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

    stem = csv_path.stem.lower()
    if "physical" in stem:
        box_color = MODEL_COLORS["physical_model"]
    elif "transfer" in stem:
        box_color = MODEL_COLORS["transfer_model"]
    elif "target" in stem:
        box_color = MODEL_COLORS["target_model"]
    elif "source" in stem:
        box_color = MODEL_COLORS["source_model"]
    else:
        box_color = "#4c72b0"

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
        boxprops={"facecolor": box_color, "alpha": 1.0, "edgecolor": "black", "linewidth": 0.8},
        medianprops={"color": "black", "linewidth": 0.8},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        flierprops={"marker": "", "alpha": 0},
        ax=ax,
    )

    y_min = float(np.nanmin(df_long["Value"]))
    y_max = float(np.nanmax(df_long["Value"]))
    y_span = max(1e-6, y_max - y_min)

    for i, key in enumerate(order):
        sub = df_long[df_long["GroupMetric"] == key].dropna(subset=["Value", "testday"])
        if sub.empty:
            continue

        x_vals = _beeswarm_x_positions(
            y_vals=sub["Value"].to_numpy(dtype=float),
            center_x=float(i),
            x_half_width=0.42,
            min_y_sep=y_span * 0.012,
            min_x_sep=0.055,
        )
        y_vals = sub["Value"].to_numpy()
        nums = sub["testday"].to_numpy()

        for x, y, n in zip(x_vals, y_vals, nums):
            ax.text(
                x,
                y,
                str(int(n)),
                ha="center",
                va="center",
                fontsize=5.5,
                color=("red" if int(n) >= 62 else "black"),
                zorder=4,
            )

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
