import argparse
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import cycle


METRICS = [
    ("aswdir_s_Wm2", "aswdir_s [W/m²]"),
    ("dni_Wm2", "DNI [W/m²]"),
    ("cos_zenith", "cos(zenith) [-]"),
    ("zenith_deg", "Zenith [deg]"),
    ("transposed_irradiance_Wm2", "Transposed irradiance [W/m²]"),
    ("poa_global_Wm2", "POA global [W/m²]"),
    ("iam", "IAM [-]"),
    ("pdc_W", "DC power [W]"),
    ("ac_W", "AC power [W]"),
]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = PROJECT_ROOT / "data" / "results" / "physical_output"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "physical"
WINDOW_START = pd.Timestamp("2025-08-26 17:30:00")
WINDOW_END = pd.Timestamp("2025-08-26 18:30:00")


def load_debug_frames(files: List[Path]) -> pd.DataFrame:
    frames = []
    for fp in files:
        if not fp.exists():
            raise FileNotFoundError(f"Debug file not found: {fp}")
        df = pd.read_csv(fp, parse_dates=["timestamp"])
        df["source"] = fp.name
        frames.append(df)
    if not frames:
        raise ValueError("No debug files loaded.")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["timestamp", "site", "source"])
    return combined


def plot_debug(df: pd.DataFrame, out_dir: Path) -> Path:
    window_mask = (df["timestamp"] >= WINDOW_START) & (df["timestamp"] <= WINDOW_END)
    df = df[window_mask].copy()

    sites = sorted(df["site"].unique())
    color_cycle = cycle(cm.get_cmap("tab10").colors)
    color_map = {site: next(color_cycle) for site in sites}

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(9, 2.4 * len(METRICS)), sharex=True)
    if len(axes) == 1:
        axes = [axes]

    for idx_site, site in enumerate(sites):
        site_df = df[df["site"] == site]
        time = site_df["timestamp"]
        color = color_map[site]
        for idx_metric, (ax, (col, label)) in enumerate(zip(axes, METRICS)):
            if col not in site_df.columns:
                raise KeyError(f"Column '{col}' missing in debug file.")
            ax.plot(
                time,
                site_df[col],
                marker="o",
                linewidth=1,
                color=color,
                label=site if idx_metric == 0 else None,
            )
            ax.set_ylabel(label)
            ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].legend(title="Site", loc="best")
    axes[-1].set_xlabel("Timestamp")
    fig.suptitle("Debug window per site")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"debug_window_{WINDOW_START:%Y%m%d_%H%M}-{WINDOW_END:%H%M}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot debug exports for the 2025-08-26 window.")
    parser.add_argument(
        "files",
        nargs="*",
        help="Paths to *_debug_20250826.csv files. "
        f"Defaults to all matching files in {INPUT_DIR}.",
    )
    parser.add_argument(
        "--outdir",
        default=str(OUTPUT_DIR),
        help=f"Output directory for the plot (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = list(INPUT_DIR.glob("*_debug_20250826.csv"))
        if not files:
            raise FileNotFoundError(f"No *_debug_20250826.csv files found in {INPUT_DIR}.")

    df = load_debug_frames(files)
    out_path = plot_debug(df, Path(args.outdir))
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
