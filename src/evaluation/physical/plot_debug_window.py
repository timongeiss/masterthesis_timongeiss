import argparse
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


METRICS: List[Tuple[str, str, str, bool]] = [
    ("aswdir_s_Wm2", "aswdir_s [W/m^2]", "#ffd900", False),
    ("dni_Wm2", "DNI [W/m^2]", "#ffd900", False),
    ("cos_zenith", "cos(zenith) [-]", "#777777", False),
    ("zenith_deg", "Zenith [deg]", "#777777", False),
    ("transposed_irradiance_Wm2", "Transp.[W/m^2]", "#ffd900", True),
    ("poa_global_Wm2", "POA [W/m^2]", "#ffd900", True),
    ("iam", "IAM [-]", "#777777", True),
    ("pdc_W", "DC power [W]", "#0aa03b", True),
    ("ac_W", "AC power [W]", "#333333", True),
]

SITE_STYLES: List[Tuple[str, str]] = [
    ("-", "o"),
    ("--", "s"),
    (":", "^"),
    ("-.", "D"),
    ((0, (3, 1, 1, 1)), "v"),
    ((0, (5, 1)), "P"),
]

PLOT_CONFIG = {
    "figsize": (11.7, 14.0),
    "dpi": 200,
    "fontsize": 12,
    "ticksize": 11,
    "linewidth": 1.8,
    "grid_alpha": 0.35,
}
NWP_SOURCE_INDEX = 0
DISCRETE_FREQ = "15min"
SUPPRESS_VALUE_LABEL_SITES = {"2", "3"}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = PROJECT_ROOT / "data" / "results" / "physical_output"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "physical"
WINDOW_START = pd.Timestamp("2025-08-19 16:00:00")
WINDOW_END = pd.Timestamp("2025-08-19 19:00:00")


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


def _site_style_map(sites: List[str]) -> Dict[str, Tuple[str, str]]:
    style_cycle = cycle(SITE_STYLES)
    return {site: next(style_cycle) for site in sites}


def _suppress_value_labels_for_site(site: object) -> bool:
    s = str(site).strip().lower()
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits in SUPPRESS_VALUE_LABEL_SITES or s in {"site2", "site_2", "site3", "site_3"}


def _select_source_and_discretize(
    df: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> tuple[pd.DataFrame, str]:
    sources = sorted(df["source"].dropna().unique())
    if not sources:
        raise ValueError("No source information in debug data.")

    preferred_source = None
    if 0 <= NWP_SOURCE_INDEX < len(sources):
        preferred_source = str(sources[NWP_SOURCE_INDEX])

    expected_times = pd.date_range(window_start, window_end, freq=DISCRETE_FREQ, inclusive="both")
    expected_sites = sorted(df["site"].dropna().unique())

    def _coverage_stats(source_name: str) -> tuple[int, int]:
        s = df[df["source"] == source_name].copy()
        s["timestamp"] = s["timestamp"].dt.floor(DISCRETE_FREQ)
        s = s[(s["timestamp"] >= window_start) & (s["timestamp"] <= window_end)]
        rows_in_window = len(s)
        if rows_in_window == 0:
            return len(expected_times) * max(1, len(expected_sites)), 0

        missing_total = 0
        expected_set = set(pd.Timestamp(t) for t in expected_times)
        for site in expected_sites:
            site_df = s[s["site"] == site]
            present = set(pd.Timestamp(t) for t in site_df["timestamp"].dropna().unique())
            missing_total += len(expected_set - present)
        return missing_total, rows_in_window

    stats = []
    for source_name in sorted((str(s) for s in sources), reverse=True):
        missing_total, rows_in_window = _coverage_stats(source_name)
        stats.append(
            {"source": source_name, "missing": missing_total, "rows": rows_in_window},
        )

    complete_sources = [s for s in stats if s["missing"] == 0 and s["rows"] > 0]
    if preferred_source is not None:
        preferred_complete = [s for s in complete_sources if s["source"] == preferred_source]
        if preferred_complete:
            selected_source = preferred_source
        elif complete_sources:
            selected_source = max(complete_sources, key=lambda s: (s["rows"], s["source"]))["source"]
        else:
            best = min(stats, key=lambda s: (s["missing"], -s["rows"], s["source"]))
            raise ValueError(
                "No source has complete 15-minute coverage in the selected window "
                f"({window_start} to {window_end}). "
                f"Best available: {best['source']} (missing points across sites: {best['missing']})."
            )
    else:
        if complete_sources:
            selected_source = max(complete_sources, key=lambda s: (s["rows"], s["source"]))["source"]
        else:
            best = min(stats, key=lambda s: (s["missing"], -s["rows"], s["source"]))
            raise ValueError(
                "No source has complete 15-minute coverage in the selected window "
                f"({window_start} to {window_end}). "
                f"Best available: {best['source']} (missing points across sites: {best['missing']})."
            )

    out = df[df["source"] == selected_source].copy()

    out["timestamp"] = out["timestamp"].dt.floor(DISCRETE_FREQ)
    agg_cols = [col for col, _label, _color, _dist in METRICS if col in out.columns]
    out = (
        out.groupby(["site", "timestamp"], as_index=False)[agg_cols]
        .mean(numeric_only=True)
        .sort_values(["timestamp", "site"])
    )
    return out, selected_source


def plot_debug(df: pd.DataFrame, out_dir: Path) -> Path:
    df, selected_source = _select_source_and_discretize(df, WINDOW_START, WINDOW_END)
    window_mask = (df["timestamp"] >= WINDOW_START) & (df["timestamp"] <= WINDOW_END)
    df = df[window_mask].copy()
    if df.empty:
        raise ValueError("No debug data in selected time window.")

    sites = sorted(df["site"].unique())
    site_style_map = _site_style_map(sites)

    fig, axes = plt.subplots(
        len(METRICS),
        1,
        figsize=PLOT_CONFIG["figsize"],
        dpi=PLOT_CONFIG["dpi"],
        sharex=True,
    )
    if len(METRICS) == 1:
        axes = [axes]

    first_site_distinction_axis = None

    for idx_metric, (ax, (col, label, metric_color, distinguish_sites)) in enumerate(zip(axes, METRICS)):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing in debug file.")

        metric_sites = sites if distinguish_sites else sites[:1]
        for site in metric_sites:
            site_df = df[df["site"] == site]
            if site_df.empty:
                continue

            linestyle, marker = site_style_map[site]
            if not distinguish_sites:
                linestyle = "-"
                marker = "o"

            ax.plot(
                site_df["timestamp"],
                site_df[col],
                linewidth=PLOT_CONFIG["linewidth"],
                linestyle=linestyle,
                marker=marker,
                markersize=4 if marker else 0,
                color=metric_color,
                alpha=0.95,
                label=site if (distinguish_sites and idx_metric == 4) else None,
            )

            if not _suppress_value_labels_for_site(site):
                for ts, val in zip(site_df["timestamp"], site_df[col]):
                    if pd.notna(val):
                        ax.annotate(
                            f"{float(val):.2f}",
                            xy=(ts, float(val)),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=max(7, PLOT_CONFIG["ticksize"] - 2),
                            color=metric_color,
                            alpha=0.95,
                        )

        ax.set_ylabel(label, fontsize=PLOT_CONFIG["fontsize"])
        ax.tick_params(axis="y", labelsize=PLOT_CONFIG["ticksize"])
        ax.grid(True, linestyle="--", alpha=PLOT_CONFIG["grid_alpha"])

        if distinguish_sites and first_site_distinction_axis is None:
            first_site_distinction_axis = idx_metric

    if first_site_distinction_axis is not None:
        legend_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=1.8,
                linestyle=site_style_map[site][0],
                marker=site_style_map[site][1],
                markersize=5,
                label=str(site),
            )
            for site in sites
        ]
        axes[first_site_distinction_axis].legend(
            handles=legend_handles,
            title="Site",
            loc="upper right",
            fontsize=PLOT_CONFIG["ticksize"],
            title_fontsize=PLOT_CONFIG["fontsize"],
        )

    axes[-1].set_xlabel("Time", fontsize=PLOT_CONFIG["fontsize"])
    quarter_hour_locator = mdates.MinuteLocator(byminute=[0, 15, 30, 45])
    formatter = mdates.DateFormatter("%H:%M")
    for ax in axes:
        ax.xaxis.set_major_locator(quarter_hour_locator)
        ax.set_xlim(WINDOW_START, WINDOW_END)
    axes[-1].xaxis.set_major_formatter(formatter)
    plt.setp(axes[-1].get_xticklabels(), rotation=0, ha="center", fontsize=PLOT_CONFIG["ticksize"])

    fig.suptitle(
        (
            f"Debug window per site ({WINDOW_START:%Y-%m-%d %H:%M} - {WINDOW_END:%H:%M})\n"
            f"NWP source: {selected_source}"
        ),
        fontsize=PLOT_CONFIG["fontsize"] + 6,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"debug_window_{WINDOW_START:%Y%m%d_%H%M}-{WINDOW_END:%H%M}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot debug exports for the selected debug window.")
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
        window_tag = WINDOW_START.strftime("%Y%m%d")
        files = list(INPUT_DIR.glob(f"*_debug_{window_tag}.csv"))
        if not files:
            files = list(INPUT_DIR.glob("*_debug_*.csv"))
        if not files:
            raise FileNotFoundError(f"No *_debug_*.csv files found in {INPUT_DIR}.")

    df = load_debug_frames(files)
    out_path = plot_debug(df, Path(args.outdir))
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
