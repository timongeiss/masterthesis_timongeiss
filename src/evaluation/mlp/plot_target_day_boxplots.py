# supplementary, no upload

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
DEFAULT_EXPORTS_DIR = PROJECT_ROOT / "data" / "results" / "target_output"

PLOT_CONFIG = {
    "figsize": (8, 6),
    "dpi": 150,
    "fontsize": 8,
    "linewidth": 1.0,
    "grid": True,
    "grid_style": ":",
    "grid_alpha": 0.5,
    "rotation": 90,
    "margin_frac": 0.01,
    "xtick_density": 45,
}

# Quick defaults for run-button usage; leave empty to rely on CLI args
SELECTED_DAY: Optional[str] = "2025-10-03"  # e.g. "2025-09-27"
SELECTED_OUTPUT: Optional[str] = ""  # e.g. "artifacts/day_boxplot.png"
SELECTED_EXPORTS_DIR: Optional[str] = ""  # default data/results/target_output

# What to plot per-timepoint across runs
BOXPLOT_SERIES = [
    ("aswdir_s_norm", "Direct irradiance (norm)", "#ffd900"),
    ("aswdifd_s_norm", "Diffuse irradiance (norm)", "#f70eff"),
    ("t2m_norm", "Temperature (norm)", "#0077ff"),
]

PV_COL_CANDS = ["y_pred"]  # MLP forecast
AVG_POWER_CANDS = ["y_true"]  # observed power (normalized)
SOL_ELEV_CANDS = ["solar_elevation_deg", "solar_elevation", "sol_elev_deg"]


def list_exports(directory: Path) -> List[Path]:
    return sorted(directory.glob("*.csv"))


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["valid_time"])
    return df.sort_values("valid_time").reset_index(drop=True)


def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def parse_day_arg(raw: str) -> pd.Timestamp:
    s = raw.strip()
    if len(s) == 8 and s.isdigit():
        return pd.to_datetime(f"{s[0:4]}-{s[4:6]}-{s[6:8]}")
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Ungültiges Datumsformat: '{raw}'. Erwarte yyyymmdd oder yyyy-mm-dd.")
    return ts.normalize()


def collect_day_frames_with_paths(exports_dir: Path, day: pd.Timestamp) -> List[Tuple[Path, pd.DataFrame]]:
    out: List[Tuple[Path, pd.DataFrame]] = []
    for fp in list_exports(exports_dir):
        df = load_data(fp)
        mask = df["valid_time"].dt.date == day.date()
        df_day = df.loc[mask].copy()
        if not df_day.empty:
            out.append((fp, df_day))
    return out


def build_time_union(frames: List[pd.DataFrame]) -> List[pd.Timestamp]:
    all_times = set()
    for df in frames:
        for t in df["valid_time"]:
            all_times.add(pd.Timestamp(t))
    return sorted(all_times)


def expected_day_times(day: pd.Timestamp) -> List[pd.Timestamp]:
    rng = pd.date_range(day.normalize(), day.normalize() + pd.Timedelta(days=1), freq="15T", inclusive="left")
    return [pd.Timestamp(t) for t in rng]


def _cols_for(df: pd.DataFrame) -> tuple[Optional[str], Optional[str], Optional[str]]:
    pv_col = find_col(df, PV_COL_CANDS)
    avg_col = find_col(df, AVG_POWER_CANDS)
    sol_col = find_col(df, SOL_ELEV_CANDS)
    return pv_col, avg_col, sol_col


def choose_complete_source(
    frames_with_paths: List[tuple[Path, pd.DataFrame]],
    exp_times: List[pd.Timestamp],
) -> Optional[tuple[pd.DataFrame, str, str]]:
    for _path, df in frames_with_paths:
        _pv, avg_col, sol_col = _cols_for(df)
        if avg_col is None or sol_col is None:
            continue
        s_avg = df.set_index("valid_time")[avg_col]
        s_sol = df.set_index("valid_time")[sol_col]
        if all(pd.notna(s_avg.get(t)) and pd.notna(s_sol.get(t)) for t in exp_times):
            return df, avg_col, sol_col
    return None


def choose_best_partial_source(
    frames_with_paths: List[tuple[Path, pd.DataFrame]],
    exp_times: List[pd.Timestamp],
) -> Optional[tuple[pd.DataFrame, str, str]]:
    best = None
    best_count = -1
    for _path, df in frames_with_paths:
        _pv, avg_col, sol_col = _cols_for(df)
        if avg_col is None or sol_col is None:
            continue
        s_avg = df.set_index("valid_time")[avg_col]
        s_sol = df.set_index("valid_time")[sol_col]
        cnt = sum(1 for t in exp_times if pd.notna(s_avg.get(t)) and pd.notna(s_sol.get(t)))
        if cnt > best_count:
            best = (df, avg_col, sol_col)
            best_count = cnt
    return best


def collect_series_by_time(
    frames: List[pd.DataFrame],
    times: List[pd.Timestamp],
    col_name: str,
) -> List[List[float]]:
    buckets: List[List[float]] = [[] for _ in times]
    lookups: List[Dict[pd.Timestamp, float]] = []
    for df in frames:
        series = df.set_index("valid_time")[col_name]
        lm: Dict[pd.Timestamp, float] = {}
        for t, v in series.items():
            try:
                if pd.notna(v):
                    lm[pd.Timestamp(t)] = float(v)
            except Exception:
                pass
        lookups.append(lm)
    for i, t in enumerate(times):
        for lm in lookups:
            if t in lm:
                buckets[i].append(lm[t])
    return buckets


def plot_day_boxplots(
    frames: List[pd.DataFrame],
    frame_paths: Optional[List[Path]],
    day: pd.Timestamp,
    title: Optional[str] = None,
) -> plt.Figure:
    if not frames:
        raise ValueError("Keine Tagesdaten gefunden (frames leer).")

    times = build_time_union(frames)
    if not times:
        raise ValueError("Keine Zeitstempel für diesen Tag gefunden.")
    exp_times = expected_day_times(day)

    frames_with_paths = list(zip(frame_paths or [Path("")] * len(frames), frames))
    chosen = choose_complete_source(frames_with_paths, exp_times) or choose_best_partial_source(
        frames_with_paths, exp_times
    )
    if chosen is None:
        raise ValueError("Keine Quelle mit vollständiger Solar elevation und y_true gefunden.")

    src_df, avg_col, sol_col = chosen
    sol_series = src_df.set_index("valid_time")[sol_col]

    # Only plot times with solar elevation > 0
    filtered_times = [t for t in times if pd.notna(sol_series.get(t)) and float(sol_series.get(t)) > 0]
    if not filtered_times:
        raise ValueError("Keine Zeitstempel mit Solar elevation > 0 gefunden.")
    times = filtered_times
    x = mdates.date2num(times)

    n_sub = len(BOXPLOT_SERIES) + 2
    fig, axes = plt.subplots(n_sub, 1, figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"], sharex=True)
    if n_sub == 1:
        axes = [axes]

    box_width = (15 / 60 / 24) * 0.8

    def _style_ax(ax, ylabel: str):
        ax.set_ylabel(ylabel, fontsize=PLOT_CONFIG["fontsize"])
        if PLOT_CONFIG["grid"]:
            ax.grid(True, linestyle=PLOT_CONFIG["grid_style"], alpha=PLOT_CONFIG["grid_alpha"])

    # Feature boxplots
    for idx, (col_key, label, color) in enumerate(BOXPLOT_SERIES):
        col = None
        for df in frames:
            col = find_col(df, [col_key])
            if col is not None:
                break
        if col is None:
            continue

        buckets = collect_series_by_time(frames, times, col)
        ax = axes[idx]
        bp = ax.boxplot(
            buckets,
            positions=x,
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
            whis=(5, 95),
            showfliers=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_color(color)
                line.set_alpha(0.9)
                line.set_linewidth(0.8)

        _style_ax(ax, label)

    # Power subplot: forecasts (y_pred) boxes, y_true line
    axp = axes[-2]
    pv_col = None
    for df in frames:
        pv_col = find_col(df, PV_COL_CANDS)
        if pv_col is not None:
            break
    pv_legend_handle = None
    if pv_col is not None:
        pv_buckets = collect_series_by_time(frames, times, pv_col)
        bp = axp.boxplot(
            pv_buckets,
            positions=x,
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
            whis=(5, 95),
            showfliers=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#0aa03b")
            patch.set_alpha(0.35)
            patch.set_edgecolor("#0aa03b")
        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_color("#0aa03b")
                line.set_alpha(0.9)
                line.set_linewidth(0.8)
        pv_legend_handle = Patch(facecolor="#0aa03b", edgecolor="#0aa03b", alpha=0.35, label="MLP y_pred")

    mean_line_handle = None
    s_avg = src_df.set_index("valid_time")[avg_col]
    y_avg = [float(s_avg.get(t)) if pd.notna(s_avg.get(t)) else None for t in times]
    axp.plot(times, y_avg, color="#333333", lw=PLOT_CONFIG["linewidth"] + 0.5, label="y_true")
    mean_line_handle = Line2D([0], [0], color="#333333", lw=PLOT_CONFIG["linewidth"] + 0.5, label="y_true")

    _style_ax(axp, "Power (norm)")

    legend_handles = [h for h in (pv_legend_handle, mean_line_handle) if h is not None]
    if legend_handles:
        axp.legend(handles=legend_handles, fontsize=PLOT_CONFIG["fontsize"], loc="upper right")

    # Solar elevation subplot
    axe = axes[-1]
    y_sol = [float(sol_series.get(t)) if pd.notna(sol_series.get(t)) else None for t in times]
    axe.plot(times, y_sol, color="#777777", lw=PLOT_CONFIG["linewidth"], label="Solar elevation [deg]")
    axe.set_ylabel("Solar elevation [deg]", fontsize=PLOT_CONFIG["fontsize"])
    axe.grid(True, linestyle=PLOT_CONFIG["grid_style"], alpha=PLOT_CONFIG["grid_alpha"])
    axe.legend(fontsize=PLOT_CONFIG["fontsize"], loc="upper right")

    xmin = min(times) - pd.Timedelta(days=PLOT_CONFIG["margin_frac"])
    xmax = max(times) + pd.Timedelta(days=PLOT_CONFIG["margin_frac"])
    axes[0].set_xlim(xmin, xmax)

    locator = mdates.AutoDateLocator(
        minticks=PLOT_CONFIG["xtick_density"] // 2,
        maxticks=PLOT_CONFIG["xtick_density"],
    )
    formatter = mdates.DateFormatter("%Y-%m-%d\n%H:%M")
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)
    plt.setp(axes[-1].get_xticklabels(), rotation=PLOT_CONFIG["rotation"], ha="center")

    ttl = title or f"MLP day overview — {day.date()} (N={len(frames)} runs)"
    fig.suptitle(ttl, fontsize=PLOT_CONFIG["fontsize"] + 2)
    fig.tight_layout()

    return fig


def main(argv: List[str]) -> None:
    day_arg = (SELECTED_DAY or "").strip()
    if not day_arg:
        if len(argv) < 2:
            print("Usage: python plot_target_day_boxplots.py <yyyymmdd|yyyy-mm-dd> [out.png] [exports_dir]")
            sys.exit(2)
        day_arg = argv[1]

    day = parse_day_arg(day_arg)

    out_arg = (SELECTED_OUTPUT or "").strip()
    if not out_arg and len(argv) > 2 and not argv[2].lower().endswith(".csv"):
        out_arg = argv[2]
    out_png: Optional[Path] = None

    dir_arg = (SELECTED_EXPORTS_DIR or "").strip()
    if not dir_arg:
        dir_arg = argv[3] if len(argv) > 3 else DEFAULT_EXPORTS_DIR
    exports_dir = Path(dir_arg)

    frames_with_paths = collect_day_frames_with_paths(exports_dir, day)
    if not frames_with_paths:
        raise FileNotFoundError(f"Keine passenden Tagesdaten in '{exports_dir}' gefunden für {day.date()}")
    frame_paths = [p for p, _ in frames_with_paths]
    frames = [df for _, df in frames_with_paths]

    title = f"MLP day overview - {day.date()}"
    fig = plot_day_boxplots(frames, frame_paths, day, title=title)

    if False:
        fig.savefig(Path("unused.png"))
    else:
        plt.show()


if __name__ == "__main__":
    main(sys.argv)
