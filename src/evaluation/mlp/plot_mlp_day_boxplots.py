import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd


# ==========================================
# Plot configuration
# ==========================================
PLOT_CONFIG = {
    "figsize": (11.7, 8.3),
    "dpi": 180,
    "fontsize": 11,
    "linewidth": 1.2,
    "grid": True,
    "grid_style": ":",
    "grid_alpha": 0.5,
    "rotation": 0,
    "margin_frac": 0.01,
    "xtick_density": 12,
}

# Model selection in header (no CLI override).
SELECTED_MODEL_OUTPUT: str = "source_output"

# Optional override for input directory; keep empty to use data/results/<SELECTED_MODEL_OUTPUT>.
SELECTED_EXPORTS_DIR: Optional[str] = ""

# Day indexing starts at this date; this date is day 1.
DAY_INDEX_START_DATE: str = "2025-07-07"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "data" / "results"
REPORTS_ROOT = PROJECT_ROOT / "reports"

ALLOWED_MODEL_OUTPUTS = {"source_output", "target_output", "transfer_output"}

# MLP feature series to plot as boxplots.
BOXPLOT_SERIES = [
    ("aswdir_s_norm", "Direct (norm)", "#ffd900"),
    ("aswdifd_s_norm", "Diffuse (norm)", "#f70eff"),
    ("t2m_norm", "Temp. (norm)", "#0077ff"),
]

PV_COL_CANDS = ["y_pred"]
AVG_POWER_CANDS = ["y_true"]
SOL_ELEV_CANDS = ["solar_elevation_deg", "solar_elevation", "sol_elev_deg"]


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def model_tag_from_output(model_output: str) -> str:
    if model_output not in ALLOWED_MODEL_OUTPUTS:
        allowed = ", ".join(sorted(ALLOWED_MODEL_OUTPUTS))
        raise ValueError(
            f"Unsupported SELECTED_MODEL_OUTPUT='{model_output}'. Allowed values: {allowed}."
        )
    return model_output.replace("_output", "")


def list_exports(directory: Path) -> List[Path]:
    return sorted(directory.rglob("*.csv"))


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["valid_time"])
    required = {"valid_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in '{path}': {sorted(missing)}")

    numeric_candidates = [
        "y_true",
        "y_pred",
        "solar_elevation_deg",
        "solar_elevation",
        "sol_elev_deg",
        "aswdir_s_norm",
        "aswdifd_s_norm",
        "t2m_norm",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = _to_float(df[col])

    return df.sort_values("valid_time").reset_index(drop=True)


def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def list_available_days(frames_with_paths: List[Tuple[Path, pd.DataFrame]]) -> List[pd.Timestamp]:
    days = set()
    for _path, df in frames_with_paths:
        if "valid_time" not in df.columns:
            continue
        for t in df["valid_time"].dropna().unique():
            days.add(pd.Timestamp(t).normalize())
    return sorted(days)


def collect_day_frames_from_loaded(
    frames_with_paths: List[Tuple[Path, pd.DataFrame]],
    day: pd.Timestamp,
) -> List[Tuple[Path, pd.DataFrame]]:
    out: List[Tuple[Path, pd.DataFrame]] = []
    for fp, df in frames_with_paths:
        mask = df["valid_time"].dt.normalize() == day.normalize()
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
    rng = pd.date_range(day.normalize(), day.normalize() + pd.Timedelta(days=1), freq="15min", inclusive="left")
    return [pd.Timestamp(t) for t in rng]


def _cols_for(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    pv_col = find_col(df, PV_COL_CANDS)
    avg_col = find_col(df, AVG_POWER_CANDS)
    sol_col = find_col(df, SOL_ELEV_CANDS)
    return pv_col, avg_col, sol_col


def choose_complete_source(
    frames_with_paths: List[Tuple[Path, pd.DataFrame]],
    exp_times: List[pd.Timestamp],
) -> Optional[Tuple[pd.DataFrame, str, str]]:
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
    frames_with_paths: List[Tuple[Path, pd.DataFrame]],
    exp_times: List[pd.Timestamp],
) -> Optional[Tuple[pd.DataFrame, str, str]]:
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
        lookup: Dict[pd.Timestamp, float] = {}
        for t, v in series.items():
            try:
                if pd.notna(v):
                    lookup[pd.Timestamp(t)] = float(v)
            except Exception:
                pass
        lookups.append(lookup)

    for i, t in enumerate(times):
        for lookup in lookups:
            if t in lookup:
                buckets[i].append(lookup[t])
    return buckets


def plot_day_boxplots(
    frames: List[pd.DataFrame],
    frame_paths: Optional[List[Path]],
    day: pd.Timestamp,
    title: Optional[str] = None,
) -> plt.Figure:
    if not frames:
        raise ValueError("No day data found (frames is empty).")

    times = build_time_union(frames)
    if not times:
        raise ValueError("No timestamps found for this day.")
    exp_times = expected_day_times(day)

    frames_with_paths = list(zip(frame_paths or [Path("")] * len(frames), frames))
    chosen = choose_complete_source(frames_with_paths, exp_times)
    if chosen is None:
        chosen = choose_best_partial_source(frames_with_paths, exp_times)
    if chosen is None:
        raise ValueError("No source with valid solar elevation and y_true found.")

    src_df, avg_col, sol_col = chosen
    sol_series = src_df.set_index("valid_time")[sol_col]

    filtered_times = [t for t in times if pd.notna(sol_series.get(t)) and float(sol_series.get(t)) > 0.0]
    if not filtered_times:
        raise ValueError("No timestamps with solar elevation > 0 found.")
    times = filtered_times
    x = mdates.date2num(times)

    n_sub = len(BOXPLOT_SERIES) + 2
    fig, axes = plt.subplots(n_sub, 1, figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"], sharex=True)
    if n_sub == 1:
        axes = [axes]

    box_width = (15 / 60 / 24) * 0.8

    def _style_ax(ax: plt.Axes, ylabel: str) -> None:
        ax.set_ylabel(ylabel, fontsize=PLOT_CONFIG["fontsize"])
        ax.tick_params(axis="y", labelsize=PLOT_CONFIG["fontsize"] - 1)
        if PLOT_CONFIG["grid"]:
            ax.grid(True, linestyle=PLOT_CONFIG["grid_style"], alpha=PLOT_CONFIG["grid_alpha"])

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

    axp = axes[-2]
    pv_col = None
    for df in frames:
        pv_col = find_col(df, PV_COL_CANDS)
        if pv_col is not None:
            break

    pred_legend_handle = None
    if pv_col is not None:
        pred_buckets = collect_series_by_time(frames, times, pv_col)
        bp = axp.boxplot(
            pred_buckets,
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
        pred_legend_handle = Patch(facecolor="#0aa03b", edgecolor="#0aa03b", alpha=0.35, label="MLP y_pred")

    mean_line_handle = None
    s_avg = src_df.set_index("valid_time")[avg_col]
    y_avg = [float(s_avg.get(t)) if pd.notna(s_avg.get(t)) else None for t in times]
    axp.plot(times, y_avg, color="#333333", lw=PLOT_CONFIG["linewidth"] + 0.5, label="y_true")
    mean_line_handle = Line2D([0], [0], color="#333333", lw=PLOT_CONFIG["linewidth"] + 0.5, label="y_true")

    _style_ax(axp, "Power (norm)")

    legend_handles = [h for h in (pred_legend_handle, mean_line_handle) if h is not None]
    if legend_handles:
        axp.legend(handles=legend_handles, fontsize=PLOT_CONFIG["fontsize"], loc="upper right")

    axe = axes[-1]
    y_sol = [float(sol_series.get(t)) if pd.notna(sol_series.get(t)) else None for t in times]
    axe.plot(times, y_sol, color="#777777", lw=PLOT_CONFIG["linewidth"])
    axe.set_ylabel("Solar elev. [deg]", fontsize=PLOT_CONFIG["fontsize"])
    axe.grid(True, linestyle=PLOT_CONFIG["grid_style"], alpha=PLOT_CONFIG["grid_alpha"])
    axe.tick_params(axis="y", labelsize=PLOT_CONFIG["fontsize"] - 1)

    xmin = min(times) - pd.Timedelta(days=PLOT_CONFIG["margin_frac"])
    xmax = max(times) + pd.Timedelta(days=PLOT_CONFIG["margin_frac"])
    axes[0].set_xlim(xmin, xmax)

    locator = mdates.AutoDateLocator(
        minticks=PLOT_CONFIG["xtick_density"] // 2,
        maxticks=PLOT_CONFIG["xtick_density"],
    )
    formatter = mdates.DateFormatter("%H:%M")
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)
    plt.setp(
        axes[-1].get_xticklabels(),
        rotation=PLOT_CONFIG["rotation"],
        ha="center",
        fontsize=PLOT_CONFIG["fontsize"] - 1,
    )

    ttl = title or f"MLP day overview - {day.date()} (N={len(frames)} runs)"
    fig.suptitle(ttl, fontsize=PLOT_CONFIG["fontsize"] + 4)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def main(argv: List[str]) -> None:
    del argv  # no CLI options for model selection by design

    model_output = SELECTED_MODEL_OUTPUT.strip()
    model_tag = model_tag_from_output(model_output)

    if SELECTED_EXPORTS_DIR and SELECTED_EXPORTS_DIR.strip():
        exports_dir = Path(SELECTED_EXPORTS_DIR.strip())
    else:
        exports_dir = RESULTS_ROOT / model_output

    output_dir = REPORTS_ROOT / model_tag

    export_files = list_exports(exports_dir)
    if not export_files:
        raise FileNotFoundError(f"No CSV exports found in '{exports_dir}'.")

    all_frames_with_paths = [(fp, load_data(fp)) for fp in export_files]
    all_days = list_available_days(all_frames_with_paths)
    if not all_days:
        raise FileNotFoundError(f"No valid timestamps found in '{exports_dir}'.")

    start_day = pd.Timestamp(DAY_INDEX_START_DATE).normalize()
    all_days = [d for d in all_days if d >= start_day]
    if not all_days:
        raise FileNotFoundError(f"No days from start date {start_day.date()} found in '{exports_dir}'.")
    if all_days[0] != start_day:
        raise FileNotFoundError(
            f"Start date {start_day.date()} not found in data (first available day: {all_days[0].date()})."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    for day_index, day in enumerate(all_days, start=1):
        day_frames_with_paths = collect_day_frames_from_loaded(all_frames_with_paths, day)
        if not day_frames_with_paths:
            skipped += 1
            print(f"Day {day_index} ({day:%Y-%m-%d}) skipped: no day data.")
            continue

        frame_paths = [p for p, _ in day_frames_with_paths]
        frames = [df for _, df in day_frames_with_paths]
        title = f"{model_tag.capitalize()} MLP day overview - {day.date()}"

        try:
            fig = plot_day_boxplots(frames, frame_paths, day, title=title)
            out_png = output_dir / f"{day_index}_day_boxplots_{day:%Y%m%d}_{model_tag}.png"
            fig.savefig(out_png)
            plt.close(fig)
            saved += 1
            print(f"Saved plot: '{out_png}'.")
        except Exception as exc:
            skipped += 1
            print(f"Day {day_index} ({day:%Y-%m-%d}) skipped: {exc}")

    print(f"Done. Saved: {saved}, skipped: {skipped}, total days: {len(all_days)}.")


if __name__ == "__main__":
    main(sys.argv)
