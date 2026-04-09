from __future__ import annotations

import sys
from dataclasses import dataclass
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
    "figsize": (11.7, 13.5),
    "dpi": 180,
    "fontsize": 10,
    "linewidth": 1.2,
    "grid": True,
    "grid_style": ":",
    "grid_alpha": 0.5,
    "rotation": 0,
    "margin_frac": 0.01,
    "xtick_density": 12,
}

DAY_INDEX_START_DATE: str = "2025-07-07"
P_INSTALLED_KW: float = 9.72

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "data" / "results"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "days"

MODEL_COLORS = {
    "transfer_model": "tab:blue",
    "physical_model": "tab:orange",
    "source_model": "tab:green",
    "target_model": "tab:red",
}

FEATURE_SPECS = [
    ("aswdir_s", "Direct [W/m^2]", "#ffd900"),
    ("aswdifd_s", "Diffuse [W/m^2]", "#f70eff"),
    ("t_2m", "Temp. [K]", "#0077ff"),
]

SOL_ELEV_CANDS = ["solar_elevation_deg", "solar_elevation", "sol_elev_deg"]
PHYS_PRED_CANDS = ["pv_power_W", "pv_power_w", "pv_power", "ac_power", "ac_power_w"]
PHYS_TRUE_CANDS = ["Mittelwertleistung [W]", "mittelwertleistung [w]", "mean_power", "avg_power"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    input_dir: Path
    pred_candidates: Sequence[str]
    true_candidates: Sequence[str]
    scale_to_kw: float
    color_key: str
    uses_decimal_comma: bool
    prefer_suffix_with_pv_power: bool = False


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        key="physical",
        label="Physical",
        input_dir=RESULTS_ROOT / "physical_output",
        pred_candidates=PHYS_PRED_CANDS,
        true_candidates=PHYS_TRUE_CANDS,
        scale_to_kw=1.0 / 1000.0,
        color_key="physical_model",
        uses_decimal_comma=True,
        prefer_suffix_with_pv_power=True,
    ),
    ModelSpec(
        key="source",
        label="Source",
        input_dir=RESULTS_ROOT / "source_output",
        pred_candidates=["y_pred"],
        true_candidates=["y_true"],
        scale_to_kw=P_INSTALLED_KW,
        color_key="source_model",
        uses_decimal_comma=False,
    ),
    ModelSpec(
        key="target",
        label="Target",
        input_dir=RESULTS_ROOT / "target_output",
        pred_candidates=["y_pred"],
        true_candidates=["y_true"],
        scale_to_kw=P_INSTALLED_KW,
        color_key="target_model",
        uses_decimal_comma=False,
    ),
    ModelSpec(
        key="transfer",
        label="Transfer",
        input_dir=RESULTS_ROOT / "transfer_output",
        pred_candidates=["y_pred"],
        true_candidates=["y_true"],
        scale_to_kw=P_INSTALLED_KW,
        color_key="transfer_model",
        uses_decimal_comma=False,
    ),
]


def _to_float(series: pd.Series, use_decimal_comma: bool) -> pd.Series:
    if use_decimal_comma:
        return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def list_exports(spec: ModelSpec) -> List[Path]:
    files = sorted(spec.input_dir.rglob("*.csv"))
    if spec.prefer_suffix_with_pv_power:
        filtered = [fp for fp in files if fp.name.endswith("_with_pv_power.csv")]
        if filtered:
            return filtered
    return files


def load_data(path: Path, spec: ModelSpec) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["valid_time"])
    if "valid_time" not in df.columns:
        raise ValueError(f"Missing 'valid_time' in {path}")

    for col in set(spec.pred_candidates) | set(spec.true_candidates) | set(SOL_ELEV_CANDS) | {"aswdir_s", "aswdifd_s", "t_2m"}:
        if col in df.columns:
            df[col] = _to_float(df[col], spec.uses_decimal_comma)

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
        for t in df["valid_time"].dropna():
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


def choose_best_dual_source(
    frames_with_paths: List[Tuple[Path, pd.DataFrame]],
    primary_candidates: Sequence[str],
    secondary_candidates: Sequence[str],
    times: List[pd.Timestamp],
) -> Optional[Tuple[pd.DataFrame, str, str]]:
    best: Optional[Tuple[pd.DataFrame, str, str]] = None
    best_count = -1

    for _path, df in frames_with_paths:
        col_a = find_col(df, primary_candidates)
        col_b = find_col(df, secondary_candidates)
        if col_a is None or col_b is None:
            continue
        s_a = df.set_index("valid_time")[col_a]
        s_b = df.set_index("valid_time")[col_b]
        cnt = sum(1 for t in times if pd.notna(s_a.get(t)) and pd.notna(s_b.get(t)))
        if cnt > best_count:
            best = (df, col_a, col_b)
            best_count = cnt
        if cnt == len(times):
            break
    return best


def collect_series_by_time(
    frames: List[pd.DataFrame],
    times: List[pd.Timestamp],
    candidates: Sequence[str],
    scale: float,
) -> List[List[float]]:
    buckets: List[List[float]] = [[] for _ in times]
    lookups: List[Dict[pd.Timestamp, float]] = []

    for df in frames:
        col = find_col(df, candidates)
        if col is None:
            continue
        series = df.set_index("valid_time")[col]
        lookup: Dict[pd.Timestamp, float] = {}
        for t, v in series.items():
            try:
                if pd.notna(v):
                    lookup[pd.Timestamp(t)] = float(v) * scale
            except Exception:
                pass
        lookups.append(lookup)

    for i, t in enumerate(times):
        for lookup in lookups:
            if t in lookup:
                buckets[i].append(lookup[t])
    return buckets


def choose_best_line_series(
    frames_with_paths: List[Tuple[Path, pd.DataFrame]],
    candidates: Sequence[str],
    times: List[pd.Timestamp],
    scale: float,
) -> List[Optional[float]]:
    best_df: Optional[pd.DataFrame] = None
    best_col: Optional[str] = None
    best_count = -1

    for _path, df in frames_with_paths:
        col = find_col(df, candidates)
        if col is None:
            continue
        series = df.set_index("valid_time")[col]
        cnt = sum(1 for t in times if pd.notna(series.get(t)))
        if cnt > best_count:
            best_count = cnt
            best_df = df
            best_col = col
        if cnt == len(times):
            break

    if best_df is None or best_col is None:
        raise ValueError(f"No valid line source found for candidates: {list(candidates)}")

    series = best_df.set_index("valid_time")[best_col]
    return [float(series.get(t)) * scale if pd.notna(series.get(t)) else None for t in times]


def draw_boxplot(
    ax: plt.Axes,
    buckets: List[List[float]],
    x: List[float],
    color: str,
    width: float,
) -> None:
    bp = ax.boxplot(
        buckets,
        positions=x,
        widths=width,
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


def style_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_ylabel(ylabel, fontsize=PLOT_CONFIG["fontsize"])
    ax.tick_params(axis="y", labelsize=PLOT_CONFIG["fontsize"] - 1)
    if PLOT_CONFIG["grid"]:
        ax.grid(True, linestyle=PLOT_CONFIG["grid_style"], alpha=PLOT_CONFIG["grid_alpha"])


def plot_day_boxplots(
    day: pd.Timestamp,
    day_frames_by_model: Dict[str, List[Tuple[Path, pd.DataFrame]]],
) -> plt.Figure:
    physical_frames_with_paths = day_frames_by_model["physical"]
    physical_frames = [df for _, df in physical_frames_with_paths]
    if not physical_frames:
        raise ValueError("No physical day frames available.")

    times = build_time_union(physical_frames)
    if not times:
        raise ValueError("No timestamps found for this day.")
    exp_times = expected_day_times(day)

    phys_ref = choose_best_dual_source(physical_frames_with_paths, PHYS_TRUE_CANDS, SOL_ELEV_CANDS, exp_times)
    if phys_ref is None:
        raise ValueError("No physical source with real power and solar elevation found.")
    phys_ref_df, _phys_true_col, phys_sol_col = phys_ref
    phys_sol_series = phys_ref_df.set_index("valid_time")[phys_sol_col]

    times = [t for t in times if pd.notna(phys_sol_series.get(t)) and float(phys_sol_series.get(t)) > 0.0]
    if not times:
        raise ValueError("No timestamps with solar elevation > 0 found.")

    x = mdates.date2num(times)
    box_width = (15 / 60 / 24) * 0.8

    n_sub = len(FEATURE_SPECS) + 1 + len(MODEL_SPECS)
    fig, axes = plt.subplots(n_sub, 1, figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"], sharex=True)
    if n_sub == 1:
        axes = [axes]

    # Feature boxplots from physical data (unnormalized).
    for idx, (feature_col, label, color) in enumerate(FEATURE_SPECS):
        ax = axes[idx]
        buckets = collect_series_by_time(physical_frames, times, [feature_col], scale=1.0)
        if any(len(b) > 0 for b in buckets):
            draw_boxplot(ax, buckets, x, color, box_width)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        style_axis(ax, label)

    # Solar elevation subplot from physical reference source.
    solar_ax = axes[len(FEATURE_SPECS)]
    y_solar = [float(phys_sol_series.get(t)) if pd.notna(phys_sol_series.get(t)) else None for t in times]
    solar_ax.plot(times, y_solar, color="#777777", lw=PLOT_CONFIG["linewidth"])
    style_axis(solar_ax, "Solar elev. [deg]")

    # Power subplots per model: prediction boxplot + real line.
    power_start = len(FEATURE_SPECS) + 1
    for i, spec in enumerate(MODEL_SPECS):
        ax = axes[power_start + i]
        model_frames_with_paths = day_frames_by_model[spec.key]
        model_frames = [df for _, df in model_frames_with_paths]
        pred_color = MODEL_COLORS[spec.color_key]

        pred_buckets = collect_series_by_time(model_frames, times, spec.pred_candidates, spec.scale_to_kw)
        pred_handle = None
        if any(len(b) > 0 for b in pred_buckets):
            draw_boxplot(ax, pred_buckets, x, pred_color, box_width)
            pred_handle = Patch(
                facecolor=pred_color,
                edgecolor=pred_color,
                alpha=0.35,
                label=f"{spec.label} prediction [kW]",
            )

        y_true = choose_best_line_series(model_frames_with_paths, spec.true_candidates, times, spec.scale_to_kw)
        ax.plot(times, y_true, color="#333333", lw=PLOT_CONFIG["linewidth"] + 0.5, label="Real power [kW]")
        real_handle = Line2D([0], [0], color="#333333", lw=PLOT_CONFIG["linewidth"] + 0.5, label="Real power [kW]")

        style_axis(ax, "Power [kW]")
        ax.set_title(spec.label, fontsize=PLOT_CONFIG["fontsize"] + 1)

        handles = [h for h in (pred_handle, real_handle) if h is not None]
        if handles:
            ax.legend(handles=handles, fontsize=PLOT_CONFIG["fontsize"] - 1, loc="upper left")

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

    fig.suptitle(f"Model day overview - {day.date()}", fontsize=PLOT_CONFIG["fontsize"] + 4)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def main(argv: List[str]) -> None:
    del argv

    frames_by_model: Dict[str, List[Tuple[Path, pd.DataFrame]]] = {}
    days_by_model: Dict[str, set[pd.Timestamp]] = {}

    for spec in MODEL_SPECS:
        files = list_exports(spec)
        if not files:
            raise FileNotFoundError(f"No CSV exports found in '{spec.input_dir}'.")

        frames = [(fp, load_data(fp, spec)) for fp in files]
        frames_by_model[spec.key] = frames
        days_by_model[spec.key] = set(list_available_days(frames))
        model_days = sorted(days_by_model[spec.key])
        print(
            f"[{spec.key}] files={len(files)} days={len(model_days)} "
            f"range={model_days[0].date()}..{model_days[-1].date()}"
        )

    common_days = set.intersection(*(days_by_model[k] for k in days_by_model))
    start_day = pd.Timestamp(DAY_INDEX_START_DATE).normalize()
    all_days = sorted(d for d in common_days if d >= start_day)
    if not all_days:
        raise FileNotFoundError(f"No common days from start date {start_day.date()} found across all models.")
    if all_days[0] != start_day:
        raise FileNotFoundError(
            f"Start date {start_day.date()} not found in common day set (first common day: {all_days[0].date()})."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    for day_index, day in enumerate(all_days, start=1):
        day_frames_by_model = {
            key: collect_day_frames_from_loaded(frames_by_model[key], day)
            for key in frames_by_model
        }

        if any(len(v) == 0 for v in day_frames_by_model.values()):
            skipped += 1
            print(f"Day {day_index} ({day:%Y-%m-%d}) skipped: missing day data in at least one model.")
            continue

        try:
            fig = plot_day_boxplots(day, day_frames_by_model)
            out_png = OUTPUT_DIR / f"{day_index}_day_boxplots_{day:%Y%m%d}.png"
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
