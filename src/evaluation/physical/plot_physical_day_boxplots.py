import sys
from pathlib import Path
from typing import List, Optional, Dict, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd

# ==========================================
# Plot-Konfiguration (wie im Einzelskript)
# ==========================================
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

# Optional oberer Konfig‑Block für einfachen Start ohne CLI
# Datum als String im Format yyyymmdd oder yyyy-mm-dd angeben
# Beispiel: "2025-07-05" oder "20250705". Leer lassen, um CLI zu verwenden.
SELECTED_DAY: Optional[str] = "20251005"
# Optionaler Ausgabepfad als PNG; leer lassen für interaktive Anzeige
SELECTED_OUTPUT: Optional[str] = ""
# Optional: Pfad zum Exports‑Ordner überschreiben; leer -> "exports"
SELECTED_EXPORTS_DIR: Optional[str] = ""

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = PROJECT_ROOT / "data" / "results" / "physical_output"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "physical"

# Boxplot‑Reihen (Zeitreihen, die je Lauf unterschiedlich sind)
BOXPLOT_SERIES = [
    ("t_2m", "Temperature [K]", "#0077ff"),
    ("aswdir_s", "Direct irradiance [W/m^2]", "#ffd900"),
    ("aswdifd_s", "Diffuse irradiance [W/m^2]", "#f70eff"),
]

# Leistungs‑Spaltenkandidaten
PV_COL_CANDS = ["pv_power_W", "pv_power_w", "pv_power", "ac_power", "ac_power_w"]

# Mittelwertleistung (real) – identisch über alle Läufe
AVG_POWER_CANDS = [
    "Mittelwertleistung [W]",
    "mittelwertleistung [w]",
    "mean_power",
    "avg_power",
]

# Sonnenhöhe – identisch über alle Läufe
SOL_ELEV_CANDS = ["solar_elevation_deg", "solar_elevation", "sol_elev_deg"]


def list_exports(directory: Path) -> List[Path]:
    files = sorted(directory.glob("*_with_pv_power.csv"))
    if files:
        return files
    return sorted(directory.glob("*.csv"))


def load_data(path: Path) -> pd.DataFrame:
    """CSV laden, Zeit parsen, Dezimalkommas in Zahlen konvertieren, sortiert zurückgeben."""
    df = pd.read_csv(path, parse_dates=["valid_time"])  # parse valid_time
    # Strings -> Zahlen (Komma als Dezimaltrennzeichen erlauben)
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
            except Exception:
                pass
    return df.sort_values("valid_time").reset_index(drop=True)


def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def parse_day_arg(raw: str) -> pd.Timestamp:
    """Akzeptiert yyyymmdd oder yyyy-mm-dd und gibt UTC‑naiven Timestamp (Mitternacht) zurück."""
    s = raw.strip()
    if len(s) == 8 and s.isdigit():
        return pd.to_datetime(f"{s[0:4]}-{s[4:6]}-{s[6:8]}")
    # Fallback: pandas parsen lassen
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Ungültiges Datumsformat: '{raw}'. Erwarte yyyymmdd oder yyyy-mm-dd.")
    return ts.normalize()


def collect_day_frames_with_paths(exports_dir: Path, day: pd.Timestamp) -> List[tuple[Path, pd.DataFrame]]:
    """Alle Tages-DataFrames inklusive ihrer Quelldatei zurückgeben."""
    out: List[tuple[Path, pd.DataFrame]] = []
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
    """96 Schritte im 15‑Minuten‑Raster für den Tag."""
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
    """Ersten Frame wählen, der für alle 96 Zeiten sowohl Mittelwertleistung als auch Sonnenhöhe hat."""
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
    """Falls kein kompletter Frame vorhanden ist, den mit der größten Abdeckung nehmen."""
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
    """Für jeden Zeitstempel alle Werte dieser Spalte über die Frames einsammeln (NaN ignorieren)."""
    buckets: List[List[float]] = [[] for _ in times]
    # Schnellzugriff: pro Frame ein Dict valid_time->Wert
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

    # Zeitachse (Vereinigung aller Timestamps)
    times = build_time_union(frames)
    if not times:
        raise ValueError("Keine Zeitstempel für diesen Tag gefunden.")
    exp_times = expected_day_times(day)

    frames_with_paths = list(zip(frame_paths or [Path("")]*len(frames), frames))
    chosen = choose_complete_source(frames_with_paths, exp_times)
    if chosen is None:
        chosen = choose_best_partial_source(frames_with_paths, exp_times)
    if chosen is None:
        raise ValueError("Keine Quelle mit vollständiger Solar elevation und Mittelwertleistung gefunden.")

    src_df, avg_col, sol_col = chosen
    sol_series = src_df.set_index("valid_time")[sol_col]

    # Nur Zeiten mit Solar elevation > 0 plotten
    filtered_times = [t for t in times if pd.notna(sol_series.get(t)) and float(sol_series.get(t)) > 0]
    if not filtered_times:
        raise ValueError("Keine Zeitstempel mit Solar elevation > 0 gefunden.")
    times = filtered_times
    x = mdates.date2num(times)

    # Plot‑Layout: Boxplots für BOXPLOT_SERIES + 1 Subplot für Leistung (Box + Linie)
    # + 1 eigener Subplot für Sonnenhöhe
    n_sub = len(BOXPLOT_SERIES) + 2
    fig, axes = plt.subplots(n_sub, 1, figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"], sharex=True)
    if n_sub == 1:
        axes = [axes]

    # Breite der Boxen (in Matplotlib‑Datums‑Einheiten = Tage)
    # 15 min ≈ 15/60/24 Tage
    box_width = (15 / 60 / 24) * 0.8

    # Hilfsfunktionen
    def _style_ax(ax, ylabel: str):
        ax.set_ylabel(ylabel, fontsize=PLOT_CONFIG["fontsize"])
        if PLOT_CONFIG["grid"]:
            ax.grid(True, linestyle=PLOT_CONFIG["grid_style"], alpha=PLOT_CONFIG["grid_alpha"])

    # Boxplots für die Wetter‑Eingänge
    for idx, (col_key, label, color) in enumerate(BOXPLOT_SERIES):
        # Spaltenname robust feststellen (muss im ersten verfügbaren Frame existieren)
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
        # Farben anpassen
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
        for element in ['whiskers', 'caps', 'medians']:
            for line in bp[element]:
                line.set_color(color)
                line.set_alpha(0.9)
                line.set_linewidth(0.8)

        _style_ax(ax, label)

    # Leistungs‑Subplot (Boxplots über Läufe + reale Mittelwertleistung als Linie)
    axp = axes[-2]
    # PV‑Spalte suchen
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
        for patch in bp['boxes']:
            patch.set_facecolor("#0aa03b")
            patch.set_alpha(0.35)
            patch.set_edgecolor("#0aa03b")
        for element in ['whiskers', 'caps', 'medians']:
            for line in bp[element]:
                line.set_color("#0aa03b")
                line.set_alpha(0.9)
                line.set_linewidth(0.8)
        pv_legend_handle = Patch(facecolor="#0aa03b", edgecolor="#0aa03b", alpha=0.35, label="PV power (forecasts)")

    # Reale Mittelwertleistung (Linie) aus der gewählten Quelle
    mean_line_handle = None
    s_avg = src_df.set_index("valid_time")[avg_col]
    y_avg = [float(s_avg.get(t)) if pd.notna(s_avg.get(t)) else None for t in times]
    axp.plot(times, y_avg, color="#333333", lw=PLOT_CONFIG["linewidth"] + 0.5, label="Mean power [W]")
    mean_line_handle = Line2D([0], [0], color="#333333", lw=PLOT_CONFIG["linewidth"] + 0.5, label="Mean power [W]")

    _style_ax(axp, "Power [W]")

    # Legende für Leistungs‑Subplot
    legend_handles = [h for h in (pv_legend_handle, mean_line_handle) if h is not None]
    if legend_handles:
        axp.legend(handles=legend_handles, fontsize=PLOT_CONFIG["fontsize"], loc="upper right")

    # Eigener Subplot für Sonnenhöhe (Linie, identisch über Läufe)
    axe = axes[-1]
    y_sol = [float(sol_series.get(t)) if pd.notna(sol_series.get(t)) else None for t in times]
    axe.plot(times, y_sol, color="#777777", lw=PLOT_CONFIG["linewidth"], label="Solar elevation [deg]")
    axe.set_ylabel("Solar elevation [deg]", fontsize=PLOT_CONFIG["fontsize"]) 
    axe.grid(True, linestyle=PLOT_CONFIG["grid_style"], alpha=PLOT_CONFIG["grid_alpha"])
    axe.legend(fontsize=PLOT_CONFIG["fontsize"], loc="upper right")

    # X‑Achse konfigurieren
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

    # Titel und Layout
    ttl = title or f"Day boxplots for {day.date()} (N={len(frames)} runs)"
    fig.suptitle(ttl, fontsize=PLOT_CONFIG["fontsize"] + 2)
    fig.tight_layout()

    return fig


def main(argv: List[str]) -> None:
    # Aufruf: python plot_physical_day_boxplots.py 20250927 [out.png] [exports_dir]
    # Oder: oben SELECTED_DAY setzen und ohne CLI starten.
    day_arg = (SELECTED_DAY or "").strip()
    if not day_arg:
        if len(argv) < 2:
            print("Usage: python plot_physical_day_boxplots.py <yyyymmdd|yyyy-mm-dd> [out.png] [exports_dir]")
            sys.exit(2)
        day_arg = argv[1]

    day = parse_day_arg(day_arg)

    out_arg = (SELECTED_OUTPUT or "").strip()
    if not out_arg and len(argv) > 2 and not argv[2].lower().endswith('.csv'):
        out_arg = argv[2]
    out_png: Optional[Path] = Path(out_arg) if out_arg else None

    dir_arg = (SELECTED_EXPORTS_DIR or "").strip()
    if not dir_arg:
        dir_arg = argv[3] if len(argv) > 3 else str(DEFAULT_DIR)
    exports_dir = Path(dir_arg)

    frames_with_paths = collect_day_frames_with_paths(exports_dir, day)
    if not frames_with_paths:
        raise FileNotFoundError(
            f"Keine passenden Tagesdaten in '{exports_dir}' gefunden für {day.date()}"
        )
    frame_paths = [p for p, _ in frames_with_paths]
    frames = [df for _, df in frames_with_paths]

    title = f"Physical model day overview — {day.date()}"
    fig = plot_day_boxplots(frames, frame_paths, day, title=title)

    if out_png is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_png = OUTPUT_DIR / f"day_boxplots_{day:%Y%m%d}.png"

    fig.savefig(out_png)
    print(f"Plot gespeichert unter: '{out_png}'.")
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv)
