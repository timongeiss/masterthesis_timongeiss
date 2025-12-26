"""
Aggregiert Ziel-PV-Leistung: Wochen-CSV lesen -> 15-Min-Mittel -> NaN-Glättung -> Speichern.
Plots werden angezeigt, aber nicht gespeichert.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Plot-Konfiguration
PLOT_CONFIG = {
    "figsize": (7, 4),
    "dpi": 150,
    "fontsize": 9,
    "linewidth": 1.0,
    "grid": True,
    "grid_style": ":",
    "grid_alpha": 0.5,
    "rotation": 90,
    "margin_frac": 0.01,
}

SERIES = {"Mittelwertleistung [W]": ("Average power [W]", "#d62728")}


def read_all_csv_files(prefix: str = "S917517808601355959525685-week-") -> pd.DataFrame:
    """Liest alle Wochen-CSV-Dateien aus data/raw/target_label_space ein."""
    folder_path = PROJECT_ROOT / "data" / "raw" / "target_label_space"
    csv_files = sorted(f for f in folder_path.iterdir() if f.name.startswith(prefix) and f.suffix == ".csv")
    dfs = []
    for file in csv_files:
        df = pd.read_csv(
            file,
            skiprows=1,
            delimiter=";",
            decimal=",",
            names=[
                "Uhrzeit",
                "Netzbezug [kW]",
                "Netzeinspeisung [kW]",
                "Stromverbrauch [kW]",
                "Akkubeladung [kW]",
                "Akkuentnahme [kW]",
                "Stromerzeugung [kW]",
                "Akku Spannung [V]",
                "Akku Stromstärke [A]",
            ],
        )
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("Keine passenden CSV-Dateien gefunden.")
    return pd.concat(dfs, ignore_index=True)


def clean_time_and_generation(df: pd.DataFrame) -> pd.DataFrame:
    """Zeit robust parsen, Erzeugung in Float konvertieren, nach Zeit sortieren."""
    out = df.copy()
    raw_time = out["Uhrzeit"].astype(str).str.strip()
    parsed = pd.to_datetime(raw_time, format="%d.%m.%Y %H:%M:%S", errors="coerce")
    mask_na = parsed.isna()
    if mask_na.any():
        parsed.loc[mask_na] = pd.to_datetime(raw_time.loc[mask_na], format="%d.%m.%Y %H:%M", errors="coerce")
    mask_na = parsed.isna()
    if mask_na.any():
        parsed.loc[mask_na] = pd.to_datetime(raw_time.loc[mask_na], errors="coerce", dayfirst=True)
    out["Uhrzeit"] = parsed
    out = out.dropna(subset=["Uhrzeit"])

    if not pd.api.types.is_numeric_dtype(out["Stromerzeugung [kW]"]):
        out["Stromerzeugung [kW]"] = out["Stromerzeugung [kW]"].astype(str).str.replace(",", ".", regex=False)
        out["Stromerzeugung [kW]"] = pd.to_numeric(out["Stromerzeugung [kW]"], errors="coerce")

    return out.sort_values("Uhrzeit")


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """15-Minuten-Mittelwerte berechnen (W -> mW -> W)."""
    df = clean_time_and_generation(df)
    s = df.set_index("Uhrzeit")["Stromerzeugung [kW]"].sort_index()

    start, end = s.index.min().ceil("15min"), s.index.max().floor("15min")
    centers = pd.date_range(start=start, end=end, freq="15min") if start < end else pd.DatetimeIndex([])

    s2 = s.reindex(s.index.union(centers)).sort_index()
    mean15 = s2.rolling("15min", center=True, min_periods=1).mean()

    out = (mean15.loc[centers] * 1000.0).reset_index()
    out.columns = ["datetime", "Mittelwertleistung [W]"]
    return out


def filter_from_july(processed_df: pd.DataFrame) -> pd.DataFrame:
    """Verwirft alle Zeilen vor dem 01.07."""
    if processed_df.empty:
        return processed_df
    df = processed_df.copy()
    months = df["datetime"].dt.month
    return df.loc[months >= 7].reset_index(drop=True)


def fill_isolated_nans_linear(processed_df: pd.DataFrame) -> pd.DataFrame:
    """NaN-Behandlung: isoliert linear, Nachtläufe auf 0, Tagläufe ffill."""
    if processed_df.empty:
        return processed_df

    df = processed_df.copy()
    s = df["Mittelwertleistung [W]"]

    mask = s.isna()
    if not mask.any():
        return df

    run_id = mask.ne(mask.shift(fill_value=False)).cumsum()
    run_len = run_id.groupby(run_id).transform("size")

    prev_vals = s.shift(1)
    next_vals = s.shift(-1)

    isolated = mask & (run_len == 1) & prev_vals.notna() & next_vals.notna()
    s.loc[isolated] = (prev_vals + next_vals) / 2.0

    mask = s.isna()
    if mask.any():
        run_id = mask.ne(mask.shift(fill_value=False)).cumsum()
        run_len = run_id.groupby(run_id).transform("size")

        hours = df["datetime"].dt.hour
        night = (hours >= 20) | (hours < 6)

        multi_night = mask & (run_len > 1) & night
        if multi_night.any():
            s.loc[multi_night] = 0.0

        mask2 = s.isna()
        if mask2.any():
            run_id2 = mask2.ne(mask2.shift(fill_value=False)).cumsum()
            run_len2 = run_id2.groupby(run_id2).transform("size")
            day_multi = mask2 & (run_len2 > 1) & (~night)
            if day_multi.any():
                ff = s.ffill()
                s.loc[day_multi] = ff.loc[day_multi]

    df["Mittelwertleistung [W]"] = s
    return df


def save_to_csv(df: pd.DataFrame, filename: str = "target_label_space_leistung.csv"):
    """Speichert nach data/processed."""
    export_dir = PROJECT_ROOT / "data" / "processed"
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / filename
    df.to_csv(path, decimal=",", index=False)
    print(f"Gesamtdatei gespeichert: {path} ({len(df)} Zeilen)")


def plot_data(raw_df: pd.DataFrame, processed_df: pd.DataFrame):
    """Plot Rohdaten + 15-Minuten-Mittelwerte (anzeige, kein Speichern)."""
    raw_df = clean_time_and_generation(raw_df)

    plt.figure(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])
    plt.plot(
        raw_df["Uhrzeit"],
        raw_df["Stromerzeugung [kW]"] * 1000,
        color="lightgray",
        linewidth=PLOT_CONFIG["linewidth"],
        label="Measurements (~5 min)",
    )
    plt.plot(
        processed_df["datetime"],
        processed_df["Mittelwertleistung [W]"],
        color=SERIES["Mittelwertleistung [W]"][1],
        linewidth=1.5,
        label=SERIES["Mittelwertleistung [W]"][0],
    )

    plt.xlabel("Time", fontsize=PLOT_CONFIG["fontsize"])
    plt.ylabel("Power [W]", fontsize=PLOT_CONFIG["fontsize"])
    plt.title("PV Power: Measurements and 15-minute Means", fontsize=PLOT_CONFIG["fontsize"] + 4)
    if PLOT_CONFIG["grid"]:
        plt.grid(True, linestyle=PLOT_CONFIG["grid_style"], alpha=PLOT_CONFIG["grid_alpha"])
    plt.xticks(rotation=PLOT_CONFIG["rotation"])
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_data_horizontal(raw_df: pd.DataFrame, processed_df: pd.DataFrame):
    """Plot Rohdaten + 15-Minuten-Mittelwerte als horizontale Linien (anzeige, kein Speichern)."""
    raw_df = clean_time_and_generation(raw_df)

    plt.figure(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])

    plt.plot(
        raw_df["Uhrzeit"],
        raw_df["Stromerzeugung [kW]"] * 1000,
        color="lightgray",
        linewidth=PLOT_CONFIG["linewidth"],
        marker="o",
        markersize=3,
        label="Measurements (~5 min)",
    )

    centers = processed_df["datetime"]
    starts = centers - pd.Timedelta(minutes=7, seconds=30)
    ends = centers + pd.Timedelta(minutes=7, seconds=30)
    values = processed_df["Mittelwertleistung [W]"]

    plt.hlines(
        values,
        xmin=starts,
        xmax=ends,
        colors=SERIES["Mittelwertleistung [W]"][1],
        linewidth=1.8,
        label=SERIES["Mittelwertleistung [W]"][0],
    )

    plt.xlabel("Time", fontsize=PLOT_CONFIG["fontsize"])
    plt.ylabel("Power [W]", fontsize=PLOT_CONFIG["fontsize"])
    plt.title("PV Power: Measurements and 15-minute Means", fontsize=PLOT_CONFIG["fontsize"] + 4)
    if PLOT_CONFIG["grid"]:
        plt.grid(True, linestyle=PLOT_CONFIG["grid_style"], alpha=PLOT_CONFIG["grid_alpha"])
    plt.xticks(rotation=PLOT_CONFIG["rotation"])
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    raw = read_all_csv_files()
    processed = process_data(raw)
    processed = filter_from_july(processed)
    processed = fill_isolated_nans_linear(processed)

    save_to_csv(processed, filename="target_label_space_leistung.csv")
    plot_data(raw, processed)
    plot_data_horizontal(raw, processed)
