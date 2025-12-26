# nicht checked aber supplement

"""Plottet Zeitreihen aus den exportierten ID-CSV-Dateien (ASWDIR_S, ASWDIFD_S, T_2M)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]


# Ordner mit den resampleten CSVs
INPUT_DIR  = PROJECT_ROOT / "data" / "processed" / "resampled_rea"


# Plot-Konfiguration
PLOT_CONFIG = {
    "figsize": (7, 6),
    "dpi": 150,
    "fontsize": 6,
    "linewidth": 1.0,
    "grid": True,
    "grid_style": ":",
    "grid_alpha": 0.5,
    "rotation": 90,
    "margin_frac": 0.01,
    "xtick_density": 45,
}

# Zu plottende Variablen mit Beschriftung und Farbe (Spaltennamen sind case-insensitive)
SERIES = [
    ("aswdir_s", "Direct [W/m^2]", "#ffd900"),
    ("aswdifd_s", "Diffuse [W/m^2]", "#f70eff"),
    ("t_2m", "Temperature [K]", "#0077ff"),
    ("power_ac", "AC Power [W]", "#d62728"),
    ("solar_elevation_deg", "Solar elevation [deg]", "#777777"),
]

# Interaktive Ansicht (Zoom) anzeigen?
SHOW_INTERACTIVE = True


def plot_id_csv(csv_path: Path) -> None:
    """Erstellt Subplots je Field und zeigt sie interaktiv an."""
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df.set_index("time").sort_index()
    plant_id = csv_path.stem.split("_")[0]

    plt.rcParams.update({"font.size": PLOT_CONFIG["fontsize"]})

    fig, axes = plt.subplots(len(SERIES), 1, figsize=PLOT_CONFIG["figsize"], sharex=True)
    fig.suptitle(plant_id)

    for ax, (col_name, label, color) in zip(axes, SERIES):
        col = next((c for c in df.columns if c.lower() == col_name.lower()), None)
        if col:
            ax.plot(df.index, df[col], linewidth=PLOT_CONFIG["linewidth"], color=color)
            ax.set_ylabel(label)
        else:
            ax.set_ylabel(f"{label} (missing)")

        if PLOT_CONFIG["grid"]:
            ax.grid(True, linestyle=PLOT_CONFIG["grid_style"], alpha=PLOT_CONFIG["grid_alpha"])
        ax.margins(x=PLOT_CONFIG["margin_frac"])

    # X-Ticks herunterdichten
    if len(df.index) > 0 and PLOT_CONFIG["xtick_density"] > 0:
        step = max(1, len(df.index) // PLOT_CONFIG["xtick_density"])
        xticks = df.index[::step]
        axes[-1].set_xticks(xticks)
        plt.setp(axes[-1].get_xticklabels(), rotation=PLOT_CONFIG["rotation"])

    axes[-1].set_xlabel("Time")
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    # Optional: Interaktive Ansicht fuer Zoom/Pan, danach speichern
    if SHOW_INTERACTIVE:
        plt.show()
    plt.close()
    print(f"Plot angezeigt: {csv_path.name}")


def main() -> None:
    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        print(f"Keine CSV-Dateien in {INPUT_DIR} gefunden.")
        return

    for csv_path in csv_files:
        plot_id_csv(csv_path)


if __name__ == "__main__":
    main()
