import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "physical_input"


# dataset: ID wie "2025070400" ODER Pfad zur CSV (Default-Ordner: data/processed/physical_input)
# save_png: Pfad fuer Ausgabe-PNG oder None fuer Anzeige
CODE_SELECTION = {
    "dataset": '2025070400',     # z.B. "2025070400" oder kompletter Pfad
    "save_png": None,    # z.B. "exports/fig_targetdatasetsample.png"
}



PLOT_CONFIG = {
    "figsize": (7, 6),        # Größe der Abbildung (Breite, Höhe)
    "dpi": 150,               # Auflösung der Abbildung
    "fontsize": 14,            # Grundschriftgröße
    "linewidth": 1.0,         # Linienbreite
    "grid": True,             # Gitterlinien aktivieren
    "grid_style": ":",        # Stil der Gitterlinien
    "grid_alpha": 0.5,        # Transparenz der Gitterlinien
    "rotation": 90,           # Drehung der X-Achsenbeschriftung
    "margin_frac": 0.01,      # Prozentualer Randbeschnitt links/rechts
    "xtick_density":45,      # Anzahl der X-Ticks
}

# Zu plottende Variablen mit Beschriftung und Farbe
SERIES = [
    ("aswdir_s", "Direct [W/m²]", "#ffd900"),
    ("aswdifd_s", "Diffuse [W/m²]", "#f70eff"),
    ("t_2m", "Temp. [K]", "#0077ff"),
    ("Mittelwertleistung [W]", "Power [W]", "#d62728"),
    ("solar_elevation_deg", "Solar elev. [°]", "#777777"),
]


def load_data(path: Path) -> pd.DataFrame:
    # CSV-Datei einlesen und Zeitspalte als Datum parsen
    df = pd.read_csv(path, parse_dates=["valid_time"])

    # Textspalten in numerische Werte konvertieren
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
            except Exception:
                pass

    # Nach Zeit sortieren und Index zurücksetzen
    return df.sort_values("valid_time").reset_index(drop=True)


def plot_data(df: pd.DataFrame, title: str = "Target dataset features"):
    # Nur vorhandene Spalten auswählen
    cols = [(col, label, color) for col, label, color in SERIES if col in df.columns]
    n = len(cols)

    # Abbildung und Subplots erzeugen
    fig, axes = plt.subplots(n, 1, figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"], sharex=True)
    if n == 1:
        axes = [axes]

    # Zeitachse extrahieren
    time = df["valid_time"]

    # Datenreihen zeichnen
    for ax, (col, label, color) in zip(axes, cols):
        ax.plot(time, df[col], color=color, lw=PLOT_CONFIG["linewidth"])
        ax.set_ylabel(label, fontsize=PLOT_CONFIG["fontsize"])
        if PLOT_CONFIG["grid"]:
            ax.grid(True, linestyle=PLOT_CONFIG["grid_style"], alpha=PLOT_CONFIG["grid_alpha"])

    # X-Achse links und rechts leicht beschneiden
    xmin, xmax = time.min(), time.max()
    span = xmax - xmin
    xmin += span * PLOT_CONFIG["margin_frac"]
    xmax -= span * PLOT_CONFIG["margin_frac"]
    axes[0].set_xlim(xmin, xmax)

    # X-Achse formatieren und Ticks definieren
    locator = mdates.AutoDateLocator(
        minticks=PLOT_CONFIG["xtick_density"] // 2,
        maxticks=PLOT_CONFIG["xtick_density"]
    )
    formatter = mdates.DateFormatter("%Y-%m-%d\n%H:%M")
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)

    # X-Achsenbeschriftung drehen
    plt.setp(axes[-1].get_xticklabels(), rotation=PLOT_CONFIG["rotation"], ha="center")

    # Titel und Layout anpassen
    fig.suptitle(title, fontsize=PLOT_CONFIG["fontsize"] + 2)
    fig.tight_layout()

    return fig


def _list_available_exports(export_dir: Path, limit: int = 20):
    files = sorted(export_dir.glob("pv_weather_*.csv"))
    return files[:limit]


def _resolve_input_to_path(arg, export_dir: Path):
    # Falls es eine direkte Datei ist
    p = Path(arg)
    if p.exists() and p.is_file():
        return p

    # Falls nur eine ID (z.B. 2025070400) übergeben wurde
    if len(arg) >= 6 and arg.isdigit():
        candidate = export_dir / f"pv_weather_{arg}.csv"
        if candidate.exists():
            return candidate

    # Fallback: fuzzy match innerhalb des Export-Ordners
    matches = sorted([fp for fp in export_dir.glob("pv_weather_*.csv") if arg in fp.name])
    if matches:
        return matches[0]

    return None


def main():
    export_dir = Path(os.getenv("ICON_EXPORT_DIR", DEFAULT_INPUT_DIR))
    export_dir.mkdir(parents=True, exist_ok=True)

    # Auswahl aus Code oder CLI entnehmen
    arg = CODE_SELECTION.get("dataset") if CODE_SELECTION else None
    if arg is None and len(sys.argv) >= 2:
        arg = sys.argv[1]
    if arg is None:
        print("Bitte im Code (CODE_SELECTION['dataset']) oder per CLI eine Dataset-ID (z.B. 2025070400) bzw. CSV-Datei angeben.")
        avail = _list_available_exports(export_dir)
        if avail:
            print("Verfügbar (Beispiele):")
            for fp in avail:
                print(" -", fp.name)
        else:
            print(f"Keine Dateien in '{export_dir}' gefunden.")
        sys.exit(1)

    in_path = _resolve_input_to_path(arg, export_dir)
    if in_path is None:
        print(f"Konnte keine Datei zu '{arg}' finden. Ordner: '{export_dir}'.")
        sys.exit(1)

    # Ausgabeziel aus Code oder CLI entnehmen
    out_png = None
    if CODE_SELECTION and CODE_SELECTION.get("save_png"):
        out_png = Path(CODE_SELECTION.get("save_png"))
    elif len(sys.argv) > 2:
        out_png = Path(sys.argv[2])

    # Daten laden
    df = load_data(in_path)

    # Plot erstellen
    title = f"Target dataset features - {in_path.name}"
    fig = plot_data(df, title=title)

    # Plot speichern oder anzeigen
    if out_png:
        fig.savefig(out_png)
        print(f"Plot wurde gespeichert unter: '{out_png}'.")
    else:
        plt.show()



if __name__ == "__main__":
    main()
