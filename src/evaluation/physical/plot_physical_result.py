import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Button

# ==========================================
# Plot-Konfiguration
# ==========================================
PLOT_CONFIG = {
    "figsize": (7, 6),        # Groesse der Abbildung (Breite, Hoehe)
    "dpi": 150,               # Aufloesung der Abbildung
    "fontsize": 8,            # Grundschriftgroesse
    "linewidth": 1.0,         # Linienbreite
    "grid": True,             # Gitterlinien aktivieren
    "grid_style": ":",        # Stil der Gitterlinien
    "grid_alpha": 0.5,        # Transparenz der Gitterlinien
    "rotation": 90,           # Drehung der X-Achsenbeschriftung
    "margin_frac": 0.01,      # Prozentualer Rand links/rechts
    "xtick_density": 45,      # Anzahl der X-Ticks
}

# Auszuwählender Exportdatensatz (Dateiname innerhalb von `exports`)
# Beispiel: "pv_weather_2025070318_with_pv_power.csv"
# Hinweis: Wenn `SELECTED_EXPORT = None` oder leer ist, startet automatisch der Browse-Modus
SELECTED_EXPORT = "pv_weather_2025092700_with_pv_power.csv"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = PROJECT_ROOT / "data" / "results" / "physical_output"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "physical"

# Zu plottende Einzel-Variablen (ohne Leistung)
SERIES = [
    ("aswdir_s", "Direct irradiance [W/m^2]", "#ffd900"),       # Direkte horizontale Strahlung
    ("aswdifd_s", "Diffuse irradiance [W/m^2]", "#f70eff"),      # Diffuse horizontale Strahlung
    ("t_2m", "Temperature [K]", "#0077ff"),                      # Lufttemperatur (Kelvin)
    ("solar_elevation_deg", "Solar elevation [deg]", "#777777"), # Sonnenhoehe (Grad)
]


def load_data(path: Path) -> pd.DataFrame:
    """CSV laden, Zeitspalte parsen, numerische Werte bereinigen."""
    df = pd.read_csv(path, parse_dates=["valid_time"])  # Datei einlesen und Zeitspalte parsen

    # Textspalten in numerische Werte konvertieren (z. B. "1,23" -> 1.23)
    for col in df.columns:                               # Alle Spalten durchgehen
        if df[col].dtype == "object":                   # Nur Textspalten verarbeiten
            try:
                df[col] = pd.to_numeric(                 # In Zahlen umwandeln
                    df[col].astype(str).str.replace(",", "."), errors="coerce"
                )
            except Exception:
                pass                                     # Bei Fehlern Spalte ignorieren

    # Nach Zeit sortieren und Index zuruecksetzen
    return df.sort_values("valid_time").reset_index(drop=True)


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Spaltennamen robust (case-insensitive) finden und Originalnamen zurueckgeben."""
    lower_map = {c.lower(): c for c in df.columns}       # Mapping lowercase -> Original
    for cand in candidates:                               # Kandidatenliste durchgehen
        key = cand.lower()
        if key in lower_map:                              # Treffer gefunden
            return lower_map[key]                          # Originalspaltennamen liefern
    return None                                           # Kein Treffer


def plot_data(df: pd.DataFrame, title: str = "Physical model inputs and power", fig: Optional[plt.Figure] = None):
    """Mehrere Subplots zeichnen und Leistung zusammenlegen."""
    # Vorhandene Einzelvariablen (ohne Leistung) bestimmen
    cols = [(col, label, color) for col, label, color in SERIES if col in df.columns]

    # Power-Spalten ermitteln (berechnet + Mittelwertleistung)
    pv_col = find_col(df, ["pv_power_W", "pv_power_w", "pv_power", "ac_power", "ac_power_w"])     # berechnete PV-Leistung
    avg_col = find_col(df, ["Mittelwertleistung [W]", "mittelwertleistung [w]", "mean_power", "avg_power"])  # Durchschnittsleistung

    # Anzahl der Subplots festlegen (zusaetzlich 1 fuer Power, falls vorhanden)
    add_power_subplot = pv_col is not None or avg_col is not None
    n = len(cols) + (1 if add_power_subplot else 0)

    # Abbildung und Achsen erstellen/neu zeichnen
    if fig is None:
        fig, axes = plt.subplots(n, 1, figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"], sharex=True)
    else:
        fig.clf()
        axes = fig.subplots(n, 1, sharex=True)
    if n == 1:
        axes = [axes]                                    # Einheitliche Liste erzwingen

    # Zeitachse extrahieren
    time = df["valid_time"]

    # Einzelvariablen zeichnen
    for ax, (col, label, color) in zip(axes, cols):
        ax.plot(time, df[col], color=color, lw=PLOT_CONFIG["linewidth"])      # Datenreihe zeichnen
        ax.set_ylabel(label, fontsize=PLOT_CONFIG["fontsize"])                 # Y-Label setzen
        if PLOT_CONFIG["grid"]:
            ax.grid(True, linestyle=PLOT_CONFIG["grid_style"], alpha=PLOT_CONFIG["grid_alpha"])  # Gitter

    # Power-Subplot anhaengen (gemeinsam fuer berechnete und Mittelwertleistung)
    if add_power_subplot:
        ax_power = axes[-1] if len(cols) > 0 else axes[0]                      # Ziel-Achse bestimmen
        lines, labels = [], []                                                 # Fuer die Legende sammeln
        if avg_col is not None:
            l1, = ax_power.plot(time, df[avg_col], color="#d62728", lw=PLOT_CONFIG["linewidth"])  # Mittelwertleistung
            lines.append(l1); labels.append("Average power [W]")
        if pv_col is not None:
            l2, = ax_power.plot(time, df[pv_col], color="#2ca02c", lw=PLOT_CONFIG["linewidth"])   # Berechnete PV-Leistung
            lines.append(l2); labels.append("Calculated PV power [W]")
        ax_power.set_ylabel("Power [W]", fontsize=PLOT_CONFIG["fontsize"])    # Gemeinsame Y-Achse
        if PLOT_CONFIG["grid"]:
            ax_power.grid(True, linestyle=PLOT_CONFIG["grid_style"], alpha=PLOT_CONFIG["grid_alpha"])  # Gitter
        if lines:
            ax_power.legend(lines, labels, fontsize=PLOT_CONFIG["fontsize"], loc="upper left")     # Legende

    # X-Achse links/rechts leicht beschneiden
    xmin, xmax = time.min(), time.max()                                       # Zeitbereich ermitteln
    span = xmax - xmin
    xmin += span * PLOT_CONFIG["margin_frac"]
    xmax -= span * PLOT_CONFIG["margin_frac"]
    axes[0].set_xlim(xmin, xmax)                                              # Achsenlimits setzen

    # X-Achse formatieren (Ticks/Labels)
    locator = mdates.AutoDateLocator(
        minticks=PLOT_CONFIG["xtick_density"] // 2,
        maxticks=PLOT_CONFIG["xtick_density"]
    )
    formatter = mdates.DateFormatter("%Y-%m-%d\n%H:%M")
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)

    # X-Labels drehen
    plt.setp(axes[-1].get_xticklabels(), rotation=PLOT_CONFIG["rotation"], ha="center")

    # Titel und Layout
    fig.suptitle(title, fontsize=PLOT_CONFIG["fontsize"] + 2)
    fig.tight_layout()

    return fig


def list_exports(directory: Path) -> List[Path]:
    # bevorzugt *_with_pv_power.csv, ansonsten alle csv
    all_csv = sorted(directory.glob("*_with_pv_power.csv"))
    if all_csv:
        return all_csv
    return sorted(directory.glob("*.csv"))


def browse_exports(exports_dir: Path):
    files = list_exports(exports_dir)
    if not files:
        raise FileNotFoundError(f"Keine CSVs in '{exports_dir}' gefunden.")

    idx = 0
    df = load_data(files[idx])

    def make_title(i: int) -> str:
        return f"Physical model inputs and power\n{files[i].name}  ({i+1}/{len(files)})"

    fig = plot_data(df, title=make_title(idx))

    # Buttons anlegen
    ax_prev = fig.add_axes([0.70, 0.01, 0.08, 0.05])
    ax_next = fig.add_axes([0.80, 0.01, 0.08, 0.05])
    ax_quit = fig.add_axes([0.60, 0.01, 0.08, 0.05])
    b_prev = Button(ax_prev, "Prev")
    b_next = Button(ax_next, "Next")
    b_quit = Button(ax_quit, "Quit")

    def show(i: int):
        nonlocal idx
        idx = i % len(files)
        df2 = load_data(files[idx])
        plot_data(df2, title=make_title(idx), fig=fig)
        fig.canvas.draw_idle()

    def on_prev(event):
        show(idx - 1)

    def on_next(event):
        show(idx + 1)

    def on_quit(event):
        plt.close(fig)

    def on_key(event):
        k = (event.key or "").lower()
        if k in ("right", "n", " "):
            on_next(event)
        elif k in ("left", "p", "backspace"):
            on_prev(event)
        elif k in ("escape", "q"):
            on_quit(event)

    b_prev.on_clicked(on_prev)
    b_next.on_clicked(on_next)
    b_quit.on_clicked(on_quit)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()


def main():
    # Standard-Eingabe: genau ein im Code vorgegebener Datensatz
    # Interaktiver Modus per Flag oder wenn SELECTED_EXPORT nicht gesetzt ist
    browse_flag = len(sys.argv) > 1 and sys.argv[1] in ("--browse", "-b", "browse")
    selected_missing = (
        SELECTED_EXPORT is None or (isinstance(SELECTED_EXPORT, str) and SELECTED_EXPORT.strip() == "")
    )
    if browse_flag or (len(sys.argv) == 1 and selected_missing):
        browse_exports(INPUT_DIR)
        return
    
    default_in = INPUT_DIR / SELECTED_EXPORT                        # Fester Datensatzname
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_in  # Optional per CLI ueberschreibbar
    out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else None        # Optionaler PNG-Ausgabepfad

    if not in_path.exists():
        raise FileNotFoundError(
            f"Ausgewählte Datei nicht gefunden: '{in_path}'. Bitte 'SELECTED_EXPORT' anpassen."
        )

    # Daten laden
    df = load_data(in_path)                                           # CSV einlesen

    # Plot erstellen
    fig = plot_data(df)                                               # Abbildung erzeugen

    # Speichern oder anzeigen
    if out_png is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_png = OUTPUT_DIR / f"{in_path.stem}.png"

    fig.savefig(out_png)
    print(f"Plot gespeichert unter: '{out_png}'.")
    plt.close(fig)


if __name__ == "__main__":
    main()
