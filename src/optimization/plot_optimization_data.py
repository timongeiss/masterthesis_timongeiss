"""
Visualisiert Optimierungsdaten (Last, reale Erzeugung, DA-Preis und reBAP-Preise)
auf einer gemeinsamen Zeitachse.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data" / "optimization" / "optimization_data.csv"

TIME_COLUMN = "datetime"
POWER_COLUMNS = ["Last [W]", "Reale Erzeugung [W]"]
PRICE_COLUMNS = [
    "DA Preis [EUR/MWh]",
    "reBAP unterdeckt [EUR/MWh]",
]
REQUIRED_COLUMNS = {TIME_COLUMN, *POWER_COLUMNS, *PRICE_COLUMNS}


def load_data(path: Path) -> pd.DataFrame:
    """Liest die Optimierungsdaten ein, validiert Pflichtspalten und bereitet sie plotbar auf."""
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    df = pd.read_csv(path, decimal=",")
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Fehlende Spalten in {path}: {missing}")

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
    for col in POWER_COLUMNS + PRICE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Doppelte Zeitstempel werden gemittelt, damit die Kurven eindeutig je Zeitpunkt sind.
    df = (
        df.dropna(subset=[TIME_COLUMN])
        .groupby(TIME_COLUMN, as_index=False)[POWER_COLUMNS + PRICE_COLUMNS]
        .mean(numeric_only=True)
        .sort_values(TIME_COLUMN)
    )
    if df.empty:
        raise ValueError("Keine gueltigen Datenzeilen nach Bereinigung vorhanden.")
    return df


def plot_data(df: pd.DataFrame, title: str) -> None:
    """Erstellt den Zeitreihen-Plot mit Leistungswerten links und Preiswerten rechts."""
    fig, ax_power = plt.subplots(figsize=(14, 6), dpi=120)
    ax_price = ax_power.twinx()

    line_last = ax_power.plot(
        df[TIME_COLUMN], df["Last [W]"], linewidth=1.0, color="#1f77b4", label="Last [W]"
    )[0]
    line_gen = ax_power.plot(
        df[TIME_COLUMN],
        df["Reale Erzeugung [W]"],
        linewidth=1.0,
        color="#ff7f0e",
        label="Reale Erzeugung [W]",
    )[0]
    line_da = ax_price.plot(
        df[TIME_COLUMN],
        df["DA Preis [EUR/MWh]"],
        linewidth=1.0,
        color="#2ca02c",
        label="DA Preis [EUR/MWh]",
    )[0]
    line_rebap_under = ax_price.plot(
        df[TIME_COLUMN],
        df["reBAP unterdeckt [EUR/MWh]"],
        linewidth=1.0,
        linestyle="-",
        color="#d027d6",
        label="reBAP unterdeckt [EUR/MWh]",
    )[0]

    ax_power.set_title(title)
    ax_power.set_xlabel("Zeit")
    ax_power.set_ylabel("Leistung [W]")
    ax_price.set_ylabel("Preis [EUR/MWh]")
    ax_power.grid(True, linestyle=":", alpha=0.5)

    handles = [line_last, line_gen, line_da, line_rebap_under]
    labels = [h.get_label() for h in handles]
    ax_power.legend(handles, labels, loc="upper right")

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Startet das Laden der Daten und zeigt den Plot fuer die visuelle Pruefung an."""
    title = "Optimierungsdaten: Last, reale Erzeugung, DA-Preis und reBAP"
    df = load_data(INPUT_CSV)
    plot_data(df, title=title)


if __name__ == "__main__":
    main()
