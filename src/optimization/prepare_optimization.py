
"""
Bereitet die Zeitreihen fuer die Optimierung auf.

Ablauf:
1) Last aus target_label_space-Wochenexporten lesen und auf 15 Minuten mitteln
2) Reale PV-Erzeugung inkl. Solar-Elevation aus physical_input laden
3) Spotpreis und reBAP laden
4) Alle Reihen auf eine gemeinsame 15-Minuten-Achse bringen
5) Als data/optimization/optimization_data.csv speichern
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_COLUMNS = [
    "datetime",
    "Last [W]",
    "Reale Erzeugung [W]",
    "solar_elevation_deg",
    "DA Preis [EUR/MWh]",
    "reBAP unterdeckt [EUR/MWh]",
    "reBAP ueberdeckt [EUR/MWh]",
]


def read_target_label_space(prefix: str = "Inverter_Target-") -> pd.DataFrame:
    """Liest alle passenden Wochen-CSVs aus data/raw/target_label_space."""
    folder = PROJECT_ROOT / "data" / "raw" / "target_label_space"
    csv_files = sorted(f for f in folder.iterdir() if f.name.startswith(prefix) and f.suffix == ".csv")

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
                "Akku Stromstaerke [A]",
            ],
        )
        dfs.append(df)

    # Wenn keine Datei passt, liefert pd.concat bewusst die Standard-Exception.
    return pd.concat(dfs, ignore_index=True)


def clean_time_and_numeric(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Parst die Zeitspalte robust und konvertiert eine Zielspalte zu float."""
    out = df.copy()

    raw_time = out["Uhrzeit"].astype(str).str.strip()

    # Mehrstufiges Datumsparsing:
    # 1) strikt mit Sekunden, 2) ohne Sekunden, 3) generisch dayfirst.
    parsed = pd.to_datetime(raw_time, format="%d.%m.%Y %H:%M:%S", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(raw_time.loc[missing], format="%d.%m.%Y %H:%M", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(raw_time.loc[missing], errors="coerce", dayfirst=True)

    out["Uhrzeit"] = parsed
    out = out.dropna(subset=["Uhrzeit"])

    # Einheitliche numerische Konvertierung fuer Komma/Punkt-Formate.
    out[value_col] = pd.to_numeric(out[value_col].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    return out.sort_values("Uhrzeit")


def process_load_15min_w(df: pd.DataFrame) -> pd.DataFrame:
    """Erzeugt 15-Minuten-Lastwerte in W aus Rohlast in kW."""
    cleaned = clean_time_and_numeric(df, value_col="Stromverbrauch [kW]")
    s = cleaned.set_index("Uhrzeit")["Stromverbrauch [kW]"].sort_index()

    # Zielraster: 15-Minuten-Zeitpunkte als zentrierte Fenster-Mittelwerte.
    start = s.index.min().ceil("15min")
    end = s.index.max().floor("15min")
    centers = pd.date_range(start=start, end=end, freq="15min") if start <= end else pd.DatetimeIndex([])

    s2 = s.reindex(s.index.union(centers)).sort_index()
    mean15 = s2.rolling("15min", center=True, min_periods=1).mean()

    out = (mean15.loc[centers] * 1000.0).reset_index()
    out.columns = ["datetime", "Last [W]"]
    return out


def read_real_generation() -> pd.DataFrame:
    """Liest reale PV + Solar-Elevation aus data/processed/physical_input/*.csv."""
    folder = PROJECT_ROOT / "data" / "processed" / "physical_input"
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Keine CSV-Dateien in {folder} gefunden.")

    parts = []
    for file_order, file in enumerate(csv_files):
        df = pd.read_csv(file, usecols=["valid_time", "Mittelwertleistung [W]", "solar_elevation_deg"])
        df["datetime"] = pd.to_datetime(df["valid_time"], errors="coerce")
        df["Mittelwertleistung [W]"] = _to_numeric(df["Mittelwertleistung [W]"])
        df["solar_elevation_deg"] = _to_numeric(df["solar_elevation_deg"])
        df["file_order"] = int(file_order)
        parts.append(df[["datetime", "Mittelwertleistung [W]", "solar_elevation_deg", "file_order"]])

    merged = pd.concat(parts, ignore_index=True)
    merged = merged.dropna(subset=["datetime"]).sort_values(["datetime", "file_order"], kind="mergesort")
    merged = merged.drop_duplicates(subset=["datetime"], keep="first")
    merged = merged.drop(columns=["file_order"])
    merged = merged.dropna(subset=["Mittelwertleistung [W]", "solar_elevation_deg"])

    return merged.rename(columns={"Mittelwertleistung [W]": "Reale Erzeugung [W]"})


def read_spot_prices() -> pd.DataFrame:
    """Liest Spotpreise (EUR/MWh) und liefert datetime + DA Preis."""
    path = PROJECT_ROOT / "data" / "raw" / "optimization" / "spot_prices_germany_2025.csv"

    # Die Quelldatei enthaelt hinter der Kopfzeile eine zweite Meta-Zeile.
    df = pd.read_csv(path, skiprows=1)

    time_col = df.columns[0]
    price_col = df.columns[1]

    df = df.rename(columns={time_col: "datetime", price_col: "DA Preis [EUR/MWh]"})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df["DA Preis [EUR/MWh]"] = pd.to_numeric(df["DA Preis [EUR/MWh]"], errors="coerce")

    # Auf lokale Marktzeit bringen und tz-Info entfernen, damit Merge auf naiver Zeitachse klappt.
    df["datetime"] = df["datetime"].dt.tz_convert("Europe/Berlin").dt.tz_localize(None)

    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    return df[["datetime", "DA Preis [EUR/MWh]"]]


def _to_numeric(series: pd.Series) -> pd.Series:
    """Konvertiert robust zu numerisch (Komma- und Punktnotation)."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def read_rebap_prices() -> pd.DataFrame:
    """Liest reBAP-Preise aus reBAP.csv und liefert datetime + unter/ueberdeckt."""
    csv_path = PROJECT_ROOT / "data" / "raw" / "optimization" / "reBAP.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {csv_path}")

    df = pd.read_csv(csv_path, sep=";", decimal=",")
    # Tabs/Mehrfachspaces in Kopfzeilen gl?tten.
    df = df.rename(columns={col: " ".join(str(col).strip().split()) for col in df.columns})

    def _pick_col(candidates: list[str]) -> str | None:
        for cand in candidates:
            if cand in df.columns:
                return cand
        return None

    date_col = _pick_col(["Datum", "date"])
    from_col = _pick_col(["von", "from", "start"])
    under_col = _pick_col(["reBAP unterdeckt", "reBAP unterdeckt [EUR/MWh]"])
    over_col = _pick_col(["reBAP ueberdeckt", "reBAP ueberdeckt [EUR/MWh]"])

    missing = [
        name
        for name, col in [
            ("Datum", date_col),
            ("von", from_col),
            ("reBAP unterdeckt", under_col),
            ("reBAP ueberdeckt", over_col),
        ]
        if col is None
    ]
    if missing:
        raise ValueError(f"Erwartete Spalten fehlen in {csv_path}: {missing}")

    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                df[date_col].astype(str).str.strip() + " " + df[from_col].astype(str).str.strip(),
                dayfirst=True,
                errors="coerce",
            ),
            "reBAP unterdeckt [EUR/MWh]": _to_numeric(df[under_col]),
            "reBAP ueberdeckt [EUR/MWh]": _to_numeric(df[over_col]),
        }
    )
    out = out.dropna(subset=["datetime", "reBAP unterdeckt [EUR/MWh]", "reBAP ueberdeckt [EUR/MWh]"])
    out = out.sort_values("datetime")
    return out


def build_optimization_dataframe(
    load_df: pd.DataFrame, generation_df: pd.DataFrame, prices_df: pd.DataFrame, rebap_df: pd.DataFrame
) -> pd.DataFrame:
    """Fuehrt Last, PV, Spotpreis und reBAP auf einer 15-Minuten-Zeitachse zusammen."""
    load_s = load_df.groupby("datetime")["Last [W]"].mean().sort_index()
    gen_s = generation_df.groupby("datetime")["Reale Erzeugung [W]"].mean().sort_index()
    solar_s = generation_df.groupby("datetime")["solar_elevation_deg"].mean().sort_index()
    price_s = prices_df.groupby("datetime")["DA Preis [EUR/MWh]"].mean().sort_index()
    rebap_under_s = rebap_df.groupby("datetime")["reBAP unterdeckt [EUR/MWh]"].mean().sort_index()
    rebap_over_s = rebap_df.groupby("datetime")["reBAP ueberdeckt [EUR/MWh]"].mean().sort_index()

    first_load, last_load = load_s.first_valid_index(), load_s.last_valid_index()
    first_gen, last_gen = gen_s.first_valid_index(), gen_s.last_valid_index()
    first_solar, last_solar = solar_s.first_valid_index(), solar_s.last_valid_index()
    first_price, last_price = price_s.first_valid_index(), price_s.last_valid_index()
    first_rebap_under, last_rebap_under = rebap_under_s.first_valid_index(), rebap_under_s.last_valid_index()
    first_rebap_over, last_rebap_over = rebap_over_s.first_valid_index(), rebap_over_s.last_valid_index()

    if any(
        x is None
        for x in [
            first_load,
            last_load,
            first_gen,
            last_gen,
            first_solar,
            last_solar,
            first_price,
            last_price,
            first_rebap_under,
            last_rebap_under,
            first_rebap_over,
            last_rebap_over,
        ]
    ):
        raise ValueError("Keine gueltigen Daten fuer Last, Erzeugung, Solarwinkel, Spotpreis oder reBAP vorhanden.")

    # Nur den gemeinsamen Ueberlappungsbereich aller Zeitreihen verwenden.
    start = max(first_load, first_gen, first_solar, first_price, first_rebap_under, first_rebap_over)
    end = min(last_load, last_gen, last_solar, last_price, last_rebap_under, last_rebap_over)
    full_index = pd.date_range(start=start, end=end, freq="15min")

    # Reindex auf gemeinsames Raster.
    out = pd.DataFrame(index=full_index)
    out["Last [W]"] = load_s.reindex(full_index)
    out["Reale Erzeugung [W]"] = gen_s.reindex(full_index)
    out["solar_elevation_deg"] = solar_s.reindex(full_index)
    out["DA Preis [EUR/MWh]"] = price_s.reindex(full_index)
    out["reBAP unterdeckt [EUR/MWh]"] = rebap_under_s.reindex(full_index)
    out["reBAP ueberdeckt [EUR/MWh]"] = rebap_over_s.reindex(full_index)

    cols = [
        "Last [W]",
        "Reale Erzeugung [W]",
        "solar_elevation_deg",
        "DA Preis [EUR/MWh]",
        "reBAP unterdeckt [EUR/MWh]",
        "reBAP ueberdeckt [EUR/MWh]",
    ]

    # Persistenzfuellung fuer einzelne Luecken im ueberlappenden Zeitraum.
    out[cols] = out[cols].ffill()
    out[cols] = out[cols].bfill()

    out = out.reset_index(names="datetime")
    return out


def validate_optimization_dataframe(df: pd.DataFrame, dt_minutes: int = 15) -> None:
    """Validiert Pflichtspalten, Vollstaendigkeit und lueckenloses Zeitraster."""
    missing_cols = [col for col in OUTPUT_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Fehlende Pflichtspalten in optimization_data: {missing_cols}")

    out = df[OUTPUT_COLUMNS].copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    numeric_cols = [col for col in OUTPUT_COLUMNS if col != "datetime"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    nan_counts = out.isna().sum()
    if int(nan_counts.sum()) > 0:
        bad = {k: int(v) for k, v in nan_counts.items() if int(v) > 0}
        raise ValueError(f"optimization_data enthaelt unvollstaendige Werte: {bad}")

    if out["datetime"].duplicated().any():
        dup_count = int(out["datetime"].duplicated().sum())
        raise ValueError(f"optimization_data enthaelt doppelte Zeitstempel: {dup_count}")

    out = out.sort_values("datetime")
    expected_step = pd.Timedelta(minutes=dt_minutes)
    deltas = out["datetime"].diff().dropna()
    invalid = deltas[deltas != expected_step]
    if not invalid.empty:
        first_idx = int(invalid.index[0])
        raise ValueError(
            f"optimization_data ist nicht lueckenlos im {dt_minutes}-Minuten-Raster "
            f"(erste Abweichung bei Zeile {first_idx}: {invalid.iloc[0]})."
        )


def save_optimization_data(df: pd.DataFrame, filename: str = "optimization_data.csv") -> Path:
    """Speichert die aufbereiteten Daten als CSV nach data/optimization."""
    out_dir = PROJECT_ROOT / "data" / "optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    df.to_csv(path, index=False, decimal=",")
    return path


def main() -> None:
    """Fuehrt den kompletten Aufbereitungsworkflow aus."""
    raw = read_target_label_space()
    load_15min = process_load_15min_w(raw)

    generation = read_real_generation()
    prices = read_spot_prices()
    rebap = read_rebap_prices()
    optimization_df = build_optimization_dataframe(load_15min, generation, prices, rebap)
    validate_optimization_dataframe(optimization_df, dt_minutes=15)

    out_path = save_optimization_data(optimization_df)
    print(f"optimization_data.csv gespeichert: {out_path} ({len(optimization_df)} Zeilen)")


if __name__ == "__main__":
    main()
