# checked 29.11.2025


"""
Resample REA6 ID-Zeitreihen auf 15 Minuten und Solar Elevation berechnen

Workflow:
- metadata.csv einlesen (Koordinaten, begin/end_ts)
- Alle CSVs in exports/ suchen (z. B. IDxxx_rea6_*.csv)
- Pro ID:
    * CSV mit wetterzeitreihe laden, Zeitspalte parsen (UTC), sortieren
    * Auf 15-min-Gitter resamplen (linear, 3 Zwischenpunkte)
    * Solar Elevation mit pvlib.solarposition.get_solarposition berechnen
    * Ergebnis als <ID>_rea6_15min.csv speichern
"""

from pathlib import Path
import pandas as pd
import pvlib



BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]


INPUT_DIR  = PROJECT_ROOT / "data" / "processed" / "extracted_rea"
EXPORT_DIR = PROJECT_ROOT / "data" / "processed" / "resampled_rea"
METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "source_label_space" / "metadata.csv"
POWER_PATH = PROJECT_ROOT / "data" / "raw" / "source_label_space" / "filtered_pv_power_measurements_ac.csv"

def load_metadata() -> pd.DataFrame:
    
    meta = pd.read_csv(METADATA_PATH, sep=";")
    meta["begin_ts"] = pd.to_datetime(meta["begin_ts"], utc=True).dt.tz_convert(None)
    meta["end_ts"] = pd.to_datetime(meta["end_ts"], utc=True).dt.tz_convert(None).dt.ceil("h")
    
    return meta.set_index("ID")


def load_power_series(target_id: str) -> pd.DataFrame | None:
    """
    Liest die Power-Daten einer ID aus dem Datensatz und resampelt auf 15min.
    """
    
    df_power = pd.read_csv(POWER_PATH, decimal=".", usecols=["DateTime", target_id],parse_dates=["DateTime"])
    
    time_index = pd.to_datetime(df_power["DateTime"], utc=True, errors="coerce").dt.tz_convert(None)
    
    df_power = df_power.set_index(time_index).drop(columns=["DateTime"])
    df_power = df_power.rename(columns={target_id: "power_ac"})
    
    df_power = df_power.resample("15min").mean()
    
    return df_power



def process_id(csv_path: Path, meta: pd.Series) -> None:
    
    df = pd.read_csv(csv_path, parse_dates=["time"])
    
    # Zeitstempel robust parsen und tz-naiv machen
    time_index = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.set_index(time_index).sort_index()
    df = df.drop(columns=["time"])

    # Wetter auf 15 Minuten resamplen (strecken)
    df = df.resample("15min").interpolate("linear")

    # Zeitfenster zuschneiden
    begin_ts = meta["begin_ts"]
    end_ts = meta["end_ts"]
    df = df.loc[(df.index >= begin_ts) & (df.index <= end_ts)]

    # Temperatur-Celsius berechnen für solarpos
    temps_c = df["T_2M"] - 273.15

    lat = float(meta["north"])
    lon = float(meta["west"])
    alt = 5.0   #utrecht etwa 5m über meer

    solar = pvlib.solarposition.get_solarposition(
        time=df.index,
        latitude=lat,
        longitude=lon,
        altitude=alt,
        temperature=temps_c,
        method="nrel_numpy",
    )
    
    df["solar_elevation_deg"] = solar["elevation"]  #get_solarpos liefert verschiedene winkel

    # Power-Daten (15min) hinzujoinen, falls vorhanden
    target_id_column = csv_path.stem.split("_")[0]
    power_series = load_power_series(target_id_column)
    if power_series is not None:
        df = df.join(power_series, how="left")

    # Zeitstempel wieder als Spalte ausgeben
    out_df = df.reset_index().rename(columns={"index": "time"})

    # Unterordner
    out_dir = EXPORT_DIR

    # Datei im Unterordner
    out_path = out_dir / f"{csv_path.stem.replace('_rea6', '')}_rea6_15min.csv"

    # Speichern
    out_df.to_csv(out_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"Gespeichert: {out_path} (Zeilen: {len(out_df)})")





def main() -> None:
    
    meta = load_metadata()

    csv_files = sorted(INPUT_DIR.glob("*_rea6.csv"))

    for csv_path in csv_files:
        # ID aus Dateinamen extrahieren: vor dem ersten '_' bis ggf. Ende
        stem_parts = csv_path.stem.split("_")
        target_id = stem_parts[0]
    
        if target_id not in meta.index:
            print(f"ID {target_id} nicht in metadata, ueberspringe {csv_path}")
            continue
        
        print(f"Verarbeite {csv_path} fuer ID {target_id}")
        process_id(csv_path, meta.loc[target_id])


if __name__ == "__main__":
    main()
