import os
import xarray as xr
import pandas as pd
import configparser
import datetime
import numpy as np

from tqdm import tqdm
import re
import sys
sys.stdout.reconfigure(line_buffering=True)


# ======================================
# KONFIGURATION
# ======================================
CONFIG_PATH = "/app/config.txt"
DATA_DIR = "/data"
EXPORT_DIR = "/app/exports"

DEBUGMODE = False

# CONFIG_PATH = "config.txt"
# DATA_DIR = "shared"
# EXPORT_DIR = "exports"

config = configparser.ConfigParser()
config.read(CONFIG_PATH)
LAT = float(config["location"]["lat"])
LON = float(config["location"]["lon"])


VAR_MAPPING = {
    "aswdir_s": "aswdir_s",  # entspricht ASWDIR_S
    "aswdifd_s": "aswdifd_s",  # entspricht ASWDIFD_S
    "t_2m": "t2m",
}



# ======================================
# DATAEXTRACT FUNKTIONEN
# ======================================


def extract_point_from_grib(filepath: str, lat: float, lon: float) -> pd.DataFrame | None:
    """
    Öffnet eine ICON-D2-GRIB2-Datei und extrahiert alle Zeitschritte (0–4 pro Datei)
    der passenden Variable am angegebenen Punkt (nearest neighbor).

    Gibt einen DataFrame mit Spalten:
    ['valid_time', 'value', 'variable', 'latitude', 'longitude', 'source_file']
    zurück oder None, falls keine Werte gefunden werden.
    """
    try:
        if DEBUGMODE:
            print(f"INFO - Beginne mit {os.path.basename(filepath)}")

        ds = xr.open_dataset(
            filepath,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
            decode_timedelta=True
        )

        # passenden Variablennamen im Dataset finden
        grib_key = None
        for desired, actual in VAR_MAPPING.items():
            for candidate in ds.data_vars.keys():
                if candidate.lower() == actual.lower():
                    grib_key = candidate
                    break
            if grib_key:
                break

        if grib_key is None:
            grib_key = list(ds.data_vars)[0]

        if DEBUGMODE:
            print(f"INFO - GRIB-Key ist {grib_key}")

        # Punkt extrahieren
        da = ds[grib_key].sel(latitude=lat, longitude=lon, method="nearest")

        # --- Variante A: Mehrere Zeitschritte vorhanden ---
        if da.ndim >= 1 and da.size > 1:
            df = da.to_dataframe().reset_index()
            if "valid_time" not in df.columns:
                if "time" in df.columns:
                    df.rename(columns={"time": "valid_time"}, inplace=True)
                elif "step" in df.columns:
                    base_time = pd.Timestamp(ds.time.values)
                    df["valid_time"] = [base_time + pd.to_timedelta(s.values) for s in df["step"]]
                else:
                    df["valid_time"] = pd.NaT
            df["value"] = df[grib_key]
            df["variable"] = grib_key.lower()

        # --- Variante B: Einzelwert (z. B. bei +48 h Dateien oder T2M) ---
        elif da.size == 1:
            ts = None
            if "time" in ds:
                ts = pd.Timestamp(ds["time"].values)
            elif "valid_time" in ds:
                ts = pd.Timestamp(ds["valid_time"].values)
            else:
                # Fallback aus Dateiname
                m = re.search(r"_(\d{10})_(\d{3})_", os.path.basename(filepath))
                if m:
                    base_str, hour_offset = m.groups()
                    ts = pd.Timestamp(datetime.strptime(base_str, "%Y%m%d%H")) + pd.Timedelta(hours=int(hour_offset))
                else:
                    ts = pd.NaT

            df = pd.DataFrame([{
                "valid_time": ts,
                "value": float(da.values),
                "variable": grib_key.lower()
            }])

        # --- Variante C: Kein gültiger Inhalt ---
        else:
            print(f"WARNUNG - Keine Werte in {os.path.basename(filepath)} gefunden.")
            return None

        # Zusatzinfos
        df["latitude"] = float(da.latitude.values)
        df["longitude"] = float(da.longitude.values)
        df["source_file"] = os.path.basename(filepath)

        if DEBUGMODE:
            print(f"INFO - {len(df)} Zeitschritte extrahiert aus {os.path.basename(filepath)}")
        return df

    except Exception as e:
        print(f"ERROR - Fehler beim Verarbeiten von {filepath}: {e}")
        return None






def reconstruct_absolute(df: pd.DataFrame, time_column: str, value_column: str) -> pd.DataFrame:
    """
    Rekonstruiert absolute (intervallweise) Werte aus kumulativen Mittelwerten M_i.
    Annahme: M_i ist der kumulative Mittelwert von t0..t_i (ICON-Logik).
    Dann gilt für das Intervall (t_{i-1}, t_i]:
        A_i = (t_i*M_i - t_{i-1}*M_{i-1}) / (t_i - t_{i-1})
    Die Ausgabereihe ist LINKS-bündig: Zeitstempel = t_{i-1}.
    """
    df = df.copy()

    # Zeiten + Zahlen bereinigen
    df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
    df[value_column] = (
        df[value_column].astype(str).str.replace(",", ".", regex=False)
    )
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")

    # sortieren, duplikate entfernen, nur gültige Zeilen behalten
    df = (
        df.dropna(subset=[time_column, value_column])
          .sort_values(time_column)
          .drop_duplicates(subset=[time_column], keep="last")
          .reset_index(drop=True)
    )

    n = len(df)
    if n < 2:
        # nichts zu rekonstruieren – einfach zurückgeben
        return df[[time_column]].rename(columns={time_column: "valid_time"}).assign(Original=df[value_column])

    t = df[time_column].to_numpy()
    # Stunden relativ zu t[0] (für stabile Numerik; Differenzen bleiben identisch)
    t_hours = (pd.to_datetime(t) - t[0]) / pd.Timedelta(hours=1)
    y = df[value_column].to_numpy(dtype=float)

    # Intervallwerte berechnen, Länge n-1, am linken Rand ausgerichtet (t[0..n-2])
    out_vals = np.full(n-1, np.nan)
    for i in range(1, n):
        dt = float(t_hours[i] - t_hours[i-1])
        if dt != 0 and np.isfinite(y[i]) and np.isfinite(y[i-1]):
            out_vals[i-1] = (t_hours[i] * y[i] - t_hours[i-1] * y[i-1]) / dt

    out_times = df[time_column].iloc[:-1].to_numpy()  # linker Rand
    out = pd.DataFrame({time_column: out_times, "Original": out_vals})
    
    return out




def resample_temperature_to_radiation(df_temp: pd.DataFrame, radiation_times: pd.Series) -> pd.DataFrame:
    """
    Bringt stündliche Temperaturdaten (t_2m) auf dieselbe Zeitbasis wie die Strahlungsdaten (z. B. 15-minütig).
    - Erstellt künstliche Stundenzeitachse beginnend bei der ersten Strahlungszeit
    - Interpoliert linear auf 15min
    - Nutzt nearest-match für finale Reindexierung
    """

    if df_temp.empty or "valid_time" not in df_temp.columns:
        print("WARNUNG - Keine gültigen Temperaturdaten.")
        return df_temp

    df_temp = df_temp.copy()
    df_temp["valid_time"] = pd.to_datetime(df_temp["valid_time"], errors="coerce")

    val_col = "t_2m" if "t_2m" in df_temp.columns else "value"
    df_temp[val_col] = df_temp[val_col].astype(str).str.replace(",", ".", regex=False)
    df_temp[val_col] = pd.to_numeric(df_temp[val_col], errors="coerce")

    df_temp = df_temp.dropna(subset=[val_col]).reset_index(drop=True)

    # ---- künstliche Stundenachse ab erster Strahlungszeit ----
    start_time = pd.to_datetime(radiation_times.iloc[0])
    n_vals = len(df_temp)
    df_temp["valid_time"] = [start_time + pd.Timedelta(hours=i) for i in range(n_vals)]

    # ---- Interpolation auf 15 Minuten ----
    df_temp = df_temp.set_index("valid_time").resample("15min").interpolate("linear")

    # ---- Auf Strahlungszeitbasis mappen (nearest) ----
    radiation_times = pd.to_datetime(radiation_times, errors="coerce").dropna().sort_values()
    df_temp = df_temp.reindex(df_temp.index.union(radiation_times)).interpolate("time")
    df_temp = df_temp.loc[radiation_times]
    df_temp = df_temp.rename(columns={val_col: "t_2m"}).reset_index().rename(columns={"index": "valid_time"})

    # ---- Debug-Export ----
    if DEBUGMODE:
        os.makedirs(EXPORT_DIR, exist_ok=True)
        debug_path = os.path.join(EXPORT_DIR, "DEBUG_temp_resampled.csv")
        df_temp.to_csv(debug_path, index=False, decimal=",")
        print(f"DEBUG: Temperatur auf Strahlungszeitbasis exportiert → {debug_path} ({len(df_temp)} Zeilen)")

    return df_temp





# ======================================
# PIPELINE-FUNKTIONEN
# ======================================


def find_unique_forecasts(files: list[str]) -> list[str]:
    """Extrahiert eindeutige Forecast-Timestamps (YYYYMMDDHH) aus Dateinamen"""
    pattern = re.compile(r"(\d{10})_")
    forecasts = sorted(set(re.search(pattern, f).group(1) for f in files if re.search(pattern, f)))
    return forecasts



def filter_files_for_forecast(files: list[str], forecast_id: str) -> list[str]:
    """Filtert alle Dateien, die zu einem Forecast-Zeitpunkt gehören"""
    return [f for f in files if forecast_id in f]



def process_forecast(forecast_id: str, files: list[str]) -> None:
    """Verarbeitet alle Dateien einer Prognose (alle Variablen, 49 Schritte)"""
    print(f"\nINFO - Verarbeite Prognose {forecast_id}")

    forecast_files = filter_files_for_forecast(files, forecast_id)
    grouped = {var: [f for f in forecast_files if var in f.lower()] for var in VAR_MAPPING.keys()}
    results = {}

    for var, flist in grouped.items():
        if not flist:
            print(f"WARNUNG - Keine Dateien gefunden für {var}")
            continue

        all_dfs = []
        for f in sorted(flist):
            full_path = os.path.join(DATA_DIR, f)
            df = extract_point_from_grib(full_path, LAT, LON)
            if df is not None and not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            continue

        # Zeitschritte sammeln
        df_var = pd.concat(all_dfs, ignore_index=True)

        # passende Wertspalte finden
        value_col = None
        for c in df_var.columns:
            if c.lower() in [var.lower(), "value", "aswdifd_s", "aswdir_s", "t_2m"]:
                value_col = c
                break
        if value_col is None:
            print(f"WARNUNG - Keine passende Wertspalte in {var} gefunden, überspringe.")
            continue

        # Vorreinigung (nur Parsen/Sortieren)
        df_var = df_var.copy()
        df_var["valid_time"] = pd.to_datetime(df_var["valid_time"], errors="coerce")
        df_var[value_col] = df_var[value_col].astype(str).str.replace(",", ".", regex=False)
        df_var[value_col] = pd.to_numeric(df_var[value_col], errors="coerce")

        # Für Strahlungsdaten: Duplikate nach Zeitstempel entfernen
        if var in ["aswdir_s", "aswdifd_s"]:
            df_var = (
                df_var.dropna(subset=["valid_time", value_col])
                    .sort_values("valid_time")
                    .drop_duplicates(subset=["valid_time"], keep="last")
                    .reset_index(drop=True)
            )
        else:
            # Für Temperatur: NICHT nach valid_time deduplizieren!
            df_var = (
                df_var.dropna(subset=[value_col])
                    .sort_values(value_col)
                    .reset_index(drop=True)
            )

        if var in ["aswdir_s", "aswdifd_s"]:
            df_rec = reconstruct_absolute(df_var, "valid_time", value_col)
            df_rec = df_rec.rename(columns={"Original": var})
            
            if DEBUGMODE:
                debug_after_path = os.path.join(EXPORT_DIR, f"DEBUG_after_{var}_{forecast_id}.csv")
                os.makedirs(EXPORT_DIR, exist_ok=True)
                df_rec.to_csv(debug_after_path, index=False, decimal=",")
                print(f"DEBUG: Nach Rekonstruktion exportiert → {debug_after_path} ({len(df_rec)} Zeilen)")
                
            df_var = df_rec
            
        else:
            if value_col != var:
                df_var = df_var.rename(columns={value_col: var})
            df_var = df_var[["valid_time", var]]

        results[var] = df_var

    # ---- Temperatur auf Strahlungszeitbasis bringen ----
    if "t_2m" in results and "aswdifd_s" in results:
        if DEBUGMODE:
            print("INFO - Resampling Temperatur auf 15min Basis...")
        df_temp = results["t_2m"]
        radiation_times = results["aswdifd_s"]["valid_time"]
        results["t_2m"] = resample_temperature_to_radiation(df_temp, radiation_times)

    # ---- Gemeinsame Zeitbasis / Merge ----
    if not results:
        print(f"ERROR - Keine Daten für Prognose {forecast_id}")
        return

    for var_name in results:
        df_clean = results[var_name].copy()
        df_clean["valid_time"] = pd.to_datetime(df_clean["valid_time"], errors="coerce")
        df_clean = (
            df_clean.dropna(subset=["valid_time"])
                    .sort_values("valid_time")
                    .groupby("valid_time", as_index=False)
                    .mean(numeric_only=True)
        )
        results[var_name] = df_clean

    time_df = results[list(results.keys())[0]][["valid_time"]].copy()

    for var_name, df in results.items():
        time_df = pd.merge(time_df, df[["valid_time", var_name]], on="valid_time", how="left")

    time_df = time_df.sort_values("valid_time").dropna(subset=list(results.keys()), how="all")

    # Export
    os.makedirs(EXPORT_DIR, exist_ok=True)
    out_path = os.path.join(EXPORT_DIR, f"forecast_{forecast_id}.csv")
    time_df.to_csv(out_path, index=False, decimal=",")
    print(f"INFO - Exportiert: {out_path} ({len(time_df)} Zeilen)")



def combine_forecasts(export_dir: str, output_name: str = "merged_dataset.csv") -> None:
    """Fügt alle Forecast-CSV-Dateien zu einer konsistenten Zeitreihe zusammen."""

    csv_files = sorted([f for f in os.listdir(export_dir) if f.startswith("forecast_") and f.endswith(".csv")])
    if not csv_files:
        print("WARNUNG - Keine Forecast-Dateien gefunden.")
        return

    # Forecast-Startzeiten extrahieren
    forecasts = []
    for f in csv_files:
        ts = re.search(r"forecast_(\d{10})", f)
        if ts:
            start = pd.to_datetime(ts.group(1), format="%Y%m%d%H")
            forecasts.append((start, f))

    forecasts = sorted(forecasts, key=lambda x: x[0])
    merged = []

    for i, (start, fname) in enumerate(forecasts):
        file_path = os.path.join(export_dir, fname)
        df = pd.read_csv(file_path, sep=",", decimal=",")
        df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce")
        df = df.dropna(subset=["valid_time"])

        # Gültigkeitsbereich definieren (erste 1h überspringen)
        valid_from = start + pd.Timedelta(hours=1)
        valid_to = forecasts[i + 1][0] if i + 1 < len(forecasts) else None

        if valid_to:
            df = df[(df["valid_time"] >= valid_from) & (df["valid_time"] < valid_to)]
        else:
            df = df[df["valid_time"] >= valid_from]

        merged.append(df)
        print(f"INFO - Forecast {start:%Y-%m-%d %H:%M} → {len(df)} Zeilen übernommen.")

    merged_df = pd.concat(merged, ignore_index=True).sort_values("valid_time").reset_index(drop=True)

    out_path = os.path.join(export_dir, output_name)
    merged_df.to_csv(out_path, index=False, decimal=",")
    print(f"\n✅ Zusammengeführter Datensatz exportiert: {out_path} ({len(merged_df)} Zeilen)")




# ======================================
# HAUPTPROGRAMM
# ======================================

def main():
    # Ordner suchen
    if not os.path.exists(DATA_DIR):
        print("ERROR - Datenverzeichnis nicht gefunden.")
        return

    # Alle Grib Files suchen
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".grib2"))
    if not files:
        print("ERROR - Keine GRIB2-Dateien gefunden.")
        return

    forecasts = find_unique_forecasts(files)
    print(f"INFO - {len(forecasts)} Prognosen gefunden.")

    # Batchweise verarbeiten
    for forecast_id in tqdm(forecasts, desc="Forecasts"):
        process_forecast(forecast_id, files)
        print(f"✅  Prognose {forecast_id} abgeschlossen.\n", flush=True)
        
    combine_forecasts(EXPORT_DIR)


if __name__ == "__main__":
    main()

