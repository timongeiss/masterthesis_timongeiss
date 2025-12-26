import os
import re
from pathlib import Path
import pandas as pd
import pvlib

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# --- Wetterdaten laden ---
def load_weather(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["valid_time"], decimal=",")
    if missing := {"valid_time", "aswdir_s", "aswdifd_s", "t_2m"} - set(df.columns):
        raise ValueError(f"{path}: fehlende Spalten {missing}")
    return df


# --- PV-Daten laden (nur Jahr 2025) ---
def load_pv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"], decimal=",")
    if missing := {"datetime", "Mittelwertleistung [W]"} - set(df.columns):
        raise ValueError(f"{path}: fehlende Spalten {missing}")
    return df[df["datetime"].dt.year == 2025]


# --- Zeitzonen angleichen ---
def convert_tz(series, from_tz, to_tz):
    idx = pd.DatetimeIndex(series)
    if idx.tz is None:
        idx = idx.tz_localize(from_tz, ambiguous="infer", nonexistent="shift_forward")
    return idx.tz_convert(to_tz).tz_localize(None)


# --- Wetter- und PV-Daten zusammenführen ---
def merge_data(weather, pv):
    df = weather.merge(pv, left_on="valid_time", right_on="datetime").drop(columns="datetime")
    cols = ["valid_time", "aswdir_s", "aswdifd_s", "t_2m", "Mittelwertleistung [W]"]
    if "source_file" in df.columns:
        cols.append("source_file")
    return df[cols].dropna(subset=["valid_time"]).sort_values("valid_time").reset_index(drop=True)


# --- NaN-Werte prüfen (nur Konsolenreport) ---
def report_nans(df):
    cols = ["aswdir_s", "aswdifd_s", "t_2m", "Mittelwertleistung [W]"]
    mask = df[cols].isna().any(axis=1)
    return df.loc[mask, ["valid_time"] + cols]


# --- Vollständigkeitsprüfung wie in combine_forecasts ---
def is_complete_forecast(df: pd.DataFrame, required_rows: int = 191) -> bool:
    return df.shape[0] == required_rows and df.count().eq(required_rows).all()


def main():
    # Verzeichnisse aus Umgebungsvariablen lesen (Defaults relativ zum Projekt)
    shared_dir = Path(os.getenv("ICON_SHARED_DIR", PROJECT_ROOT / "data" / "raw" / "target_feature_space"))
    export_dir = Path(os.getenv("ICON_EXPORT_DIR", PROJECT_ROOT / "data" / "processed" / "physical_input"))

    # Pfad zur PV-Datei
    pv_path = PROJECT_ROOT / "data" / "processed" / "target_label_space_leistung.csv"

    # Zeitzonen
    local_tz = "Europe/Berlin"
    weather_tz = "UTC"
    pv_tz = local_tz

    # Ausgabeverzeichnis vorbereiten
    os.makedirs(export_dir, exist_ok=True)

    # PV laden und in lokale Zeit überführen
    pv = load_pv(pv_path)
    pv["datetime"] = convert_tz(pv["datetime"], pv_tz, local_tz)

    # Alle Forecast-Dateien finden
    all_files = sorted(
        f for f in os.listdir(shared_dir)
        if f.startswith("forecast_") and f.endswith(".csv")
    )
    if not all_files:
        print("WARNUNG - Keine Forecast-Dateien im shared-Ordner gefunden.")
        return

    processed, skipped = 0, 0

    for fname in all_files:
        m = re.search(r"forecast_(\d{10})", fname)
        if not m:
            skipped += 1
            continue
        ts_str = m.group(1)
        fpath = Path(shared_dir) / fname

        try:
            weather = load_weather(fpath)
        except Exception as e:
            print(f"SKIP {fname} - Lese-/Spaltenfehler: {e}")
            skipped += 1
            continue

        # Vollständigkeit prüfen (191 Zeilen, keine NaNs in Spalten)
        if not is_complete_forecast(weather, 191):
            print(f"SKIP {fname} - unvollständig (!=191 Zeilen oder NaNs)")
            skipped += 1
            continue

        # Zeiten in lokale Zeit konvertieren
        weather = weather.copy()
        weather["valid_time"] = convert_tz(weather["valid_time"], weather_tz, local_tz)
        weather["source_file"] = fname

        # Prüfen: PV muss den gesamten Forecast-Zeitraum (191 Zeitpunkte) abdecken
        expected = pd.DatetimeIndex(weather["valid_time"])  # 191 Zeitpunkte
        pv_idx = pd.DatetimeIndex(pv["datetime"])          # verfügbare PV-Zeitpunkte
        missing_mask = ~expected.isin(pv_idx)
        if missing_mask.any():
            missing_count = int(missing_mask.sum())
            print(
                f"SKIP {fname} - PV-Realleistung unvollständig: {missing_count}/" \
                f"{len(expected)} Zeitpunkte fehlen"
            )
            skipped += 1
            continue

        # Mergen mit PV
        merged = merge_data(weather, pv)

        # Sicherheitscheck: Ergebnis muss 191 Zeilen haben
        if len(merged) != 191:
            print(f"SKIP {fname} - gemergtes Ergebnis hat {len(merged)} statt 191 Zeilen")
            skipped += 1
            continue

        # Sonnenhöhe berechnen
        lat, lon, alt = 48.54728845042677, 11.273884308087036, 452
        temps = merged["t_2m"].to_numpy()
        temps_c = temps - 273.15 if pd.Series(temps).median() > 200 else temps
        times = pd.DatetimeIndex(merged["valid_time"]).tz_localize(
            local_tz, nonexistent="shift_forward", ambiguous="infer"
        )
        merged["solar_elevation_deg"] = pvlib.solarposition.get_solarposition(
            time=times, latitude=lat, longitude=lon, altitude=alt,
            temperature=temps_c, method="nrel_numpy"
        )["apparent_elevation"].values

        # Optional: NaN-Kurzreport in Konsole
        issues = report_nans(merged)
        if not issues.empty:
            missing_cols = ["aswdir_s", "aswdifd_s", "t_2m", "Mittelwertleistung [W]"]
            n = len(issues)
            print(f"WARNUNG {fname} - {n} Zeilen mit NaNs in {missing_cols}")

        # Ergebnis je Forecast speichern
        out_name = f"pv_weather_{ts_str}.csv"
        out_path = Path(export_dir) / out_name
        merged.to_csv(out_path, index=False, decimal=",")
        print(f"OK {fname} → {out_path} ({len(merged)} Zeilen)")
        processed += 1

    print(f"Fertig: {processed} Dateien erzeugt, {skipped} übersprungen.")


if __name__ == "__main__":
    main()
