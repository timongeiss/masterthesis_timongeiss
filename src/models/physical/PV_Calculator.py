import os
import pandas as pd
#import numpy as np
#import pvlib
import configparser
from pathlib import Path

from PV_model import (
    create_location_pvlib,
    preprocess_weather,
    transposition_model,
    temperature_model,
    cell_model,
    scale_module_to_system_pdc,
    calculate_acpower_from_pdc,
)


# -----------------------------------
# CONFIGS
# -----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config_physical_pv_model.txt"

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

# Standortparameter
LAT = float(config["location"]["lat"])   # Breitengrad
LON = float(config["location"]["lon"])   # Längengrad

# Pfade
SHARED_DIR = PROJECT_ROOT / "data" / "processed" / "physical_input"
EXPORT_DIR = PROJECT_ROOT / "data" / "results" / "physical_output"

os.makedirs(EXPORT_DIR, exist_ok=True)

# Default DC rating per site inverter (W). Can be overridden per site in config via key 'dc_max'.
SITE_INVERTER_DC_MAX = 3500.0

# Debug window for targeted diagnostics (local time, naive timestamps)
DEBUG_START = pd.Timestamp("2025-08-19 16:00:00")
DEBUG_END = pd.Timestamp("2025-08-19 19:00:00")

def process_single_file(
    input_file: str,
    PVLIBlocation,
    inverter_meta: dict,
    modultyp_meta: dict,
):
    # Wetterdaten einlesen
    df_in = pd.read_csv(input_file, parse_dates=["valid_time"])  # CSV lesen und Zeitspalte parsen

    # Spalten robust auf Kleinbuchstaben abbilden
    df_in = df_in.rename(columns={c: c.lower() for c in df_in.columns})

    # Numerische Spalten robust in float konvertieren (auch Komma-Decimals)
    def _coerce_num(series: pd.Series) -> pd.Series:
        if series.dtype.kind in ("i", "u", "f"):
            return pd.to_numeric(series, errors="coerce")
        # strings/objects: erst Komma als Dezimalpunkt, dann konvertieren
        return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")

    # Sicherstellen, dass erwartete Spalten vorhanden sind
    required_cols = {"valid_time", "aswdir_s", "aswdifd_s", "t_2m"}
    missing = required_cols.difference(set(df_in.columns))
    if missing:
        raise ValueError(f"Missing required columns in input: {sorted(missing)}")

    # Konvertieren und bereinigen
    for col in ["aswdir_s", "aswdifd_s", "t_2m", "wind_speed"]:
        if col in df_in.columns:
            df_in[col] = _coerce_num(df_in[col])

    # Ungültige Temperaturen verwerfen (ohne Temperatur keine Solarpositionskorrektur)
    df_in = df_in[df_in["t_2m"].notna()].copy()
    if df_in.empty:
        raise ValueError("No valid numeric rows after coercion (t_2m all NaN)")

    # Wetter-DataFrame für das physikalische Modell
    df_weather = pd.DataFrame(
        {
            "timestamp": df_in["valid_time"],  # PV_model erwartet 'timestamp'
            "aswdir_s": df_in["aswdir_s"].fillna(0).clip(lower=0),
            "aswdifd_s": df_in["aswdifd_s"].fillna(0).clip(lower=0),
            "t_2m": df_in["t_2m"],  # Kelvin
            # Wind optional: falls vorhanden übernehmen, sonst 0.0
            "wind_speed": df_in["wind_speed"].fillna(0.0) if "wind_speed" in df_in.columns else 0.0,
        }
    )

    # Standort und Wettervorverarbeitung
    df_weather_pp, PVLIBsolpos = preprocess_weather(df_weather.copy(), PVLIBlocation)

    # AC-Leistung aufsummieren (über alle Sites; je Site eigener kleiner WR)
    ac_total = pd.Series(0.0, index=df_weather_pp.index)
    debug_rows = []
    debug_mask = None
    if DEBUG_START is not None and DEBUG_END is not None:
        debug_mask = (df_weather_pp.index >= DEBUG_START) & (df_weather_pp.index <= DEBUG_END)

    for section in config.sections():
        if not section.startswith("site"):
            continue

        site_cfg = config[section]
        pvsite_meta = {
            "Name": section,
            "Tilt": float(site_cfg["tilt"]),
            "Azimuth": float(site_cfg["azimuth"]),
            "quantity": int(site_cfg.get("quantity", 1)),
        }

        transposed_irradiance, PVLIBpoa, PVLIBiam = transposition_model(
            df_weather_pp, PVLIBsolpos, pvsite_meta
        )
        PVLIBcelltemperature = temperature_model(PVLIBpoa, df_weather_pp)
        PVLIBmpp = cell_model(modultyp_meta, PVLIBcelltemperature, transposed_irradiance)
        pdc_site = scale_module_to_system_pdc(PVLIBmpp, pvsite_meta)

        # Je Site eigener (kleiner) Wechselrichter
        site_dc_max = float(site_cfg.get("dc_max", SITE_INVERTER_DC_MAX))
        inverter_site = {
            "dc_max": site_dc_max,
            "eta_max": inverter_meta["eta_max"],
        }
        ac_site = calculate_acpower_from_pdc(pdc_site, inverter_site)

        ac_total = ac_total.add(ac_site, fill_value=0.0)

        if debug_mask is not None and debug_mask.any():
            timestamps = df_weather_pp.index[debug_mask]
            weather_slice = df_weather_pp.loc[debug_mask]
            zenith_slice = PVLIBsolpos.loc[debug_mask, "apparent_zenith"]
            debug_rows.append(
                pd.DataFrame(
                    {
                        "timestamp": timestamps,
                        "site": section,
                        "aswdir_s_Wm2": weather_slice["aswdir_s"].values,
                        "dni_Wm2": weather_slice["dni"].values,
                        "cos_zenith": weather_slice["cos_zenith"].values,
                        "zenith_deg": zenith_slice.values,
                        "transposed_irradiance_Wm2": transposed_irradiance.loc[debug_mask].values,
                        "iam": PVLIBiam.loc[debug_mask].values,
                        "poa_global_Wm2": PVLIBpoa.loc[debug_mask, "poa_global"].values,
                        "pdc_W": pdc_site.loc[debug_mask].values,
                        "ac_W": ac_site.loc[debug_mask].values,
                    }
                )
            )

    # Originaldaten an Zeitachse des Modells anpassen und Ergebnisspalte hinzufügen
    df_out = df_in.set_index("valid_time")
    df_out = df_out.reindex(ac_total.index)
    df_out["pv_power_W"] = ac_total.values

    # Zeitindex wieder als Spalte (spaltenname erzwingen)
    df_out.index.name = "valid_time"
    df_out = df_out.reset_index()

    # Exportpfad bestimmen: gleicher Basename + Suffix
    base = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(EXPORT_DIR, f"{base}_with_pv_power.csv")
    df_out.to_csv(output_file, index=False)

    if debug_rows:
        debug_df = pd.concat(debug_rows, ignore_index=True)
        debug_file = os.path.join(EXPORT_DIR, f"{base}_debug_20250826.csv")
        debug_df.to_csv(debug_file, index=False)
        print(f"DEBUG export written: {debug_file}")

    print(f"DONE - {os.path.basename(input_file)} -> {output_file}")
    return output_file


# -----------------------------------
# Batch-Verarbeitung aller CSVs im shared-Ordner
# -----------------------------------


def main():

    # Inverter-/Standort-Metadaten (einmalig)
    inverter_meta = {
        "Lat": LAT,
        "Lon": LON,
        "Name": "Inverter",
        "dc_max": float(config["inverter"]["dc_max"]),
        "eta_max": float(config["inverter"]["eta_max"]),
    }

    # Modulparameter (ein Modultyp für alle Sites) – einmalig
    module_cfg = config["moduletype"]
    modultyp_meta = {
        "v_mp": float(module_cfg["v_mp"]),
        "i_mp": float(module_cfg["i_mp"]),
        "v_oc": float(module_cfg["v_oc"]),
        "i_sc": float(module_cfg["i_sc"]),
        "alpha_sc": float(module_cfg["alpha_sc"]),
        "beta_voc": float(module_cfg["beta_voc"]),
    }

    # Standortobjekt (einmalig)
    PVLIBlocation = create_location_pvlib(inverter_meta)

    # Alle CSV-Dateien im shared-Ordner finden
    input_files = [
        os.path.join(SHARED_DIR, f)
        for f in os.listdir(SHARED_DIR)
        if f.lower().endswith(".csv")
    ]

    if not input_files:
        print(f"No CSV files found in {SHARED_DIR}")
    else:
        print(f"Found {len(input_files)} CSV files. Processing...")
        for fp in sorted(input_files):
            try:
                process_single_file(fp, PVLIBlocation, inverter_meta, modultyp_meta)
            except Exception as e:
                print(f"ERROR processing {os.path.basename(fp)}: {e}")


if __name__ == "__main__":
    main()