# checked 29.11.2025

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]

EXPORT_DIR = PROJECT_ROOT / "data" / "processed" / "prepared_rea_windows"
METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "source_label_space" / "metadata.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "normalized_rea_windows"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config_norm.yaml"

FEATURES = ["ASWDIR_S", "ASWDIFD_S", "T_2M"]


def load_norm_config(path: Path) -> dict:
    """Minimal parser for simple key: value floats in the norm config."""
    values = {}
    if not path.exists():
        raise FileNotFoundError(f"Norm-Config nicht gefunden: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        try:
            values[key.strip()] = float(val.strip())
        except ValueError:
            raise ValueError(f"Konnte Wert nicht als float parsen in {path}: '{raw_line}'")
    return values


norm_cfg = load_norm_config(CONFIG_PATH)

# Parameter norm
ASWDIR_NORM = norm_cfg["ASWDIR_NORM"]
ASWDIF_NORM = norm_cfg["ASWDIF_NORM"]
T2M_LOWER_NORM = norm_cfg["T2M_LOWER_NORM"]  # verschiebt -10C -> 0
T2M_RANGE = norm_cfg["T2M_RANGE"]       # (40 - (-10)) = 50C

# Min/Max-Grenzen
FEATURE_BOUNDS = {
    "ASWDIR_S": {"min": 0.0, "max": ASWDIR_NORM},
    "ASWDIFD_S": {"min": 0.0, "max": ASWDIF_NORM},
    "T_2M": {"min": 273.15-T2M_LOWER_NORM,   "max": 273.15+T2M_RANGE-T2M_LOWER_NORM},  # Kelvin
}



# -----------------------------
# HELPER
# -----------------------------

def parse_run_start(run_id: str) -> pd.Timestamp:
    # Erwartet z.B. ID001_2025070311 und gibt den Startzeitpunkt aus Window ID zurueck
    
    ts = run_id.split("_")[1]
    
    return pd.to_datetime(ts, format="%Y%m%d%H")


def window_within_bounds(df_window: pd.DataFrame) -> bool:
    # Prueft, ob alle benoetigten Spalten innerhalb definierter Grenzen liegen
    
    for col in FEATURES:
        bounds = FEATURE_BOUNDS[col]
        if df_window[col].lt(bounds["min"]).any() or df_window[col].gt(bounds["max"]).any():
            return False
        
    return True


def load_peakpower() -> dict:
    #Lädt die maxpower aus der Metadatei in ein Dictionary
    
    meta = pd.read_csv(METADATA_PATH, sep=";")

    return dict(zip(meta["ID"], meta["estimated_dc_capacity"]))



# -----------------------------
# MAIN
# -----------------------------


def main():
    peakpower_map = load_peakpower()    # kommt in Watt

    for path in EXPORT_DIR.glob("*_windows.csv"):
        
        df = pd.read_csv(path, parse_dates=["valid_time"])

        id_prefix = path.stem.split("_")[0]
        
        capacity = peakpower_map.get(id_prefix)

        kept = []
        total_windows = 0
        
        for run_id, grp in df.groupby("run_id"):    #alle window id gruppieren und durchlaufen -> iteration über alle windows
            total_windows += 1
            
            if not window_within_bounds(grp):   # verwerfen wenn werte extrema haben, schleife eins weiter (fenster nicht gespeichert)
                continue

            local_start = parse_run_start(run_id)   # startzeit für leadtime value
            grp = grp.copy()
            grp["lead_time_hours"] = (grp["valid_time"] - local_start).dt.total_seconds() / 3600.0  # leadtime in dezimal berechnen


            # Normalisierungen: vorhandene Spalten überschreiben
            grp["power_ac"] = grp["power_ac"] / float(capacity)
            if grp["power_ac"].gt(1.0).any():
                continue  # Fenster verwerfen, wenn Power-Norm > 1  # kann nicht real sein da maxpower installiert

            grp["ASWDIR_S"] = grp["ASWDIR_S"] / ASWDIR_NORM
            grp["ASWDIFD_S"] = grp["ASWDIFD_S"] / ASWDIF_NORM
            t2m_c = grp["T_2M"] - 273.15
            grp["T_2M"] = ((t2m_c + T2M_LOWER_NORM) / T2M_RANGE).clip(0.0, 1.0)

            kept.append(grp)


        if not kept:
            print(f"Skipping {path.name}: keine Fenster innerhalb Bounds.")
            continue

        out_df = pd.concat(kept, ignore_index=True)
        out_path = OUT_DIR / f"{path.stem}_norm.csv"
        out_df.to_csv(out_path, index=False)
        
        print(f"Saved {out_path} with {len(out_df)} rows ({len(kept)}/{total_windows} Fenster behalten).")



if __name__ == "__main__":
    main()
