# checked 29.11.2025

import pandas as pd
from pathlib import Path

ALLOWED_HOURS = {2, 5, 8, 11, 14, 17, 20, 23}
STEP = pd.Timedelta(minutes=15)
WINDOW = pd.Timedelta(hours=48)
ROWS_PER_WINDOW = int(WINDOW / STEP)  # 192


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]

def extract_windows(path: Path) -> pd.DataFrame:
    """
    Liest alle gültigen 48h-Fenster aus einer CSV-Datei
    """
    
    # Laedt die Daten und sortiert sie nach Zeit
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Nacht: Sonnenhoehe < 0 -> Strahlung/Leistung auf 0 setzen
    is_night = df["solar_elevation_deg"] < 0
    df.loc[is_night, ["ASWDIR_S", "ASWDIFD_S", "power_ac"]] = 0.0

    windows = [] #leere Liste für alle gefundenen 48h-Fenster

    # Startzeit auf volle Viertelstunden runden und letztes erlaubtes Startfenster bestimmen
    candidate = df["time"].min().floor("15min")
    last_allowed = df["time"].max() - WINDOW     #spätester erlaubter Startzeitpunkt

    # Schleife über alle möglichen Startzeitpunkte im 15-Minuten-Raster.
    while candidate <= last_allowed:
        
        # Nur volle Stunden, die im erlaubten Raster liegen
        if candidate.minute == 0 and candidate.hour in ALLOWED_HOURS:
            
            end = candidate + WINDOW - STEP #letzter Timestamp des 48h-Fensters.
            mask = (df["time"] >= candidate) & (df["time"] <= end)
            seg = df.loc[mask]  # Ausschnitt der Daten für dieses Fenster

            if len(seg) == ROWS_PER_WINDOW:  #nur weiter, wenn genau 192 Zeilen
                
                time_diff = seg["time"].diff().dropna()
                has_regular_steps = (time_diff == STEP).all()   #prüfen, ob alle Abstände exakt 15 Minuten sind
                has_nan = seg.isna().any().any()                #prüfen, ob irgendwo NaNs enthalten sind

                #Fenster ist nur gültig, wenn Schritte regelmäßig sind und keine Lücken/NaNs
                if has_regular_steps and not has_nan:
                    # Window ID erstellen aus der PV ID und dem Window start datetime
                    run_id = f"{path.stem.split('_')[0]}_{candidate:%Y%m%d%H}"
                    window = seg.copy()
                    window.insert(0, "run_id", run_id)  # jedem wert eine window ID geben 
                    window = window.rename(columns={"time": "valid_time"})
                    windows.append(window)
        candidate += STEP

    if windows:
        return pd.concat(windows, ignore_index=True)

    return pd.DataFrame()




def main():
    
    # Zielordner fuer die Exportdateien
    
    input_dir  = PROJECT_ROOT / "data" / "processed" / "resampled_rea"
    
    export_dir =  PROJECT_ROOT / "data" / "processed" / "prepared_rea_windows"


    # Iteriert ueber alle Quelldateien und schreibt gueltige Fenster
    
    for path in input_dir.glob("*_rea6_15min.csv"):
        
        df = extract_windows(path)
        
        if df.empty:
            print(f"Skipping {path.name}: no valid windows.")
            continue

        id_prefix = path.stem.split("_")[0]
        out_path = export_dir / f"{id_prefix}_windows.csv"
        df.to_csv(out_path, index=False)
        windows_here = len(df) // ROWS_PER_WINDOW
        print(f"Saved {out_path} with {len(df)} rows ({windows_here} windows).")


if __name__ == "__main__":
    main()
