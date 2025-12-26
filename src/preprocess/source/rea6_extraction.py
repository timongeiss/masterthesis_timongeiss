# lösche noch die auskommentierungen 1,2 ,3 -> Ansonsten checked 29.11.2025 

"""
REA6-Zeitreihen je Standort extrahieren und als CSV speichern.

Logik:
- metadata.csv lesen, alle IDs mit begin_ts / end_ts und Koordinaten (north, west)
- fuer jede ID und jedes Feld (ASWDIR_S, ASWDIFD_S, T_2M):
    * alle GRIB-Dateien von begin_ts bis end_ts unter shared/rea6/<PARAM>/<YEAR>/ laden
    * naechsten Gridpunkt zu (lat=North, lon=West) finden
    * Zeitreihe herausziehen, auf Zeitfenster begin_ts bis end_ts beschneiden
    * Konsolen-Log: angefragte Koordinate vs. gewaehlt je GRIB-Datei
- Alle drei Felder (ASWDIR_S, ASWDIFD_S, T_2M) auf gemeinsamen Zeitindex mergen und als CSV pro ID speichern.
"""

from pathlib import Path
import cfgrib
import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]

# Basisverzeichnis mit entpackten GRIB-Dateien
BASE_REA6_DIR = PROJECT_ROOT / "data" / "raw" / "source_feature_space" / "rea6"

# Parameter
PARAMETERS = ["ASWDIR_S", "ASWDIFD_S", "T_2M"]

# Pfad zur Metadatei
METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "source_label_space" / "metadata.csv"

# Ausgabeordner
EXPORT_DIR = PROJECT_ROOT / "data" / "processed" / "extracted_rea"


# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def get_dataset_variable_name(ds, requested_name):
    """
    Bestimmt den Variablennamen im Dataset
    """
    var_mapping = {
        "ASWDIR_S": "aswdir_s",
        "ASWDIFD_S": "aswdifd_s",
        "T_2M": "t2m",
    }
    
    requested_upper = requested_name.upper()
    candidate = var_mapping.get(requested_upper, requested_name).lower()

    for ds_var in ds.data_vars:
        if ds_var.lower() == candidate:
            return ds_var

    # 1
    # print("Fehler: Gewuenschte Variable wurde nicht gefunden.")
    # print("Verfuegbare Variablen im Dataset:", list(ds.data_vars))
    # first_name = list(ds.data_vars)[0]
    # print(f"Nutze stattdessen: {first_name}")
    # return first_name


def find_nearest_grid_index(ds, lat, lon):
    """
    Sucht den Index des nächsten Gridpunkts über Lat/Lon-Matrizen
    """
    lat_mat = ds["latitude"].values
    lon_mat = ds["longitude"].values
    dist = np.sqrt((lat_mat - lat) ** 2 + (lon_mat - lon) ** 2) #ür jeden Punkt im 2D-Gitter wird der Abstand zum gesuchten Punkt (lat, lon) berechnet -> Distanzkarte
    flat_index = np.argmin(dist)    # gibt den Index des kleinsten Werts zurück
    i_lat, i_lon = np.unravel_index(flat_index, dist.shape) # flachen Index zurück in 2D-Koordinaten umrechnen
    actual_lat = float(lat_mat[i_lat, i_lon])   # Tatsächliche Latitude/Longitude dieses Gridpunkts
    actual_lon = float(lon_mat[i_lat, i_lon])
    return i_lat, i_lon, actual_lat, actual_lon


def open_dataset(grib_file):
    """
    Oeffnet eine GRIB-Datei mit cfgrib
    """
    try:
        return xr.open_dataset(grib_file, engine="cfgrib", backend_kwargs={"indexpath": ""})
    
    except cfgrib.dataset.DatasetBuildError as exc:
        if "uvRelativeToGrid" in str(exc):  # versucht bei uvRelativeToGrid-Konflikten beide Filterwerte
            for val in (0, 1):
                
                try:
                    print(f"  -> erneuter Versuch mit filter_by_keys uvRelativeToGrid={val}")
                    return xr.open_dataset(
                        grib_file,
                        engine="cfgrib",
                        backend_kwargs={"indexpath": "", "filter_by_keys": {"uvRelativeToGrid": val}},
                    )
                    
                except Exception:
                    continue
        raise



def load_timeseries_at_point(grib_file, var_name, lat, lon):
    """
    Lädt Zeitreihe am nächsten Gridpunkt zu lat/lon, gibt DataFrame und Variablennamen und gewählte Koordinate
    """
    
    print(f"Oeffne Datei: {grib_file}")
    ds = open_dataset(grib_file)    #Rückgabe ist xarray
    ds_var_name = get_dataset_variable_name(ds, var_name) # wird benötigt als zugnangskey in die grib message -> Daten
    da = ds[ds_var_name] # zugang in die grib message über gribkey

    i_lat, i_lon, actual_lat, actual_lon = find_nearest_grid_index(ds, lat, lon) # nearest neighbor datenpunkt zur anlage finden
    point_da = da.isel(y=i_lat, x=i_lon)
    
    # 2
    # if "latitude" in da.dims and "longitude" in da.dims:
    #     point_da = da.isel(latitude=i_lat, longitude=i_lon)
    #     print("lalo")
        
    # elif "y" in da.dims and "x" in da.dims:
    #     point_da = da.isel(y=i_lat, x=i_lon)
    #     print("yx")
    # else:
    #     first_dim, second_dim = list(da.dims)[:2]
    #     point_da = da.isel({first_dim: i_lat, second_dim: i_lon})
    #     print("fisedim")

    print(
        f"  Anfrage lat/lon: ({lat:.4f}, {lon:.4f}) -> gewaehlt: ({actual_lat:.4f}, {actual_lon:.4f})"
    )

    df = point_da.to_dataframe().reset_index()
    time_col = "time"
    
    # 3
    # if "time" in df.columns:
    #     time_col = "time"
    #     print("time")    
    # elif "valid_time" in df.columns:
    #     time_col = "valid_time"
    #     print("validtime")
    # else:
    #     raise ValueError("Keine geeignete Zeitspalte gefunden (weder 'time' noch 'valid_time').")


    df["time"] = pd.to_datetime(df[time_col], utc=True).dt.tz_convert(None)
    df = df.sort_values("time").reset_index(drop=True)
    df = df[["time", ds_var_name]]
    
    return df, ds_var_name



# ------------------------------------------------------------
# Sammellogik pro Standort
# ------------------------------------------------------------

def collect_for_id(target_id, lat, lon, begin_ts, end_ts):
    """
    Sammelt alle drei Felder für eine ID im Zeitfenster begin_ts nis end_ts.
    """
    data_per_var = {}

    years = range(begin_ts.year, end_ts.year + 1)

    for param in PARAMETERS:
        dfs = []
        
        for year in years:
            year_dir = BASE_REA6_DIR / param / str(year)
            if not year_dir.exists():
                continue
            
            for grib_file in sorted(year_dir.glob("*.grb")):    # grib files alle finden die benötigt
                try:
                    df, ds_var = load_timeseries_at_point(grib_file, param, lat, lon)   #zeitreihe je file
                    df = df.rename(columns={ds_var: param})
                    dfs.append(df)
                    
                except Exception as exc:  # pragma: no cover
                    print(f"Fehler bei {grib_file}: {exc}")
                    continue

        if dfs: #Wenn zeitreihe zumindest teilweise erfolgreich extrahiert muss verknüpft werden
            df_param = pd.concat(dfs, ignore_index=True)
            df_param = (
                df_param.dropna(subset=["time"])
                .sort_values("time")
                .drop_duplicates(subset=["time"], keep="last")  #keine duplikate
                .set_index("time")
            )
            
            # Zeitfenster beschneiden
            df_param = df_param.loc[(df_param.index >= begin_ts) & (df_param.index < end_ts)]
            data_per_var[param] = df_param

    if not data_per_var:
        raise RuntimeError(f"Keine Daten fuer ID {target_id} gefunden.")

    
    # Alle drei Fields sammeln und mergen
    merged = None
    for param, df_param in data_per_var.items():
        merged = df_param if merged is None else merged.join(df_param, how="outer")

    merged = merged.sort_index()
    return merged


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    meta = pd.read_csv(METADATA_PATH, sep=";")
    meta["begin_ts"] = pd.to_datetime(meta["begin_ts"], utc=True).dt.tz_convert(None)
    meta["end_ts"] = pd.to_datetime(meta["end_ts"], utc=True).dt.tz_convert(None)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    #iteration über alle ANlagen
    SKIP_ROWS = 122   # batchweise gearbeitet

    for i, (_, row) in enumerate(meta.iterrows()):
        if i < SKIP_ROWS:
            continue 
        
        target_id = row["ID"]
        begin_ts = row["begin_ts"]
        end_ts = row["end_ts"].ceil("H")
        lat = float(row["north"])
        lon = float(row["west"])

        print(f"\n=== Verarbeite ID {target_id} (lat={lat}, lon={lon}) ===")

        # alle drei Fields zeitreihen eines standorts
        merged = collect_for_id(target_id, lat, lon, begin_ts, end_ts)

        # abspeichern unter ID
        out_path = EXPORT_DIR / f"{target_id}_rea6.csv"
        merged.to_csv(out_path, index=True, date_format="%Y-%m-%d %H:%M:%S")
        print(f"CSV gespeichert fuer {target_id}: {out_path} (Zeilen: {len(merged)})")


if __name__ == "__main__":
    main()
