'''Skript zur Evaluation eines vortrainierten Source-MLP Modells auf dem Target-Datensatz. Kein Transfer Learning, nur reine forward feed und Metriken-Berechnung.'''

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

# ---------------------------------------------------------
# Pfade
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
TEST_CSV = PROJECT_ROOT / "data" / "processed" / "target_mlp_input" /"ann_test.csv"
SAVE_PATH = PROJECT_ROOT / "data" / "artifacts" / "source_mlp_checkpoint.pt"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config_target_system.yaml"
TEST_DAYS_PATH = PROJECT_ROOT / "configs" / "config_testdays.yaml"
TEST_METRICS_PATH = PROJECT_ROOT / "data" / "artifacts" / "source_model_testday_metrics.csv"
LOOKUP_SOURCES = [TEST_CSV]
CONFIG_TARGET_MLP = PROJECT_ROOT / "configs" / "config_target_mlp.yaml"

# Projekteigenen Code importierbar machen
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.mlp import mlp_model as mlp



# ---------------------------------------------------------
# Konfig und Hyperparameter
# ---------------------------------------------------------

EXPECTED_STEPS = mlp.EXPECTED_STEPS

cfg_target_mlp = mlp.load_mlp_config(CONFIG_TARGET_MLP)

FEATURES_PER_STEP = len(cfg_target_mlp["feature_cols"])
INPUT_DIM = mlp.compute_input_dim(cfg_target_mlp["feature_cols"], cfg_target_mlp["meta_dim"])

cfg_target = mlp.load_target_config(CONFIG_PATH)

TEST_DATES = mlp.load_test_days(TEST_DAYS_PATH)

HIDDEN_SIZES = [256, 256, 128, 128, 64, 64]




# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def evaluate_on_test(model, device, meta_tuple, label_lookup):
    """
    Evaluiert das Modell auf ann_test.csv und gibt einen DataFrame mit Metriken zurueck.
    """

    run_ids, init_labels, features, targets, times, model_starts = load_test_runs(TEST_CSV, cfg_target_mlp["feature_cols"], cfg_target_mlp["target_col"], EXPECTED_STEPS, label_lookup)

    # sammelt pro run_id die Vorhersagen und Targets
    records = []
    
    # Kein Rechengraph, kein Speicher fuer Gradienten
    with torch.no_grad():
        
        # fensterweise iterieren
        for run_id, init_label, feats, tgt, t_series, m_start in zip(run_ids, init_labels, features, targets, times, model_starts):
            
            # features flatten aus (191 x 5) -> (955,)
            feats_flat = feats.reshape(-1)
            # meta features zusammenfuehren [start_power, dc_capacity, tilt_norm, azimuth_norm, dc_eff]
            meta_vec = np.array([init_label, *meta_tuple], dtype=np.float32)
            # x tensor erstellen Enddimension: 960
            x = torch.from_numpy(np.concatenate([meta_vec, feats_flat]).astype(np.float32)).to(device)
            
            # Forward-Pass
            pred = model(x).cpu().numpy()

            # DataFrame fuer einen Lauf mit Zeiten, Vorhersagen und label
            run_df = pd.DataFrame(
                {
                    "run_id": run_id,
                    "model_start": m_start,
                    "valid_time": t_series,
                    "y_true": tgt,
                    "y_pred": pred,
                    "solar_elevation_deg": feats[:, cfg_target_mlp["feature_cols"].index("solar_elevation_deg")],
                }
            )
            
            # windowweise Ergebnis anhaengen
            records.append(run_df)
    
    # alle ergebnisse zusammenketten        
    df_all = pd.concat(records, ignore_index=True)
    
    # Nacht raus
    df = df_all[df_all["solar_elevation_deg"] >= 0].copy()
    # minuten entfernen, auf tage bringen
    df["date"] = df["valid_time"].dt.date
    df["model_start_date"] = df["model_start"].dt.date
    # sortieren in die zwei gruppen
    df["run_type"] = np.where(df["model_start_date"] < df["date"], "day_ahead", "intra_day")

    rows = []
    # nach tagen sortieren
    for day, g in df.groupby("date"):
        
        # gruppieren in drei zielgruppem
        for label, subset in [
            ("all", g),
            ("day_ahead", g[g["run_type"] == "day_ahead"]),
            ("intra_day", g[g["run_type"] == "intra_day"]),
        ]:
            # metriken berechnen mit numpy
            if len(subset) == 0:    #null wenn zb keine dayahead existiert
                nrmse = np.nan
                nmae = np.nan
            else:
                nrmse = float(np.sqrt(np.mean((subset["y_pred"] - subset["y_true"]) ** 2)))
                nmae = float(np.mean(np.abs(subset["y_pred"] - subset["y_true"])))
                
            # Ergebnisse speichern fuer jeden Tag und subset
            rows.append(
                {
                    "date": day,
                    "group": label,
                    "nRMSE": nrmse,
                    "nMAE": nmae,
                    "count": len(subset),
                }
            )
            
    
    metrics = pd.DataFrame(rows)
    # nur definierte Testtage behalten
    metrics = metrics[metrics["date"].isin(TEST_DATES)]
    
    return metrics



def load_test_runs(csv_path: Path, feature_cols: list, target_cols: str, expected_steps: int, label_lookup: dict):
    """
    Laedt Testfenster und liefert run_id, init_label, features, targets, times, model_start.
    """
    # test csv laden
    df = pd.read_csv(csv_path, parse_dates=["valid_time", "model_start"])
    
    run_ids = []          # speichert run_id pro Fenster
    init_labels = []      # Startleistung 15 min vor Fensterstart
    feature_list = []     # Features je Fenster (191 x feature_dim)
    target_list = []      # Targets je Fenster
    times_list = []       # Zeitachsen je Fenster
    model_starts = []     # model_start je Fenster
    
    # gehe ueber alle windows
    for run_id, group in df.groupby("run_id"):
        
        group = group.sort_values("valid_time") # sortiere window nach validzeit
        
        # pruefe erneut ob laenge passt
        if len(group) != expected_steps:
            print(f"Skipping run {run_id}: expected {expected_steps} rows, got {len(group)}")
            continue
        
        start_time = group["valid_time"].iloc[0] #kleinste zeit = start
        
        # Leistungsfeature finden, sonst skip
        prior_time = start_time - pd.Timedelta(minutes=15) 
        init_label = label_lookup.get(prior_time)
        
        if init_label is None:
            print(f"Skipping run {run_id}: no preceding power_norm found at {prior_time}")
            continue
        
        # sammeln der windows
        run_ids.append(run_id)
        init_labels.append(init_label)
        feature_list.append(group[feature_cols].to_numpy(dtype=np.float32))
        target_list.append(group[target_cols].to_numpy(dtype=np.float32))
        times_list.append(group["valid_time"].reset_index(drop=True))
        model_starts.append(group["model_start"].iloc[0])
        
    return run_ids, init_labels, feature_list, target_list, times_list, model_starts





# ---------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------

def main():
    
    mlp.seed_everything()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Konstanten Feature-Block aus Vorgabe
    meta_tuple = (
        cfg_target["DC_CAPACITY_KWP"],
        cfg_target["TILT_DEG"] / 90.0,                  # normierung
        (cfg_target["AZIMUTH_DEG"] % 360.0) / 360.0,    # normierung
        cfg_target["DC_EFF_PER_KWP"],
    )
    
    label_lookup = mlp.build_label_lookup(LOOKUP_SOURCES, cfg_target_mlp["target_col"])

    # Modellobjekt erstellen und Gewichte laden
    model = mlp.WindowMLP(HIDDEN_SIZES, INPUT_DIM).to(device)

    # Zustand laden
    state = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(state)
    # Eval-Modus
    model.eval()

    # Test-Auswertung
    test_metrics = evaluate_on_test(model, device, meta_tuple, label_lookup)
    
    # csv exportieren
    TEST_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_metrics.to_csv(TEST_METRICS_PATH, index=False)
    print("Testmetriken gespeichert:", TEST_METRICS_PATH)



if __name__ == "__main__":
    main()
