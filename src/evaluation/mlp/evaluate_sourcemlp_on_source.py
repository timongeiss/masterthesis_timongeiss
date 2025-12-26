"""skript zur evaluation eines source-mlp auf dem source-datensatz"""

from pathlib import Path
from typing import Dict, List, Tuple
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------
# Pfade 
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "mlp_input"# Ordner mit train/val/test
METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "source_label_space" / "metadata.csv"# Metadaten der Anlagen
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "artifacts" / "source_mlp_checkpoint.pt"
CONFIG_SOURCE = PROJECT_ROOT / "configs" / "config_source_mlp.yaml"


# Projekteigenen Code importierbar machen
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.mlp import mlp_model  as mlp



#---------------------------------------------------------
# Konfig und Hyperparameter
#---------------------------------------------------------

cfg_source = mlp.load_mlp_config(CONFIG_SOURCE)

INPUT_DIM = mlp.compute_input_dim(cfg_source["feature_cols"], cfg_source["meta_dim"])


BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")





# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------



def _infer_hidden_sizes(state: Dict[str, torch.Tensor]) -> List[int]:
    """Hidden-Layer-Groessen aus Gewichtsmatrizen rekonstruieren."""
    
    # Annahme: state keys enden auf "weight" fuer Gewichtsmatrizen
    weight_keys = [k for k in state.keys() if k.endswith("weight")]
    hidden = []
    for key in weight_keys[:-1]:  # letzte ist Output-Schicht
        hidden.append(state[key].shape[0])
        
    return hidden


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[Dict[str, torch.Tensor], List[int]]:
    """Laedt Modell-Zustand und Hidden-Sizes."""

    # Modell Laden
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    hidden_sizes = _infer_hidden_sizes(state)
    
    return state, hidden_sizes


def prepare_test_loader() -> Tuple[DataLoader, List[str]]:
    """
    Baut DataLoader fuer Testfenster  und liefert IDs/Kapazitaeten.
    """
    
    meta_map = mlp.load_metadata_source(METADATA_PATH, 1.0)  # DC_EFF_PER_KWP nicht relevant fuer Evaluation
    
    test_paths = list((DATA_DIR / "test").glob("*_windows_norm.csv"))

    X_list = []
    y_list = []
    caps = []
    ids = []

    total = len(test_paths)
    for idx, path in enumerate(sorted(test_paths), start=1):
        
        anlagen_id = path.stem.split("_")[0]
        
        meta = meta_map.get(anlagen_id) # Metadaten fuer Anlage laden
        if meta is None:
            continue
        
        df = pd.read_csv(path, parse_dates=["valid_time"])
        if df.empty:
            continue
        
        samples = mlp.collect_windows_source(df, meta, cfg_source)
        cap_kwp = float(meta[0]) / 1000.0  # Metadaten in Watt -> auf kWp umrechnen

        for f, t in samples:
            X_list.append(f)
            y_list.append(t)
            caps.append(cap_kwp)
            ids.append(anlagen_id)
        print(f"[prepare_test_loader] {idx}/{total} Anlagen verarbeitet ({anlagen_id}), Fenster: {len(samples)}", flush=True)

    if not X_list:
        raise ValueError("Keine Test-Fenster gefunden.")

    X = torch.from_numpy(np.stack(X_list))
    Y = torch.from_numpy(np.stack(y_list))
    caps_t = torch.tensor(caps, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X, Y, caps_t),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    return loader, ids


def evaluate(loader: DataLoader, model: nn.Module, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Berechnet Vorhersagen, Targets und Kapazitaeten.
    """
    
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    caps: List[torch.Tensor] = []
    model.eval()
    
    # evaluationsmode keine gradients
    with torch.no_grad():
        # iteriert über den Test-DataLoader, schickt die Features auf das angegebene Device
        for features, y, cap in loader:
            features = features.to(device)
            y = y.to(device)
            #macht einen Forward-Pass
            out = model(features)
            preds.append(out.cpu())
            targets.append(y.cpu())
            # wird später gebraucht, um die normierten Vorhersagen/Targets auf absolute Werte zu skalieren und nMAE/nRMSE pro Anlage
            # Der DataLoader liefert Fenster, keine Anlagen; so weiß jedes Sample sofort seine Kapazität (zum Denormalisieren von y_true/y_pred)
            caps.append(cap.cpu())
    
    return (
        torch.cat(preds, dim=0).numpy(),
        torch.cat(targets, dim=0).numpy(),
        torch.cat(caps, dim=0).numpy(),
    )


def compute_metrics(preds: np.ndarray, targets: np.ndarray, caps: np.ndarray) -> Dict[str, float]:
    """
    MAE/RMSE auf den denormalisierten Werten über alle daten.
    gewichtet implizit nach Fensteranzahl je Anlage.
    """
    
    # absolute Werte rekonstruieren
    preds_abs = preds * caps[:, None]
    targets_abs = targets * caps[:, None]
    
    # Fehler berechnen (absolut)
    mse = float(np.mean((preds_abs - targets_abs) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds_abs - targets_abs)))
    
    return {"mae (kW)": mae, "rmse (kW)": rmse}


def compute_normalized_metrics_per_id(preds: np.ndarray, targets: np.ndarray, caps: np.ndarray, ids: List[str]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    nMAE/nRMSE pro Anlage sowie Mittelwert ueber alle Anlagen, nur als information welche Anlage wie performant ist.

    ids und caps sind parallel zu preds/targets aufgebaut:
    - enumerate(ids) iteriert ueber alle Fenster
    - i ist der Fenster-Index
    - ids[i] die zugehoerige Anlagen-ID,
    - caps[i] die passende Kapazitaet
    - preds[i]/targets[i] die Fenster-Vorhersage/-Labels.
    - die zweite Schleife ueber per_id aggregiert dann ueber Anlagen-IDs (d.h. ueber alle Fenster derselben Anlage).
    """
    
    # sammelcontainer pro anlage
    per_id: Dict[str, List[float]] = {}
    per_id_rmse: Dict[str, List[float]] = {}

    # iteriere über alle samples
    for i, anlagen_id in enumerate(ids):
        cap = float(caps[i]) #if caps is not None else 1.0
        
        # absolute fehler berechnen
        err_abs = preds[i] * cap - targets[i] * cap
        mae = float(np.mean(np.abs(err_abs)))
        rmse = float(np.sqrt(np.mean(err_abs ** 2)))
        
        # normalisierte fehler rückberechnen
        nmae = mae / cap
        nrmse = rmse / cap
        
        # in sammelcontainer ablegen
        per_id.setdefault(anlagen_id, []).append(nmae)
        per_id_rmse.setdefault(anlagen_id, []).append(nrmse)

    # sammelcontainer je anlage
    per_id_metrics: Dict[str, Dict[str, float]] = {}
    for anlagen_id in per_id:
        per_id_metrics[anlagen_id] = {
            "nMAE": float(np.mean(per_id[anlagen_id])),
            "nRMSE": float(np.mean(per_id_rmse[anlagen_id])),
        }

    mean_metrics = {
        "nMAE_mean": float(np.mean([v["nMAE"] for v in per_id_metrics.values()])),
        "nRMSE_mean": float(np.mean([v["nRMSE"] for v in per_id_metrics.values()])),
    }
    return per_id_metrics, mean_metrics


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    
    mlp.seed_everything()
    
    print("Baue Test-Loader")
    test_loader, id_list = prepare_test_loader()
    

    state, hidden_sizes = load_checkpoint(CHECKPOINT_PATH, DEVICE)
    print(f"Hidden-Sizes aus Checkpoint: {hidden_sizes}")

    model = mlp.WindowMLP(hidden_sizes, INPUT_DIM).to(DEVICE)
    model.load_state_dict(state)

    preds, targets, caps = evaluate(test_loader, model, DEVICE)
    metrics = compute_metrics(preds, targets, caps)
    per_id_metrics, mean_metrics = compute_normalized_metrics_per_id(preds, targets, caps, id_list)

    print("Evaluation abgeschlossen.")
    
    print(f"Samples: {len(preds)} Fenster x {preds.shape[1]} Schritte")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
        
    print("nMAE / nRMSE pro Anlage:")
    for anlagen_id in sorted(per_id_metrics.keys()):
        vals = per_id_metrics[anlagen_id]
        print(f"  {anlagen_id}: nMAE={vals['nMAE']:.6f}, nRMSE={vals['nRMSE']:.6f}")
    print(f"Mittelwert ueber Anlagen: nMAE={mean_metrics['nMAE_mean']:.6f}, nRMSE={mean_metrics['nRMSE_mean']:.6f}")


if __name__ == "__main__":
    main()
