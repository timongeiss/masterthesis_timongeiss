"""Skript zum Trainieren eines MLP-Modells auf den Source-Daten."""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import mlp_model as mlp



# ---------------------------------------------------------
# Pfade
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "mlp_input"
METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "source_label_space" / "metadata.csv"
SAVE_PATH = PROJECT_ROOT / "data" / "artifacts" / "source_mlp_checkpoint.pt"

# ---------------------------------------------------------
# Konmfiguration und Hyperparameter
# ---------------------------------------------------------

CONFIG_SOURCE = PROJECT_ROOT / "configs" / "config_source_mlp.yaml"
cfg = mlp.load_mlp_config(CONFIG_SOURCE)

DC_EFF_PER_KWP = 1

BATCH_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 1e-4
HIDDEN_SIZES = [256, 256, 128, 128, 64, 64]



# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------


def build_dataset_source(paths, meta_map):
    """
    Liest CSVs ein und baut X- und y-Tensoren.
    """
    
    all_features = []
    all_targets = []
    total = len(list(paths))
    processed = 0

    for idx, path in enumerate(sorted(paths), start=1):
        anlagen_id = path.stem.split("_")[0]    # aus dateinamen
        meta = meta_map.get(anlagen_id)         # metadaten ziehen
        if meta is None:
            continue

        df = pd.read_csv(path, parse_dates=["valid_time"])  # datensatz in dataframe laden
        if df.empty:
            continue

        samples = mlp.collect_windows_source(df, meta, cfg)  # sammeln aller möglichen features und targets vektoren (ready für batch picking aus samples)
        processed += 1
        print(f"[build_dataset_source] {idx}/{total} Dateien verarbeitet ({path.name}) - Fenster: {len(samples)}", flush=True)

        # iterieren über alle dateien, einen großen datensatz kreieren
        for f, t in samples:
            all_features.append(f)
            all_targets.append(t)

    if not all_features:
        raise ValueError("Keine Fenster nach dem Filtern übrig.")

    # konvertiert NumPy-Arrays in PyTorch Tensoren (nD-Array (2D, 3D, 4D, …) mit GPU/Gradienten)
    X = torch.from_numpy(np.stack(all_features)) # vorher jede sample shape: (960,)
    Y = torch.from_numpy(np.stack(all_targets)) # vorher jede sample shape: (191,)
    
    return X, Y




def prepare_loaders_source(data_dir: Path, meta_map, batch_size: int):
    """
    Erstellt DataLoader für train und val.
    """
    
    #Alle csv hinterlegen
    train_paths = list((data_dir / "train").glob("*_windows_norm.csv"))
    val_paths = list((data_dir / "val").glob("*_windows_norm.csv"))

    # Dataset bauen
    X_train, y_train = build_dataset_source(train_paths, meta_map)
    X_val, y_val = build_dataset_source(val_paths, meta_map)

    # TensorDataset -> Baut ein Dataset-Objekt, das aus zwei Tensoren besteht (X,Y)
    # DataLoader -> shufflet und liefert mini batches automatisch
    
    train_loader = DataLoader(TensorDataset(X_train, y_train),        
                            batch_size=batch_size, shuffle=True, num_workers=0)          # Bei mehreren Workern werden die Batches schneller vorbereitet, was die Trainingszeit insbesondere bei größeren Datensätzen deutlich reduziert.                              
    
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader




# ---------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------

def main():
    
    # Initale Einstellungen
    
    mlp.seed_everything()
    feature_cols = cfg["feature_cols"]
    meta_dim = cfg["meta_dim"]
    input_dim = mlp.compute_input_dim(feature_cols, meta_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loader laden")


    # Laden der Windows und Tensoren vorbereiten
    
    meta_map = mlp.load_metadata_source(METADATA_PATH, DC_EFF_PER_KWP)

    train_loader, val_loader = prepare_loaders_source(
        DATA_DIR, meta_map, BATCH_SIZE
    ) 


    # Modellobjekt erstellen und analysieren
    
    model = mlp.WindowMLP(HIDDEN_SIZES, input_dim).to(device)
    print("Aufbau des MLP")
    print(model)
    
    
    # Optimierer adam wählen, lernrate übergeben
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Lossfunktion auswählen
    criterion = nn.MSELoss()

    # der kleinste bisher gesehene Loss Wert
    best_val = float("inf") #initial auf unendlich gesetzt -> Damit jede echte Validierungs-Loss im ersten Epochendurchlauf automatisch kleiner
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Training durchführen über anzahl der geforderten epochen
    
    for epoch in range(1, EPOCHS + 1):
        
        # Python übergibt Objektreferenzen, keine Kopien -> Model wird geupdatet
        train_loss = mlp.run_epoch(train_loader, model, criterion, optimizer, device)   # durchschnittlichen Loss über die komplette Epoche
        val_loss = mlp.run_epoch(val_loader, model, criterion, None, device)
        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print("  Modell gespeichert:", SAVE_PATH)


if __name__ == "__main__":
    main()





# ---------------------------------------------------------
# Extern aufrufbarer Helper für Grid/Experimente
# ---------------------------------------------------------
def prepare_data_for_training(batch_size=None):
    """
    Bereitet train/val DataLoader vor (separat nutzbar für Grid Search).
    """
    bs = batch_size if batch_size is not None else BATCH_SIZE
    meta_map = mlp.load_metadata_source(METADATA_PATH, DC_EFF_PER_KWP)
    return prepare_loaders_source(DATA_DIR, meta_map, bs)


def train_and_eval(lr, epochs, hidden_sizes, save_path, batch_size=None, patience=None, loaders=None):
    """
    Parameterisierte Variante der main-Logik:
    - lr: Lernrate
    - epochs: Anzahl Epochen
    - hidden_sizes: Liste der Hidden-Layer-Größen
    - save_path: Pfad zum Speichern des Modells
    - batch_size: optional andere Batchgröße
    - patience: optional Early-Stopping-Patience (Anzahl Epochen ohne Verbesserung)
    """
    mlp.seed_everything()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bs = batch_size if batch_size is not None else BATCH_SIZE
    stop_patience = patience if patience is not None else epochs  # kein Early-Stopping wenn nicht gesetzt
    feature_cols = cfg["feature_cols"]
    meta_dim = cfg["meta_dim"]
    input_dim = mlp.compute_input_dim(feature_cols, meta_dim)

    # kein neubau der loaders wenn schon übergeben -> laufzeitoptimierung bei Grid Search
    if loaders is None:
        train_loader, val_loader = prepare_data_for_training(bs)
    else:
        train_loader, val_loader = loaders

    # Modell erstellen
    model = mlp.WindowMLP(hidden_sizes, input_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Early Stopping Logik
    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        train_loss = mlp.run_epoch(train_loader, model, criterion, optimizer, device)
        val_loss = mlp.run_epoch(val_loader, model, criterion, None, device)

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        if epochs_no_improve >= stop_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val, None
