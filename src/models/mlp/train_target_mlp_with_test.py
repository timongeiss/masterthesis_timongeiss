'''Skript zum Trainieren eines MLP-Modells auf dem Target-System mit anschließender Test-Auswertung.'''

from pathlib import Path
import torch
from torch import nn
import mlp_model as mlp

# ---------------------------------------------------------
# Pfade
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_TARGET_SYSTEM = PROJECT_ROOT / "configs" / "config_target_system.yaml"
CONFIG_TARGET_MLP = PROJECT_ROOT / "configs" / "config_target_mlp.yaml"

ANN_ALL_CSV = PROJECT_ROOT / "data" / "processed" / "target_mlp_input" / "ann_all.csv"
TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "target_mlp_input" / "ann_train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "processed" / "target_mlp_input" / "ann_val.csv"
TEST_CSV = PROJECT_ROOT / "data" / "processed" / "target_mlp_input" / "ann_test.csv"

SAVE_PATH = PROJECT_ROOT / "data" / "artifacts" / "target_mlp_checkpoint.pt"
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "target_output"

# ---------------------------------------------------------
# Config und hyperparameter
# ---------------------------------------------------------

cfg_target = mlp.load_target_config(CONFIG_TARGET_SYSTEM)
cfg_mlp = mlp.load_mlp_config(CONFIG_TARGET_MLP)


BATCH_SIZE = 16
EPOCHS = 513
LEARNING_RATE = 0.0001 
HIDDEN_SIZES = [256, 256, 128, 128, 64, 64] 



# ---------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------

def main():
    
    # Initale Einstellungen
    mlp.seed_everything()
    device, input_dim, meta_tuple, label_lookup = mlp.build_target_runtime_context(
        cfg_mlp, cfg_target, ANN_ALL_CSV
    )

    train_loader, val_loader = mlp.prepare_loaders_target(TRAIN_CSV, VAL_CSV, meta_tuple, label_lookup, BATCH_SIZE, cfg_mlp)


    # Modellobjekt erstellen und analysieren
    model = mlp.WindowMLP(HIDDEN_SIZES, input_dim).to(device)
    print("Aufbau des MLP")
    print(model)
    
    # Optimierer adam, lernrate geben
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Lossfunktion
    criterion = nn.MSELoss()

    # der kleinste bisher gesehene Loss Wert
    best_val = float("inf") #initial auf unendlich gesetzt -> Damit jede echte Validierungs-Loss im ersten Epochendurchlauf automatisch kleiner
    best_state = None
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Training durchführen über anzahl der geforderten epochen
    
    for epoch in range(1, EPOCHS + 1):
        
        # Python uebergibt Objektreferenzen, keine Kopien -> Model wird geupdatet
        train_loss = mlp.run_epoch(train_loader, model, criterion, optimizer, device)   # durchschnittlichen Loss über die komplette Epoche
        val_loss = mlp.run_epoch(val_loader, model, criterion, None, device)
        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            torch.save(model.state_dict(), SAVE_PATH)
            print("Modell gespeichert:", SAVE_PATH)

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test-Auswertung
    test_metrics = mlp.evaluate_on_test(
        model,
        device,
        meta_tuple,
        label_lookup,
        cfg_mlp,
        TEST_CSV,
        export_predictions=True,
        results_dir=RESULTS_DIR,
    )
    print("Testmetriken berechnet:", len(test_metrics), "Zeilen")
    if not test_metrics.empty:
        print(test_metrics.to_string(index=False))



if __name__ == "__main__":
    main()





# ---------------------------------------------------------
# Extern aufrufbarer Helper für Grid/Experimente und Meta-Skripte
# ---------------------------------------------------------

def train_and_eval(
    lr,
    epochs,
    hidden_sizes,
    save_path=None,
    batch_size=None,
    patience=None,
    save_test_metrics=False,
    save_predictions=False,
    results_dir=None,
):
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
    device, input_dim, meta_tuple, label_lookup = mlp.build_target_runtime_context(
        cfg_mlp, cfg_target, ANN_ALL_CSV
    )

    bs = batch_size if batch_size is not None else BATCH_SIZE
    stop_patience = patience if patience is not None else epochs  # kein Early-Stopping wenn nicht gesetzt

    train_loader, val_loader = mlp.prepare_loaders_target(
        TRAIN_CSV, VAL_CSV, meta_tuple, label_lookup, bs, cfg_mlp
    )

    model = mlp.WindowMLP(hidden_sizes, input_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    save_path_obj = Path(save_path) if save_path is not None else None
    if save_path_obj is not None:
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        train_loss = mlp.run_epoch(train_loader, model, criterion, optimizer, device) # muss laufen wegen backprop
        val_loss = mlp.run_epoch(val_loader, model, criterion, None, device)

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= stop_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if save_path_obj is not None:
            mlp.save_checkpoint_with_retry(best_state, save_path_obj)
        
    export_dir = Path(results_dir) if results_dir is not None else None
    test_metrics = mlp.evaluate_on_test(
        model,
        device,
        meta_tuple,
        label_lookup,
        cfg_mlp,
        TEST_CSV,
        export_predictions=save_predictions,
        results_dir=export_dir,
    )

    if save_test_metrics:
        print(
            "Hinweis: save_test_metrics ist veraltet. "
            "Aggregation der Metrics erfolgt im Meta-Skript."
        )

    return best_val, test_metrics


def evaluate_checkpoint_on_test(
    checkpoint_path,
    hidden_sizes=None,
    save_predictions=False,
    results_dir=None,
):
    """
    Lädt einen Checkpoint und evaluiert ausschließlich auf ann_test.csv.
    Gedacht für Freeze-Phase ohne erneutes Training.
    """
    mlp.seed_everything()
    device, input_dim, meta_tuple, label_lookup = mlp.build_target_runtime_context(
        cfg_mlp, cfg_target, ANN_ALL_CSV
    )
    hs = hidden_sizes if hidden_sizes is not None else HIDDEN_SIZES
    model = mlp.WindowMLP(hs, input_dim).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    export_dir = Path(results_dir) if results_dir is not None else None
    test_metrics = mlp.evaluate_on_test(
        model,
        device,
        meta_tuple,
        label_lookup,
        cfg_mlp,
        TEST_CSV,
        export_predictions=save_predictions,
        results_dir=export_dir,
    )
    return test_metrics
