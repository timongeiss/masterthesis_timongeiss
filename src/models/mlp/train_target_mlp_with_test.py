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
CONFIG_TESTDAYS_PATH = PROJECT_ROOT / "configs" / "config_testdays.yaml"
CONFIG_TARGET_SYSTEM = PROJECT_ROOT / "configs" / "config_target_system.yaml"
CONFIG_TARGET_MLP = PROJECT_ROOT / "configs" / "config_target_mlp.yaml"

TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "target_mlp_input" / "ann_train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "processed" / "target_mlp_input" / "ann_val.csv"
TEST_CSV = PROJECT_ROOT / "data" / "processed" / "target_mlp_input" / "ann_test.csv"
LOOKUP_SOURCES = [
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
]

SAVE_PATH = PROJECT_ROOT / "data" / "artifacts" / "target_mlp_checkpoint.pt"
TEST_METRICS_PATH = PROJECT_ROOT / "data" / "artifacts" / "target_model_testday_metrics.csv"
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "target_output"

# ---------------------------------------------------------
# Config und hyperparameter
# ---------------------------------------------------------

cfg_target = mlp.load_target_config(CONFIG_TARGET_SYSTEM)
TEST_DATES = mlp.load_test_days(CONFIG_TESTDAYS_PATH)
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
    feature_cols = cfg_mlp["feature_cols"]
    target_col = cfg_mlp["target_col"]
    meta_dim = cfg_mlp["meta_dim"]
    
    input_dim = mlp.compute_input_dim(feature_cols, meta_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Konstanten Feature-Block aus Vorgabe
    meta_tuple = (
        cfg_target["DC_CAPACITY_KWP"],
        cfg_target["TILT_DEG"] / 90.0,                  # normierung
        (cfg_target["AZIMUTH_DEG"] % 360.0) / 360.0,    # normierung
        cfg_target["DC_EFF_PER_KWP"],
    )

    label_lookup = mlp.build_label_lookup(LOOKUP_SOURCES, target_col)

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
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Training durchführen über anzahl der geforderten epochen
    
    for epoch in range(1, EPOCHS + 1):
        
        # Python uebergibt Objektreferenzen, keine Kopien -> Model wird geupdatet
        train_loss = mlp.run_epoch(train_loader, model, criterion, optimizer, device)   # durchschnittlichen Loss über die komplette Epoche
        val_loss = mlp.run_epoch(val_loader, model, criterion, None, device)
        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print("Modell gespeichert:", SAVE_PATH)

    # Test-Auswertung
    test_metrics = mlp.evaluate_on_test(model, device, meta_tuple, label_lookup, cfg_mlp, TEST_CSV, TEST_DATES, export_predictions=True, results_dir=RESULTS_DIR)
    
    # csv exportieren
    TEST_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_metrics.to_csv(TEST_METRICS_PATH, index=False)
    print("Testmetriken gespeichert:", TEST_METRICS_PATH)



if __name__ == "__main__":
    main()





# ---------------------------------------------------------
# Extern aufrufbarer Helper für Grid/Experimente
# ---------------------------------------------------------

def train_and_eval(lr, epochs, hidden_sizes, save_path, batch_size=None, patience=None, save_test_metrics=True, save_predictions=False):
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
    feature_cols = cfg_mlp["feature_cols"]
    target_col = cfg_mlp["target_col"]
    meta_dim = cfg_mlp["meta_dim"]
    input_dim = mlp.compute_input_dim(feature_cols, meta_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_tuple = (
        cfg_target["DC_CAPACITY_KWP"],
        cfg_target["TILT_DEG"] / 90.0,                  # normierung
        (cfg_target["AZIMUTH_DEG"] % 360.0) / 360.0,    # normierung
        cfg_target["DC_EFF_PER_KWP"],
    )

    label_lookup = mlp.build_label_lookup(LOOKUP_SOURCES, target_col)

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
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        train_loss = mlp.run_epoch(train_loader, model, criterion, optimizer, device) # muss laufen wegen backprop
        val_loss = mlp.run_epoch(val_loader, model, criterion, None, device)

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= stop_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        
    test_metrics = mlp.evaluate_on_test(model, device, meta_tuple, label_lookup, cfg_mlp, TEST_CSV, TEST_DATES, export_predictions=False, results_dir=None)
    
    
    if save_test_metrics and test_metrics is not None and not test_metrics.empty:
        TEST_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        test_metrics.to_csv(TEST_METRICS_PATH, index=False)

    return best_val, test_metrics
