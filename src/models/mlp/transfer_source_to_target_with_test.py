'''Skript zum Transferlernen von einem vortrainierten MLP auf ein Zielsystem.'''

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

PRETRAINED_PATH = PROJECT_ROOT / "data" / "artifacts" / "source_mlp_checkpoint.pt"
SAVE_PATH = PROJECT_ROOT / "data" / "artifacts" / "transfer_mlp_checkpoint.pt"
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "transfer_output"

# ---------------------------------------------------------
# Konfigurationen und Hyperparameter
# ---------------------------------------------------------

cfg_target = mlp.load_target_config(CONFIG_TARGET_SYSTEM)
cfg_mlp = mlp.load_mlp_config(CONFIG_TARGET_MLP)

# mischung aus source und target mlp configs!
BATCH_SIZE = 16 # klein weil target data
EPOCHS = 1000    # testen!
LEARNING_RATE = 0.0001 # wie bei target mlp, da target data
HIDDEN_SIZES = [256, 256, 128, 128, 64, 64] # struktur des source mlp
TRANSFER_HIDDEN_SIZE = 64   # größe der neuen transfer schicht



# ---------------------------------------------------------
# Transferklasse
# ---------------------------------------------------------


class TransferWindowMLP(nn.Module):
    def __init__(self, backbone: mlp.WindowMLP, transfer_hidden_size: int):
        """
        Wrappt das vortrainierte MLP und fuegt eine zusaetzliche Schicht vor
        dem bisherigen Output-Layer ein.
        """
        super().__init__()

        layers = list(backbone.net.children()) #Return an iterator over immediate children modules -> Architektur der layer des source model
        last_linear = layers[-1]
        self.backbone = nn.Sequential(*layers[:-1])  # alles bis zur letzten Linear-Schicht -> Referenzen auf dieselben Layer-Instanzen, die auch im ursprünglichen backbone -> gleiche gewichte
        self.transfer_layer = nn.Linear(last_linear.in_features, transfer_hidden_size) # Linear(64 → 191) --> last_linear.in_features == 64
        self.output_layer = nn.Linear(transfer_hidden_size, mlp.EXPECTED_STEPS)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.backbone(x)
        x = self.activation(self.transfer_layer(x))
        return self.output_layer(x)



def build_transfer_model(hidden_sizes, checkpoint_path, transfer_hidden_size, input_dim, device):
    """
    Laedt das bestehende Source-MLP aus dem Checkpoint und steckt einen neuen
    Layer vor den Output, so dass anschliessend transfer-trainiert werden kann.
    """

    backbone = mlp.WindowMLP(hidden_sizes, input_dim)
    state_dict = torch.load(checkpoint_path, map_location=device)
    backbone.load_state_dict(state_dict)

    model = TransferWindowMLP(backbone, transfer_hidden_size).to(device)

    # Backbone einfrieren, so dass nur Transfer-/Output-Layer lernen
    for param in model.backbone.parameters():
        param.requires_grad = False
    return model


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------


def main():
    """Manueller Einzel-Run (legacy)."""
    mlp.seed_everything()
    device, input_dim, meta_tuple, label_lookup = mlp.build_target_runtime_context(
        cfg_mlp, cfg_target, ANN_ALL_CSV
    )

    train_loader, val_loader = mlp.prepare_loaders_target(
        TRAIN_CSV, VAL_CSV, meta_tuple, label_lookup, BATCH_SIZE, cfg_mlp
    )
    
    # Modellobjekt erstellen und analysieren
    
    print("Lade vortrainiertes MLP von", PRETRAINED_PATH)
    model = build_transfer_model(HIDDEN_SIZES, PRETRAINED_PATH, TRANSFER_HIDDEN_SIZE, input_dim, device)
    print("Aufbau des MLP")
    for layer in list(model.backbone) + [model.transfer_layer, model.activation, model.output_layer]:
        print(layer)

    # Optimierer adam wählen, lernrate übergeben
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE)
    
        
    # Lossfunktion auswählen
    criterion = nn.MSELoss()

    # der kleinste bisher gesehene Loss Wert
    best_val = float("inf") #initial auf unendlich gesetzt -> Damit jede echte Validierungs-Loss im ersten Epochendurchlauf automatisch kleiner
    best_state = None

    # Training über anzahl der geforderten epochen
    for epoch in range(1, EPOCHS + 1):
        
        # Objektreferenzen, keine Kopien -> Model wird geupdatet
        train_loss = mlp.run_epoch(train_loader, model, criterion, optimizer, device)
        val_loss = mlp.run_epoch(val_loader, model, criterion, None, device)
        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
        mlp.save_checkpoint_with_retry(best_state, SAVE_PATH)
        print("Modell gespeichert:", SAVE_PATH)

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


if __name__ == "__main__":
    main()


# ---------------------------------------------------------
# Extern aufrufbare Helper fuer Meta-/Grid-Laufs
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
    Trainiert Transfermodell auf ann_train/ann_val und evaluiert auf ann_test.
    """
    # Initiale Einstellungen und gemeinsamer Laufzeitkontext
    mlp.seed_everything()
    device, input_dim, meta_tuple, label_lookup = mlp.build_target_runtime_context(
        cfg_mlp, cfg_target, ANN_ALL_CSV
    )

    # Hyperparameter defaults fuer diesen Lauf
    bs = batch_size if batch_size is not None else BATCH_SIZE
    stop_patience = patience if patience is not None else epochs

    # DataLoader fuer Train/Val vorbereiten
    train_loader, val_loader = mlp.prepare_loaders_target(
        TRAIN_CSV, VAL_CSV, meta_tuple, label_lookup, bs, cfg_mlp
    )

    # Transfermodell aufbauen (Source-Backbone + Transfer-Layer)
    model = build_transfer_model(hidden_sizes, PRETRAINED_PATH, TRANSFER_HIDDEN_SIZE, input_dim, device)

    # Optimierer und Lossfunktion
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    criterion = nn.MSELoss()

    # Bestes Modell waehrend Training merken
    best_val = float("inf")
    best_state = None
    save_path_obj = Path(save_path) if save_path is not None else None

    # Trainingsschleife mit Early-Stopping
    epochs_no_improve = 0
    for _ in range(1, epochs + 1):
        train_loss = mlp.run_epoch(train_loader, model, criterion, optimizer, device)
        val_loss = mlp.run_epoch(val_loader, model, criterion, None, device)

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= stop_patience:
            break

    # Bestes Modell laden und optional Checkpoint speichern
    if best_state is not None:
        model.load_state_dict(best_state)
        if save_path_obj is not None:
            mlp.save_checkpoint_with_retry(best_state, save_path_obj)

    # Test-Auswertung und optionaler Prediction-Export
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
    Laedt einen Transfer-Checkpoint und evaluiert nur auf ann_test.
    Gedacht fuer Freeze-Phase ohne erneutes Training.
    """
    # Initiale Einstellungen und gemeinsamer Laufzeitkontext
    mlp.seed_everything()
    device, input_dim, meta_tuple, label_lookup = mlp.build_target_runtime_context(
        cfg_mlp, cfg_target, ANN_ALL_CSV
    )
    hs = hidden_sizes if hidden_sizes is not None else HIDDEN_SIZES

    # Modellstruktur aus Source-Backbone + Transfer-Layer erzeugen,
    # dann Transfer-Checkpoint laden.
    model = build_transfer_model(hs, PRETRAINED_PATH, TRANSFER_HIDDEN_SIZE, input_dim, device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Test-Auswertung und optionaler Prediction-Export
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
