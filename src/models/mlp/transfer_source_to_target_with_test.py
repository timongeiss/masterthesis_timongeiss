'''Skript zum Transferlernen von einem vortrainierten MLP auf ein Zielsystem mit Testtagen.'''

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

PRETRAINED_PATH = PROJECT_ROOT / "data" / "artifacts" / "source_mlp_checkpoint.pt"
SAVE_PATH = PROJECT_ROOT / "data" / "artifacts" / "transfer_mlp_checkpoint.pt"
TEST_METRICS_PATH = PROJECT_ROOT / "data" / "artifacts" / "transfer_model_testday_metrics.csv"
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "transfer_output"

# ---------------------------------------------------------
# Konfigurationen und Hyperparameter
# ---------------------------------------------------------

cfg_target = mlp.load_target_config(CONFIG_TARGET_SYSTEM)
TEST_DATES = mlp.load_test_days(CONFIG_TESTDAYS_PATH)
cfg_mlp = mlp.load_mlp_config(CONFIG_TARGET_MLP)

# mischung aus source und target mlp configs!
BATCH_SIZE = 16 # klein weil target data
EPOCHS = 513    # testen!
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
    Laedt das bestehende MLP aus dem Checkpoint und steckt einen neuen Layer
    vor den Output, so dass anschliessend weiter trainiert werden kann.
    """

    backbone = mlp.WindowMLP(hidden_sizes, input_dim)
    state_dict = torch.load(checkpoint_path, map_location=device)
    backbone.load_state_dict(state_dict)

    model = TransferWindowMLP(backbone, transfer_hidden_size).to(device)
    
    # Backbone einfrieren, so dass nur die neue Transfer-Schicht und der Output lernen
    for param in model.backbone.parameters():
        param.requires_grad = False
    return model




# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    
    # Initale Einstellungen
    
    mlp.seed_everything()
    feature_cols = cfg_mlp["feature_cols"]
    target_col = cfg_mlp["target_col"]
    meta_dim = cfg_mlp.get("meta_dim", mlp.DEFAULT_META_DIM)
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
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Training über anzahl der geforderten epochen
    
    for epoch in range(1, EPOCHS + 1):
        
        # Objektreferenzen, keine Kopien -> Model wird geupdatet
        train_loss = mlp.run_epoch(train_loader, model, criterion, optimizer, device)   # durchschnittlichen Loss Epoche
        val_loss = mlp.run_epoch(val_loader, model, criterion, None, device)
        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print("  Modell gespeichert:", SAVE_PATH)


    # Test-Auswertung
    test_metrics = mlp.evaluate_on_test(model, device, meta_tuple, label_lookup, cfg_mlp, TEST_CSV, TEST_DATES, export_predictions=True, results_dir=RESULTS_DIR)
    
    # csv exportieren
    TEST_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_metrics.to_csv(TEST_METRICS_PATH, index=False)
    print("Testmetriken gespeichert:", TEST_METRICS_PATH)



if __name__ == "__main__":
    main()
