'''Skript zur Evaluation eines vortrainierten Source-MLP Modells auf dem Target-Datensatz.
Kein Transfer Learning, nur reine Forward-Paesse und Metrik-Berechnung.'''

from pathlib import Path
import torch
import sys

# ---------------------------------------------------------
# Pfade
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
TEST_CSV = PROJECT_ROOT / "data" / "processed" / "target_mlp_input" / "ann_test.csv"
ANN_ALL_CSV = PROJECT_ROOT / "data" / "processed" / "target_mlp_input" / "ann_all.csv"
SAVE_PATH = PROJECT_ROOT / "data" / "artifacts" / "source_mlp_checkpoint.pt"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config_target_system.yaml"
CONFIG_TARGET_MLP = PROJECT_ROOT / "configs" / "config_target_mlp.yaml"
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "source_output"
LOOKUP_SOURCES = [ANN_ALL_CSV]

# Projekteigenen Code importierbar machen
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.mlp import mlp_model as mlp


# ---------------------------------------------------------
# Konfig und Hyperparameter
# ---------------------------------------------------------

cfg_target_mlp = mlp.load_mlp_config(CONFIG_TARGET_MLP)
cfg_target = mlp.load_target_config(CONFIG_PATH)
HIDDEN_SIZES = [256, 256, 128, 128, 64, 64]


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------


def _build_runtime_context(lookup_sources=None):
    """
    Baut gemeinsame Laufzeitobjekte fuer Eval:
    device, input_dim, meta_tuple, label_lookup.
    """
    feature_cols = cfg_target_mlp["feature_cols"]
    meta_dim = cfg_target_mlp["meta_dim"]
    target_col = cfg_target_mlp["target_col"]

    input_dim = mlp.compute_input_dim(feature_cols, meta_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_tuple = (
        cfg_target["DC_CAPACITY_KWP"],
        cfg_target["TILT_DEG"] / 90.0,
        (cfg_target["AZIMUTH_DEG"] % 360.0) / 360.0,
        cfg_target["DC_EFF_PER_KWP"],
    )

    lookup_paths = lookup_sources if lookup_sources is not None else LOOKUP_SOURCES
    label_lookup = mlp.build_label_lookup(lookup_paths, target_col)

    return device, input_dim, meta_tuple, label_lookup


def evaluate_checkpoint_on_test(
    checkpoint_path,
    hidden_sizes=None,
    save_predictions=False,
    results_dir=None,
    lookup_sources=None,
):
    """
    Laedt Source-Checkpoint und evaluiert auf ann_test.csv.
    Gedacht fuer Coldstart-Meta-Lauf (ohne Training).
    """
    mlp.seed_everything()
    device, input_dim, meta_tuple, label_lookup = _build_runtime_context(
        lookup_sources=lookup_sources
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
        cfg_target_mlp,
        TEST_CSV,
        export_predictions=save_predictions,
        results_dir=export_dir,
    )
    return test_metrics


# ---------------------------------------------------------
# Hauptfunktion (manuell)
# ---------------------------------------------------------


def main():
    """Manueller Einzelaufruf (legacy)."""
    test_metrics = evaluate_checkpoint_on_test(
        checkpoint_path=SAVE_PATH,
        hidden_sizes=HIDDEN_SIZES,
        save_predictions=True,
        results_dir=RESULTS_DIR,
        lookup_sources=LOOKUP_SOURCES,
    )
    print("Testmetriken berechnet:", len(test_metrics), "Zeilen")


if __name__ == "__main__":
    main()
