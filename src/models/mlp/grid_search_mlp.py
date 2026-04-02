"""Unified grid search for source/target MLPs."""

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

#---------------------------------------------------------
# Pfade
#---------------------------------------------------------


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]

# make project src importable when run as script
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.mlp import train_source_mlp
from models.mlp import train_target_mlp_with_test


MODE_CONFIG: Dict[str, Dict[str, Any]] = {
    "source": {
        "batch_sizes": [32, 64],
        "hidden_space": [[128, 64], [256, 128, 64]],
        "lr_space": [1e-3, 1e-4],
        "epochs": 50,
        "patience": 15,
        "train_fn": train_source_mlp.train_and_eval,
        "prepare_loaders": train_source_mlp.prepare_data_for_training,
        "artifact_dir": PROJECT_ROOT / "data" / "artifacts" / "grid_search_source_mlp",
        "test_metrics_path": None,
        "use_loader_cache": True,
    },
    "target": {
        "batch_sizes": [16],
        "hidden_space": [[256, 256, 128, 128, 64, 64]],
        "lr_space": [1e-4],
        "epochs": 500,
        "patience": 25,
        "train_fn": train_target_mlp_with_test.train_and_eval,
        "prepare_loaders": None,  # handled inside train_fn
        "artifact_dir": PROJECT_ROOT / "data" / "artifacts" / "grid_search_target_mlp",
        "test_metrics_path": None,  # grid search schreibt keine finalen Testmetriken
        "use_loader_cache": False,
    },
}


def run_grid_search(mode: str) -> None:
    '''
    Skript für Grid-Search über verschiedene Hyperparameter-Kombinationen.
    Unterstützt "source" und "target" MLPs.
    Speichert das beste Modell basierend auf Validierungsverlust.
    '''
    
    # Lade Konfiguration für den Modus Target/Source und Pfade
    cfg = MODE_CONFIG[mode]
    artifact_dir: Path = cfg["artifact_dir"]
    best_model_path = artifact_dir / "best_model.pt"
    artifact_dir.mkdir(parents=True, exist_ok=True)


    best_val = float("inf") # initial auf unendlich gesetzt -> Damit jede echte Validierungs-Loss im ersten Epochendurchlauf automatisch kleiner
    # Speicherobjekte der besten Konfiguration
    best_cfg: Optional[Dict[str, Any]] = None
    loader_cache: Dict[int, Tuple[Any, Any]] = {}

    # iteration über alle Kombinationen der Hyperparameter
    for bs, hidden, lr in itertools.product(cfg["batch_sizes"], cfg["hidden_space"], cfg["lr_space"]): #bildet das kartesische Produkt dieser drei Listen: Ergebnis: ein Iterator über alle Kombinationen 
        
        # Definiere eindeutigen Namen und Speicherpfad für das Modell
        cfg_name = f"bs{bs}_h{'-'.join(map(str, hidden))}_lr{lr}"
        save_path = artifact_dir / f"{cfg_name}.pt"
        print(f"==> Starte {mode} {cfg_name} (epochs={cfg['epochs']}, patience={cfg['patience']})")

        # hält das zurückgegebene Tupel aus prepare_loaders (i.d.R. (train_loader, val_loader))
        # Damit nicht für jede Hyperparameter-Kombi die gleichen Loader neu gebaut werden -> Laufzeitoptimierung
        # bei neuer batch_size muss neu gebaut werden
        
        loaders = None
        
        if cfg["use_loader_cache"]:
            if bs not in loader_cache:
                loader_cache[bs] = cfg["prepare_loaders"](batch_size=bs)
            loaders = loader_cache[bs]

        if cfg["use_loader_cache"]:
            val_loss, test_metrics = cfg["train_fn"](
                lr=lr,
                epochs=cfg["epochs"],
                hidden_sizes=hidden,
                save_path=save_path,
                batch_size=bs,
                patience=cfg["patience"],
                loaders=loaders,
            )
            
        else:
            # Target: keine finalen Testmetriken aus der Grid-Search schreiben
            val_loss, test_metrics = cfg["train_fn"](
                lr=lr,
                epochs=cfg["epochs"],
                hidden_sizes=hidden,
                save_path=save_path,
                batch_size=bs,
                patience=cfg["patience"],
                save_test_metrics=False,
                save_predictions=False,
            )

        record = {
            "mode": mode,
            "config": cfg_name,
            "batch_size": bs,
            "hidden": "-".join(map(str, hidden)),
            "lr": lr,
            "val_loss": val_loss,
            "model_path": str(save_path),
            "test_metrics_csv": str(cfg["test_metrics_path"]) if cfg["test_metrics_path"] else "",
        }

        if val_loss < best_val:
            best_val = val_loss
            best_cfg = record
            if best_model_path.exists():
                best_model_path.unlink()
            save_path.replace(best_model_path)
            print(f"Neues bestes Modell -> {best_model_path}")

    print("Bestes Config:", best_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search fuer Source/Target MLP")
    parser.add_argument(
        "--mode",
        choices=MODE_CONFIG.keys(),
        default="source",
        help="Welches MLP suchen (source/target). Default: source",
    )
    args = parser.parse_args()
    run_grid_search(args.mode)


if __name__ == "__main__":
    main()
