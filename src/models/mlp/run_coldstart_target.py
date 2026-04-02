# ready for push 01.04.2026

"""Cold-start runner (target + transfer + source): adaptive phase + freeze phase.

Ablauf:
1) Testtage aus model_start ableiten (erste 4 und letzte 2 Tage verwerfen).
2) Adaptive Phase: pro Testtag neu trainieren (Train <= t-3, Test == t).
3) Freeze Phase: einmal trainieren (am ersten Freeze-Tag), danach nur Checkpoint-Eval.
4) Source-Modell: immer eval-only (kein Training).
5) Finale Tagesmetriken zentral aus allen Iterations-Prediction-CSVs berechnen.
"""

from __future__ import annotations

import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import train_target_mlp_with_test as target_train
import transfer_source_to_target_with_test as transfer_train

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.mlp import evaluate_sourcemlp_on_target as source_eval

ANN_ALL_CSV = PROJECT_ROOT / "data" / "processed" / "target_mlp_input" / "ann_all.csv"
TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "target_mlp_input" / "ann_train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "processed" / "target_mlp_input" / "ann_val.csv"
TEST_CSV = PROJECT_ROOT / "data" / "processed" / "target_mlp_input" / "ann_test.csv"

DROP_INITIAL_DAYS = 4
DROP_FINAL_DAYS = 2
FREEZE_TEST_DAYS = 30
VAL_FRACTION = 0.2
RANDOM_SEED = 42

MODEL_SPECS = [
    {
        "name": "target",
        "module": target_train,
        "mode": "trainable",
        "results_dir": PROJECT_ROOT / "data" / "results" / "target_output",
        "metrics_path": PROJECT_ROOT / "data" / "artifacts" / "target_model_testday_metrics.csv",
        "checkpoint_path": PROJECT_ROOT / "data" / "artifacts" / "target_mlp_checkpoint.pt",
    },
    {
        "name": "transfer",
        "module": transfer_train,
        "mode": "trainable",
        "results_dir": PROJECT_ROOT / "data" / "results" / "transfer_output",
        "metrics_path": PROJECT_ROOT / "data" / "artifacts" / "transfer_model_testday_metrics.csv",
        "checkpoint_path": PROJECT_ROOT / "data" / "artifacts" / "transfer_mlp_checkpoint.pt",
    },
    {
        "name": "source",
        "module": source_eval,
        "mode": "eval_only",
        "results_dir": PROJECT_ROOT / "data" / "results" / "source_output",
        "metrics_path": PROJECT_ROOT / "data" / "artifacts" / "source_model_testday_metrics.csv",
        "checkpoint_path": PROJECT_ROOT / "data" / "artifacts" / "source_mlp_checkpoint.pt",
    },
]


def load_base_dataset(path: Path) -> pd.DataFrame:
    """
    Load and validate ann_all.csv
    """
    
    if not path.exists():
        raise FileNotFoundError(f"Base dataset not found: {path}")

    df = pd.read_csv(path, parse_dates=["valid_time", "model_start"])
    if df.empty:
        raise ValueError(f"Base dataset is empty: {path}")

    required_cols = {"run_id", "model_start", "valid_time", "power_norm", "solar_elevation_deg"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in ann_all.csv: {sorted(missing)}")

    return df.sort_values(["model_start", "valid_time"]).reset_index(drop=True)


def get_test_days(df: pd.DataFrame, drop_initial_days: int, drop_final_days: int) -> list[pd.Timestamp]:
    """
    Compute usable test days after dropping initial/final days
    """
    
    run_days = sorted(pd.to_datetime(df["model_start"]).dt.normalize().unique())
    min_required = drop_initial_days + drop_final_days + 1
    if len(run_days) < min_required:
        raise ValueError(f"Need at least {min_required} run days, found only {len(run_days)}.")
    return list(run_days[drop_initial_days : len(run_days) - drop_final_days])


def split_train_val_run_ids(run_ids: list[str], val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    """
    Split run_ids into train/val sets with reproducible random shuffle
    """
    
    if len(run_ids) < 2:
        return set(), set()

    ids = list(run_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    val_count = max(1, int(round(len(ids) * val_fraction)))
    val_ids = set(ids[:val_count])
    train_ids = set(ids[val_count:])

    if not train_ids:
        moved = next(iter(val_ids))
        val_ids.remove(moved)
        train_ids.add(moved)

    return train_ids, val_ids


def write_iteration_split_csvs(
    df: pd.DataFrame,
    test_day: pd.Timestamp,
    val_fraction: float,
    seed: int,
) -> tuple[int, int, int]:
    
    """
    Write ann_train/ann_val/ann_test for one iteration test day
    """
    
    run_start_day = pd.to_datetime(df["model_start"]).dt.normalize()
    train_cutoff_day = test_day - pd.Timedelta(days=3)

    train_candidate_ids = (
        df.loc[run_start_day <= train_cutoff_day, "run_id"]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )
    test_ids = (
        df.loc[run_start_day == test_day, "run_id"]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )

    train_ids, val_ids = split_train_val_run_ids(train_candidate_ids, val_fraction, seed)
    if not train_ids or not val_ids or not test_ids:
        return 0, 0, 0

    run_id_str = df["run_id"].astype(str)
    train_df = df[run_id_str.isin(train_ids)].reset_index(drop=True)
    val_df = df[run_id_str.isin(val_ids)].reset_index(drop=True)
    test_df = df[run_id_str.isin(test_ids)].reset_index(drop=True)

    TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    return len(train_ids), len(val_ids), len(test_ids)


def write_test_csv_for_day(df: pd.DataFrame, test_day: pd.Timestamp) -> int:
    """
    Write ann_test.csv for one day (used in freeze inference-only iterations)
    """
    
    run_start_day = pd.to_datetime(df["model_start"]).dt.normalize()
    test_ids = (
        df.loc[run_start_day == test_day, "run_id"]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )
    if not test_ids:
        return 0

    run_id_str = df["run_id"].astype(str)
    test_df = df[run_id_str.isin(test_ids)].reset_index(drop=True)
    TEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(TEST_CSV, index=False)
    return len(test_ids)


def prepare_iteration_results_dir(base_results_dir: Path, test_day: pd.Timestamp) -> Path:
    """
    Create iteration result dir and clear old per-run CSVs in that dir
    """
    
    iteration_dir = base_results_dir / f"iter_{test_day.strftime('%Y-%m-%d')}"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    for old_csv in iteration_dir.glob("*.csv"):
        old_csv.unlink()
    return iteration_dir


def collect_iteration_prediction_files(iteration_dir: Path) -> list[Path]:
    """
    Return all per-run prediction CSVs from folder iter_2025 for one iteration for the metrics aggregation step
    """
    return sorted(iteration_dir.glob("*.csv"))


def compute_daily_metrics_from_prediction_files(prediction_files: list[Path]) -> pd.DataFrame:
    """
    Compute daily all/day_ahead/intra_day metrics from per-run prediction CSVs
    """
    
    if not prediction_files:
        raise RuntimeError("No prediction files collected for aggregation.")

    # vorfiltern der CSVs, die nicht lesbar sind oder nicht die erwarteten Spalten haben, um Fehler in der Aggregation zu vermeiden
    # sammelt pro run_id-Datei die Vorhersagen und Targets
    frames = []
    # fensterweise (Datei pro run_id) iterieren
    for path in prediction_files:
        try:
            df = pd.read_csv(path, parse_dates=["valid_time", "model_start"])
        except Exception:
            continue
        required = {"valid_time", "model_start", "y_true", "y_pred", "solar_elevation_deg"}
        if not required.issubset(df.columns):
            continue
        frames.append(df[["valid_time", "model_start", "y_true", "y_pred", "solar_elevation_deg"]].copy())

    if not frames:
        raise RuntimeError("No readable prediction records found for aggregation.")

    # alle Ergebnisse zusammenketten
    all_pred = pd.concat(frames, ignore_index=True)
    # numerische Spalten robust casten
    all_pred["y_true"] = pd.to_numeric(all_pred["y_true"], errors="coerce")
    all_pred["y_pred"] = pd.to_numeric(all_pred["y_pred"], errors="coerce")
    all_pred["solar_elevation_deg"] = pd.to_numeric(all_pred["solar_elevation_deg"], errors="coerce")
    all_pred = all_pred.dropna(subset=["valid_time", "model_start", "y_true", "y_pred", "solar_elevation_deg"])

    # Nacht raus
    all_pred = all_pred[all_pred["solar_elevation_deg"] >= 0].copy()
    if all_pred.empty:
        raise RuntimeError("All aggregated prediction rows were filtered out as night/invalid.")

    # Minuten entfernen, auf Tage bringen
    all_pred["date"] = all_pred["valid_time"].dt.date
    all_pred["model_start_date"] = all_pred["model_start"].dt.date
    # sortieren in die zwei Gruppen
    all_pred["run_type"] = np.where(
        all_pred["model_start_date"] < all_pred["date"], "day_ahead", "intra_day"
    )

    rows = []
    # nach Tagen sortieren
    for day, day_df in all_pred.groupby("date"):
        # gruppieren in drei Zielgruppen
        for group, subset in [
            ("all", day_df),
            ("day_ahead", day_df[day_df["run_type"] == "day_ahead"]),
            ("intra_day", day_df[day_df["run_type"] == "intra_day"]),
        ]:
            # Metriken berechnen mit numpy
            if subset.empty:
                nmae = np.nan
                nrmse = np.nan
                count = 0
            else:
                err = subset["y_pred"] - subset["y_true"]
                nmae = float(np.mean(np.abs(err)))
                nrmse = float(np.sqrt(np.mean(err**2)))
                count = int(len(subset))

            # Ergebnisse speichern fuer jeden Tag und Subset
            rows.append(
                {
                    "date": day,
                    "group": group,
                    "nMAE": nmae,
                    "nRMSE": nrmse,
                    "count": count,
                }
            )

    # DataFrame final aufbauen und sortieren
    metrics = pd.DataFrame(rows)
    return metrics.sort_values(["date", "group"]).reset_index(drop=True)


def run_train_eval_for_model(
    spec: dict,
    test_day: pd.Timestamp,
    idx: int,
    phase_label: str,
) -> list[Path]:
    """
    Run one train+eval iteration for one trainable model spec
    """
    
    module = spec["module"]
    
    # legt ergebnis ornder an und löscht alte csvs
    iteration_dir = prepare_iteration_results_dir(spec["results_dir"], test_day)

    best_val, test_metrics = module.train_and_eval(
        lr=module.LEARNING_RATE,
        epochs=module.EPOCHS,
        hidden_sizes=module.HIDDEN_SIZES,
        save_path=spec["checkpoint_path"],
        batch_size=module.BATCH_SIZE,
        patience=module.EPOCHS,
        save_predictions=True,
        results_dir=iteration_dir,
    )
    print(
        f"[{spec['name']} {phase_label} {idx}] {test_day.date()}: "
        f"best_val={best_val:.6f}, raw_metric_rows={len(test_metrics)}"
    )
    return collect_iteration_prediction_files(iteration_dir)


def run_eval_only_for_model(spec: dict, test_day: pd.Timestamp, idx: int, phase_label: str) -> list[Path]:
    """
    Run one eval-only iteration for one model spec.
    """
    
    module = spec["module"]
    iteration_dir = prepare_iteration_results_dir(spec["results_dir"], test_day)

    test_metrics = module.evaluate_checkpoint_on_test(
        checkpoint_path=spec["checkpoint_path"],
        hidden_sizes=module.HIDDEN_SIZES,
        save_predictions=True,
        results_dir=iteration_dir,
    )
    print(
        f"[{spec['name']} {phase_label} {idx}] {test_day.date()}: "
        f"raw_metric_rows={len(test_metrics)}"
    )
    return collect_iteration_prediction_files(iteration_dir)


def main() -> None:
    """
    Run cold-start v2 for target+transfer+source and write aggregated metrics.
    """
    # Schritt 1: Testtage ableiten
    df_all = load_base_dataset(ANN_ALL_CSV)
    test_days = get_test_days(df_all, DROP_INITIAL_DAYS, DROP_FINAL_DAYS)

    # Schritt 2: Adaptive Phase + Freeze Phase festlegen
    total_test_days = len(test_days)
    freeze_days_count = min(FREEZE_TEST_DAYS, total_test_days)
    adaptive_days = test_days[: total_test_days - freeze_days_count]
    freeze_days = test_days[total_test_days - freeze_days_count :]

    print(
        f"Total test days={total_test_days}, adaptive={len(adaptive_days)}, "
        f"freeze={len(freeze_days)}"
    )

    # schritt 3: pro Modell eine Liste mit Prediction-Dateipfaden sammeln (für die finale Metriken-Berechnung am Ende)
    collected_prediction_files = {spec["name"]: [] for spec in MODEL_SPECS}

    trainable_specs = [s for s in MODEL_SPECS if s["mode"] == "trainable"]
    eval_only_specs = [s for s in MODEL_SPECS if s["mode"] == "eval_only"]

    # schritt 4:
    # Phase A: tägliches Retraining (trainable) + eval-only (source)
    for i, day in enumerate(adaptive_days, start=1):
        train_n, val_n, test_n = write_iteration_split_csvs(
            df_all,
            test_day=day,
            val_fraction=VAL_FRACTION,
            seed=RANDOM_SEED,
        )
        if train_n == 0 or val_n == 0 or test_n == 0:
            print(f"[A {i}] Skip {day.date()}: train={train_n}, val={val_n}, test={test_n}")
            continue

        print(
            f"[A {i}] {day.date()}: train_runs={train_n}, "
            f"val_runs={val_n}, test_runs={test_n}"
        )

        # schleife über Modelle: train+eval für trainable, eval-only für source -> gibt eine Liste dieser CSV-Dateipfade zurück (files)
        for spec in trainable_specs:
            files = run_train_eval_for_model(spec, day, i, "A")
            collected_prediction_files[spec["name"]].extend(files)

        for spec in eval_only_specs:
            files = run_eval_only_for_model(spec, day, i, "A-eval")
            collected_prediction_files[spec["name"]].extend(files)

    # Phase B: ein letztes Modell trainieren, dann nur noch eval
    if freeze_days:
        first_freeze_day = freeze_days[0]

        # einmaliger split mit allen daten bis vor den ersten freeze-tag als train+val, erster freeze-tag als test, dann schleife zum testen der tage einzeln
        train_n, val_n, test_n = write_iteration_split_csvs(
            df_all,
            test_day=first_freeze_day,
            val_fraction=VAL_FRACTION,
            seed=RANDOM_SEED,
        )
        if train_n > 0 and val_n > 0 and test_n > 0:
            print(
                f"[F 1] Freeze-train {first_freeze_day.date()}: train_runs={train_n}, "
                f"val_runs={val_n}, test_runs={test_n}"
            )
            
            # schleife über Modelle: train+eval für trainable, eval-only für source -> gibt eine Liste dieser CSV-Dateipfade zurück (files)
            for spec in trainable_specs:
                files = run_train_eval_for_model(spec, first_freeze_day, len(adaptive_days) + 1, "F-train")
                collected_prediction_files[spec["name"]].extend(files)

            # Source bleibt eval-only, auch auf erstem Freeze-Tag
            for spec in eval_only_specs:
                files = run_eval_only_for_model(spec, first_freeze_day, len(adaptive_days) + 1, "F-eval")
                collected_prediction_files[spec["name"]].extend(files)
        else:
            print(
                f"[F 1] Skip-train {first_freeze_day.date()}: "
                f"train={train_n}, val={val_n}, test={test_n}"
            )
        
        # iteration über die restlichen freeze-tage mit eval-only für alle modelle (auch source), da kein neues training mehr stattfindet
        for offset, day in enumerate(freeze_days[1:], start=2):
            iter_idx = len(adaptive_days) + offset
            
            # Schreibt nur ann_test.csv für diesen Tag (kein train/val mehr)
            test_n = write_test_csv_for_day(df_all, day)
            if test_n == 0:
                print(f"[F {offset}] Skip-eval {day.date()}: test=0")
                continue
            
            print(f"[F {offset}] Freeze-eval {day.date()}: test_runs={test_n}")
            for spec in trainable_specs:
                files = run_eval_only_for_model(spec, day, iter_idx, "F-eval")
                collected_prediction_files[spec["name"]].extend(files)

            for spec in eval_only_specs:
                files = run_eval_only_for_model(spec, day, iter_idx, "F-eval")
                collected_prediction_files[spec["name"]].extend(files)

    # Zentrale finale Tagesmetriken je Modell aus allen Prediction-Dateien berechnen
    for spec in MODEL_SPECS:
        model_name = spec["name"]
        pred_files = collected_prediction_files[model_name]
        final_metrics = compute_daily_metrics_from_prediction_files(pred_files)
        spec["metrics_path"].parent.mkdir(parents=True, exist_ok=True)
        final_metrics.to_csv(spec["metrics_path"], index=False)

        print(f"[{model_name}] Collected prediction files: {len(pred_files)}")
        print(f"[{model_name}] Saved aggregated metrics: {spec['metrics_path']}")
        print(f"[{model_name}] Checkpoint used/final: {spec['checkpoint_path']}")


if __name__ == "__main__":
    main()
