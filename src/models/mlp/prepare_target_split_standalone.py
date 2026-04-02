"""Standalone split builder for target MLP input CSVs.

Erzeugt aus ann_all.csv die drei Dateien:
- ann_train.csv
- ann_val.csv
- ann_test.csv

Logik (coldstart-nah):
- Nutzt model_start-Tage, verwirft die ersten 4 und letzten 2 Tage.
- Testtage = letzte min(30, verfuegbare Tage).
- Train/Val-Tage = Tage <= erster Testtag - 3 Tage, davon die letzten min(61, ...).
- Train/Val-Split auf run_id-Basis mit val_fraction=0.2 und seed=42.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]

ANN_ALL_CSV = PROJECT_ROOT / "data" / "processed" / "target_mlp_input" / "ann_all.csv"
TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "target_mlp_input" / "ann_train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "processed" / "target_mlp_input" / "ann_val.csv"
TEST_CSV = PROJECT_ROOT / "data" / "processed" / "target_mlp_input" / "ann_test.csv"

DROP_INITIAL_DAYS = 4
DROP_FINAL_DAYS = 2
TRAINVAL_DAYS_TARGET = 61
TEST_DAYS_TARGET = 30
TRAIN_TEST_GAP_DAYS = 3
VAL_FRACTION = 0.2
RANDOM_SEED = 42


def load_base_dataset(path: Path) -> pd.DataFrame:
    """Load ann_all.csv and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Base dataset not found: {path}")

    df = pd.read_csv(path, parse_dates=["valid_time", "model_start"])
    if df.empty:
        raise ValueError(f"Base dataset is empty: {path}")

    required_cols = {"run_id", "model_start", "valid_time"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in ann_all.csv: {sorted(missing)}")

    return df.sort_values(["model_start", "valid_time"]).reset_index(drop=True)


def split_train_val_run_ids(run_ids: list[str], val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    """Split run_ids into train/val sets with reproducible random shuffle."""
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


def _format_day_range(days: list[pd.Timestamp]) -> str:
    if not days:
        return "n/a"
    return f"{days[0].date()} -> {days[-1].date()}"


def main() -> None:
    df = load_base_dataset(ANN_ALL_CSV)

    run_days = sorted(pd.to_datetime(df["model_start"]).dt.normalize().unique())
    usable_days = list(run_days[DROP_INITIAL_DAYS : len(run_days) - DROP_FINAL_DAYS])
    if not usable_days:
        raise ValueError(
            "No usable run days after dropping boundaries "
            f"(drop_initial={DROP_INITIAL_DAYS}, drop_final={DROP_FINAL_DAYS})."
        )

    test_days_count = min(TEST_DAYS_TARGET, len(usable_days))
    test_days = usable_days[-test_days_count:]
    first_test_day = test_days[0]

    trainval_cutoff_day = first_test_day - pd.Timedelta(days=TRAIN_TEST_GAP_DAYS)
    trainval_candidate_days = [d for d in usable_days if d <= trainval_cutoff_day]
    trainval_days_count = min(TRAINVAL_DAYS_TARGET, len(trainval_candidate_days))
    trainval_days = trainval_candidate_days[-trainval_days_count:]

    if not trainval_days:
        raise ValueError(
            "No train/val days available with configured gap. "
            f"Need days <= {trainval_cutoff_day.date()}."
        )

    if max(trainval_days) > min(test_days) - pd.Timedelta(days=TRAIN_TEST_GAP_DAYS):
        raise RuntimeError("Split integrity failed: train/val days violate 3-day gap to test days.")

    run_start_day = pd.to_datetime(df["model_start"]).dt.normalize()
    trainval_day_set = set(trainval_days)
    test_day_set = set(test_days)

    trainval_candidate_ids = (
        df.loc[run_start_day.isin(trainval_day_set), "run_id"].drop_duplicates().astype(str).tolist()
    )
    test_ids = df.loc[run_start_day.isin(test_day_set), "run_id"].drop_duplicates().astype(str).tolist()

    train_ids, val_ids = split_train_val_run_ids(trainval_candidate_ids, VAL_FRACTION, RANDOM_SEED)
    test_id_set = set(test_ids)

    if not train_ids or not val_ids or not test_id_set:
        raise ValueError(
            "Split resulted in empty subset(s): "
            f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_id_set)}"
        )

    overlap = (train_ids & val_ids) | (train_ids & test_id_set) | (val_ids & test_id_set)
    if overlap:
        raise RuntimeError(f"Split integrity failed: overlapping run_ids found ({len(overlap)}).")

    run_id_str = df["run_id"].astype(str)
    train_df = df[run_id_str.isin(train_ids)].reset_index(drop=True)
    val_df = df[run_id_str.isin(val_ids)].reset_index(drop=True)
    test_df = df[run_id_str.isin(test_id_set)].reset_index(drop=True)

    TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print("Standalone target split complete.")
    print(
        "Run days: total=",
        len(run_days),
        ", usable=",
        len(usable_days),
        ", dropped(initial/final)=",
        f"{DROP_INITIAL_DAYS}/{DROP_FINAL_DAYS}",
        sep="",
    )
    print(f"Train/Val days: {len(trainval_days)} ({_format_day_range(trainval_days)})")
    print(f"Test days: {len(test_days)} ({_format_day_range(test_days)})")
    if trainval_days_count < TRAINVAL_DAYS_TARGET:
        print(
            f"Note: train/val days auto-shortened to {trainval_days_count} "
            f"(target {TRAINVAL_DAYS_TARGET})."
        )
    if test_days_count < TEST_DAYS_TARGET:
        print(f"Note: test days auto-shortened to {test_days_count} (target {TEST_DAYS_TARGET}).")
    print(
        "Run counts: "
        f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_id_set)}"
    )
    print(
        "Row counts: "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    print("Output written:")
    print(f"- {TRAIN_CSV}")
    print(f"- {VAL_CSV}")
    print(f"- {TEST_CSV}")


if __name__ == "__main__":
    main()
