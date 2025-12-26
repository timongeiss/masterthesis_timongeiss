import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]

BASE = PROJECT_ROOT / "data" / "processed" / "mlp_input"
TRAIN_DIR = BASE / "train"
VAL_DIR = BASE / "val"
TEST_DIR = BASE / "test"

TARGET_DIR = PROJECT_ROOT / "data" / "processed" / "target_mlp_input"
TARGET_FILES = {
    "Train": TARGET_DIR / "ann_train.csv",
    "Validation": TARGET_DIR / "ann_val.csv",
    "Test": TARGET_DIR / "ann_test.csv",
}

def load_split(id_prefix: str, directory: Path, split_label: str) -> pd.DataFrame:
    path = directory / f"{id_prefix}_windows_norm.csv"
    if not path.exists():
        return pd.DataFrame(columns=["run_id", "valid_time", "split"])
    df = pd.read_csv(path, parse_dates=["valid_time"], usecols=["run_id", "valid_time"])
    df = df.drop_duplicates(subset=["run_id", "valid_time"])
    df["split"] = split_label
    return df


def plot_id(id_prefix: str) -> None:
    frames = [
        load_split(id_prefix, TRAIN_DIR, "Train"),
        load_split(id_prefix, VAL_DIR, "Validation"),
        load_split(id_prefix, TEST_DIR, "Test"),
    ]
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        print(f"{id_prefix}: keine Daten gefunden.")
        return

    run_ids = sorted(combined["run_id"].unique())
    run_to_y = {rid: idx for idx, rid in enumerate(run_ids)}
    combined["y"] = combined["run_id"].map(run_to_y)

    colors = {"Train": "#1f77b4", "Validation": "#ff7f0e", "Test": "#2ca02c"}
    fig_height = max(3, 0.2 * len(run_ids) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    for split_label, df_split in combined.groupby("split"):
        if df_split.empty:
            continue
        ax.scatter(
            df_split["valid_time"],
            df_split["y"],
            color=colors.get(split_label, "#444444"),
            s=10,
            alpha=0.8,
            label=split_label,
        )

    ax.set_yticks(range(len(run_ids)))
    ax.set_yticklabels(run_ids)
    ax.set_xlabel("valid_time")
    ax.set_ylabel("run_id")
    ax.set_title(f"valid_time coverage per run_id {id_prefix}")
    ax.set_ylim(-0.5, len(run_ids) - 0.5)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.legend()
    fig.autofmt_xdate()

    plt.show()


def plot_target_ann() -> None:
    """Plot coverage for target ann_{train,val,test}.csv."""
    frames = []
    for split_label, path in TARGET_FILES.items():
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["valid_time"], usecols=["run_id", "valid_time"])
        df = df.drop_duplicates(subset=["run_id", "valid_time"])
        df["split"] = split_label
        frames.append(df)

    if not frames:
        print("Keine target ann_* Dateien gefunden.")
        return

    combined = pd.concat(frames, ignore_index=True)
    run_ids = sorted(combined["run_id"].unique())
    run_to_y = {rid: idx for idx, rid in enumerate(run_ids)}
    combined["y"] = combined["run_id"].map(run_to_y)

    colors = {"Train": "#1f77b4", "Validation": "#ff7f0e", "Test": "#2ca02c"}
    fig_height = max(3, 0.2 * len(run_ids) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    for split_label, df_split in combined.groupby("split"):
        if df_split.empty:
            continue
        ax.scatter(
            df_split["valid_time"],
            df_split["y"],
            color=colors.get(split_label, "#444444"),
            s=10,
            alpha=0.8,
            label=split_label,
        )

    ax.set_yticks(range(len(run_ids)))
    ax.set_yticklabels(run_ids)
    ax.set_xlabel("valid_time")
    ax.set_ylabel("run_id")
    ax.set_title("valid_time coverage per run_id (target ann_*)")
    ax.set_ylim(-0.5, len(run_ids) - 0.5)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.legend()
    fig.autofmt_xdate()

    plt.show()


def main() -> None:
    if not TRAIN_DIR.exists() and not VAL_DIR.exists() and not TEST_DIR.exists():
        raise SystemExit("Keine Splits gefunden. Bitte zuerst split_windows.py ausfuehren.")

    # Plot target ann_* first (if present)
    plot_target_ann()

    ids = {p.stem.replace("_windows_norm", "") for p in TRAIN_DIR.glob("*_windows_norm.csv")}
    ids |= {p.stem.replace("_windows_norm", "") for p in VAL_DIR.glob("*_windows_norm.csv")}
    ids |= {p.stem.replace("_windows_norm", "") for p in TEST_DIR.glob("*_windows_norm.csv")}

    if not ids:
        raise SystemExit("Keine Dateien fuer Coverage-Plots gefunden.")

    for id_prefix in sorted(ids):
        plot_id(id_prefix)


if __name__ == "__main__":
    main()
