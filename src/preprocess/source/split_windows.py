# checked 29.11.2025

import random
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]


EXPORT_NORM_DIR = PROJECT_ROOT / "data" / "processed" / "normalized_rea_windows"
OUT_BASE = PROJECT_ROOT / "data" / "processed" / "mlp_input"

# Unterordner fuer die drei Gruppen
TEST_DIR = OUT_BASE / "test"   # Zielordner fuer Test
TRAIN_DIR = OUT_BASE / "train" # Zielordner fuer Train
VAL_DIR = OUT_BASE / "val"     # Zielordner fuer Val


# Testdaten: ein Block mit 15% der Daten, Rest: Train 70% und Val 15%
BLOCK_FRACTION = 0.15
WINDOW_HOURS = 48  # Fenster reichen 48h
ROWS_PER_WINDOW = 192  # 48h @ 15min
RANDOM_SEED = 42


#---------------------
# Helper
#---------------------


def parse_run_start(run_id: str) -> pd.Timestamp:
    # Gibt den Startzeitpunkt aus der run_id zurueck (z.B. ID001_2025070311)
    
    ts = run_id.split("_")[1]                  # Zweiter Teil nach Unterstrich nehmen
    
    return pd.to_datetime(ts, format="%Y%m%d%H")  # In Timestamp umwandeln


def pick_one_block(n_rows: int, block_size: int):
    # Waehlt genau einen zufaelligen Block (Start- und Endindex)
    
    if block_size <= 0 or block_size > n_rows: # Falls Blockgroesse ungueltig, abbrechen
        return None
    
    random.seed(RANDOM_SEED)                   # Zufalls-Seed setzen
    start = random.randint(0, n_rows - block_size)  # Zufallsstart bestimmen
    end = start + block_size                  # Ende ist Start + Groesse
    
    return (start, end)                       # Tupel mit Start/Ende zurueckgeben


def select_test_data(df: pd.DataFrame, run_starts: pd.Series):
    # Waehlt einen Testblock, filtert nach 48h-Puffer und liefert Testdaten + blockierte run_ids
    
    block_size = int(len(df) * BLOCK_FRACTION)     # Groesse fuer 15% der Zeilen
    block = pick_one_block(len(df), block_size)    # Zufallsblock bestimmen
    if block is None:                              # Wenn kein Block passt
        return None, set()                         # Nichts zurueckgeben

    run_ids_blocked = set()                        # run_ids, die im Block liegen
    test_parts = []                                # Liste fuer gueltige Testfenster
    window_delta = pd.Timedelta(hours=WINDOW_HOURS)  # Laenge eines Fensters

    s, e = block                                   # Start und Ende auspacken
    block_df = df.iloc[s:e]                        # Zeilen fuer den Block nehmen
    run_ids_blocked.update(block_df["run_id"].unique())  # Alle run_ids im Block merken

    block_start_time = block_df["valid_time"].min()      # Fruehester Zeitpunkt im Block
    earlier_starts = run_starts[run_starts < block_start_time]  # Modellstarts davor
    cutoff = block_start_time                        # Standard-Cutoff ist Blockstart
    if not earlier_starts.empty:                     # Falls vorherige Starts existieren
        cutoff = earlier_starts.max() + window_delta # Cutoff auf Ende des letzten Fensters setzen

    trimmed = block_df[block_df["valid_time"] >= cutoff]  # Alles vor Cutoff entfernen
    if not trimmed.empty:                              # Wenn noch etwas uebrig ist
        full = trimmed.groupby("run_id").filter(lambda g: len(g) == ROWS_PER_WINDOW)  # Nur volle Fenster behalten
        if not full.empty:                             # Falls volle Fenster existieren
            test_parts.append(full)                    # Zu den Testteilen hinzufuegen

    if test_parts:                                     # Wenn etwas gesammelt wurde
        return pd.concat(test_parts, ignore_index=True), run_ids_blocked  # Testdaten und blockierte IDs zurueck
    return None, run_ids_blocked                       # Sonst keine Testdaten, aber blockierte IDs zurueck


def split_train_val(rest_df: pd.DataFrame):
    # Teilt die verbleibenden Fenster in Train und Val auf
    
    run_ids = (                                   # Alle run_ids holen
        rest_df["run_id"]
        .drop_duplicates()                        # Dubletten entfernen
        .sample(frac=1.0, random_state=RANDOM_SEED)  # Zufaellig mischen
        .tolist()                                 # Als Liste speichern
    )
    
    train_ids_count = int(len(run_ids) * (70 / 85))  # 70% gesamt -> 70/85 vom Rest
    train_ids = set(run_ids[:train_ids_count])    # Erste Portion fuer Train
    val_ids = set(run_ids[train_ids_count:])      # Rest fuer Val

    train_df = rest_df[rest_df["run_id"].isin(train_ids)]  # Zeilen fuer Train filtern
    val_df = rest_df[rest_df["run_id"].isin(val_ids)]      # Zeilen fuer Val filter
    
    return train_df, val_df                                # Beide DataFrames zurueckgeben



def process_file(path: Path) -> None:
    # Durchlaeuft eine CSV (eine Anlage) und splittet in Test/Train/Val
    
    # CSV laden und nach Zeit sortieren, damit der Block zeitlich zusammenhaengt
    df = pd.read_csv(path, parse_dates=["valid_time"])          # Datei einlesen, Zeitspalte als Datum
    df = df.sort_values("valid_time").reset_index(drop=True)    # Nach Zeit sortieren, Index neu setzen

    # run_id -> Modellstart berechnen, um spaeter Abstaende pruefen zu koennen
    run_starts = (                                              # Serie mit Startzeiten bauen
        df[["run_id"]]                                          # Nur run_id Spalte nehmen
        .drop_duplicates()                                      # Doppelte run_ids entfernen
        .assign(run_start=lambda x: x["run_id"].apply(parse_run_start))  # Startzeit berechnen
        .set_index("run_id")["run_start"]                       # run_id als Index setzen und Spalte holen
    )

    # Testdaten bestimmen
    test_df, run_ids_blocked = select_test_data(df, run_starts) # Testblock holen und blockierte IDs sammeln
    if test_df is not None:                                    # Wenn Testdaten existieren
        test_df.to_csv(TEST_DIR / path.name, index=False)      # Testdaten speichern
        print(f"{path.name}: Test gespeichert ({test_df['run_id'].nunique()} Fenster, 1 Block).")  # Info ausgeben
    else:                                                      # Keine Testdaten
        print(f"{path.name}: Keine Test-Daten nach 48h-Ausschluss.")                              # Hinweis ausgeben

    # Uebrige Daten fuer Train/Val (ohne run_ids im Testblock)
    rest_df = df.loc[~df["run_id"].isin(run_ids_blocked)].reset_index(drop=True)  # Alles ausser Test-run_ids
    if not rest_df.empty:                                        # Wenn noch Daten da sind
        rest_df = rest_df.groupby("run_id").filter(lambda g: len(g) == ROWS_PER_WINDOW)  # Nur volle Fenster behalten
    if rest_df.empty:                                            # Wenn nichts mehr uebrig
        print(f"{path.name}: Keine Daten fuer Train/Val.")       # Hinweis ausgeben
        return                                                   # Funktion beenden

    # Uebrige run_ids mischen und in Train/Val splitten
    train_df, val_df = split_train_val(rest_df)                  # Train- und Val-DataFrames erzeugen

    train_df.to_csv(TRAIN_DIR / path.name, index=False)          # Train speichern
    val_df.to_csv(VAL_DIR / path.name, index=False)              # Val speichern
    print(                                                       # Kurze Zusammenfassung ausgeben
        f"{path.name}: Train {train_df['run_id'].nunique()} Fenster, "
        f"Val {val_df['run_id'].nunique()} Fenster."
    )


#---------------------
# MAIN
#---------------------

def main() -> None:
    for path in EXPORT_NORM_DIR.glob("*_windows_norm.csv"):   # Alle passenden Dateien durchgehen
        process_file(path)                                    # Jede Datei verarbeiten


if __name__ == "__main__":
    main()
