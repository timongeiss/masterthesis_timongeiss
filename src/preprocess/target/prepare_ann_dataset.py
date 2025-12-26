# ready for upload 23_11_2025

import random
from datetime import datetime
from pathlib import Path
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]

# ---- Konfiguration / Einstellungen ----

# Pfade
CONFIG_NORM_PATH = PROJECT_ROOT / "configs" / "config_norm.yaml"
CONFIG_TESTDAYS_PATH = PROJECT_ROOT / "configs" / "config_testdays.yaml"
INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "physical_input"
OUTPUT_TRAIN = PROJECT_ROOT / "data" / "processed" / "target_mlp_input" / "ann_train.csv"
OUTPUT_VAL = PROJECT_ROOT / "data" / "processed" / "target_mlp_input" / "ann_val.csv"
OUTPUT_TEST = PROJECT_ROOT / "data" / "processed" / "target_mlp_input" / "ann_test.csv"   # Output fuer Test-Tage

# Erwartete Anzahl Zeilen pro Lauf (48h Vorhersage in 15-Minuten-Schritten) abzueglich letzten 15 min
EXPECTED_STEPS = 48 * 4 - 1

# Weitere Einstellungen
VAL_FRACTION = 0.2          # Anteil der Laeufe fuer Validierung in prozent
P_INSTALLED_KWP = 9.73      # installierte Leistung in kWp fuer normierung
UTC_TO_LOCAL_OFFSET_H = 2   # gueltige Zeit ist UTC+2, run_id bleibt in UTC


# ---- Hilfsfunktionen ----

def parse_run_id_from_filename(path):
    """
    Extrahiert den Modell-Startzeitpunkt aus dem Dateinamen.
    Erwartetes Format: pv_weather_YYYYMMDDHH.csv
    """
    stem = path.stem
    ts = stem.split("_")[-1]
    return datetime.strptime(ts, "%Y%m%d%H")


def load_norm_config(path: Path) -> dict:
    """Liest die Normierungs-Parameter aus config_norm.yaml."""
    if not path.exists():
        raise FileNotFoundError(f"Norm-Config nicht gefunden: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return cfg


def load_test_days(path: Path) -> set:
    """Liest Testtage (dd.mm.yyyy) aus der YAML-Liste."""
    if not path.exists():
        raise FileNotFoundError(f"Testday-Config nicht gefunden: {path}")
    days = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {datetime.strptime(day, "%d.%m.%Y").date() for day in days}


norm_cfg = load_norm_config(CONFIG_NORM_PATH)
ASWDIR_NORM = norm_cfg["ASWDIR_NORM"]
ASWDIF_NORM = norm_cfg["ASWDIF_NORM"]
T2M_LOWER_NORM = norm_cfg["T2M_LOWER_NORM"]
T2M_UPPER_NORM = T2M_LOWER_NORM + norm_cfg.get("T2M_RANGE", 0.0)
TEST_DATES = load_test_days(CONFIG_TESTDAYS_PATH)


def overlaps_test_days(valid_times, test_dates):
    """
    Prueft, ob eine Zeitreihe (valid_times) irgendeinen der Test-Tage enthaelt.
    Gibt True zurueck, wenn es eine Ueberschneidung gibt.
    """
    return bool(set(valid_times.dt.date) & test_dates)


def prepare_one_file(path, p_installed_w, test_dates):
    """
    Liest eine CSV-Datei ein, bereitet sie fuer das ANN-Training/Test vor
    und gibt einen DataFrame zurueck.
    Die Aufteilung in Train/Val/Test passiert NICHT hier,
    sondern spaeter in main().
    """
    # Modellstart aus Dateinamen (UTC)
    run_start = parse_run_id_from_filename(path)
    # Lokale Startzeit = UTC + Offset
    local_start = run_start + pd.Timedelta(hours=UTC_TO_LOCAL_OFFSET_H)

    # CSV einlesen, Dezimaltrennzeichen ist Komma, valid_time direkt als Datum/Zeit parsen
    df = pd.read_csv(path, decimal=",", parse_dates=["valid_time"])

    # Wichtige Spalten, die vorhanden sein muessen
    required_cols = [
        "valid_time",
        "aswdir_s",
        "aswdifd_s",
        "t_2m",
        "Mittelwertleistung [W]",
        "solar_elevation_deg",
    ]

    # Pruefen ob es in diesen Spalten NaN-Werte gibt
    nan_mask = df[required_cols].isna()
    if nan_mask.any().any():
        print(f"Warnung: Datei {path.name} enthaelt fehlende Werte (NaN) in den wichtigen Spalten!")
        print("Betroffene Spalten (Anzahl NaNs):")
        print(df[required_cols].isna().sum())

    # Nach Zeit sortieren und Index neu setzen
    df = df.sort_values("valid_time").reset_index(drop=True)

    # Lead-Time in Stunden relativ zur lokalen Startzeit berechnen
    df["lead_time_hours"] = (
        (df["valid_time"] - local_start).dt.total_seconds() / 3600.0
    )

    # Pruefen, ob alle Zeilen innerhalb des erwarteten Zeitfensters liegen (0-48 h)
    outside_mask = ~((df["lead_time_hours"] >= 0) & (df["lead_time_hours"] < 48.0))
    if outside_mask.any():
        print(f"Warnung: Datei {path.name} enthaelt Zeilen ausserhalb des 48h-Bereichs!")
        print("Anzahl der betroffenen Zeilen:", outside_mask.sum())
        print(df.loc[outside_mask, ["valid_time", "lead_time_hours"]])

    # Nacht bestimmen (Sonnenhoehe < 0 Grad)
    is_night = df["solar_elevation_deg"] < 0
    # Bei Nacht direkte/diffuse Einstrahlung und Leistung auf 0 setzen
    df.loc[is_night, ["aswdir_s", "aswdifd_s", "Mittelwertleistung [W]"]] = 0.0

    # Warnung, wenn die Anzahl der Zeilen im Zeitfenster nicht der Erwartung entspricht
    if len(df) != EXPECTED_STEPS:
        print(
            f"Warning: {path.name} has {len(df)} rows in 48h window (expected {EXPECTED_STEPS})"
        )

    # ---- Normalisierung der Groessen ----

    # Leistung auf installierte Leistung normieren
    df["power_norm"] = df["Mittelwertleistung [W]"] / float(p_installed_w)

    # Einstrahlung normieren
    df["aswdir_s_norm"] = df["aswdir_s"] / ASWDIR_NORM
    df["aswdifd_s_norm"] = df["aswdifd_s"] / ASWDIF_NORM

    # Temperatur: Kelvin -> Celsius
    t2m_c = df["t_2m"] - 273.15
    # Temperatur auf Bereich [-10, 40] degC normieren und auf [0,1] begrenzen
    df["t2m_norm"] = ((t2m_c + T2M_LOWER_NORM) / T2M_UPPER_NORM).clip(lower=0.0, upper=1.0)

    # ---- Metadaten zum Lauf hinzufuegen ----

    df["model_start"] = local_start         # lokale Modellstartzeit (UTC+2)
    df["run_id"] = run_start.strftime("%Y%m%d%H")  # run_id im UTC-Format

    # Nur die relevanten Spalten fuer das ANN-Training/Test behalten
    cols = [
        "run_id",
        "model_start",
        "valid_time",
        "lead_time_hours",
        "solar_elevation_deg",
        "aswdir_s_norm",
        "aswdifd_s_norm",
        "t2m_norm",
        "power_norm",
    ]
    return df[cols].reset_index(drop=True)


def split_train_val_by_run(df, val_fraction):
    """
    Teilt die Daten in Trainings- und Validierungsdaten auf
    Die Aufteilung passiert auf Basis der run_id (ganze Laeufe werden getrennt)
    """
    run_ids = df["run_id"].unique().tolist()
    rng = random.Random(42)
    rng.shuffle(run_ids)

    val_count = max(1, int(round(len(run_ids) * val_fraction)))
    val_ids = set(run_ids[:val_count])

    val_df = df[df["run_id"].isin(val_ids)].reset_index(drop=True)
    train_df = df[~df["run_id"].isin(val_ids)].reset_index(drop=True)

    return train_df, val_df


# ---- Hauptlaeufer ----

def main():
    """
    - CSV-Dateien einlesen
    - Daten vorbereiten und kombinieren
    - Trainings-, Validierungs- und Testdatensatz erzeugen
    - Ergebnisse als CSV speichern
    """
    # installierte Leistung von kWp in W umrechnen
    p_installed_w = P_INSTALLED_KWP * 1000.0

    # Testtage aus Config
    test_dates = TEST_DATES

    # Alle passenden CSV-Dateien im Eingabeordner finden
    files = sorted(INPUT_DIR.glob("pv_weather_*.csv"))

    frames = []        # Train/Val
    test_frames = []   # Test-Kandidaten (Runs, die Testtage beruehren)

    # Jede Datei nacheinander verarbeiten
    for f in files:
        try:
            frame = prepare_one_file(f, p_installed_w, test_dates)
        except Exception as e:
            raise RuntimeError(f"Failed processing {f}: {e}") from e

        # Laeufe mit Testtagen -> Test-Kandidaten, sonst Train/Val
        if overlaps_test_days(frame["valid_time"], test_dates):
            test_frames.append(frame)
        else:
            frames.append(frame)

    # ---- Trainings- / Validierungsdaten aufbereiten ----

    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined = combined.sort_values(["model_start", "valid_time"]).reset_index(drop=True)

    train_df, val_df = split_train_val_by_run(
        combined, val_fraction=VAL_FRACTION
    )

    OUTPUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUTPUT_TRAIN, index=False)
    val_df.to_csv(OUTPUT_VAL, index=False)

    # ---- Testdaten aufbereiten ----
    # Jetzt zusaetzliche Bedingung:
    # In ann_test.csv sollen NUR run_ids landen,
    # deren model_start-Datum selbst ein Testtag ist.
    combined_test = None
    if test_frames:
        combined_test_all = pd.concat(test_frames, axis=0, ignore_index=True)
        combined_test_all = combined_test_all.sort_values(
            ["model_start", "valid_time"]
        ).reset_index(drop=True)

        # run_ids auswaehlen, deren Startdatum ein Testtag ist
        keep_run_ids = []
        for run_id, group in combined_test_all.groupby("run_id"):
            start_date = group["model_start"].iloc[0].date()
            if start_date in test_dates:
                keep_run_ids.append(run_id)

        combined_test = combined_test_all[
            combined_test_all["run_id"].isin(keep_run_ids)
        ].reset_index(drop=True)

        if not combined_test.empty:
            OUTPUT_TEST.parent.mkdir(parents=True, exist_ok=True)
            combined_test.to_csv(OUTPUT_TEST, index=False)
        else:
            print(
                "Hinweis: Es gibt zwar Laeufe, die Testtage beruehren, "
                "aber kein run_id startet an einem Testtag. "
                "ann_test.csv wird nicht geschrieben."
            )
            combined_test = None

    # ---- Zusammenfassung auf der Konsole ausgeben ----
    print(f"Total runs read: {len(files)}")
    print(
        "Kept train/val runs:",
        combined["run_id"].nunique(),
        "-> train",
        train_df["run_id"].nunique(),
        "val",
        val_df["run_id"].nunique(),
    )
    print(
        "Rows:",
        "train",
        len(train_df),
        "val",
        len(val_df),
    )
    if combined_test is not None:
        print("Test runs (start on test days):", combined_test["run_id"].nunique())
        print("Rows: test", len(combined_test))
        print("Test output written to:", OUTPUT_TEST)
    else:
        print("No test runs with model_start on test days written.")

    print("Outputs written to:", OUTPUT_TRAIN, OUTPUT_VAL, OUTPUT_TEST, sep="\n- ")


if __name__ == "__main__":
    main()
