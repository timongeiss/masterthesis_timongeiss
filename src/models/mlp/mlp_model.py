'''Zentrale Funktionen und Klassen für MLP-Modelle.'''

import pandas as pd
import numpy as np
import random
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from pathlib import Path


# ---------------------------------------------------------
# globale MLP-Konstanten und Config Helper
# ---------------------------------------------------------

EXPECTED_STEPS = 48 * 4 - 1  # 191 Zeitpunkte pro Fenster
DEFAULT_META_DIM = 5  # start_power + 4 meta features
RANDOM_SEED = 42

def load_mlp_config(path: Path) -> dict:
    """
    Liest eine MLP-Config (feature_cols, target_col, meta_features, meta_dim).
    genutzt in:
    train_source_mlp.py,
    train_target_mlp_with_test.py,
    transfer_source_to_target_with_test.py,
    evaluate_sourcemlp_on_source.py,
    evaluate_sourcemlp_on_target.py,
    grid_search_mlp.py
    """
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_target_config(path: Path) -> dict:
    """
    Liest die Target-System-Config als Dict.
    genutzt in:
    train_target_mlp_with_test.py,
    transfer_source_to_target_with_test.py,
    evaluate_sourcemlp_on_target.py
    """
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_metadata_source(path: Path, dc_eff_per_kwp: float) -> dict:
    """
    Lädt die Metadaten und gibt ein dict[AID] = (cap, tilt_norm, azimuth_norm, dc_eff_per_kwp) zurück.
    """
    
    meta_df = pd.read_csv(path, sep=";")
    meta_df = meta_df.drop_duplicates(subset=["ID"]).set_index("ID")

    meta_map = {}
    
    for idx, row in meta_df.iterrows():
        cap = float(row["estimated_dc_capacity"])
        tilt = float(row["tilt"]) / 90.0        # normierung  
        az = (float(row["azimuth"]) % 360.0) / 360.0    # modulo macht wenn größer als 360 wieder unter 360, dann norm
        meta_map[idx] = (cap, tilt, az, dc_eff_per_kwp)

    return meta_map


# ---------------------------------------------------------
# Modellklasse
# ---------------------------------------------------------

class WindowMLP(nn.Module):             # erbt aus nn.Module Klasse
    """
    Einfaches MLP-Modell für Fenster-Vorhersagen.
    genutzt in:
    train_source_mlp.py,
    train_target_mlp_with_test.py,
    transfer_source_to_target_with_test.py,
    evaluate_sourcemlp_on_source.py,
    evaluate_sourcemlp_on_target.py,
    grid_search_mlp.py
    """
    
    def __init__(self, hidden_sizes, input_dim):   # Konstruktor
        super().__init__()

        layers = []
        prev = input_dim

        for size in hidden_sizes:   # zb [128, 64]
            layers.append(nn.Linear(prev, size))    # Lineare Schicht: Erste durchlauf eingabe layer 960 auf zb 128, zweite durchlauf 128 auf 64
            layers.append(nn.ReLU())                # nichtlinearisieren mit aktivierungsfunk ReLu
            prev = size                             # zweite layer überschreibt eingabe -> 128

        layers.append(nn.Linear(prev, EXPECTED_STEPS))  # Letzte ist ausgabe layer mit 191
        self.net = nn.Sequential(*layers)           # Liste layers und kettet alle Schichten der Reihe nach zusammen

    def forward(self, x):           # sollte überschrieben werden
        # x: [batch, INPUT_DIM]
        return self.net(x)      # vorwärtsdurchlauf durch modell mit Tensor x -> ý
    
    
    

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def seed_everything() -> None:
    """
    Sorgt für reproduzierbare Zufallszahlen.
    genutzt in:
    train_source_mlp.py,
    train_target_mlp_with_test.py,
    transfer_source_to_target_with_test.py,
    grid_search_mlp.py,
    evaluate_sourcemlp_on_source.py,
    evaluate_sourcemlp_on_target.py.
    """
    seed=RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_input_dim(feature_cols: list, meta_dim: int = DEFAULT_META_DIM, expected_steps: int = EXPECTED_STEPS) -> int:
    """
    Berechnet INPUT_DIM aus Feature-Anzahl, Meta-Dim und Zeitschritten.
    genutzt in:
    train_source_mlp.py,
    train_target_mlp_with_test.py,
    transfer_source_to_target_with_test.py,
    evaluate_sourcemlp_on_source.py,
    evaluate_sourcemlp_on_target.py,
    grid_search_mlp.py.
    """
    return expected_steps * len(feature_cols) + meta_dim


def build_label_lookup(paths, target_col) -> dict[pd.Timestamp, float]:
    """
    Baut ein Lookup valid_time -> power_norm aus den bereitgestellten CSVs.
    Wörterbuch, das Zeitstempel (valid_time) auf den zugehörigen Zielwert (power_norm) abbildet
    genutzt in:
    train_target_mlp_with_test.py,
    transfer_source_to_target_with_test.py,
    evaluate_sourcemlp_on_target.py.
    """
    # leere dict
    lookup = {}
    # für jede übergebene csv
    for path in paths:
        # einlesen validtime als index
        df = pd.read_csv(path, parse_dates=["valid_time"])
        # power_norm an validtime
        lookup.update({ts: val for ts, val in zip(df["valid_time"], df[target_col])})
        
    return lookup


def build_target_runtime_context(cfg_mlp: dict, cfg_target: dict, ann_all_csv: Path):
    """
    Baut gemeinsame Laufzeitobjekte fuer Target-Workflows:
    device, input_dim, meta_tuple, label_lookup.
    Lookup-Quelle ist immer ann_all.csv.
    """
    feature_cols = cfg_mlp["feature_cols"]
    target_col = cfg_mlp["target_col"]
    meta_dim = cfg_mlp.get("meta_dim", DEFAULT_META_DIM)
    input_dim = compute_input_dim(feature_cols, meta_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Metadaten für das Ziel-System
    meta_tuple = (
        cfg_target["DC_CAPACITY_KWP"],
        cfg_target["TILT_DEG"] / 90.0,  #normierung auf [0,1]
        (cfg_target["AZIMUTH_DEG"] % 360.0) / 360.0,    #normierung auf [0,1]
        cfg_target["DC_EFF_PER_KWP"],
    )
    label_lookup = build_label_lookup([ann_all_csv], target_col)
    return device, input_dim, meta_tuple, label_lookup


def save_checkpoint_with_retry(state_dict, save_path: Path, retries: int = 5, sleep_seconds: float = 0.2) -> None:
    """
    Speichert robust mit Retries und atomarem Replace.
    Hilft bei transienten File-Locks unter Windows.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
    last_error = None

    for _ in range(retries):
        try:
            torch.save(state_dict, tmp_path)
            tmp_path.replace(save_path)
            return
        except Exception as err:
            last_error = err
            time.sleep(sleep_seconds)

    raise RuntimeError(
        f"Checkpoint konnte nicht gespeichert werden: {save_path} (retries={retries})"
    ) from last_error


def make_feature_block(window: pd.DataFrame, meta_tuple: tuple, start_power: float, feature_cols: list, target_col: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Baut Feature- und Target-Vektoren aus einem Fenster DataFrame.
    genutzt in:
    train_source_mlp.py,
    train_target_mlp_with_test.py,
    transfer_source_to_target_with_test.py.
    """
    
    # window: 191 Zeilen
    step_slice = window.iloc[:EXPECTED_STEPS]

    # Zeitabhaengige Features (191 x feature_dim)
    feature_part = step_slice[feature_cols].to_numpy(dtype=np.float32)
    
    # Flatten der Zeitreihe: (191 x feature_dim) -> (191*feature_dim,)
    time_series = feature_part.reshape(-1)

    # Metadaten (5 Werte)
    cap, tilt, az, eff = meta_tuple
    meta_vec = np.array([start_power, cap, tilt, az, eff], dtype=np.float32)

    # Metadaten vorne anhaengen
    features = np.concatenate([meta_vec, time_series])

    # Targets bleiben 191 Werte
    targets = step_slice[target_col].to_numpy(dtype=np.float32)
    
    # 191 * 5 + 5 = 960 Werte

    return features, targets


def collect_windows_target(
    df: pd.DataFrame,
    meta_tuple: tuple,
    label_lookup: dict,
    feature_cols: list,
    target_col: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Zerlegt df in Fenster pro run_id und holt die Startleistung
    aus dem Lookup (15 Minuten vor Fensterbeginn). Erwartet EXACT_EXPECTED_STEPS Zeilen.
    genutzt in:
    train_target_mlp_with_test.py (über build_dataset_target),
    transfer_source_to_target_with_test.py,
    evaluate_sourcemlp_on_target.py.

    """
    samples: list[tuple[np.ndarray, np.ndarray]] = []

    grouped = {rid: grp.sort_values("valid_time") for rid, grp in df.groupby("run_id")}
    for _, grp in sorted(grouped.items(), key=lambda kv: kv[0]):
        if len(grp) != EXPECTED_STEPS:
            continue
        if grp[feature_cols + [target_col]].isna().any().any():
            continue

        window_start = grp["valid_time"].min()
        start_power = label_lookup.get(window_start - pd.Timedelta(minutes=15))
        if start_power is None:
            continue

        feat, targ = make_feature_block(
            grp.reset_index(drop=True),
            meta_tuple,
            float(start_power),
            feature_cols,
            target_col,
        )
        samples.append((feat, targ))

    return samples


def collect_windows_source(df: pd.DataFrame, meta_tuple: tuple, cfg: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Zerlegt df in Fenster pro run_id und holt die Startleistung
    als power_ac 15 Minuten vor Fensterbeginn aus dem selben DataFrame.

    Rückgabe:
        samples: Liste von (features, targets)
    """
    
    # zum sammeln aller features und targets vektoren (ready für batch picking aus samples)
    samples = []
    feature_cols = cfg["feature_cols"]
    target_col = cfg["target_col"]

    # run_id-Gruppen bilden und nach Zeit sortieren
    grouped = {}
    for rid, grp in df.groupby("run_id"):
        grouped[rid] = grp.sort_values("valid_time")

    # stabile Reihenfolge über run_id
    runs = sorted(grouped.items(), key=lambda kv: kv[0])

    for run_id, grp in runs:        #grp entspricht einer runid 192 zeilen
        
        # richtige Fensterlänge prüfen
        if len(grp) != EXPECTED_STEPS + 1:
            continue

        # NaNs in relevanten Spalten -> Fenster verwerfen
        if grp[feature_cols + [target_col]].isna().any().any():
            continue

        # Fensterbeginn (frühester valid_time in diesem run)
        window_start = grp["valid_time"].min()

        # Zeitpunkt 15 Minuten vor Fensterstart
        start_time = window_start - pd.Timedelta(minutes=15)

        # Im gesamten df nach diesem Zeitpunkt suchen
        # (gleiche Anlage, gleicher Datensatz)
        candidates = df.loc[df["valid_time"] == start_time, target_col]


        # Das später noch fixen!
        
        if candidates.empty:
            # kein Messwert 15 min vor Start -> dieses Fenster nicht verwenden
            continue

        # erste gefundene Leistung als Startleistung benutzen
        start_power = float(candidates.iloc[0])

        # Features und Targets für dieses Fenster erzeugen
        feat, targ = make_feature_block(
            grp.reset_index(drop=True),
            meta_tuple,
            start_power,
            feature_cols,
            target_col,
        )

        samples.append((feat, targ))

    return samples


def build_dataset_target(
    csv_path: Path,
    meta_tuple: tuple,
    label_lookup: dict,
    feature_cols: list,
    target_col: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Liest eine einzelne CSV und baut X-/y-Tensoren auf Basis collect_windows_target.
    genutzt in:
    train_target_mlp_with_test.py,
    transfer_source_to_target_with_test.py.
    """
    all_features: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    df = pd.read_csv(csv_path, parse_dates=["valid_time"])
    if not df.empty:
        samples = collect_windows_target(df, meta_tuple, label_lookup, feature_cols, target_col)
        for f, t in samples:
            all_features.append(f)
            all_targets.append(t)

    if not all_features:
        raise ValueError("Keine Fenster nach dem Filtern übrig.")

    X = torch.from_numpy(np.stack(all_features))
    Y = torch.from_numpy(np.stack(all_targets))
    return X, Y



def prepare_loaders_target(train_path: Path, val_path: Path, meta_tuple, label_lookup, batch_size: int, cfg_mlp: dict):
    """
    Erstellt DataLoader für train und val.
    genutzt in:
    train_target_mlp_with_test.py,
    transfer_source_to_target_with_test.py.
    """

    X_train, y_train = build_dataset_target(train_path, meta_tuple, label_lookup,cfg_mlp["feature_cols"], cfg_mlp["target_col"])
    X_val, y_val = build_dataset_target(val_path, meta_tuple, label_lookup,cfg_mlp["feature_cols"], cfg_mlp["target_col"])

    # TensorDataset -> Baut ein Dataset-Objekt, das aus zwei Tensoren besteht (X,Y)
    # DataLoader -> shufflet und liefert mini batches automatisch
    
    train_loader = DataLoader(TensorDataset(X_train, y_train),        
                            batch_size=batch_size, shuffle=True, num_workers=0)          # Bei mehreren Workern werden die Batches schneller vorbereitet, was die Trainingszeit insbesondere bei grÇôÇYeren DatensÇÏtzen deutlich reduziert.                              
    
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader



def run_epoch(loader, model, criterion, optimizer, device):
    """
    Fuehrt eine Trainings- oder Validierungsepoche aus.
    optimizer=None -> Val-Modus.
    genutzt in:
    train_source_mlp.py,
    train_target_mlp_with_test.py,
    transfer_source_to_target_with_test.py,
    grid_search_mlp.py
    """
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_samples = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        if train_mode:
            optimizer.zero_grad()

        preds = model(X)
        loss = criterion(preds, y)

        if train_mode:
            loss.backward()
            optimizer.step()

        bs = X.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / total_samples



# --------------------------------------------------------
# Evaluation Functions
# ---------------------------------------------------------

def load_test_runs_target(csv_path: Path, feature_cols: list, target_col: str, label_lookup: dict, expected_steps: int = EXPECTED_STEPS):
    """
    Lädt Testfenster und liefert run_id, init_label, features, targets, times, model_start.
    genutzt in:
    train_target_mlp_with_test.py,
    transfer_source_to_target_with_test.py,
    evaluate_sourcemlp_on_target.py
    """
    # test csv laden
    df = pd.read_csv(csv_path, parse_dates=["valid_time", "model_start"])
    
    run_ids = []          # speichert run_id pro Fenster
    init_labels = []      # Startleistung 15 min vor Fensterstart
    feature_list = []     # Features je Fenster (191 x feature_dim)
    target_list = []      # Targets je Fenster
    times_list = []       # Zeitachsen je Fenster
    model_starts = []     # model_start je Fenster
    
    #gehe über alle windows
    for run_id, group in df.groupby("run_id"):
        
        group = group.sort_values("valid_time") # sortiere window nach validzeit
        
        #prüfe erneut ob länge passt
        if len(group) != expected_steps:
            print(f"Skipping run {run_id}: expected {expected_steps} rows, got {len(group)}")
            continue
        
        start_time = group["valid_time"].iloc[0] #kleinste zeit = start
        
        # Leistungsfeature finden, sonst skip
        prior_time = start_time - pd.Timedelta(minutes=15) 
        init_label = label_lookup.get(prior_time)
        
        if init_label is None:
            print(f"Skipping run {run_id}: no preceding power_norm found at {prior_time}")
            continue
        
        # sammeln der windows
        run_ids.append(run_id)
        init_labels.append(init_label)
        feature_list.append(group[feature_cols].to_numpy(dtype=np.float32))
        target_list.append(group[target_col].to_numpy(dtype=np.float32))
        times_list.append(group["valid_time"].reset_index(drop=True))
        model_starts.append(group["model_start"].iloc[0])
        
    return run_ids, init_labels, feature_list, target_list, times_list, model_starts


def export_predictions_per_run(df_all: pd.DataFrame, results_dir: Path) -> None:
    """
    Schreibt pro run_id ein CSV mit valid_time/y_true/y_pred etc.
    genutzt in:
    train_target_mlp_with_test.py (optional),
    mlp_model.evaluate_on_test (optional)
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for run_id, grp in df_all.groupby("run_id"):
        run_id_str = str(run_id)
        fname = f"{run_id_str}.csv"
        grp.sort_values("valid_time").to_csv(results_dir / fname, index=False)


def evaluate_on_test(
    model,
    device,
    meta_tuple,
    label_lookup,
    cfg_mlp,
    test_csv_path,
    test_dates=None,
    export_predictions: bool = False,
    results_dir: Path = None,
):
    """
    Evaluiert das Modell auf ann_test.csv und gibt einen DataFrame mit Metriken zurück.
    Speichert optional Vorhersagen pro run_id als CSV.
    genutzt in:
    train_target_mlp_with_test.py,
    transfer_source_to_target_with_test.py
    """
    feature_cols = cfg_mlp["feature_cols"]
    target_col = cfg_mlp["target_col"]
    
    run_ids, init_labels, features, targets, times, model_starts = load_test_runs_target(
        test_csv_path, feature_cols, target_col, label_lookup, EXPECTED_STEPS
    )

    # sammelt pro run_id die Vorhersagen und Targets
    records = []
    
    # Kein Rechengraph, kein Speicher für Gradienten
    with torch.no_grad():
        
        # fensterweise iterieren
        for run_id, init_label, feats, tgt, t_series, m_start in zip(run_ids, init_labels, features, targets, times, model_starts):
            
            # features flatten aus (191 × 5) -> (955,)
            feats_flat = feats.reshape(-1)
            # meta features zusammenführen [start_power, dc_capacity, tilt_norm, azimuth_norm, dc_eff]
            meta_vec = np.array([init_label, *meta_tuple], dtype=np.float32)
            # x tensor erstellen Enddimension: 960
            x = torch.from_numpy(np.concatenate([meta_vec, feats_flat]).astype(np.float32)).to(device)
            
            # Forward-Pass
            pred = model(x).cpu().numpy()

            feature_cols_data = {col: feats[:, idx] for idx, col in enumerate(feature_cols)}

            # DataFrame für einen Lauf mit Zeiten, Vorhersagen und label
            run_df = pd.DataFrame(
                {
                    "run_id": run_id,
                    "model_start": m_start,
                    "valid_time": t_series,
                    "y_true": tgt,
                    "y_pred": pred,
                    **feature_cols_data,
                }
            )
            
            # windowweise ergebnis anhängen
            records.append(run_df)
    
    # alle ergebnisse zusammenketten und in data\results ablegen
    if not records:
        return pd.DataFrame(columns=["date", "group", "nRMSE", "nMAE", "count"])
    df_all = pd.concat(records, ignore_index=True)
    if export_predictions:
        export_predictions_per_run(df_all, results_dir)
    
    # Nacht raus
    df = df_all[df_all["solar_elevation_deg"] >= 0].copy()
    # minuten entfernen, auf tage bringen
    df["date"] = df["valid_time"].dt.date
    df["model_start_date"] = df["model_start"].dt.date
    # sortieren in die zwei gruppen
    df["run_type"] = np.where(df["model_start_date"] < df["date"], "day_ahead", "intra_day")

    rows = []
    # nach tagen sortieren
    for day, g in df.groupby("date"):
        
        # gruppieren in drei zielgruppem
        for label, subset in [
            ("all", g),
            ("day_ahead", g[g["run_type"] == "day_ahead"]),
            ("intra_day", g[g["run_type"] == "intra_day"]),
        ]:
            # metriken berechnen mit numpy
            if len(subset) == 0:    #null wenn zb keine dayahead existiert
                nrmse = np.nan
                nmae = np.nan
            else:
                nrmse = float(np.sqrt(np.mean((subset["y_pred"] - subset["y_true"]) ** 2)))
                nmae = float(np.mean(np.abs(subset["y_pred"] - subset["y_true"])))
                
            # ergebnisse speichern für jeden tag und subset
            rows.append(
                {
                    "date": day,
                    "group": label,
                    "nRMSE": nrmse,
                    "nMAE": nmae,
                    "count": len(subset),
                }
            )
            
    
    metrics = pd.DataFrame(rows)
    # Optional: nur definierte Testtage behalten
    if test_dates is not None:
        metrics = metrics[metrics["date"].isin(test_dates)]
    
    return metrics




