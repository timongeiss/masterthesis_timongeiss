"""
Erstellt die Preislogik-Auswertungen fuer die vier Optimierungsmodelle.

Das Skript liest die vorhandenen Dispatch-Ergebnisse der Modelle `physical`,
`source`, `target` und `transfer` ein. Jede Viertelstunde wird danach in eine
fachliche Fallklasse eingeordnet:

- Forecastfehler: Overforecast oder Underforecast
- Vorzeichen des Day-Ahead-Preises
- Vorzeichen des reBAP-Preises
- Vergleich Day-Ahead gegen reBAP
- geplanter Netzfluss
- realisierter Netzfluss

Auf Basis dieser Klassifikation wird eine CSV-Datei erzeugt:

- `data/optimization/dispatch_combination_occurrences.csv`
- `data/optimization/dispatch_combination_summary.csv`

Die zugehoerigen PNG-Tabellen werden bewusst in das separate Skript
`plot_price_logik.py` ausgelagert. Dieses Skript kuemmert sich daher
ausschliesslich um Klassifikation, Aggregation und CSV-Export.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPTIMIZATION_DIR = PROJECT_ROOT / "data" / "optimization"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config_optimization.yaml"

MODELS = ("physical", "source", "target", "transfer")

DISPATCH_FILES = {
    "physical": OPTIMIZATION_DIR / "physical" / "optimization_dispatch_physical.csv",
    "source": OPTIMIZATION_DIR / "source" / "optimization_dispatch_source.csv",
    "target": OPTIMIZATION_DIR / "target" / "optimization_dispatch_target.csv",
    "transfer": OPTIMIZATION_DIR / "transfer" / "optimization_dispatch_transfer.csv",
}

OCCURRENCES_CSV = OPTIMIZATION_DIR / "dispatch_combination_occurrences.csv"
SUMMARY_CSV = OPTIMIZATION_DIR / "dispatch_combination_summary.csv"


# ---------------------------------------------------------------------------
# Fachliche Reihenfolgen und feste Zuordnungen
# ---------------------------------------------------------------------------

SELECTED_COVER = (
    ("physical", "2025-08-10"),
    ("source", "2025-09-24"),
    ("target", "2025-07-21"),
    ("transfer", "2025-09-14"),
)

FORECAST_STATE_ORDER = ("over", "under")
PRICE_SIGN_ORDER = ("neg", "pos")
DA_VS_IMB_ORDER = ("DA_gt_imb", "DA_lt_imb")
GRID_STATE_ORDER = ("export", "import", "no_flow")

FORECAST_CLASS_CODES = {"over": "1", "under": "2"}
PRICE_SIGN_COMBO_CODES = {
    ("neg", "neg"): "1",
    ("pos", "pos"): "2",
    ("neg", "pos"): "3",
    ("pos", "neg"): "4",
}
DA_VS_IMB_CODES = {"DA_gt_imb": "1", "DA_lt_imb": "2"}
GRID_STATE_CODES = {"export": "1", "import": "2", "no_flow": "3"}


# ---------------------------------------------------------------------------
# Grundeinstellungen fuer die Berechnung
# ---------------------------------------------------------------------------

POWER_EPS_W = 1e-3  # Schwellwert fuer die Unterscheidung von Import/Export/No-Flow in Watt
STEP_HOURS = 0.25


# ---------------------------------------------------------------------------
# Spaltenlisten
# ---------------------------------------------------------------------------

REQUIRED_DISPATCH_COLUMNS = [
    "dispatch_day",
    "datetime",
    "input_price_EUR_MWh",
    "real_imbalance_price_eur_mwh",
    "input_P_pv_forecast_W",
    "real_P_pv_W",
    "opt_P_grid_in_W",
    "opt_P_grid_out_W",
    "opt_P_grid_net_W",
    "real_P_grid_in_W",
    "real_P_grid_out_W",
    "real_P_grid_net_W",
    "real_imbalance_W",
    "real_grid_fee_step_eur",           # nur die reale Grid Fee
    "real_rebap_settlement_step_eur",   # nur das Imbalance-/reBAP-Settlement
    "real_revenue_step_eur",            # DA/Fahrplanerlös minus reale Grid Fee, kein aging
]

OCCURRENCE_COLUMNS = [
    "model",
    "dispatch_day",
    "datetime",
    "forecast_state",
    "da_price_sign",
    "imbalance_price_sign",
    "da_vs_imb",
    "planned_grid_state",
    "realized_grid_state",
    "grid_transition",
    "combination_key",
    "input_price_EUR_MWh",
    "real_imbalance_price_eur_mwh",
    "input_P_pv_forecast_W",
    "real_P_pv_W",
    "forecast_error_W",
    "opt_P_grid_in_W",
    "opt_P_grid_out_W",
    "opt_P_grid_net_W",
    "real_P_grid_in_W",
    "real_P_grid_out_W",
    "real_P_grid_net_W",
    "real_imbalance_W",
    "real_rebap_settlement_step_eur",
    "real_revenue_step_eur",
    "abs_imbalance_energy_step_kwh",
    "impact_delta_grid_fee_step_eur",
    "impact_rebap_settlement_step_eur",
    "impact_imbalance_da_reference_step_eur",
    "impact_imbalance_spread_step_eur",
    "impact_total_step_eur",
]

SUMMARY_ORDER_COLUMNS = [
    "forecast_state",
    "da_price_sign",
    "imbalance_price_sign",
    "da_vs_imb",
    "planned_grid_state",
    "realized_grid_state",
]

SUMMARY_COUNT_COLUMNS = [
    "step_count",
    "day_count",
    "model_day_count",
    "physical_step_count",
    "source_step_count",
    "target_step_count",
    "transfer_step_count",
]

SUMMARY_IMPACT_COLUMNS = [
    "impact_total_eur",
    "impact_delta_grid_fee_eur",
    "impact_rebap_settlement_eur",
    "impact_imbalance_da_reference_eur",
    "impact_imbalance_spread_eur",
]

SUMMARY_ENERGY_COLUMNS = [
    "energy_kwh",
]

SUMMARY_EXPORT_COLUMNS = [
    "classification",
    "forecast_state",
    "da_price_sign",
    "imbalance_price_sign",
    "da_vs_imb",
    "planned_grid_state",
    "realized_grid_state",
    "grid_transition",
    "combination_key",
    "step_count",
    "day_count",
    "model_day_count",
    "physical_step_count",
    "source_step_count",
    "target_step_count",
    "transfer_step_count",
    "energy_kwh",
    "physical_energy_kwh",
    "source_energy_kwh",
    "target_energy_kwh",
    "transfer_energy_kwh",
    "impact_total_eur",
    "impact_delta_grid_fee_eur",
    "impact_rebap_settlement_eur",
    "impact_imbalance_da_reference_eur",
    "impact_imbalance_spread_eur",
    "physical_impact_total_eur",
    "physical_impact_delta_grid_fee_eur",
    "physical_impact_rebap_settlement_eur",
    "physical_impact_imbalance_da_reference_eur",
    "physical_impact_imbalance_spread_eur",
    "source_impact_total_eur",
    "source_impact_delta_grid_fee_eur",
    "source_impact_rebap_settlement_eur",
    "source_impact_imbalance_da_reference_eur",
    "source_impact_imbalance_spread_eur",
    "target_impact_total_eur",
    "target_impact_delta_grid_fee_eur",
    "target_impact_rebap_settlement_eur",
    "target_impact_imbalance_da_reference_eur",
    "target_impact_imbalance_spread_eur",
    "transfer_impact_total_eur",
    "transfer_impact_delta_grid_fee_eur",
    "transfer_impact_rebap_settlement_eur",
    "transfer_impact_imbalance_da_reference_eur",
    "transfer_impact_imbalance_spread_eur",
    "models",
    "selected_cover_model_days",
]


# ---------------------------------------------------------------------------
# Kleine Basis-Helfer
# ---------------------------------------------------------------------------

def ensure_required_columns(
    data: pd.DataFrame,
    required_columns: list[str],
    source_name: str,
) -> None:
    """
    Prueft, ob eine eingelesene Tabelle alle benoetigten Spalten enthaelt.

    Eingaben:
    - `data`: bereits eingelesene Tabelle
    - `required_columns`: Liste aller Pflichtspalten
    - `source_name`: sprechender Name fuer die Fehlermeldung

    Rueckgabe:
    - keine; die Funktion wirft nur bei fehlenden Spalten einen Fehler
    """

    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Fehlende Spalten in {source_name}: {missing_text}")


def load_grid_fee_eur_per_mwh(config_path: Path = CONFIG_PATH) -> float:
    """
    Liest die konfigurierte Netzentgelt-Konstante aus der YAML-Datei.

    Eingaben:
    - `config_path`: Pfad zur Optimierungs-Konfiguration

    Rueckgabe:
    - ein Float-Wert in EUR pro MWh
    """

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return float(config["optimization"]["grid_fee_eur_per_mwh"])


def build_combination_key(data: pd.DataFrame) -> pd.Series:
    """
    Baut den textuellen Schluessel fuer eine vollstaendige Fallkombination.

    Eingaben:
    - `data`: Tabelle mit den bereits klassifizierten Kernspalten

    Rueckgabe:
    - eine `pd.Series` mit einem eindeutigen Schluessel pro Zeile
    """

    return (
        data["forecast_state"]
        + "__da_"
        + data["da_price_sign"]
        + "__imb_"
        + data["imbalance_price_sign"]
        + "__"
        + data["da_vs_imb"]
        + "__plan_"
        + data["planned_grid_state"]
        + "__real_"
        + data["realized_grid_state"]
    )


def build_classification_id(data: pd.DataFrame) -> pd.Series:
    """
    Baut die kompakte numerische Klassen-ID fuer die Summary.

    Eingaben:
    - `data`: Tabelle mit Forecast-, Preis- und Grid-State-Spalten

    Rueckgabe:
    - eine `pd.Series` im Format `a.b.c.d.e`
    """

    forecast_code = data["forecast_state"].map(FORECAST_CLASS_CODES)
    price_pairs = pd.Series(
        list(
            zip(
                data["da_price_sign"].astype(str),
                data["imbalance_price_sign"].astype(str),
            )
        ),
        index=data.index,
        dtype="object",
    )
    price_code = price_pairs.map(PRICE_SIGN_COMBO_CODES)
    da_vs_imb_code = data["da_vs_imb"].map(DA_VS_IMB_CODES)
    planned_code = data["planned_grid_state"].map(GRID_STATE_CODES)
    realized_code = data["realized_grid_state"].map(GRID_STATE_CODES)

    missing_mask = (
        forecast_code.isna()
        | price_code.isna()
        | da_vs_imb_code.isna()
        | planned_code.isna()
        | realized_code.isna()
    )
    if missing_mask.any():
        raise ValueError("Die Spalte 'classification' konnte nicht vollstaendig erzeugt werden.")

    return (
        forecast_code.astype(str)
        + "."
        + price_code.astype(str)
        + "."
        + da_vs_imb_code.astype(str)
        + "."
        + planned_code.astype(str)
        + "."
        + realized_code.astype(str)
    )


# ---------------------------------------------------------------------------
# Laden der Eingabedaten
# ---------------------------------------------------------------------------

def load_dispatch_tables() -> pd.DataFrame:
    """
    Liest die vier Dispatch-CSV-Dateien ein und fuegt sie zusammen.

    Eingaben:
    - keine; die Pfade sind oben als Konstanten definiert

    Rueckgabe:
    - eine grosse Tabelle mit allen Viertelstunden aller Modelle
    """

    frames: list[pd.DataFrame] = []

    for model in MODELS:
        path = DISPATCH_FILES[model]
        if not path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {path}")

        frame = pd.read_csv(
            path,
            decimal=",",
            parse_dates=["dispatch_day", "datetime"],
        )
        ensure_required_columns(frame, REQUIRED_DISPATCH_COLUMNS, path.name)
        frame["model"] = model
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Klassifikation der Zeitschritte
# ---------------------------------------------------------------------------

def classify_grid_state(
    grid_in_w: pd.Series,
    grid_out_w: pd.Series,
    power_eps_w: float = POWER_EPS_W,
) -> pd.Series:
    """
    Bestimmt den Netzstatus aus Import- und Exportleistung.

    Eingaben:
    - `grid_in_w`: geplante oder realisierte Importleistung in Watt
    - `grid_out_w`: geplante oder realisierte Exportleistung in Watt
    - `power_eps_w`: Schwellwert, unterhalb dessen Werte als Null gelten

    Rueckgabe:
    - eine `pd.Series` mit `import`, `export`, `no_flow`, `both` oder `unknown`
    """

    import_mask = pd.to_numeric(grid_in_w, errors="coerce") > power_eps_w
    export_mask = pd.to_numeric(grid_out_w, errors="coerce") > power_eps_w

    states = np.select(
        [
            import_mask & ~export_mask, # nur Import
            export_mask & ~import_mask, # nur Export
            ~import_mask & ~export_mask,# kein Fluss
            import_mask & export_mask,  # sowohl Import als auch Export, darf nicht auftreten, dowstream verhindert das
        ],
        ["import", "export", "no_flow", "both"],
        default="unknown",
    )
    return pd.Series(states, index=grid_in_w.index, dtype="object")


def classify_rows(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ergaenzt die rohe Dispatch-Tabelle um alle fachlichen Klassifikationsspalten.

    Eingaben:
    - `data`: zusammengefuehrte Dispatch-Tabelle aller Modelle

    Rueckgabe:
    - neue Tabelle mit Forecast-, Preis-, Grid-State- und Impact-Spalten
    """

    classified = data.copy()
    grid_fee_eur_per_mwh = load_grid_fee_eur_per_mwh()

    classified["forecast_error_W"] = (
        classified["input_P_pv_forecast_W"] - classified["real_P_pv_W"]
    )

    classified["forecast_state"] = np.select(
        [classified["forecast_error_W"] > 0, classified["forecast_error_W"] < 0],
        ["over", "under"],
        default="equal",
    )

    classified["da_price_sign"] = np.select(
        [classified["input_price_EUR_MWh"] > 0, classified["input_price_EUR_MWh"] < 0],
        ["pos", "neg"],
        default="zero",
    )

    classified["imbalance_price_sign"] = np.select(
        [
            classified["real_imbalance_price_eur_mwh"] > 0,
            classified["real_imbalance_price_eur_mwh"] < 0,
        ],
        ["pos", "neg"],
        default="zero",
    )

    price_difference = (
        classified["input_price_EUR_MWh"]
        - classified["real_imbalance_price_eur_mwh"]
    )
    classified["da_vs_imb"] = np.select(
        [price_difference > 0, price_difference < 0],
        ["DA_gt_imb", "DA_lt_imb"],
        default="DA_eq_imb",
    )

    classified["planned_grid_state"] = classify_grid_state(
        classified["opt_P_grid_in_W"],
        classified["opt_P_grid_out_W"],
    )
    classified["realized_grid_state"] = classify_grid_state(
        classified["real_P_grid_in_W"],
        classified["real_P_grid_out_W"],
    )
    classified["grid_transition"] = (
        classified["planned_grid_state"] + "_to_" + classified["realized_grid_state"]
    )

    planned_grid_fee_step_eur = (
        grid_fee_eur_per_mwh
        * pd.to_numeric(classified["opt_P_grid_in_W"], errors="coerce")
        * STEP_HOURS
        / 1e6
    )
    real_grid_fee_step_eur = pd.to_numeric(
        classified["real_grid_fee_step_eur"],
        errors="coerce",
    )
    classified["impact_delta_grid_fee_step_eur"] = (
        planned_grid_fee_step_eur - real_grid_fee_step_eur
    )

    classified["impact_rebap_settlement_step_eur"] = pd.to_numeric(
        classified["real_rebap_settlement_step_eur"],
        errors="coerce",
    )
    classified["abs_imbalance_energy_step_kwh"] = (
        pd.to_numeric(classified["real_imbalance_W"], errors="coerce").abs()
        * STEP_HOURS
        / 1e3
    )
    imbalance_energy_mwh = (
        pd.to_numeric(classified["real_imbalance_W"], errors="coerce") * STEP_HOURS / 1e6
    )
    da_price_eur_mwh = pd.to_numeric(classified["input_price_EUR_MWh"], errors="coerce")
    imbalance_price_eur_mwh = pd.to_numeric(
        classified["real_imbalance_price_eur_mwh"],
        errors="coerce",
    )
    classified["impact_imbalance_da_reference_step_eur"] = (
        -imbalance_energy_mwh * da_price_eur_mwh
    )
    classified["impact_imbalance_spread_step_eur"] = (
        -imbalance_energy_mwh * (imbalance_price_eur_mwh - da_price_eur_mwh)
    )
    classified["impact_total_step_eur"] = (
        classified["impact_delta_grid_fee_step_eur"]
        + classified["impact_rebap_settlement_step_eur"]
    )

    classified["combination_key"] = build_combination_key(classified)
    return classified


def validate_grid_state_domain(data: pd.DataFrame) -> None:
    """
    Prueft, ob nach der Klassifikation nur gueltige Grid-States vorliegen.

    Eingaben:
    - `data`: bereits klassifizierte Tabelle

    Rueckgabe:
    - keine; bei ungueltigen Zustanden wird ein Fehler geworfen
    """

    valid_states = set(GRID_STATE_ORDER)
    invalid_mask = (
        ~data["planned_grid_state"].isin(valid_states)
        | ~data["realized_grid_state"].isin(valid_states)
    )
    if invalid_mask.any():
        invalid_count = int(invalid_mask.sum())
        raise RuntimeError(
            f"Ungueltige Grid-State-Klassifikation in {invalid_count} Zeilen."
        )


def filter_core_cases(data: pd.DataFrame) -> pd.DataFrame:
    """
    Filtert alle Zeitschritte auf die fuer die Preislogik relevanten Kernfaelle.

    Eingaben:
    - `data`: vollstaendig klassifizierte Tabelle

    Rueckgabe:
    - reduzierte Tabelle mit genau den Exportspalten aus `OCCURRENCE_COLUMNS`
    """

    mask = (
        data["forecast_state"].isin(FORECAST_STATE_ORDER)
        & data["da_price_sign"].isin(PRICE_SIGN_ORDER)
        & data["imbalance_price_sign"].isin(PRICE_SIGN_ORDER)
        & data["da_vs_imb"].isin(DA_VS_IMB_ORDER)
    )

    filtered = data.loc[mask].copy()
    validate_grid_state_domain(filtered)
    filtered = filtered.loc[:, OCCURRENCE_COLUMNS].copy()
    filtered.sort_values(["dispatch_day", "datetime", "model"], inplace=True)
    return filtered


def add_occurrence_classification(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ergaenzt die Occurrence-Tabelle um die numerische Klassen-ID.

    Eingaben:
    - `data`: Occurrence-Tabelle ohne Exportformatierung

    Rueckgabe:
    - Kopie mit zusaetzlicher Spalte `classification`
    """

    enriched = data.copy()
    enriched["classification"] = build_classification_id(enriched)
    return enriched


def format_occurrences_for_export(data: pd.DataFrame) -> pd.DataFrame:
    """
    Formatiert Datums- und Zeitspalten fuer den finalen CSV-Export.

    Eingaben:
    - `data`: Occurrence-Tabelle nach dem Kernfilter

    Rueckgabe:
    - dieselbe Tabelle mit String-Darstellung fuer `dispatch_day` und `datetime`
    """

    formatted = add_occurrence_classification(data)
    formatted["dispatch_day"] = pd.to_datetime(formatted["dispatch_day"]).dt.strftime(
        "%Y-%m-%d"
    )
    formatted["datetime"] = pd.to_datetime(formatted["datetime"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    ordered_columns = ["classification"] + OCCURRENCE_COLUMNS
    return formatted.loc[:, ordered_columns].copy()


# ---------------------------------------------------------------------------
# Aufbau von Occurrence- und Summary-Tabellen
# ---------------------------------------------------------------------------

def build_full_summary_index() -> pd.DataFrame:
    """
    Erzeugt die vollstaendige 144er-Fallmatrix fuer die Summary.

    Eingaben:
    - keine; alle benoetigten Reihenfolgen sind oben fest definiert

    Rueckgabe:
    - Tabelle mit allen moeglichen Kombinationen der Kernfaelle
    """

    index_df = pd.DataFrame(
        product(
            FORECAST_STATE_ORDER,
            PRICE_SIGN_ORDER,
            PRICE_SIGN_ORDER,
            DA_VS_IMB_ORDER,
            GRID_STATE_ORDER,
            GRID_STATE_ORDER,
        ),
        columns=SUMMARY_ORDER_COLUMNS,
    )
    index_df["grid_transition"] = (
        index_df["planned_grid_state"] + "_to_" + index_df["realized_grid_state"]
    )
    index_df["combination_key"] = build_combination_key(index_df)
    index_df["classification"] = build_classification_id(index_df)
    return index_df


def apply_summary_sort_order(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Sortiert die Summary mit festen fachlichen Kategorien.

    Eingaben:
    - `summary`: Summary-Tabelle vor der finalen Sortierung

    Rueckgabe:
    - sortierte Summary-Tabelle
    """

    ordered = summary.copy()

    ordered["forecast_state"] = pd.Categorical(
        ordered["forecast_state"],
        categories=FORECAST_STATE_ORDER,
        ordered=True,
    )
    ordered["da_price_sign"] = pd.Categorical(
        ordered["da_price_sign"],
        categories=PRICE_SIGN_ORDER,
        ordered=True,
    )
    ordered["imbalance_price_sign"] = pd.Categorical(
        ordered["imbalance_price_sign"],
        categories=PRICE_SIGN_ORDER,
        ordered=True,
    )
    ordered["da_vs_imb"] = pd.Categorical(
        ordered["da_vs_imb"],
        categories=DA_VS_IMB_ORDER,
        ordered=True,
    )
    ordered["planned_grid_state"] = pd.Categorical(
        ordered["planned_grid_state"],
        categories=GRID_STATE_ORDER,
        ordered=True,
    )
    ordered["realized_grid_state"] = pd.Categorical(
        ordered["realized_grid_state"],
        categories=GRID_STATE_ORDER,
        ordered=True,
    )

    ordered.sort_values(SUMMARY_ORDER_COLUMNS, inplace=True)
    ordered.reset_index(drop=True, inplace=True)

    for column in SUMMARY_ORDER_COLUMNS:
        ordered[column] = ordered[column].astype(str)

    return ordered


def build_cover_string(
    cover_times: dict[str, list[str]],
    selected_labels: list[str],
) -> str:
    """
    Baut den Text fuer die Spalte `selected_cover_model_days`.

    Eingaben:
    - `cover_times`: Zuordnung von Modell-Tag-Labels zu Uhrzeiten
    - `selected_labels`: feste Ausgabereihenfolge der Modell-Tage

    Rueckgabe:
    - ein Pipe-getrennter String
    """

    parts: list[str] = []

    for label in selected_labels:
        if label not in cover_times:
            continue
        time_text = ",".join(cover_times[label])
        parts.append(f"{label} [{time_text}]")

    return "|".join(parts)


def add_selected_cover_columns(
    summary: pd.DataFrame,
    occurrences: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ergaenzt die Summary um die vorbereiteten Cover-Tage mit Uhrzeiten.

    Eingaben:
    - `summary`: bereits aggregierte Summary-Tabelle
    - `occurrences`: Occurrence-Tabelle mit allen beobachteten Zeitpunkten

    Rueckgabe:
    - Kopie der Summary mit der Spalte `selected_cover_model_days`
    """

    covered = occurrences.copy()
    selected_labels: list[str] = []
    cover_time_map: dict[str, dict[str, list[str]]] = {}

    for model, day in SELECTED_COVER:
        label = f"{model}:{day}"
        selected_labels.append(label)

        selected_rows = covered[
            (covered["model"] == model) & (covered["dispatch_day"] == day)
        ][["combination_key", "datetime"]].copy()

        if selected_rows.empty:
            continue

        selected_rows["time_label"] = pd.to_datetime(selected_rows["datetime"]).dt.strftime(
            "%H:%M"
        )
        grouped_rows = (
            selected_rows.groupby("combination_key", sort=False)["time_label"]
            .apply(list)
            .to_dict()
        )

        for combination_key, time_list in grouped_rows.items():
            unique_times = list(dict.fromkeys(time_list))
            if combination_key not in cover_time_map:
                cover_time_map[combination_key] = {}
            cover_time_map[combination_key][label] = unique_times

    summary_with_cover = summary.copy()
    summary_with_cover["selected_cover_model_days"] = summary_with_cover["combination_key"].map(
        lambda key: build_cover_string(cover_time_map.get(key, {}), selected_labels)
    )
    return summary_with_cover


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert die Occurrence-Daten zur finalen Summary-Tabelle.

    Eingaben:
    - `data`: formatierte Occurrence-Tabelle der Kernfaelle

    Rueckgabe:
    - vollstaendige Summary mit Zero Rows, Count-Spalten und Impact-Summen
    """

    working = data.copy()
    working["model_day_key"] = working["model"] + ":" + working["dispatch_day"]

    observed_summary = (
        working.groupby("combination_key", sort=False)
        .agg(
            step_count=("datetime", "size"),
            day_count=("dispatch_day", "nunique"),
            model_day_count=("model_day_key", "nunique"),
            models=("model", lambda s: ",".join(sorted(pd.unique(s)))),
            energy_kwh=("abs_imbalance_energy_step_kwh", "sum"),
            impact_total_eur=("impact_total_step_eur", "sum"),
            impact_delta_grid_fee_eur=("impact_delta_grid_fee_step_eur", "sum"),
            impact_rebap_settlement_eur=("impact_rebap_settlement_step_eur", "sum"),
            impact_imbalance_da_reference_eur=("impact_imbalance_da_reference_step_eur", "sum"),
            impact_imbalance_spread_eur=("impact_imbalance_spread_step_eur", "sum"),
        )
        .reset_index()
    )

    per_model_counts = (
        working.groupby(["combination_key", "model"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=MODELS, fill_value=0)
        .reset_index()
    )
    per_model_counts.rename(
        columns={model: f"{model}_step_count" for model in MODELS},
        inplace=True,
    )
    observed_summary = observed_summary.merge(
        per_model_counts,
        on="combination_key",
        how="left",
    )

    per_model_impacts = (
        working.groupby(["combination_key", "model"], sort=False)
        .agg(
            energy_step_kwh=("abs_imbalance_energy_step_kwh", "sum"),
            impact_total_step_eur=("impact_total_step_eur", "sum"),
            impact_delta_grid_fee_step_eur=("impact_delta_grid_fee_step_eur", "sum"),
            impact_rebap_settlement_step_eur=("impact_rebap_settlement_step_eur", "sum"),
            impact_imbalance_da_reference_step_eur=("impact_imbalance_da_reference_step_eur", "sum"),
            impact_imbalance_spread_step_eur=("impact_imbalance_spread_step_eur", "sum"),
        )
        .unstack(fill_value=0.0)
    )
    per_model_impacts.columns = [
        f"{model}_{metric.replace('_step_', '_')}"
        for metric, model in per_model_impacts.columns
    ]
    per_model_impacts = per_model_impacts.reset_index()
    observed_summary = observed_summary.merge(
        per_model_impacts,
        on="combination_key",
        how="left",
    )

    summary = build_full_summary_index().merge(
        observed_summary,
        on="combination_key",
        how="left",
    )
    summary["models"] = summary["models"].fillna("")

    for column in SUMMARY_COUNT_COLUMNS:
        summary[column] = (
            pd.to_numeric(summary[column], errors="coerce").fillna(0).astype(int)
        )

    energy_columns = SUMMARY_ENERGY_COLUMNS + [
        f"{model}_energy_kwh"
        for model in MODELS
    ]
    impact_columns = SUMMARY_IMPACT_COLUMNS + [
        f"{model}_{metric}"
        for model in MODELS
        for metric in (
            "impact_total_eur",
            "impact_delta_grid_fee_eur",
            "impact_rebap_settlement_eur",
            "impact_imbalance_da_reference_eur",
            "impact_imbalance_spread_eur",
        )
    ]
    for column in energy_columns:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0.0)
    for column in impact_columns:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0.0)

    summary = add_selected_cover_columns(summary, working)
    summary = apply_summary_sort_order(summary)
    return summary[SUMMARY_EXPORT_COLUMNS]


# ---------------------------------------------------------------------------
# Exporte und Einstiegspunkt
# ---------------------------------------------------------------------------

def export_tables(summary: pd.DataFrame, occurrences: pd.DataFrame) -> None:
    """
    Schreibt Occurrence- und Summary-CSV an ihre festen Zielpfade.

    Eingaben:
    - `summary`: aggregierte 144er-Summary
    - `occurrences`: exportformatierte Occurrence-Tabelle

    Rueckgabe:
    - keine; die Dateien werden direkt gespeichert
    """

    occurrences.to_csv(OCCURRENCES_CSV, sep=";", decimal=",", index=False)
    summary.to_csv(SUMMARY_CSV, sep=";", decimal=",", index=False)


def main() -> None:
    """
    Fuehrt den kompletten Workflow des Skripts aus.

    Eingaben:
    - keine; alle Quellen und Ziele sind als Konstanten definiert

    Rueckgabe:
    - keine; das Skript schreibt Dateien und gibt die wichtigsten Pfade aus

    Ablauf:
    1. Dispatch-Daten laden
    2. Zeitschritte klassifizieren
    3. Kernfaelle filtern
    4. Occurrence-Tabelle aufbereiten
    5. Summary erstellen
    6. Occurrence- und Summary-CSV exportieren
    """

    raw_data = load_dispatch_tables()
    classified = classify_rows(raw_data)
    core_cases = filter_core_cases(classified)
    formatted_occurrences = format_occurrences_for_export(core_cases)
    summary = build_summary(formatted_occurrences)

    export_tables(summary, formatted_occurrences)

    print(f"occurrences_rows={len(formatted_occurrences)}")
    print(f"summary_rows={len(summary)}")
    print(OCCURRENCES_CSV)
    print(SUMMARY_CSV)


if __name__ == "__main__":
    main()
