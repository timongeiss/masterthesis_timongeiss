"""
Export scenario-wise variable sums for days 2..91 (2025-07-08 .. 2025-10-05).

Output:
- data/optimization/scenario_variable_sums_days_2_91.csv

The output table contains one row per variable from the provided post-calculated
and decision-variable tables (including rows for variables that are not stored in
dispatch CSVs). Scenario columns hold summed values over the configured period.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPT_DIR = PROJECT_ROOT / "data" / "optimization"
OUT_CSV = OPT_DIR / "scenario_variable_sums_days_2_91.csv"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config_optimization.yaml"

FIXED_DT_HOURS = 0.25

SCENARIOS = ("ideal", "physical", "source", "target", "transfer", "standard")
START_TS = pd.Timestamp("2025-07-08 00:00:00")
END_TS = pd.Timestamp("2025-10-05 23:45:00")
SCENARIO_LABELS = {
    "ideal": "Ideal",
    "physical": "Physical",
    "source": "Source",
    "target": "Target",
    "transfer": "Transfer",
    "standard": "Standard",
}
SCENARIO_COLORS = {
    "ideal": "black",
    "physical": "tab:orange",
    "source": "tab:green",
    "target": "tab:red",
    "transfer": "tab:blue",
    "standard": "tab:gray",
}


@dataclass(frozen=True)
class VariableSpec:
    """Configuration record for one exported variable."""

    table_group: str
    latex_symbol: str
    python_variable: str
    original_unit: str
    aggregation_mode: str
    sum_unit: str
    source_column: str | None = None
    reference_column: str | None = None


VARIABLE_SPECS: tuple[VariableSpec, ...] = (
    # Input variables
    VariableSpec("input", r"P_t^{PV}", "input_P_pv_forecast_W", "W", "integrate_w_to_mwh", "MWh"),
    VariableSpec("input", r"E_t^{PV,over}", "forecast_over_mwh", "W", "integrate_positive_diff_w_to_mwh", "MWh", source_column="input_P_pv_forecast_W", reference_column="real_P_pv_W"),
    VariableSpec("input", r"E_t^{PV,under}", "forecast_under_mwh", "W", "integrate_negative_diff_w_to_mwh", "MWh", source_column="input_P_pv_forecast_W", reference_column="real_P_pv_W"),
    # Decision variables
    VariableSpec("decision", r"P_t^{*grid,in}", "opt_P_grid_in_W", "W", "integrate_w_to_mwh", "MWh"),
    VariableSpec("decision", r"P_t^{*grid,out}", "opt_P_grid_out_W", "W", "integrate_w_to_mwh", "MWh"),
    VariableSpec("decision", r"P_t^{*ch}", "opt_P_ch_W", "W", "integrate_w_to_mwh", "MWh"),
    VariableSpec("decision", r"P_t^{*dis}", "opt_P_dis_W", "W", "integrate_w_to_mwh", "MWh"),
    VariableSpec("decision", r"P_t^{*curt}", "opt_P_curt_W", "W", "integrate_w_to_mwh", "MWh"),
    VariableSpec("decision", r"E_t^{*}", "opt_E_end_Wh", "Wh", "sum_raw", "Wh"),
    VariableSpec("decision", r"P_t^{*ch,1}", "opt_P_ch_seg1_W", "W", "integrate_w_to_mwh", "MWh"),
    VariableSpec("decision", r"P_t^{*ch,2}", "opt_P_ch_seg2_W", "W", "integrate_w_to_mwh", "MWh"),
    VariableSpec("decision", r"P_t^{*dis,1}", "opt_P_dis_seg1_W", "W", "integrate_w_to_mwh", "MWh"),
    VariableSpec("decision", r"P_t^{*dis,2}", "opt_P_dis_seg2_W", "W", "integrate_w_to_mwh", "MWh"),
    VariableSpec("decision", r"E_t^{*1}", "opt_E_seg1_end_Wh", "Wh", "sum_raw", "Wh"),
    VariableSpec("decision", r"E_t^{*2}", "opt_E_seg2_end_Wh", "Wh", "sum_raw", "Wh"),
    # Post-calculated variables
    VariableSpec("post_calc", r"R_t^{*market}", "opt_revenue_step_eur", "EUR", "sum_raw", "EUR"),
    VariableSpec("post_calc", r"C_t^{*aging}", "opt_cycle_aging_cost_step_eur", "EUR", "sum_raw", "EUR"),
    VariableSpec("post_calc", r"R_t^{*net}", "opt_revenue_step_after_aging_eur", "EUR", "sum_raw", "EUR"),
    VariableSpec("post_calc", r"P_t^{*grid,net}", "opt_P_grid_net_W", "W", "integrate_w_to_mwh", "MWh"),
    VariableSpec("post_calc", r"P_t^{real,grid,in}", "real_P_grid_in_W", "W", "integrate_w_to_mwh", "MWh"),
    VariableSpec("post_calc", r"P_t^{real,grid,out}", "real_P_grid_out_W", "W", "integrate_w_to_mwh", "MWh"),
    VariableSpec("post_calc", r"R_t^{real,DA}", "real_revenue_step_da_eur", "EUR", "sum_raw", "EUR"),
    VariableSpec("post_calc", r"C_t^{real,grid}", "real_grid_fee_step_eur", "EUR", "sum_raw", "EUR"),
    VariableSpec("post_calc", r"C_t^{real,aging}", "real_cycle_aging_cost_step_eur", "EUR", "sum_raw", "EUR"),
    VariableSpec("post_calc", r"P_t^{real,grid,net}", "real_P_grid_net_W", "W", "integrate_w_to_mwh", "MWh"),
    VariableSpec("post_calc", r"P_t^{real,curt}", "real_P_curt_W", "W", "integrate_w_to_mwh", "MWh"),
    VariableSpec("post_calc", r"P_t^{real,imb}", "real_imbalance_W", "W", "integrate_w_to_mwh", "MWh"),
    VariableSpec("post_calc", r"E_t^{real,imb}", "real_imbalance_energy_mwh", "MWh", "sum_raw", "MWh"),
    VariableSpec("post_calc", r"R_t^{real,imb,+}", "real_rebap_revenue_step_eur", "EUR", "sum_positive_raw", "EUR", source_column="real_rebap_settlement_step_eur"),
    VariableSpec("post_calc", r"R_t^{real,imb,-}", "real_rebap_penalty_step_eur", "EUR", "sum_negative_raw", "EUR", source_column="real_rebap_settlement_step_eur"),
    VariableSpec("post_calc", r"R_t^{real,imb}", "real_rebap_settlement_step_eur", "EUR", "sum_raw", "EUR"),
    VariableSpec("post_calc", r"R_t^{real}", "real_revenue_step_eur", "EUR", "sum_raw", "EUR"),
    VariableSpec("post_calc", r"R_t^{real,net}", "real_revenue_step_net_eur", "EUR", "sum_raw", "EUR"),
    VariableSpec("post_calc", r"R_t^{real,net,aging}", "real_revenue_step_net_after_aging_eur", "EUR", "sum_raw", "EUR"),
)


def dispatch_path_for_scenario(scenario: str) -> Path:
    """Build the expected dispatch CSV path for a scenario."""
    return OPT_DIR / scenario / f"optimization_dispatch_{scenario}.csv"


def load_grid_fee_eur_per_mwh(config_path: Path = CONFIG_PATH) -> float:
    """Load the configured grid fee used for fallback calculations."""

    with config_path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    return float(cfg["optimization"]["grid_fee_eur_per_mwh"])

def load_filtered_dispatch(scenario: str) -> pd.DataFrame:
    """Load and time-filter one scenario dispatch file."""
    # laden der Szenario-Datei
    path = dispatch_path_for_scenario(scenario)
    if not path.exists():
        raise FileNotFoundError(f"Dispatch CSV not found for '{scenario}': {path}")

    # einlesen
    df = pd.read_csv(path, decimal=",")
    if "datetime" not in df.columns:
        raise ValueError(f"Missing 'datetime' column in {path}")

    # Datum umwandeln
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    # filtern nach Zeitraum Tage 2..91
    mask = (df["datetime"] >= START_TS) & (df["datetime"] <= END_TS)
    return df.loc[mask].copy().reset_index(drop=True)


def aggregate_series(
    values: pd.Series,
    aggregation_mode: str,
    reference_values: pd.Series | None = None,
) -> float:
    """Aggregate a numeric series according to a configured rule."""
    # in numerische Werte umwandeln, damit ungueltige Eintraege zu NaN werden
    numeric = pd.to_numeric(values, errors="coerce")
    # rohe Summe bilden, NaN wird ignoriert; min_count=1 vermeidet ein 0.0 bei komplett leeren Reihen
    raw_sum = numeric.sum(min_count=1)
    if pd.isna(raw_sum):
        return float("nan")

    # auf float normalisieren fuer die nachfolgenden Aggregationspfade
    raw_sum = float(raw_sum)
    # rohe Summe direkt zurueckgeben
    if aggregation_mode == "sum_raw":
        return raw_sum
    # nur positive Settlement-/Revenue-Anteile aufsummieren
    if aggregation_mode == "sum_positive_raw":
        return float(numeric.clip(lower=0.0).sum(min_count=1))
    # nur negative Settlement-/Penalty-Anteile aufsummieren
    if aggregation_mode == "sum_negative_raw":
        return float(numeric.clip(upper=0.0).sum(min_count=1))
    # Leistung in Energie umrechnen
    if aggregation_mode == "integrate_w_to_mwh":
        return raw_sum * FIXED_DT_HOURS / 1e6

    # fuer differenzbasierte Modi werden Referenzwerte benoetigt
    if aggregation_mode in {"integrate_positive_diff_w_to_mwh", "integrate_negative_diff_w_to_mwh"}:
        if reference_values is None:
            return float("nan")

        # Quell- und Referenzwerte als numerische Arrays aufbereiten
        source = numeric.to_numpy(dtype=float)
        reference = pd.to_numeric(reference_values, errors="coerce").to_numpy(dtype=float)
        # nur gueltige Wertepaarungen fuer die Differenzbildung verwenden
        finite = np.isfinite(source) & np.isfinite(reference)
        if not np.any(finite):
            return float("nan")

        # Differenz bilden und je nach Modus nur positive oder negative Abweichungen integrieren
        diff_w = source[finite] - reference[finite]
        if aggregation_mode == "integrate_positive_diff_w_to_mwh":
            return float(np.maximum(diff_w, 0.0).sum() * FIXED_DT_HOURS / 1e6)
        return float(np.maximum(-diff_w, 0.0).sum() * FIXED_DT_HOURS / 1e6)

    raise ValueError(f"Unsupported aggregation mode: {aggregation_mode}")


def build_real_grid_fee_fallback(df: pd.DataFrame, grid_fee_eur_per_mwh: float) -> pd.Series | None:
    """Rebuild real grid fee from real imports when the explicit column is missing."""

    if "real_P_grid_in_W" not in df.columns:
        return None
    real_grid_in_w = pd.to_numeric(df["real_P_grid_in_W"], errors="coerce")
    return real_grid_in_w * grid_fee_eur_per_mwh * FIXED_DT_HOURS / 1e6


def build_real_aging_fallback(df: pd.DataFrame) -> pd.Series | None:
    """Fallback to planned aging when no real MPC aging is available."""

    if "opt_cycle_aging_cost_step_eur" not in df.columns:
        return None
    return pd.to_numeric(df["opt_cycle_aging_cost_step_eur"], errors="coerce")

def build_sum_table() -> pd.DataFrame:
    """Build the scenario-wise aggregation table for all configured variables."""
    rows: list[dict[str, object]] = []
    grid_fee_eur_per_mwh = load_grid_fee_eur_per_mwh()

    # alle benoetigten Dispatch-Tabellen einmal laden und cachen
    dispatch_by_scenario: dict[str, pd.DataFrame] = {}
    for scenario in SCENARIOS:
        dispatch_by_scenario[scenario] = load_filtered_dispatch(scenario)

    # fuer jede konfigurierte Variable eine Tabellenzeile aufbauen
    for spec in VARIABLE_SPECS:
        row: dict[str, object] = {
            "table_group": spec.table_group,
            "latex_symbol": spec.latex_symbol,
            "python_variable": spec.python_variable,
            "original_unit": spec.original_unit,
            "aggregation_mode": spec.aggregation_mode,
            "sum_unit": spec.sum_unit,
        }

        for scenario in SCENARIOS:
            df = dispatch_by_scenario[scenario]
            # Standardfall: Quelle ist die gleichnamige Spalte wie die Python-Variable
            source_column = spec.source_column if spec.source_column is not None else spec.python_variable
            if source_column not in df.columns:
                if spec.python_variable == "real_grid_fee_step_eur":
                    fallback_values = build_real_grid_fee_fallback(df, grid_fee_eur_per_mwh)
                    if fallback_values is not None:
                        row[scenario] = aggregate_series(
                            values=fallback_values,
                            aggregation_mode=spec.aggregation_mode,
                        )
                        continue
                if spec.python_variable == "real_cycle_aging_cost_step_eur":
                    fallback_values = build_real_aging_fallback(df)
                    if fallback_values is not None:
                        row[scenario] = aggregate_series(
                            values=fallback_values,
                            aggregation_mode=spec.aggregation_mode,
                        )
                        continue
                row[scenario] = np.nan
                continue

            reference_values: pd.Series | None = None
            if spec.reference_column is not None:
                # fuer diff-basierte Aggregationen die Referenzspalte bereitstellen
                if spec.reference_column not in df.columns:
                    row[scenario] = np.nan
                    continue
                reference_values = df[spec.reference_column]

            # Szenariowert gemaess Aggregationsregel berechnen
            row[scenario] = aggregate_series(
                values=df[source_column],
                aggregation_mode=spec.aggregation_mode,
                reference_values=reference_values,
            )

        rows.append(row)

    # finale Tabelle mit stabiler Spaltenreihenfolge zurueckgeben
    out_df = pd.DataFrame(rows)
    ordered_cols = [
        "table_group",
        "latex_symbol",
        "python_variable",
        "original_unit",
        "aggregation_mode",
        "sum_unit",
        *SCENARIOS,
    ]
    return out_df.reindex(columns=ordered_cols)


def main() -> None:
    """Build and save the scenario summary CSV for days 2..91."""
    # Summen-Tabelle erzeugen
    out_df = build_sum_table()
    # Zielordner sicherstellen und CSV im deutschen Zahlenformat schreiben
    OPT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False, sep=";", decimal=",", na_rep="NA")
    print(f"Saved: {OUT_CSV}")
    print(f"Rows: {len(out_df)}")
    print(f"Columns: {len(out_df.columns)}")


if __name__ == "__main__":
    main()
