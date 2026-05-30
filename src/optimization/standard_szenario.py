"""
Builds a rule-based standard dispatch scenario without optimization solver usage.

The script creates a deterministic baseline:
- no lookahead,
- no market-reactive charging/discharging strategy,
- simple physical priority logic per 15-minute step.

Dispatch rule per step:
1) PV covers load first.
2) Remaining PV charges battery.
3) Remaining PV is exported to grid at fixed feed-in tariff.
4) Remaining load is covered by battery discharge.
5) Remaining load is imported from grid at (fixed electricity price + grid fee).
"""

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config_optimization.yaml"
INPUT_DATA_PATH = PROJECT_ROOT / "data" / "optimization" / "optimization_data.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "optimization" / "standard"
OUTPUT_DISPATCH_PATH = OUTPUT_DIR / "optimization_dispatch_standard.csv"
OUTPUT_SUMMARY_PATH = OUTPUT_DIR / "optimization_run_summary_standard.csv"

SCENARIO_NAME = "standard"

DATE_START = pd.Timestamp("2025-07-08")
DATE_END = pd.Timestamp("2025-10-05")

STEP_MINUTES = 15
STEP = pd.Timedelta(minutes=STEP_MINUTES)
DT_H = STEP_MINUTES / 60.0
DISPATCH_STEPS = 96

FEED_IN_TARIFF_CT_PER_KWH = 5.0 
FEED_IN_TARIFF_EUR_PER_MWH = FEED_IN_TARIFF_CT_PER_KWH * 10.0

# Strompreis ohne vpp [EUR/kWh] -> bereinigen von umsatzsteuer und stromsteuer, da diese in den anderen scenario simulationen auch nicht berücksichtigt werden
ELECTRICITY_COST_CT_PER_KWH_RAW = 36.06
ELECTRICITY_TAX = 0.049               # Stromsteuer
VAT = 0.16                          # Umsatzsteuer
ELECTRICITY_COST_CT_PER_KWH = ELECTRICITY_COST_CT_PER_KWH_RAW * (1 - ELECTRICITY_TAX - VAT)
ELECTRICITY_COST_EUR_PER_MWH = ELECTRICITY_COST_CT_PER_KWH * 10.0


COL_DATETIME = "datetime"
COL_LOAD = "Last [W]"
COL_REAL_PV = "Reale Erzeugung [W]"
COL_SOLAR_ELEV = "solar_elevation_deg"
COL_DA_PRICE = "DA Preis [EUR/MWh]"

DISPATCH_COLUMNS = [
    "run_id",
    "dispatch_day",
    "scenario",
    "datetime",
    "input_P_load_W",
    "input_P_pv_forecast_W",
    "input_price_EUR_MWh",
    "opt_P_grid_in_W",
    "opt_P_grid_out_W",
    "opt_P_ch_W",
    "opt_P_dis_W",
    "opt_P_ch_seg1_W",
    "opt_P_ch_seg2_W",
    "opt_P_dis_seg1_W",
    "opt_P_dis_seg2_W",
    "opt_P_curt_W",
    "opt_u_charging",
    "opt_E_start_Wh",
    "opt_E_end_Wh",
    "opt_E_seg1_start_Wh",
    "opt_E_seg2_start_Wh",
    "opt_E_seg1_end_Wh",
    "opt_E_seg2_end_Wh",
    "opt_revenue_step_eur",
    "opt_cycle_aging_cost_step_eur",
    "opt_revenue_step_after_aging_eur",
    "real_P_pv_W",
    "real_P_curt_W",
    "real_P_grid_in_W",
    "real_P_grid_out_W",
    "real_revenue_step_da_eur",
    "real_grid_fee_step_eur",
    "opt_P_grid_net_W",
    "real_P_grid_net_W",
    "real_imbalance_W",
    "real_imbalance_price_eur_mwh",
    "real_imbalance_energy_mwh",
    "real_rebap_settlement_step_eur",
    "real_revenue_step_eur",
    "real_revenue_step_net_eur",
    "real_revenue_step_net_after_aging_eur",
]

SUMMARY_COLUMNS = [
    "run_id",
    "horizon_start",
    "horizon_end_exclusive",
    "dispatch_day",
    "initial_soc_wh",
    "soc_next_publish_wh",
    "objective_horizon_eur",
    "opt_revenue_dispatch_day_eur",
    "opt_cycle_aging_cost_dispatch_day_eur",
    "opt_revenue_dispatch_day_after_aging_eur",
    "solver_status",
    "solver_termination",
    "n_horizon_steps",
    "n_dispatch_steps",
    "terminal_soc_enforced",
    "real_revenue_dispatch_day_eur",
    "real_rebap_settlement_dispatch_day_eur",
    "real_revenue_dispatch_day_net_eur",
    "real_revenue_dispatch_day_net_after_aging_eur",
    "scenario",
]


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Loads optimization configuration from YAML."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_optimization_input(data_path: Path = INPUT_DATA_PATH) -> pd.DataFrame:
    """Loads optimization_data.csv and converts required columns to numeric."""
    df = pd.read_csv(data_path, decimal=",")
    df[COL_DATETIME] = pd.to_datetime(df[COL_DATETIME])

    for col in (COL_LOAD, COL_REAL_PV, COL_SOLAR_ELEV, COL_DA_PRICE):
        df[col] = pd.to_numeric(df[col])

    return df.sort_values(COL_DATETIME).reset_index(drop=True)


def build_clipped_real_pv_series(data_df: pd.DataFrame) -> pd.Series:
    """Builds the real PV series with negative and night-time clipping."""
    pv = data_df[COL_REAL_PV].clip(lower=0.0).copy()
    pv.loc[data_df[COL_SOLAR_ELEV] < 0.0] = 0.0
    pv = pv.clip(lower=0.0)
    return pd.Series(pv.to_numpy(dtype=float), index=data_df[COL_DATETIME], name=COL_REAL_PV)


def _initial_soc_segments(cfg: dict[str, Any]) -> tuple[float, float, float, float, float]:
    """
    Computes initial battery state and segment decomposition.

    Returns:
        initial_soc_wh, e_min_wh, e_max_wh, seg1_init_wh, seg2_init_wh
    """
    bat_cfg = cfg["battery"]
    opt_cfg = cfg["optimization"]

    e_min_wh = float(bat_cfg["e_min_kwh"]) * 1000.0
    e_max_wh = float(bat_cfg["e_max_kwh"]) * 1000.0
    initial_soc_wh = float(opt_cfg["initial_soc_fraction"]) * e_max_wh

    usable_wh = e_max_wh - e_min_wh
    seg2_cap_wh = usable_wh * 0.5
    initial_usable_wh = max(initial_soc_wh - e_min_wh, 0.0)
    seg2_init_wh = min(initial_usable_wh, seg2_cap_wh)
    seg1_init_wh = max(initial_usable_wh - seg2_cap_wh, 0.0)

    return initial_soc_wh, e_min_wh, e_max_wh, seg1_init_wh, seg2_init_wh


def simulate_standard_dispatch(cfg: dict[str, Any], input_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulates the standard scenario day by day without optimization.

    The simulation uses deterministic local rules and writes output fields
    aligned with the other scenario dispatch/summary CSV structures.
    """
    opt_cfg = cfg["optimization"]
    bat_cfg = cfg["battery"]

    p_ch_max_w = float(bat_cfg["p_ch_max_kw"]) * 1000.0
    p_dis_max_w = float(bat_cfg["p_dis_max_kw"]) * 1000.0
    eta_ch = float(bat_cfg["eta_ch"])
    eta_dis = float(bat_cfg["eta_dis"])
    omega_1 = float(opt_cfg["cycle_aging_omega_1_eur_per_mwh"])
    omega_2 = float(opt_cfg["cycle_aging_omega_2_eur_per_mwh"])
    grid_fee = float(opt_cfg["grid_fee_eur_per_mwh"])

    initial_soc_wh, e_min_wh, e_max_wh, e_seg1_wh, e_seg2_wh = _initial_soc_segments(cfg)

    usable_wh = e_max_wh - e_min_wh
    seg1_cap_wh = usable_wh * 0.5
    seg2_cap_wh = usable_wh - seg1_cap_wh

    real_pv_series = build_clipped_real_pv_series(input_df)
    indexed = input_df.set_index(COL_DATETIME).sort_index()
    dispatch_days = pd.date_range(start=DATE_START, end=DATE_END, freq="D")

    dispatch_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    current_soc_wh = initial_soc_wh

    for run_id, dispatch_day in enumerate(dispatch_days):
        day_start = pd.Timestamp(dispatch_day).normalize()
        day_index = pd.date_range(day_start, periods=DISPATCH_STEPS, freq=STEP)

        load_values = indexed.loc[day_index, COL_LOAD].to_numpy(dtype=float)
        price_values = indexed.loc[day_index, COL_DA_PRICE].to_numpy(dtype=float)
        pv_values = real_pv_series.reindex(day_index).to_numpy(dtype=float)

        day_rows: list[dict[str, Any]] = []
        day_initial_soc = current_soc_wh

        for t in range(DISPATCH_STEPS):
            ts = day_index[t]
            load_w = max(0.0, float(load_values[t]))
            pv_w = max(0.0, float(pv_values[t]))
            _da_price = float(price_values[t])

            e_start_wh = e_min_wh + e_seg1_wh + e_seg2_wh

            # PV -> load
            p_pv_to_load_w = min(pv_w, load_w)
            remaining_pv_w = pv_w - p_pv_to_load_w
            remaining_load_w = load_w - p_pv_to_load_w

            # remaining PV -> battery
            headroom_wh = max(0.0, e_max_wh - e_start_wh)
            p_ch_limit_by_energy_w = headroom_wh / (eta_ch * DT_H) if eta_ch > 0.0 else 0.0
            p_ch_w = min(remaining_pv_w, p_ch_max_w, p_ch_limit_by_energy_w)
            remaining_pv_w -= p_ch_w

            # remaining PV -> grid export
            p_grid_out_w = max(0.0, remaining_pv_w)

            # remaining load -> battery
            available_wh = max(0.0, e_start_wh - e_min_wh)
            p_dis_limit_by_energy_w = (available_wh * eta_dis / DT_H) if eta_dis > 0.0 else 0.0
            p_dis_w = min(remaining_load_w, p_dis_max_w, p_dis_limit_by_energy_w)
            remaining_load_w -= p_dis_w

            # remaining load -> grid import
            p_grid_in_w = max(0.0, remaining_load_w)

            # Segment-based charge split: segment2 first, then segment1
            charge_wh = eta_ch * p_ch_w * DT_H
            seg2_headroom_wh = max(0.0, seg2_cap_wh - e_seg2_wh)
            ch_seg2_wh = min(charge_wh, seg2_headroom_wh)
            ch_seg1_wh = max(charge_wh - ch_seg2_wh, 0.0)

            p_ch_seg2_w = ch_seg2_wh / (eta_ch * DT_H) if eta_ch > 0.0 else 0.0
            p_ch_seg1_w = ch_seg1_wh / (eta_ch * DT_H) if eta_ch > 0.0 else 0.0

            # Segment-based discharge split: segment1 first, then segment2
            dis_wh = (p_dis_w / eta_dis) * DT_H if eta_dis > 0.0 else 0.0
            dis_seg1_wh = min(dis_wh, e_seg1_wh)
            dis_seg2_wh = max(dis_wh - dis_seg1_wh, 0.0)

            p_dis_seg1_w = dis_seg1_wh * eta_dis / DT_H if DT_H > 0.0 else 0.0
            p_dis_seg2_w = dis_seg2_wh * eta_dis / DT_H if DT_H > 0.0 else 0.0

            # Update segment energies
            e_seg1_end_wh = min(max(e_seg1_wh + ch_seg1_wh - dis_seg1_wh, 0.0), seg1_cap_wh)
            e_seg2_end_wh = min(max(e_seg2_wh + ch_seg2_wh - dis_seg2_wh, 0.0), seg2_cap_wh)
            e_end_wh = e_min_wh + e_seg1_end_wh + e_seg2_end_wh

            import_price = ELECTRICITY_COST_EUR_PER_MWH + grid_fee
            opt_revenue_step_eur = (
                (FEED_IN_TARIFF_EUR_PER_MWH * p_grid_out_w) - (import_price * p_grid_in_w)
            ) * DT_H / 1e6
            real_revenue_step_da_eur = (
                (FEED_IN_TARIFF_EUR_PER_MWH * p_grid_out_w)
                - (ELECTRICITY_COST_EUR_PER_MWH * p_grid_in_w)
            ) * DT_H / 1e6
            real_grid_fee_step_eur = grid_fee * p_grid_in_w * DT_H / 1e6
            opt_cycle_aging_cost_step_eur = (
                (omega_1 * p_dis_seg1_w) + (omega_2 * p_dis_seg2_w)
            ) * DT_H / 1e6
            opt_revenue_step_after_aging_eur = opt_revenue_step_eur - opt_cycle_aging_cost_step_eur

            row = {
                "run_id": run_id,
                "dispatch_day": day_start,
                "scenario": SCENARIO_NAME,
                "datetime": ts,
                "input_P_load_W": load_w,
                "input_P_pv_forecast_W": pv_w,
                "input_price_EUR_MWh": ELECTRICITY_COST_EUR_PER_MWH,
                "opt_P_grid_in_W": p_grid_in_w,
                "opt_P_grid_out_W": p_grid_out_w,
                "opt_P_ch_W": p_ch_w,
                "opt_P_dis_W": p_dis_w,
                "opt_P_ch_seg1_W": p_ch_seg1_w,
                "opt_P_ch_seg2_W": p_ch_seg2_w,
                "opt_P_dis_seg1_W": p_dis_seg1_w,
                "opt_P_dis_seg2_W": p_dis_seg2_w,
                "opt_P_curt_W": 0.0,
                "opt_u_charging": int(p_ch_w > 0.0),
                "opt_E_start_Wh": e_start_wh,
                "opt_E_end_Wh": e_end_wh,
                "opt_E_seg1_start_Wh": e_seg1_wh,
                "opt_E_seg2_start_Wh": e_seg2_wh,
                "opt_E_seg1_end_Wh": e_seg1_end_wh,
                "opt_E_seg2_end_Wh": e_seg2_end_wh,
                "opt_revenue_step_eur": opt_revenue_step_eur,
                "opt_cycle_aging_cost_step_eur": opt_cycle_aging_cost_step_eur,
                "opt_revenue_step_after_aging_eur": opt_revenue_step_after_aging_eur,
                "real_P_pv_W": pv_w,
                "real_P_curt_W": 0.0,
                "real_P_grid_in_W": p_grid_in_w,
                "real_P_grid_out_W": p_grid_out_w,
                "real_revenue_step_da_eur": real_revenue_step_da_eur,
                "real_grid_fee_step_eur": real_grid_fee_step_eur,
                "opt_P_grid_net_W": p_grid_in_w - p_grid_out_w,
                "real_P_grid_net_W": p_grid_in_w - p_grid_out_w,
                "real_imbalance_W": 0.0,
                "real_imbalance_price_eur_mwh": 0.0,
                "real_imbalance_energy_mwh": 0.0,
                "real_rebap_settlement_step_eur": 0.0,
                "real_revenue_step_eur": opt_revenue_step_eur,
                "real_revenue_step_net_eur": opt_revenue_step_eur,
                "real_revenue_step_net_after_aging_eur": opt_revenue_step_after_aging_eur,
            }
            day_rows.append(row)

            e_seg1_wh = e_seg1_end_wh
            e_seg2_wh = e_seg2_end_wh
            current_soc_wh = e_end_wh

        dispatch_rows.extend(day_rows)
        day_df = pd.DataFrame(day_rows)

        summary_rows.append(
            {
                "run_id": run_id,
                "horizon_start": day_start,
                "horizon_end_exclusive": day_start + DISPATCH_STEPS * STEP,
                "dispatch_day": day_start,
                "initial_soc_wh": float(day_initial_soc),
                "soc_next_publish_wh": float(day_df.iloc[-1]["opt_E_end_Wh"]),
                "objective_horizon_eur": float(day_df["opt_revenue_step_after_aging_eur"].sum()),
                "opt_revenue_dispatch_day_eur": float(day_df["opt_revenue_step_eur"].sum()),
                "opt_cycle_aging_cost_dispatch_day_eur": float(day_df["opt_cycle_aging_cost_step_eur"].sum()),
                "opt_revenue_dispatch_day_after_aging_eur": float(day_df["opt_revenue_step_after_aging_eur"].sum()),
                "solver_status": "rule_based",
                "solver_termination": "no_optimization",
                "n_horizon_steps": DISPATCH_STEPS,
                "n_dispatch_steps": DISPATCH_STEPS,
                "terminal_soc_enforced": False,
                "real_revenue_dispatch_day_eur": float(day_df["real_revenue_step_eur"].sum()),
                "real_rebap_settlement_dispatch_day_eur": 0.0,
                "real_revenue_dispatch_day_net_eur": float(day_df["real_revenue_step_net_eur"].sum()),
                "real_revenue_dispatch_day_net_after_aging_eur": float(
                    day_df["real_revenue_step_net_after_aging_eur"].sum()
                ),
                "scenario": SCENARIO_NAME,
            }
        )

    dispatch_df = pd.DataFrame(dispatch_rows).reindex(columns=DISPATCH_COLUMNS)
    summary_df = pd.DataFrame(summary_rows).reindex(columns=SUMMARY_COLUMNS)
    return dispatch_df, summary_df


def write_outputs(
    dispatch_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    dispatch_path: Path = OUTPUT_DISPATCH_PATH,
    summary_path: Path = OUTPUT_SUMMARY_PATH,
) -> tuple[Path, Path]:
    """Writes scenario outputs to CSV files with comma decimal separator."""
    dispatch_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    dispatch_df.to_csv(dispatch_path, index=False, decimal=",")
    summary_df.to_csv(summary_path, index=False, decimal=",")

    return dispatch_path, summary_path


def main() -> None:
    """Runs the complete rule-based standard scenario simulation."""
    cfg = load_config()
    input_df = load_optimization_input()
    dispatch_df, summary_df = simulate_standard_dispatch(cfg=cfg, input_df=input_df)
    dispatch_path, summary_path = write_outputs(dispatch_df=dispatch_df, summary_df=summary_df)

    print(f"[standard] dispatch={dispatch_path} ({len(dispatch_df)} rows)")
    print(f"[standard] summary={summary_path} ({len(summary_df)} rows)")


if __name__ == "__main__":
    main()
