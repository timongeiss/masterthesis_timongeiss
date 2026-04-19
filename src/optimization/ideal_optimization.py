"""
Builds ideal rolling optimization CSV outputs without defensive fallback logic.

Workflow:
1) Load config.
2) Load optimization input.
3) Build ideal PV from real generation (night clipping).
4) Run daily 35h optimization windows.
5) Keep only first 24h dispatch per run.
6) Apply settlement post-calculation with one reBAP series.
7) Write dispatch and summary CSV files.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from optimization import (
    aggregate_run_realized_metrics,
    compute_dispatch_settlement,
    solve_milp_for_horizon,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config_optimization.yaml"
INPUT_DATA_PATH = PROJECT_ROOT / "data" / "optimization" / "optimization_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "optimization" / "ideal"
OUTPUT_DISPATCH_PATH = OUTPUT_DIR / "optimization_dispatch_ideal.csv"
OUTPUT_SUMMARY_PATH = OUTPUT_DIR / "optimization_run_summary_ideal.csv"

SCENARIO_NAME = "ideal"

DATE_START = pd.Timestamp("2025-07-08")
DATE_END = pd.Timestamp("2025-10-05")

STEP_MINUTES = 15
STEP = pd.Timedelta(minutes=STEP_MINUTES)
HORIZON_STEPS = 140
DISPATCH_STEPS = 96

COL_DATETIME = "datetime"
COL_LOAD = "Last [W]"
COL_REAL_PV = "Reale Erzeugung [W]"
COL_SOLAR_ELEV = "solar_elevation_deg"
COL_DA_PRICE = "DA Preis [EUR/MWh]"
COL_REBAP_UNDER = "reBAP unterdeckt [EUR/MWh]"

MODEL_COL_PV = "P_pv_forecast_W"
SETTLEMENT_REBAP_COL = "reBAP_unterdeckt_eur_mwh"


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """
    Loads the optimization YAML configuration.
    """
    
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_optimization_input(data_path: Path = INPUT_DATA_PATH) -> pd.DataFrame:
    """
    Loads `data/optimization/optimization_data.csv` and performs only the
    minimal conversions needed by the optimizer.
    """
    
    df = pd.read_csv(data_path, decimal=",")

    df[COL_DATETIME] = pd.to_datetime(df[COL_DATETIME])
    for col in (COL_LOAD, COL_REAL_PV, COL_SOLAR_ELEV, COL_DA_PRICE, COL_REBAP_UNDER):
        df[col] = pd.to_numeric(df[col])

    return df.sort_values(COL_DATETIME).reset_index(drop=True)


def build_ideal_pv_series(data_df: pd.DataFrame) -> pd.Series:
    """
    Builds the ideal PV series from real generation values.

    Rules:
    - Start from `Reale Erzeugung [W]`.
    - Set all values to 0 whenever `solar_elevation_deg < 0`.
    - Clip any remaining negative values to 0.

    Args:
        data_df: Input dataframe.

    Returns:
        Datetime-indexed ideal PV series in W.
    """
    
    pv = data_df[COL_REAL_PV].clip(lower=0.0).copy()
    pv.loc[data_df[COL_SOLAR_ELEV] < 0.0] = 0.0
    pv = pv.clip(lower=0.0)
    return pd.Series(pv.to_numpy(dtype=float), index=data_df[COL_DATETIME], name=COL_REAL_PV)


def build_rebap_series(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the single reBAP series used for settlement.

    Mapping:
    - input: `reBAP unterdeckt [EUR/MWh]`
    - output column: `reBAP_unterdeckt_eur_mwh`

    Args:
        data_df: Input dataframe.

    Returns:
        Datetime-indexed dataframe with one settlement column.
    """
    
    return pd.DataFrame(
        {SETTLEMENT_REBAP_COL: data_df[COL_REBAP_UNDER].to_numpy(dtype=float)},
        index=data_df[COL_DATETIME],
    )


def build_model_input(data_df: pd.DataFrame, ideal_pv: pd.Series) -> pd.DataFrame:
    """
    Builds the exact optimizer input table.

    Args:
        data_df: Input dataframe.
        ideal_pv: Datetime-indexed ideal PV series.

    Returns:
        DataFrame with model-required columns.
    """
    
    return pd.DataFrame(
        {
            COL_DATETIME: data_df[COL_DATETIME].to_numpy(),
            COL_LOAD: data_df[COL_LOAD].to_numpy(dtype=float),
            MODEL_COL_PV: ideal_pv.to_numpy(dtype=float),
            COL_DA_PRICE: data_df[COL_DA_PRICE].to_numpy(dtype=float),
        }
    )


def _initial_soc_from_config(cfg: dict[str, Any]) -> float:
    """
    Computes initial state of charge in Wh from configuration.

    Args:
        cfg: Configuration dictionary.

    Returns:
        Initial SoC in Wh.
    """
    
    return float(cfg["optimization"]["initial_soc_fraction"]) * float(cfg["battery"]["e_max_kwh"]) * 1000.0


def build_summary_row(
    run_id: int,
    dispatch_day: pd.Timestamp,
    horizon_start: pd.Timestamp,
    current_soc_wh: float,
    solver_result: dict[str, Any],
    settled_dispatch: pd.DataFrame,
) -> dict[str, Any]:
    """
    Builds one summary row for one daily run.

    The objective value is from the full 35h horizon. Day KPIs are aggregated
    only over the fixed 24h dispatch block.

    Args:
        run_id: Sequential run id.
        dispatch_day: Fixed dispatch day.
        horizon_start: Horizon start timestamp.
        current_soc_wh: SoC at run start.
        solver_result: Result dict returned by optimizer.
        settled_dispatch: Settled 24h dispatch table for this run.

    Returns:
        Dictionary representing one summary CSV row.
    """
    
    realized_row = aggregate_run_realized_metrics(settled_dispatch).iloc[0]

    return {
        "run_id": run_id,
        "horizon_start": horizon_start,
        "horizon_end_exclusive": horizon_start + HORIZON_STEPS * STEP,
        "dispatch_day": dispatch_day,
        "initial_soc_wh": float(current_soc_wh),
        "soc_next_publish_wh": float(settled_dispatch.iloc[-1]["opt_E_end_Wh"]),
        "objective_horizon_eur": float(solver_result["objective_eur"]),
        "opt_revenue_dispatch_day_eur": float(settled_dispatch["opt_revenue_step_eur"].sum()),
        "opt_cycle_aging_cost_dispatch_day_eur": float(settled_dispatch["opt_cycle_aging_cost_step_eur"].sum()),
        "opt_revenue_dispatch_day_after_aging_eur": float(
            settled_dispatch["opt_revenue_step_after_aging_eur"].sum()
        ),
        "solver_status": str(solver_result["status"]),
        "solver_termination": str(solver_result["termination"]),
        "n_horizon_steps": HORIZON_STEPS,
        "n_dispatch_steps": DISPATCH_STEPS,
        "terminal_soc_enforced": False,
        "real_revenue_dispatch_day_eur": float(realized_row["real_revenue_dispatch_day_eur"]),
        "real_rebap_settlement_dispatch_day_eur": float(realized_row["real_rebap_settlement_dispatch_day_eur"]),
        "real_revenue_dispatch_day_net_eur": float(realized_row["real_revenue_dispatch_day_net_eur"]),
        "scenario": SCENARIO_NAME,
    }


def run_ideal_rolling_optimization(
    cfg: dict[str, Any],
    input_df: pd.DataFrame,
    start_day: pd.Timestamp = DATE_START,
    end_day: pd.Timestamp = DATE_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs the full ideal rolling optimization loop.

    Per day:
    - optimize 35h window,
    - keep first 24h fixed dispatch,
    - compute settlement,
    - carry final fixed-step SoC into next day.

    Args:
        cfg: Configuration dictionary.
        input_df: Optimization input data.
        start_day: First dispatch day.
        end_day: Last dispatch day.

    Returns:
        Tuple `(dispatch_df, summary_df)`.
    """
    
    ideal_pv = build_ideal_pv_series(input_df)
    rebap_prices = build_rebap_series(input_df)
    model_input = build_model_input(input_df, ideal_pv)

    indexed_model_input = model_input.set_index(COL_DATETIME).sort_index()
    dispatch_days = pd.date_range(start=start_day, end=end_day, freq="D")

    dispatch_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    current_soc_wh = _initial_soc_from_config(cfg)

    for run_id, dispatch_day in enumerate(dispatch_days):
        dispatch_day = pd.Timestamp(dispatch_day).normalize()
        horizon_index = pd.date_range(start=dispatch_day, periods=HORIZON_STEPS, freq=STEP)
        horizon_df = indexed_model_input.loc[horizon_index].reset_index().rename(columns={"index": COL_DATETIME})

        solver_result = solve_milp_for_horizon(
            horizon_df=horizon_df,
            cfg=cfg,
            initial_soc_wh=current_soc_wh,
            enforce_terminal_soc=False,
        )

        fixed_dispatch = solver_result["df"].iloc[:DISPATCH_STEPS].copy().reset_index(drop=True)
        settled_dispatch = compute_dispatch_settlement(
            dispatch_df=fixed_dispatch,
            real_pv_series=ideal_pv,
            cfg=cfg,
            rebap_prices=rebap_prices,
        )

        settled_dispatch["run_id"] = run_id
        settled_dispatch["dispatch_day"] = dispatch_day
        settled_dispatch["scenario"] = SCENARIO_NAME

        metadata_cols = ["run_id", "dispatch_day", "scenario"]
        settled_dispatch = settled_dispatch[metadata_cols + [c for c in settled_dispatch.columns if c not in metadata_cols]]

        summary_rows.append(
            build_summary_row(
                run_id=run_id,
                dispatch_day=dispatch_day,
                horizon_start=dispatch_day,
                current_soc_wh=current_soc_wh,
                solver_result=solver_result,
                settled_dispatch=settled_dispatch,
            )
        )
        dispatch_frames.append(settled_dispatch)

        current_soc_wh = float(settled_dispatch.iloc[-1]["opt_E_end_Wh"])

    return pd.concat(dispatch_frames, ignore_index=True), pd.DataFrame(summary_rows)


def write_outputs(
    dispatch_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    dispatch_path: Path = OUTPUT_DISPATCH_PATH,
    summary_path: Path = OUTPUT_SUMMARY_PATH,
) -> tuple[Path, Path]:
    """
    Writes both output CSV files with comma-decimal formatting.

    Args:
        dispatch_df: Dispatch output table.
        summary_df: Run summary output table.
        dispatch_path: Target path for dispatch CSV.
        summary_path: Target path for summary CSV.

    Returns:
        Written file paths.
    """
    
    dispatch_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    dispatch_df.to_csv(dispatch_path, index=False, decimal=",")
    summary_df.to_csv(summary_path, index=False, decimal=",")

    return dispatch_path, summary_path


def main() -> None:
    """
    Runs the complete ideal optimization script end-to-end.
    """
    
    cfg = load_config()
    input_df = load_optimization_input()
    dispatch_df, summary_df = run_ideal_rolling_optimization(cfg=cfg, input_df=input_df)
    dispatch_path, summary_path = write_outputs(dispatch_df=dispatch_df, summary_df=summary_df)

    print(f"Saved dispatch CSV: {dispatch_path} ({len(dispatch_df)} rows)")
    print(f"Saved summary CSV: {summary_path} ({len(summary_df)} rows)")


if __name__ == "__main__":
    main()
