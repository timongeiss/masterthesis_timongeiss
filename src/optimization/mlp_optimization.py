"""
Builds rolling optimization CSV outputs for MLP scenarios: source, target, transfer.

Workflow per scenario:
1) Load forecast files from a fixed relative path list.
2) Build forecast PV from y_pred (scaled to installed PV power).
3) Clip negative forecast values and night-time values to zero.
4) Run rolling optimization with available horizon length per day.
5) Fix first 96 steps per run and apply settlement with real PV.
6) Save dispatch and summary CSV files for the scenario.
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
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
OPTIMIZATION_DIR = PROJECT_ROOT / "data" / "optimization"

MLP_SCENARIOS = ("source", "target", "transfer")
PV_INSTALLED_W = 9720.0

DATE_START = pd.Timestamp("2025-07-08")
DATE_END = pd.Timestamp("2025-10-05")

STEP_MINUTES = 15
STEP = pd.Timedelta(minutes=STEP_MINUTES)
MAX_HORIZON_STEPS = 140
DISPATCH_STEPS = 96

COL_DATETIME = "datetime"
COL_LOAD = "Last [W]"
COL_REAL_PV = "Reale Erzeugung [W]"
COL_DA_PRICE = "DA Preis [EUR/MWh]"
COL_REBAP_UNDER = "reBAP unterdeckt [EUR/MWh]"

COL_FORECAST_TIME = "valid_time"
COL_FORECAST_SOLAR_ELEV = "solar_elevation_deg"
COL_FORECAST_PRED = "y_pred"

MODEL_COL_PV = "P_pv_forecast_W"
SETTLEMENT_REBAP_COL = "reBAP_unterdeckt_eur_mwh"


MLP_FORECAST_REL_PATHS = [
    "iter_2025-07-07/2025070709.csv",
    "iter_2025-07-08/2025070809.csv",
    "iter_2025-07-09/2025070909.csv",
    "iter_2025-07-10/2025071009.csv",
    "iter_2025-07-11/2025071109.csv",
    "iter_2025-07-12/2025071209.csv",
    "iter_2025-07-13/2025071309.csv",
    "iter_2025-07-14/2025071409.csv",
    "iter_2025-07-15/2025071509.csv",
    "iter_2025-07-16/2025071609.csv",
    "iter_2025-07-17/2025071709.csv",
    "iter_2025-07-18/2025071809.csv",
    "iter_2025-07-19/2025071909.csv",
    "iter_2025-07-20/2025072009.csv",
    "iter_2025-07-21/2025072109.csv",
    "iter_2025-07-22/2025072209.csv",
    "iter_2025-07-23/2025072309.csv",
    "iter_2025-07-24/2025072409.csv",
    "iter_2025-07-25/2025072509.csv",
    "iter_2025-07-26/2025072609.csv",
    "iter_2025-07-27/2025072709.csv",
    "iter_2025-07-28/2025072809.csv",
    "iter_2025-07-29/2025072909.csv",
    "iter_2025-07-30/2025073009.csv",
    "iter_2025-07-31/2025073109.csv",
    "iter_2025-08-01/2025080109.csv",
    "iter_2025-08-02/2025080209.csv",
    "iter_2025-08-03/2025080309.csv",
    "iter_2025-08-04/2025080409.csv",
    "iter_2025-08-05/2025080509.csv",
    "iter_2025-08-06/2025080609.csv",
    "iter_2025-08-07/2025080709.csv",
    "iter_2025-08-08/2025080809.csv",
    "iter_2025-08-09/2025080909.csv",
    "iter_2025-08-10/2025081009.csv",
    "iter_2025-08-11/2025081109.csv",
    "iter_2025-08-12/2025081209.csv",
    "iter_2025-08-13/2025081309.csv",
    "iter_2025-08-14/2025081409.csv",
    "iter_2025-08-15/2025081509.csv",
    "iter_2025-08-16/2025081609.csv",
    "iter_2025-08-17/2025081709.csv",
    "iter_2025-08-18/2025081809.csv",
    "iter_2025-08-19/2025081909.csv",
    "iter_2025-08-20/2025082009.csv",
    "iter_2025-08-21/2025082109.csv",
    "iter_2025-08-22/2025082209.csv",
    "iter_2025-08-23/2025082306.csv",
    "iter_2025-08-24/2025082409.csv",
    "iter_2025-08-25/2025082509.csv",
    "iter_2025-08-26/2025082609.csv",
    "iter_2025-08-27/2025082709.csv",
    "iter_2025-08-28/2025082809.csv",
    "iter_2025-08-29/2025082909.csv",
    "iter_2025-08-30/2025083009.csv",
    "iter_2025-08-31/2025083109.csv",
    "iter_2025-09-01/2025090109.csv",
    "iter_2025-09-02/2025090209.csv",
    "iter_2025-09-03/2025090309.csv",
    "iter_2025-09-04/2025090400.csv",
    "iter_2025-09-05/2025090509.csv",
    "iter_2025-09-06/2025090609.csv",
    "iter_2025-09-07/2025090709.csv",
    "iter_2025-09-08/2025090809.csv",
    "iter_2025-09-09/2025090909.csv",
    "iter_2025-09-10/2025091009.csv",
    "iter_2025-09-11/2025091109.csv",
    "iter_2025-09-12/2025091209.csv",
    "iter_2025-09-13/2025091309.csv",
    "iter_2025-09-14/2025091409.csv",
    "iter_2025-09-15/2025091509.csv",
    "iter_2025-09-16/2025091609.csv",
    "iter_2025-09-17/2025091709.csv",
    "iter_2025-09-18/2025091809.csv",
    "iter_2025-09-19/2025091909.csv",
    "iter_2025-09-20/2025092009.csv",
    "iter_2025-09-21/2025092109.csv",
    "iter_2025-09-22/2025092209.csv",
    "iter_2025-09-23/2025092309.csv",
    "iter_2025-09-24/2025092409.csv",
    "iter_2025-09-25/2025092509.csv",
    "iter_2025-09-26/2025092609.csv",
    "iter_2025-09-27/2025092709.csv",
    "iter_2025-09-28/2025092809.csv",
    "iter_2025-09-29/2025092909.csv",
    "iter_2025-09-30/2025093009.csv",
    "iter_2025-10-01/2025100109.csv",
    "iter_2025-10-02/2025100209.csv",
    "iter_2025-10-03/2025100309.csv",
    "iter_2025-10-04/2025100409.csv",
]


def _to_numeric(series: pd.Series) -> pd.Series:
    """Converts a series to float, robust for comma and dot decimal strings."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Loads the optimization YAML configuration."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_optimization_input(data_path: Path = INPUT_DATA_PATH) -> pd.DataFrame:
    """Loads optimization input data and converts needed columns to numeric types."""
    df = pd.read_csv(data_path, decimal=",")
    df[COL_DATETIME] = pd.to_datetime(df[COL_DATETIME])

    for col in (COL_LOAD, COL_REAL_PV, COL_DA_PRICE, COL_REBAP_UNDER):
        df[col] = pd.to_numeric(df[col])

    return df.sort_values(COL_DATETIME).reset_index(drop=True)


def load_mlp_forecast_horizon(scenario: str, forecast_rel_path: str, dispatch_day: pd.Timestamp) -> pd.DataFrame:
    """Loads one MLP forecast file and returns available horizon for a dispatch day."""
    forecast_path = RESULTS_DIR / f"{scenario}_output" / Path(forecast_rel_path)

    df = pd.read_csv(forecast_path)
    df[COL_FORECAST_TIME] = pd.to_datetime(df[COL_FORECAST_TIME])
    df[COL_FORECAST_SOLAR_ELEV] = _to_numeric(df[COL_FORECAST_SOLAR_ELEV])
    df[COL_FORECAST_PRED] = _to_numeric(df[COL_FORECAST_PRED])

    pv = (df[COL_FORECAST_PRED] * PV_INSTALLED_W).clip(lower=0.0)
    pv.loc[df[COL_FORECAST_SOLAR_ELEV] < 0.0] = 0.0
    pv = pv.clip(lower=0.0)

    out = pd.DataFrame(
        {
            COL_DATETIME: df[COL_FORECAST_TIME].to_numpy(),
            MODEL_COL_PV: pv.to_numpy(dtype=float),
        }
    ).sort_values(COL_DATETIME)

    horizon_start = dispatch_day
    horizon_end = dispatch_day + (MAX_HORIZON_STEPS - 1) * STEP
    out = out[(out[COL_DATETIME] >= horizon_start) & (out[COL_DATETIME] <= horizon_end)].copy()

    return out.reset_index(drop=True)


def build_rebap_series(data_df: pd.DataFrame) -> pd.DataFrame:
    """Builds the single reBAP settlement series from optimization input."""
    return pd.DataFrame(
        {SETTLEMENT_REBAP_COL: data_df[COL_REBAP_UNDER].to_numpy(dtype=float)},
        index=data_df[COL_DATETIME],
    )


def build_real_pv_series(data_df: pd.DataFrame) -> pd.Series:
    """Builds the real PV time series used for settlement post calculation."""
    return pd.Series(data_df[COL_REAL_PV].to_numpy(dtype=float), index=data_df[COL_DATETIME], name=COL_REAL_PV)


def _initial_soc_from_config(cfg: dict[str, Any]) -> float:
    """Computes initial state of charge in Wh from configuration."""
    return float(cfg["optimization"]["initial_soc_fraction"]) * float(cfg["battery"]["e_max_kwh"]) * 1000.0


def build_summary_row(
    scenario: str,
    run_id: int,
    dispatch_day: pd.Timestamp,
    horizon_start: pd.Timestamp,
    n_horizon_steps: int,
    current_soc_wh: float,
    solver_result: dict[str, Any],
    settled_dispatch: pd.DataFrame,
) -> dict[str, Any]:
    """Builds one summary row for one rolling-optimization run."""
    realized_row = aggregate_run_realized_metrics(settled_dispatch).iloc[0]

    return {
        "run_id": run_id,
        "horizon_start": horizon_start,
        "horizon_end_exclusive": horizon_start + n_horizon_steps * STEP,
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
        "n_horizon_steps": int(n_horizon_steps),
        "n_dispatch_steps": DISPATCH_STEPS,
        "terminal_soc_enforced": False,
        "real_revenue_dispatch_day_eur": float(realized_row["real_revenue_dispatch_day_eur"]),
        "real_rebap_settlement_dispatch_day_eur": float(realized_row["real_rebap_settlement_dispatch_day_eur"]),
        "real_revenue_dispatch_day_net_eur": float(realized_row["real_revenue_dispatch_day_net_eur"]),
        "scenario": scenario,
    }


def run_mlp_rolling_optimization(scenario: str, cfg: dict[str, Any], input_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs rolling optimization for one MLP scenario."""
    indexed_input = input_df.set_index(COL_DATETIME).sort_index()
    rebap_prices = build_rebap_series(input_df)
    real_pv_series = build_real_pv_series(input_df)

    dispatch_days = pd.date_range(start=DATE_START, end=DATE_END, freq="D")

    dispatch_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    current_soc_wh = _initial_soc_from_config(cfg)

    for run_id, forecast_rel_path in enumerate(MLP_FORECAST_REL_PATHS):
        dispatch_day = pd.Timestamp(dispatch_days[run_id]).normalize()
        forecast_horizon = load_mlp_forecast_horizon(
            scenario=scenario,
            forecast_rel_path=forecast_rel_path,
            dispatch_day=dispatch_day,
        )

        horizon_times = pd.DatetimeIndex(forecast_horizon[COL_DATETIME])
        horizon_df = pd.DataFrame(
            {
                COL_DATETIME: horizon_times.to_numpy(),
                COL_LOAD: indexed_input.loc[horizon_times, COL_LOAD].to_numpy(dtype=float),
                MODEL_COL_PV: forecast_horizon[MODEL_COL_PV].to_numpy(dtype=float),
                COL_DA_PRICE: indexed_input.loc[horizon_times, COL_DA_PRICE].to_numpy(dtype=float),
            }
        )

        solver_result = solve_milp_for_horizon(
            horizon_df=horizon_df,
            cfg=cfg,
            initial_soc_wh=current_soc_wh,
            enforce_terminal_soc=False,
        )

        fixed_dispatch = solver_result["df"].iloc[:DISPATCH_STEPS].copy().reset_index(drop=True)

        settled_dispatch = compute_dispatch_settlement(
            dispatch_df=fixed_dispatch,
            real_pv_series=real_pv_series,
            cfg=cfg,
            rebap_prices=rebap_prices,
        )

        settled_dispatch["run_id"] = run_id
        settled_dispatch["dispatch_day"] = dispatch_day
        settled_dispatch["scenario"] = scenario

        metadata_cols = ["run_id", "dispatch_day", "scenario"]
        settled_dispatch = settled_dispatch[metadata_cols + [c for c in settled_dispatch.columns if c not in metadata_cols]]

        summary_rows.append(
            build_summary_row(
                scenario=scenario,
                run_id=run_id,
                dispatch_day=dispatch_day,
                horizon_start=dispatch_day,
                n_horizon_steps=len(horizon_df),
                current_soc_wh=current_soc_wh,
                solver_result=solver_result,
                settled_dispatch=settled_dispatch,
            )
        )
        dispatch_frames.append(settled_dispatch)

        current_soc_wh = float(settled_dispatch.iloc[-1]["opt_E_end_Wh"])

    return pd.concat(dispatch_frames, ignore_index=True), pd.DataFrame(summary_rows)


def write_outputs(scenario: str, dispatch_df: pd.DataFrame, summary_df: pd.DataFrame) -> tuple[Path, Path]:
    """Writes scenario dispatch and summary outputs to CSV."""
    scenario_dir = OPTIMIZATION_DIR / scenario
    dispatch_path = scenario_dir / f"optimization_dispatch_{scenario}.csv"
    summary_path = scenario_dir / f"optimization_run_summary_{scenario}.csv"

    scenario_dir.mkdir(parents=True, exist_ok=True)
    dispatch_df.to_csv(dispatch_path, index=False, decimal=",")
    summary_df.to_csv(summary_path, index=False, decimal=",")

    return dispatch_path, summary_path


def main() -> None:
    """Runs source, target and transfer rolling optimization sequentially."""
    cfg = load_config()
    input_df = load_optimization_input()

    for scenario in MLP_SCENARIOS:
        dispatch_df, summary_df = run_mlp_rolling_optimization(scenario=scenario, cfg=cfg, input_df=input_df)
        dispatch_path, summary_path = write_outputs(scenario=scenario, dispatch_df=dispatch_df, summary_df=summary_df)

        print(f"[{scenario}] dispatch={dispatch_path} ({len(dispatch_df)} rows)")
        print(f"[{scenario}] summary={summary_path} ({len(summary_df)} rows)")


if __name__ == "__main__":
    main()
