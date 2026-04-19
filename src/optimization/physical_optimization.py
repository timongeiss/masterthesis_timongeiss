"""
Builds physical rolling optimization CSV outputs with a fixed forecast file list.

Workflow:
1) Load config.
2) Load optimization input (load, DA price, real PV, reBAP).
3) For each dispatch day, load the mapped physical forecast file.
4) Build forecast PV from `pv_power_W` with solar-elevation clipping.
5) Optimize with available horizon from 00:00 dispatch day up to max 34:45.
6) Fix first 96 steps (dispatch day), then run settlement with real PV.
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
FORECAST_DIR = PROJECT_ROOT / "data" / "results" / "physical_output"

OUTPUT_DIR = PROJECT_ROOT / "data" / "optimization" / "physical"
OUTPUT_DISPATCH_PATH = OUTPUT_DIR / "optimization_dispatch_physical.csv"
OUTPUT_SUMMARY_PATH = OUTPUT_DIR / "optimization_run_summary_physical.csv"

SCENARIO_NAME = "physical"

#dispatch days from 2025-07-08 to 2025-10-05, inclusive, total 90 days, with available forecast files for each day
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
COL_FORECAST_PV = "pv_power_W"

MODEL_COL_PV = "P_pv_forecast_W"
SETTLEMENT_REBAP_COL = "reBAP_unterdeckt_eur_mwh"

# zwei fallback files mit 09 uhr 1x und 6 uhr 1x, da 09 uhr files für 2025-08-23 und 2025-09-04 fehlen, siehe
PHYSICAL_FORECAST_FILES = [
    "pv_weather_2025070709_with_pv_power.csv",
    "pv_weather_2025070809_with_pv_power.csv",
    "pv_weather_2025070909_with_pv_power.csv",
    "pv_weather_2025071009_with_pv_power.csv",
    "pv_weather_2025071109_with_pv_power.csv",
    "pv_weather_2025071209_with_pv_power.csv",
    "pv_weather_2025071309_with_pv_power.csv",
    "pv_weather_2025071409_with_pv_power.csv",
    "pv_weather_2025071509_with_pv_power.csv",
    "pv_weather_2025071609_with_pv_power.csv",
    "pv_weather_2025071709_with_pv_power.csv",
    "pv_weather_2025071809_with_pv_power.csv",
    "pv_weather_2025071909_with_pv_power.csv",
    "pv_weather_2025072009_with_pv_power.csv",
    "pv_weather_2025072109_with_pv_power.csv",
    "pv_weather_2025072209_with_pv_power.csv",
    "pv_weather_2025072309_with_pv_power.csv",
    "pv_weather_2025072409_with_pv_power.csv",
    "pv_weather_2025072509_with_pv_power.csv",
    "pv_weather_2025072609_with_pv_power.csv",
    "pv_weather_2025072709_with_pv_power.csv",
    "pv_weather_2025072809_with_pv_power.csv",
    "pv_weather_2025072909_with_pv_power.csv",
    "pv_weather_2025073009_with_pv_power.csv",
    "pv_weather_2025073109_with_pv_power.csv",
    "pv_weather_2025080109_with_pv_power.csv",
    "pv_weather_2025080209_with_pv_power.csv",
    "pv_weather_2025080309_with_pv_power.csv",
    "pv_weather_2025080409_with_pv_power.csv",
    "pv_weather_2025080509_with_pv_power.csv",
    "pv_weather_2025080609_with_pv_power.csv",
    "pv_weather_2025080709_with_pv_power.csv",
    "pv_weather_2025080809_with_pv_power.csv",
    "pv_weather_2025080909_with_pv_power.csv",
    "pv_weather_2025081009_with_pv_power.csv",
    "pv_weather_2025081109_with_pv_power.csv",
    "pv_weather_2025081209_with_pv_power.csv",
    "pv_weather_2025081309_with_pv_power.csv",
    "pv_weather_2025081409_with_pv_power.csv",
    "pv_weather_2025081509_with_pv_power.csv",
    "pv_weather_2025081609_with_pv_power.csv",
    "pv_weather_2025081709_with_pv_power.csv",
    "pv_weather_2025081809_with_pv_power.csv",
    "pv_weather_2025081909_with_pv_power.csv",
    "pv_weather_2025082009_with_pv_power.csv",
    "pv_weather_2025082109_with_pv_power.csv",
    "pv_weather_2025082209_with_pv_power.csv",
    "pv_weather_2025082306_with_pv_power.csv",
    "pv_weather_2025082409_with_pv_power.csv",
    "pv_weather_2025082509_with_pv_power.csv",
    "pv_weather_2025082609_with_pv_power.csv",
    "pv_weather_2025082709_with_pv_power.csv",
    "pv_weather_2025082809_with_pv_power.csv",
    "pv_weather_2025082909_with_pv_power.csv",
    "pv_weather_2025083009_with_pv_power.csv",
    "pv_weather_2025083109_with_pv_power.csv",
    "pv_weather_2025090109_with_pv_power.csv",
    "pv_weather_2025090209_with_pv_power.csv",
    "pv_weather_2025090309_with_pv_power.csv",
    "pv_weather_2025090400_with_pv_power.csv",
    "pv_weather_2025090509_with_pv_power.csv",
    "pv_weather_2025090609_with_pv_power.csv",
    "pv_weather_2025090709_with_pv_power.csv",
    "pv_weather_2025090809_with_pv_power.csv",
    "pv_weather_2025090909_with_pv_power.csv",
    "pv_weather_2025091009_with_pv_power.csv",
    "pv_weather_2025091109_with_pv_power.csv",
    "pv_weather_2025091209_with_pv_power.csv",
    "pv_weather_2025091309_with_pv_power.csv",
    "pv_weather_2025091409_with_pv_power.csv",
    "pv_weather_2025091509_with_pv_power.csv",
    "pv_weather_2025091609_with_pv_power.csv",
    "pv_weather_2025091709_with_pv_power.csv",
    "pv_weather_2025091809_with_pv_power.csv",
    "pv_weather_2025091909_with_pv_power.csv",
    "pv_weather_2025092009_with_pv_power.csv",
    "pv_weather_2025092109_with_pv_power.csv",
    "pv_weather_2025092209_with_pv_power.csv",
    "pv_weather_2025092309_with_pv_power.csv",
    "pv_weather_2025092409_with_pv_power.csv",
    "pv_weather_2025092509_with_pv_power.csv",
    "pv_weather_2025092609_with_pv_power.csv",
    "pv_weather_2025092709_with_pv_power.csv",
    "pv_weather_2025092809_with_pv_power.csv",
    "pv_weather_2025092909_with_pv_power.csv",
    "pv_weather_2025093009_with_pv_power.csv",
    "pv_weather_2025100109_with_pv_power.csv",
    "pv_weather_2025100209_with_pv_power.csv",
    "pv_weather_2025100309_with_pv_power.csv",
    "pv_weather_2025100409_with_pv_power.csv",
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


def load_forecast_horizon(forecast_file: str, dispatch_day: pd.Timestamp) -> pd.DataFrame:
    """
    Loads one physical forecast file and returns the available forecast horizon
    for the optimization run starting at 00:00 of the dispatch day.
    """
    df = pd.read_csv(FORECAST_DIR / forecast_file)
    df[COL_FORECAST_TIME] = pd.to_datetime(df[COL_FORECAST_TIME])
    df[COL_FORECAST_SOLAR_ELEV] = _to_numeric(df[COL_FORECAST_SOLAR_ELEV])
    df[COL_FORECAST_PV] = _to_numeric(df[COL_FORECAST_PV])

    pv = df[COL_FORECAST_PV].clip(lower=0.0)
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
    run_id: int,
    dispatch_day: pd.Timestamp,
    horizon_start: pd.Timestamp,
    n_horizon_steps: int,
    current_soc_wh: float,
    solver_result: dict[str, Any],
    settled_dispatch: pd.DataFrame,
) -> dict[str, Any]:
    """Builds one summary row for one physical rolling-optimization run."""
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
        "scenario": SCENARIO_NAME,
    }


def run_physical_rolling_optimization(cfg: dict[str, Any], input_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs physical rolling optimization using fixed forecast files and variable
    available horizon length per day.
    """
    indexed_input = input_df.set_index(COL_DATETIME).sort_index()
    rebap_prices = build_rebap_series(input_df)
    real_pv_series = build_real_pv_series(input_df)

    dispatch_days = pd.date_range(start=DATE_START, end=DATE_END, freq="D")

    dispatch_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    current_soc_wh = _initial_soc_from_config(cfg)

    for run_id, forecast_file in enumerate(PHYSICAL_FORECAST_FILES):
        dispatch_day = pd.Timestamp(dispatch_days[run_id]).normalize()
        forecast_horizon = load_forecast_horizon(forecast_file=forecast_file, dispatch_day=dispatch_day)

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
        settled_dispatch["scenario"] = SCENARIO_NAME

        metadata_cols = ["run_id", "dispatch_day", "scenario"]
        settled_dispatch = settled_dispatch[metadata_cols + [c for c in settled_dispatch.columns if c not in metadata_cols]]

        summary_rows.append(
            build_summary_row(
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


def write_outputs(
    dispatch_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    dispatch_path: Path = OUTPUT_DISPATCH_PATH,
    summary_path: Path = OUTPUT_SUMMARY_PATH,
) -> tuple[Path, Path]:
    """Writes dispatch and summary outputs to CSV with comma decimal separator."""
    dispatch_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    dispatch_df.to_csv(dispatch_path, index=False, decimal=",")
    summary_df.to_csv(summary_path, index=False, decimal=",")

    return dispatch_path, summary_path


def main() -> None:
    """Runs the complete physical optimization workflow end-to-end."""
    cfg = load_config()
    input_df = load_optimization_input()
    dispatch_df, summary_df = run_physical_rolling_optimization(cfg=cfg, input_df=input_df)
    dispatch_path, summary_path = write_outputs(dispatch_df=dispatch_df, summary_df=summary_df)

    print(f"Saved dispatch CSV: {dispatch_path} ({len(dispatch_df)} rows)")
    print(f"Saved summary CSV: {summary_path} ({len(summary_df)} rows)")


if __name__ == "__main__":
    main()
