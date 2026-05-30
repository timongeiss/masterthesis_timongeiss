"""
Create one day-wise dispatch PNG per scenario with the exact same 5-panel
plot layout as plot_all_scenarios_full_period.py.

Output:
- reports/optimization/dispatch_day_comparison/<scenario>/<YYYYMMDD>_<scenario>_dispatch.png
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "reports" / "optimization" / "dispatch_day_comparison"

SCENARIOS = ("ideal", "physical", "source", "target", "transfer", "standard")
SCENARIO_LABELS = {
    "ideal": "Ideal",
    "physical": "Physical",
    "source": "Source",
    "target": "Target",
    "transfer": "Transfer",
    "standard": "Standard",
}
SCENARIO_FORECAST_COLORS = {
    "transfer": "tab:blue",
    "physical": "tab:orange",
    "source": "tab:green",
    "target": "tab:red",
    "standard": "tab:gray",
    "ideal": "black",
}
PV_REAL_COLOR = "#ffd900"
DA_PRICE_COLOR = "#00E5FF"
REBAP_PRICE_COLOR = "#6F00FF"
REBAP_COLOR = "tab:blue"
AGING_COLOR = "tab:green"
STANDARD_FEED_IN_COLOR = "#7f7f7f"
STANDARD_ELECTRICITY_PRICE_COLOR = "#1f4e79"
STANDARD_FEED_IN_TARIFF_EUR_MWH = 80.0
STANDARD_ELECTRICITY_PRICE_EUR_MWH = 285.0
ZERO_LINE_COLOR = "#222222"
ZERO_LINE_WIDTH = 1.0
MARKER_COLORS = {
    "best": "#228B22",
    "worst": "#C62828",
}
MARKER_LINESTYLE = "--"
MARKER_LINEWIDTH = 1.4
MARKER_LABEL_Y = {
    "best": 1.05,
    "worst": 1.16,
}

FIXED_DAY_MARKERS = {
    "physical": {
        "2025-09-01": [
            {"time": "11:45", "classification": "12212", "kind": "worst"},
            {"time": "13:15", "classification": "22231", "kind": "best"},
        ],
    },
    "source": {
        "2025-08-07": [
            {"time": "10:15", "classification": "12211", "kind": "worst"},
            {"time": "17:45", "classification": "22131", "kind": "best"},
        ],
    },
    "target": {
        "2025-07-27": [
            {"time": "15:00", "classification": "12132", "kind": "worst"},
            {"time": "17:30", "classification": "22211", "kind": "best"},
        ],
    },
    "transfer": {
        "2025-08-28": [
            {"time": "10:00", "classification": "22111", "kind": "best"},
            {"time": "13:30", "classification": "12232", "kind": "worst"},
        ],
    },
}

DISPATCH_FILES = {
    "ideal": PROJECT_ROOT / "data" / "optimization" / "ideal" / "optimization_dispatch_ideal.csv",
    "physical": PROJECT_ROOT / "data" / "optimization" / "physical" / "optimization_dispatch_physical.csv",
    "source": PROJECT_ROOT / "data" / "optimization" / "source" / "optimization_dispatch_source.csv",
    "target": PROJECT_ROOT / "data" / "optimization" / "target" / "optimization_dispatch_target.csv",
    "transfer": PROJECT_ROOT / "data" / "optimization" / "transfer" / "optimization_dispatch_transfer.csv",
    "standard": PROJECT_ROOT / "data" / "optimization" / "standard" / "optimization_dispatch_standard.csv",
}

PLOT_START = pd.Timestamp("2025-07-08 00:00:00")
PLOT_END = pd.Timestamp("2025-10-05 23:45:00")

FIGSIZE = (11.69, 8.27)  # A4 landscape in inches
EXPORT_DPI = 300
SUBPLOT_TITLE_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 13
TICK_FONTSIZE = 11
LEGEND_FONTSIZE = 11


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    """Converts listed columns to numeric in-place when they exist."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _legend_unique(ax: plt.Axes, handles=None, labels=None, anchor_x: float = 1.01) -> None:
    """Draws a legend without duplicate labels."""
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()

    uniq_handles = []
    uniq_labels = []
    seen = set()
    for handle, label in zip(handles, labels):
        if not label or label == "_nolegend_" or label in seen:
            continue
        seen.add(label)
        uniq_handles.append(handle)
        uniq_labels.append(label)

    if uniq_handles:
        ax.legend(
            uniq_handles,
            uniq_labels,
            loc="center left",
            bbox_to_anchor=(anchor_x, 0.5),
            borderaxespad=0.0,
            fontsize=LEGEND_FONTSIZE,
        )


def load_dispatch(path: Path) -> pd.DataFrame:
    """Loads one dispatch CSV and applies basic typing and filtering."""
    if not path.exists():
        raise FileNotFoundError(f"Dispatch file not found: {path}")

    df = pd.read_csv(path, decimal=",")
    if "datetime" not in df.columns:
        raise ValueError(f"Missing 'datetime' column in {path}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    _to_numeric(
        df,
        [
            "input_P_pv_forecast_W",
            "real_P_pv_W",
            "real_P_curt_W",
            "input_P_load_W",
            "input_price_EUR_MWh",
            "real_imbalance_price_eur_mwh",
            "opt_P_ch_W",
            "opt_P_dis_W",
            "opt_P_curt_W",
            "opt_E_end_Wh",
            "opt_P_grid_in_W",
            "opt_P_grid_out_W",
            "real_P_grid_in_W",
            "real_P_grid_out_W",
            "real_grid_fee_step_eur",
            "real_revenue_step_eur",
            "real_rebap_settlement_step_eur",
            "opt_cycle_aging_cost_step_eur",
            "real_revenue_step_net_after_aging_eur",
        ],
    )

    if "real_imbalance_price_eur_mwh" not in df.columns:
        df["real_imbalance_price_eur_mwh"] = 0.0
    for col in (
        "opt_P_curt_W",
        "real_P_curt_W",
        "real_revenue_step_eur",
        "real_rebap_settlement_step_eur",
        "opt_cycle_aging_cost_step_eur",
    ):
        if col not in df.columns:
            df[col] = 0.0

    if "real_revenue_step_net_after_aging_eur" not in df.columns:
        df["real_revenue_step_net_after_aging_eur"] = (
            df["real_revenue_step_eur"]
            + df["real_rebap_settlement_step_eur"]
            - df["opt_cycle_aging_cost_step_eur"]
        )

    mask = (df["datetime"] >= PLOT_START) & (df["datetime"] <= PLOT_END)
    out = df.loc[mask].copy()
    out = out.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return out


def _scenario_days(df: pd.DataFrame) -> list[pd.Timestamp]:
    days = df["datetime"].dt.normalize().dropna().drop_duplicates().sort_values()
    return [pd.Timestamp(d).normalize() for d in days]


def _filter_day(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    day = pd.Timestamp(day).normalize()
    day_end = day + pd.Timedelta(days=1)
    day_df = df.loc[(df["datetime"] >= day) & (df["datetime"] < day_end)].copy()
    day_df = day_df.sort_values("datetime").reset_index(drop=True)
    return day_df


def _expand_post_step(x: pd.Series | pd.Index, y: np.ndarray) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Expand a time series to explicit coordinates for post-step fills."""
    x_index = pd.DatetimeIndex(pd.to_datetime(x))
    y_values = np.asarray(y, dtype=float)
    if len(x_index) != len(y_values):
        raise ValueError("x and y must have the same length for post-step expansion.")
    if len(x_index) == 0:
        return pd.DatetimeIndex([]), y_values
    x_step = pd.DatetimeIndex(np.repeat(x_index.to_numpy(), 2)[1:])
    y_step = np.repeat(y_values, 2)[:-1]
    return x_step, y_step


def _day_marker_specs(scenario: str, day: pd.Timestamp) -> list[dict[str, str]]:
    """Return fixed marker specs for one scenario/day pair."""
    return FIXED_DAY_MARKERS.get(scenario, {}).get(f"{pd.Timestamp(day):%Y-%m-%d}", [])


def _draw_day_markers(axes: np.ndarray, x: pd.Series, scenario: str, day: pd.Timestamp) -> None:
    """Draw fixed vertical markers across all subplots and label them above the price subplot."""
    marker_specs = _day_marker_specs(scenario, day)
    if not marker_specs:
        return

    x_index = pd.DatetimeIndex(pd.to_datetime(x))
    x_values = set(x_index)
    ax_price = axes[3]

    for spec in marker_specs:
        marker_time = pd.Timestamp(f"{pd.Timestamp(day):%Y-%m-%d} {spec['time']}")
        if marker_time not in x_values:
            print(f"[{scenario} {day:%Y-%m-%d}] marker timestamp missing in dispatch data: {marker_time}")
            continue

        color = MARKER_COLORS.get(spec["kind"], "black")
        for ax in axes:
            ax.axvline(
                marker_time,
                color=color,
                linestyle=MARKER_LINESTYLE,
                linewidth=MARKER_LINEWIDTH,
                alpha=0.95,
                zorder=20,
            )

        ax_price.text(
            marker_time,
            MARKER_LABEL_Y.get(spec["kind"], -0.24),
            spec["classification"],
            color=color,
            fontsize=TICK_FONTSIZE,
            fontweight="bold",
            ha="center",
            va="bottom",
            transform=ax_price.get_xaxis_transform(),
            clip_on=False,
        )


def plot_scenario_day(df: pd.DataFrame, scenario: str, day: pd.Timestamp, out_file: Path) -> None:
    """Creates one 5-panel day plot and stores it as PNG."""
    forecast_color = SCENARIO_FORECAST_COLORS.get(scenario, "black")
    is_standard = scenario == "standard"
    x = df["datetime"]

    fig, axes = plt.subplots(5, 1, sharex=True, figsize=FIGSIZE)
    ax_pv, ax_bat, ax_grid, ax_price, ax_rev = axes

    pv_forecast_kw = df["input_P_pv_forecast_W"] / 1000.0
    pv_real_kw = df["real_P_pv_W"] / 1000.0
    load_kw_neg = -(df["input_P_load_W"] / 1000.0)

    p_ch_w = df["opt_P_ch_W"].clip(lower=0.0).to_numpy(dtype=float)
    p_dis_w = df["opt_P_dis_W"].clip(lower=0.0).to_numpy(dtype=float)
    p_curt_w = df["opt_P_curt_W"].clip(lower=0.0).to_numpy(dtype=float)
    real_curt_w = df["real_P_curt_W"].clip(lower=0.0).to_numpy(dtype=float)
    pv_forecast_w = df["input_P_pv_forecast_W"].clip(lower=0.0).to_numpy(dtype=float)
    pv_real_w = df["real_P_pv_W"].clip(lower=0.0).to_numpy(dtype=float)
    load_w = df["input_P_load_W"].clip(lower=0.0).to_numpy(dtype=float)

    effective_pv_w = np.maximum(pv_forecast_w - p_curt_w, 0.0)
    pv_surplus_w = np.maximum(effective_pv_w - load_w, 0.0)
    load_def_after_pv_w = np.maximum(load_w - effective_pv_w, 0.0)

    p_ch_from_pv_w = np.minimum(p_ch_w, pv_surplus_w)
    p_ch_from_grid_w = np.maximum(p_ch_w - p_ch_from_pv_w, 0.0)
    ch_from_pv_kw = p_ch_from_pv_w / 1000.0
    ch_from_grid_kw = p_ch_from_grid_w / 1000.0

    p_dis_to_load_w = np.minimum(p_dis_w, load_def_after_pv_w)
    p_dis_to_grid_w = np.maximum(p_dis_w - p_dis_to_load_w, 0.0)
    dis_to_load_kw = p_dis_to_load_w / 1000.0
    dis_to_grid_kw = p_dis_to_grid_w / 1000.0

    # 1) PV + load
    ax_pv.fill_between(x, 0.0, load_kw_neg, step="post", color="red", alpha=0.12, zorder=0, label="_nolegend_")
    ax_pv.fill_between(x, 0.0, pv_real_kw, step="post", color=PV_REAL_COLOR, alpha=0.28, zorder=1, label="_nolegend_")
    pv_diff_low_kw = np.minimum(pv_real_kw.to_numpy(dtype=float), pv_forecast_kw.to_numpy(dtype=float))
    pv_diff_high_kw = np.maximum(pv_real_kw.to_numpy(dtype=float), pv_forecast_kw.to_numpy(dtype=float))
    ax_pv.fill_between(
        x,
        pv_diff_low_kw,
        pv_diff_high_kw,
        step="post",
        facecolor="none",
        edgecolor=forecast_color,
        hatch="////",
        linewidth=0.0,
        zorder=2,
        label="_nolegend_",
    )
    real_pv_after_curt_kw = np.maximum((pv_real_w - real_curt_w) / 1000.0, 0.0)
    ax_pv.fill_between(
        x,
        real_pv_after_curt_kw,
        pv_real_kw,
        step="post",
        color="black",
        alpha=0.45,
        zorder=3,
        label="Curtailment",
    )
    ax_pv.step(x, pv_forecast_kw, where="post", label="PV forecast", linewidth=0.9, color=forecast_color, zorder=4)
    ax_pv.step(x, pv_real_kw, where="post", label="PV", linewidth=1.0, color=PV_REAL_COLOR, zorder=5)
    ax_pv.step(x, load_kw_neg, where="post", label="Load", linewidth=0.9, color="red", zorder=1)
    ax_pv.set_ylabel("kW", fontsize=AXIS_LABEL_FONTSIZE)
    ax_pv.set_title("PV & Load (+ generation, - demand)", loc="left", fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax_pv.grid(True, alpha=0.3)
    _legend_unique(ax_pv)

    # 2) Battery + SOC
    ax_soc = ax_bat.twinx()
    battery_signed_kw = (df["opt_P_dis_W"] - df["opt_P_ch_W"]) / 1000.0

    ax_bat.fill_between(x, 0.0, dis_to_load_kw, step="post", color="red", alpha=0.35, label="Load")
    ax_bat.fill_between(
        x,
        dis_to_load_kw,
        dis_to_load_kw + dis_to_grid_kw,
        step="post",
        color=REBAP_COLOR,
        alpha=0.45,
        label="Grid",
    )
    ax_bat.fill_between(x, -ch_from_grid_kw, 0.0, step="post", color=REBAP_COLOR, alpha=0.45, label="_nolegend_")
    ax_bat.fill_between(
        x,
        -(ch_from_grid_kw + ch_from_pv_kw),
        -ch_from_grid_kw,
        step="post",
        color=PV_REAL_COLOR,
        alpha=0.45,
        label="PV",
    )
    ax_bat.step(x, battery_signed_kw, where="post", color="black", linewidth=0.9, label="Edge")
    ax_bat.axhline(0.0, color="black", linewidth=0.7, alpha=0.7)

    soc_series_kwh = df["opt_E_end_Wh"] / 1000.0
    ax_soc.plot(x, soc_series_kwh, label="SOC", linewidth=1.0, color="green")
    soc_max = float(soc_series_kwh.max()) if not soc_series_kwh.empty else 1.0
    if soc_max <= 0.0:
        soc_max = 1.0
    ax_soc.set_ylim(-soc_max, soc_max)
    ax_soc.set_yticks([0.0, soc_max * 0.25, soc_max * 0.5, soc_max * 0.75, soc_max])

    ax_bat.set_ylabel("kW", fontsize=AXIS_LABEL_FONTSIZE)
    ax_soc.set_ylabel("kWh", fontsize=AXIS_LABEL_FONTSIZE)
    ax_bat.set_title("Battery (+ Discharge, - Charge)", loc="left", fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax_bat.grid(True, alpha=0.3)
    left_handles, left_labels = ax_bat.get_legend_handles_labels()
    right_handles, right_labels = ax_soc.get_legend_handles_labels()
    _legend_unique(ax_bat, left_handles + right_handles, left_labels + right_labels, anchor_x=1.12)

    # 3) Grid split
    grid_in_w = df["opt_P_grid_in_W"].clip(lower=0.0).to_numpy(dtype=float)
    grid_out_w = df["opt_P_grid_out_W"].clip(lower=0.0).to_numpy(dtype=float)
    if "real_P_grid_in_W" in df.columns and "real_P_grid_out_W" in df.columns:
        grid_in_w = df["real_P_grid_in_W"].clip(lower=0.0).to_numpy(dtype=float)
        grid_out_w = df["real_P_grid_out_W"].clip(lower=0.0).to_numpy(dtype=float)
    grid_signed_kw = (grid_in_w - grid_out_w) / 1000.0

    grid_in_to_storage_w = np.minimum(grid_in_w, p_ch_from_grid_w)
    grid_in_to_load_w = np.maximum(grid_in_w - grid_in_to_storage_w, 0.0)
    grid_out_from_storage_w = np.minimum(grid_out_w, p_dis_to_grid_w)
    grid_out_from_pv_w = np.maximum(grid_out_w - grid_out_from_storage_w, 0.0)

    grid_in_to_load_kw = grid_in_to_load_w / 1000.0
    grid_in_to_storage_kw = grid_in_to_storage_w / 1000.0
    grid_out_from_pv_kw = grid_out_from_pv_w / 1000.0
    grid_out_from_storage_kw = grid_out_from_storage_w / 1000.0

    ax_grid.fill_between(x, 0.0, grid_in_to_load_kw, step="post", color="red", alpha=0.35, label="Load")
    ax_grid.fill_between(
        x,
        grid_in_to_load_kw,
        grid_in_to_load_kw + grid_in_to_storage_kw,
        step="post",
        color="green",
        alpha=0.35,
        label="Storage",
    )
    ax_grid.fill_between(x, -grid_out_from_pv_kw, 0.0, step="post", color=PV_REAL_COLOR, alpha=0.35, label="PV")
    ax_grid.fill_between(
        x,
        -(grid_out_from_pv_kw + grid_out_from_storage_kw),
        -grid_out_from_pv_kw,
        step="post",
        color="green",
        alpha=0.35,
        label="_nolegend_",
    )
    ax_grid.step(x, grid_signed_kw, where="post", color="black", linewidth=1.0, label="Edge")
    ax_grid.axhline(0.0, color="black", linewidth=0.7, alpha=0.7)
    ax_grid.set_ylabel("kW", fontsize=AXIS_LABEL_FONTSIZE)
    ax_grid.set_title("Grid real (+ Import, - Export)", loc="left", fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax_grid.grid(True, alpha=0.3)
    _legend_unique(ax_grid)

    max_abs_kw = max(
        float(np.nanmax(np.abs(arr)))
        for arr in (
            pv_forecast_kw.to_numpy(dtype=float),
            pv_real_kw.to_numpy(dtype=float),
            load_kw_neg.to_numpy(dtype=float),
            battery_signed_kw.to_numpy(dtype=float),
            (dis_to_load_kw + dis_to_grid_kw),
            -(ch_from_grid_kw + ch_from_pv_kw),
            grid_signed_kw,
            (grid_in_to_load_kw + grid_in_to_storage_kw),
            -(grid_out_from_pv_kw + grid_out_from_storage_kw),
        )
    )
    if not np.isfinite(max_abs_kw) or max_abs_kw <= 0.0:
        max_abs_kw = 1.0
    kw_ylim = max_abs_kw * 1.05
    ax_pv.set_ylim(-kw_ylim, kw_ylim)
    ax_bat.set_ylim(-kw_ylim, kw_ylim)
    ax_grid.set_ylim(-kw_ylim, kw_ylim)

    # 4) DA + reBAP prices
    da_price_eur_mwh = df["input_price_EUR_MWh"].to_numpy(dtype=float)
    rebap_price_eur_mwh = df["real_imbalance_price_eur_mwh"].to_numpy(dtype=float)
    da_price_plot = np.clip(da_price_eur_mwh, -500.0, 500.0)
    if is_standard:
        feed_in_tariff_plot = np.full(len(df), STANDARD_FEED_IN_TARIFF_EUR_MWH, dtype=float)
        electricity_price_plot = np.full(len(df), STANDARD_ELECTRICITY_PRICE_EUR_MWH, dtype=float)
        ax_price.step(x, da_price_plot, where="post", label="DA", linewidth=0.9, color=DA_PRICE_COLOR)
        ax_price.step(
            x,
            feed_in_tariff_plot,
            where="post",
            label="Feed-in tariff",
            linewidth=0.9,
            color=STANDARD_FEED_IN_COLOR,
        )
        ax_price.step(
            x,
            electricity_price_plot,
            where="post",
            label="Electricity price",
            linewidth=0.9,
            color=STANDARD_ELECTRICITY_PRICE_COLOR,
        )
        price_stack = np.concatenate([da_price_plot, feed_in_tariff_plot, electricity_price_plot])
    else:
        rebap_price_plot = np.clip(rebap_price_eur_mwh, -500.0, 500.0)
        # `fill_between(..., where=..., step="post")` leaves tiny white gaps at
        # mask changes. Expanding both series to explicit post-step coordinates
        # keeps each 15-minute interval on one side of the comparison mask.
        x_price_step, da_price_step = _expand_post_step(x, da_price_plot)
        _, rebap_price_step = _expand_post_step(x, rebap_price_plot)
        da_higher = da_price_step >= rebap_price_step
        ax_price.fill_between(
            x_price_step,
            da_price_step,
            rebap_price_step,
            where=da_higher,
            color=DA_PRICE_COLOR,
            alpha=0.12,
            label="_nolegend_",
        )
        ax_price.fill_between(
            x_price_step,
            da_price_step,
            rebap_price_step,
            where=~da_higher,
            color=REBAP_PRICE_COLOR,
            alpha=0.12,
            label="_nolegend_",
        )
        ax_price.step(x, da_price_plot, where="post", label="DA", linewidth=0.9, color=DA_PRICE_COLOR)
        ax_price.step(x, rebap_price_plot, where="post", label="reBAP", linewidth=0.9, color=REBAP_PRICE_COLOR)
        price_stack = np.concatenate([da_price_plot, rebap_price_plot])
    ax_price.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=ZERO_LINE_WIDTH, alpha=0.9)
    price_min = float(np.nanmin(price_stack)) if price_stack.size else -500.0
    price_max = float(np.nanmax(price_stack)) if price_stack.size else 500.0
    if not np.isfinite(price_min) or not np.isfinite(price_max):
        price_min, price_max = -500.0, 500.0
    if np.isclose(price_min, price_max):
        pad = max(5.0, abs(price_min) * 0.1 + 1.0)
    else:
        pad = 0.05 * (price_max - price_min)
    y_price_min = max(-500.0, price_min - pad)
    y_price_max = min(500.0, price_max + pad)
    if y_price_min >= y_price_max:
        y_price_min, y_price_max = -500.0, 500.0
    ax_price.set_ylim(y_price_min, y_price_max)
    ax_price.set_ylabel("EUR/MWh", fontsize=AXIS_LABEL_FONTSIZE)
    price_title = "Tariff & Price" if is_standard else "DA & reBAP"
    ax_price.set_title(price_title, loc="left", fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax_price.grid(True, alpha=0.3)
    _legend_unique(ax_price)

    # 5) Revenue split + net
    revenue_step_eur = df["real_revenue_step_eur"].to_numpy(dtype=float)
    real_grid_fee_step_eur = df["real_grid_fee_step_eur"].clip(lower=0.0).to_numpy(dtype=float)
    imbalance_step_eur = df["real_rebap_settlement_step_eur"].to_numpy(dtype=float)
    aging_cost_eur = df["opt_cycle_aging_cost_step_eur"].to_numpy(dtype=float)
    net_after_aging_eur = df["real_revenue_step_net_after_aging_eur"].to_numpy(dtype=float)

    imb_pos = np.maximum(imbalance_step_eur, 0.0)
    imb_neg = np.minimum(imbalance_step_eur, 0.0)
    aging_neg = -np.abs(aging_cost_eur)

    neg_base = np.zeros(len(df), dtype=float)
    if is_standard:
        revenue_pos = np.maximum(revenue_step_eur, 0.0)
        revenue_neg = np.minimum(revenue_step_eur, 0.0)
        revenue_neg_bottom = neg_base + revenue_neg
        ax_rev.fill_between(
            x,
            neg_base,
            revenue_neg_bottom,
            step="post",
            color=STANDARD_ELECTRICITY_PRICE_COLOR,
            alpha=0.45,
            label="Electricity price",
        )
        neg_base = revenue_neg_bottom
        positive_top = revenue_pos
    else:
        # `real_revenue_step_eur` already nets DA energy and real grid fee.
        # For readability, split it back into a pure DA component and a separate grid fee cost.
        da_energy_step_eur = revenue_step_eur + real_grid_fee_step_eur
        da_energy_pos = np.maximum(da_energy_step_eur, 0.0)
        da_energy_neg = np.minimum(da_energy_step_eur, 0.0)
        grid_fee_neg = -np.abs(real_grid_fee_step_eur)

        da_energy_neg_bottom = neg_base + da_energy_neg
        ax_rev.fill_between(
            x,
            neg_base,
            da_energy_neg_bottom,
            step="post",
            color=DA_PRICE_COLOR,
            alpha=0.45,
            label="DA",
        )
        neg_base = da_energy_neg_bottom

        grid_fee_bottom = neg_base + grid_fee_neg
        ax_rev.fill_between(
            x,
            neg_base,
            grid_fee_bottom,
            step="post",
            color=REBAP_COLOR,
            alpha=0.45,
            label="Grid fee",
        )
        neg_base = grid_fee_bottom

        imb_neg_bottom = neg_base + imb_neg
        ax_rev.fill_between(
            x,
            neg_base,
            imb_neg_bottom,
            step="post",
            color=REBAP_PRICE_COLOR,
            alpha=0.45,
            label="reBAP",
        )
        neg_base = imb_neg_bottom
        pos_base = np.zeros(len(df), dtype=float)
        da_energy_pos_top = pos_base + da_energy_pos
        ax_rev.fill_between(
            x,
            pos_base,
            da_energy_pos_top,
            step="post",
            color=DA_PRICE_COLOR,
            alpha=0.45,
            label="_nolegend_",
        )
        pos_base = da_energy_pos_top
        positive_top = pos_base + imb_pos
        ax_rev.fill_between(
            x,
            pos_base,
            positive_top,
            step="post",
            color=REBAP_PRICE_COLOR,
            alpha=0.45,
            label="_nolegend_",
        )

    aging_bottom = neg_base + aging_neg
    ax_rev.fill_between(x, neg_base, aging_bottom, step="post", color=AGING_COLOR, alpha=0.45, label="Aging")

    if is_standard:
        ax_rev.fill_between(
            x,
            np.zeros(len(df), dtype=float),
            positive_top,
            step="post",
            color=STANDARD_FEED_IN_COLOR,
            alpha=0.45,
            label="Feed-in tariff",
        )

    ax_rev.step(x, net_after_aging_eur, where="post", color="black", linewidth=1.0, label="Net")
    ax_rev.axhline(0.0, color="black", linewidth=0.7, alpha=0.7)
    stack_min = float(np.nanmin(np.minimum(aging_bottom, 0.0))) if aging_bottom.size else 0.0
    stack_max = float(np.nanmax(np.maximum(positive_top, 0.0))) if positive_top.size else 0.0
    if not np.isfinite(stack_min) or not np.isfinite(stack_max):
        stack_min, stack_max = -1.0, 1.0
    if np.isclose(stack_min, stack_max):
        pad = max(0.5, abs(stack_min) * 0.1 + 0.1)
    else:
        pad = 0.05 * (stack_max - stack_min)
    ax_rev.set_ylim(stack_min - pad, stack_max + pad)
    ax_rev.set_ylabel("EUR/step", fontsize=AXIS_LABEL_FONTSIZE)
    ax_rev.set_title("Revenue (+ Revenue, - Costs)", loc="left", fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax_rev.grid(True, alpha=0.3)
    if is_standard:
        handles, labels = ax_rev.get_legend_handles_labels()
        handle_map = {label: handle for handle, label in zip(handles, labels) if label and label != "_nolegend_"}
        ordered_labels = ["Feed-in tariff", "Electricity price", "Aging", "Net"]
        ordered_handles = [handle_map[label] for label in ordered_labels if label in handle_map]
        _legend_unique(ax_rev, ordered_handles, ordered_labels)
    else:
        handles, labels = ax_rev.get_legend_handles_labels()
        handle_map = {label: handle for handle, label in zip(handles, labels) if label and label != "_nolegend_"}
        ordered_labels = ["DA", "Grid fee", "reBAP", "Aging", "Net"]
        ordered_handles = [handle_map[label] for label in ordered_labels if label in handle_map]
        _legend_unique(ax_rev, ordered_handles, ordered_labels)

    locator = mdates.AutoDateLocator(minticks=8, maxticks=20)
    ax_rev.xaxis.set_major_locator(locator)
    ax_rev.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    plt.setp(ax_rev.get_xticklabels(), rotation=25, ha="right")
    ax_rev.set_xlabel("Time", fontsize=AXIS_LABEL_FONTSIZE)

    _draw_day_markers(axes, x, scenario=scenario, day=day)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax_soc.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    fig.tight_layout(rect=(0.0, 0.0, 0.965, 1.0))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=EXPORT_DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    saved = 0
    for scenario in SCENARIOS:
        path = DISPATCH_FILES[scenario]
        df = load_dispatch(path)
        days = _scenario_days(df)
        if not days:
            print(f"[{scenario}] no days found in plot window, skipped.")
            continue

        scenario_output_dir = OUTPUT_ROOT / scenario
        scenario_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{scenario}] exporting {len(days)} daily plots...")

        for day in days:
            day_df = _filter_day(df, day)
            if day_df.empty:
                continue
            out_file = scenario_output_dir / f"{day:%Y%m%d}_{scenario}_dispatch.png"
            plot_scenario_day(day_df, scenario=scenario, day=day, out_file=out_file)
            saved += 1
            print(f"Saved: {out_file}")

    print(f"Done. Saved {saved} PNGs in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
