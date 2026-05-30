"""
Plot optimization revenue sums for days 2..91 from scenario_variable_sums_days_2_91.csv.

Output:
- reports/optimization/optimization_waterfall_revenue.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_OVERALL = PROJECT_ROOT / "configs" / "config_overall.yaml"
INPUT_CSV = PROJECT_ROOT / "data" / "optimization" / "scenario_variable_sums_days_2_91.csv"
REPORTS_DIR = PROJECT_ROOT / "reports" / "optimization"
OUT_PNG = REPORTS_DIR / "optimization_waterfall_revenue.png"
FIGSIZE_A4_HALF_PAGE_INCH = (8.27, 5.0)
FONT_SIZE_ANNOTATION = 8.8
FONT_SIZE_XTICKS = 11.0
FONT_SIZE_YTICKS = 10.5
FONT_SIZE_AXIS_LABEL = 12.0
FONT_SIZE_LEGEND = 10.0
EURO_SYMBOL = "\u20ac"
STANDARD_PROCUREMENT_FACE = "#D9D9D9"
STANDARD_PROCUREMENT_EDGE = "#666666"
STANDARD_PROCUREMENT_HATCH = "++"

MODEL_OUTPUT_ORDER = ("ideal", "standard", "physical", "source", "target", "transfer")
SCENARIO_TO_COLOR_KEY = {
    "physical": "physical_model",
    "source": "source_model",
    "target": "target_model",
    "transfer": "transfer_model",
    "standard": "standard_model",
    "ideal": "ideal_model",
}
SCENARIO_FALLBACK_COLORS = {
    "ideal": "black",
    "physical": "tab:orange",
    "source": "tab:green",
    "target": "tab:red",
    "transfer": "tab:blue",
    "standard": "tab:gray",
}
SCENARIO_LABELS = {
    "ideal": "Ideal",
    "physical": "Physical",
    "source": "Source",
    "target": "Target",
    "transfer": "Transfer",
    "standard": "Standard",
}


def load_model_colors(path: Path = CONFIG_OVERALL) -> dict:
    """Load model colors from config_overall.yaml."""
    if not path.exists():
        raise FileNotFoundError(f"Overall config not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    colors = cfg.get("model_colors", {})
    if not isinstance(colors, dict):
        raise ValueError("config_overall.yaml: 'model_colors' must be a dictionary.")
    return colors


def scenario_color(scenario: str, model_colors: dict) -> str:
    """Return the configured plot color for a scenario."""
    key = SCENARIO_TO_COLOR_KEY.get(scenario)
    if key is None:
        return str(SCENARIO_FALLBACK_COLORS.get(scenario, "black"))
    return str(model_colors.get(key, SCENARIO_FALLBACK_COLORS.get(scenario, "black")))


def load_sum_table(path: Path = INPUT_CSV) -> pd.DataFrame:
    """Load the scenario sum table from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Scenario sums CSV not found: {path}")
    df = pd.read_csv(path, sep=";", decimal=",")
    required_cols = {"python_variable", *MODEL_OUTPUT_ORDER}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df


def values_for_variable(table: pd.DataFrame, python_variable: str, scenarios: tuple[str, ...]) -> np.ndarray:
    """Extract scenario totals for one variable."""
    row = table.loc[table["python_variable"] == python_variable]
    if row.empty:
        raise ValueError(f"Variable '{python_variable}' not found in {INPUT_CSV}.")
    values = [pd.to_numeric(pd.Series([row.iloc[0][scenario]]), errors="coerce").iloc[0] for scenario in scenarios]
    return np.array(values, dtype=float)


def format_number(value: float, *, signed: bool = False, suffix: str = "") -> str:
    """Format numbers without decimals using German separators."""
    format_spec = "+,.0f" if signed else ",.0f"
    formatted = format(value, format_spec)
    formatted = formatted.replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{formatted}{suffix}"


def export_waterfall_png(table: pd.DataFrame, out_path: Path = OUT_PNG) -> Path:
    """Export the optimization revenue waterfall plot."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_colors = load_model_colors(CONFIG_OVERALL)
    scenarios = MODEL_OUTPUT_ORDER

    real_revenue_sum = values_for_variable(table, "real_revenue_step_eur", scenarios)                   # geplanter DA-Fahrplanerlös minus reale Grid Fee
    aging_cost_sum = values_for_variable(table, "real_cycle_aging_cost_step_eur", scenarios)            # entspricht dem geplanten Aging
    rebap_pos = values_for_variable(table, "real_rebap_revenue_step_eur", scenarios)
    rebap_neg = values_for_variable(table, "real_rebap_penalty_step_eur", scenarios)
    energy_revenue_sum = real_revenue_sum - aging_cost_sum                                              # DA + reale Grid Fee + Aging, ohne Imbalance
    real_net_sum = values_for_variable(table, "real_revenue_step_net_after_aging_eur", scenarios)       # enthält DA/Fahrplanerlös, reale Grid Fee, Imbalance-Settlement und Aging

    labels = [SCENARIO_LABELS.get(scenario, scenario) for scenario in scenarios]
    colors = [scenario_color(scenario, model_colors) for scenario in scenarios]

    eur_formatter = FuncFormatter(lambda v, _: format_number(v, suffix=f" {EURO_SYMBOL}"))
    waterfall_points: list[float] = [0.0]
    for energy_sum, pos_sum, neg_sum, real_sum in zip(
        energy_revenue_sum,
        rebap_pos,
        rebap_neg,
        real_net_sum,
    ):
        waterfall_points.extend(
            [
                energy_sum,
                energy_sum + pos_sum,
                energy_sum + pos_sum + neg_sum,
                real_sum,
            ]
        )
    all_values = np.array(waterfall_points, dtype=float)
    finite_vals = all_values[np.isfinite(all_values)]
    if finite_vals.size == 0:
        y_min, y_max = -1.0, 1.0
    else:
        y_min = float(finite_vals.min())
        y_max = float(finite_vals.max())
        if np.isclose(y_min, y_max):
            pad = max(1.0, abs(y_min) * 0.1)
            y_min -= pad
            y_max += pad
        else:
            pad = 0.10 * (y_max - y_min)
            y_min -= pad
            y_max += pad

    def _annotate_value(ax: plt.Axes, x_pos: float, y_value: float, value: float, *, signed: bool = False) -> None:
        offset = 0.015 * max(1.0, y_max - y_min)
        if np.isclose(value, 0.0) or not np.isfinite(value):
            return
        y_pos = y_value + offset if value >= 0 else y_value - offset
        va = "bottom" if value >= 0 else "top"
        label = format_number(value, signed=signed)
        ax.text(x_pos, y_pos, label, ha="center", va=va, fontsize=FONT_SIZE_ANNOTATION)

    fig, ax = plt.subplots(figsize=FIGSIZE_A4_HALF_PAGE_INCH, dpi=300)
    ax.set_axisbelow(True)
    ax.axhline(0.0, color="#222222", linewidth=0.9, zorder=1)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, alpha=0.30, zorder=0)
    ax.yaxis.set_major_formatter(eur_formatter)
    ax.tick_params(axis="x", labelsize=FONT_SIZE_XTICKS)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_YTICKS)

    cluster_centers = np.arange(len(scenarios), dtype=float) * 1.55
    offsets = np.array([-0.36, -0.12, 0.12, 0.36], dtype=float)
    bar_width = 0.16

    legend_handles = [
        Rectangle((0.0, 0.0), 1.0, 1.0, facecolor="#777777", edgecolor="#555555", linewidth=0.8, alpha=0.95),
        Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            facecolor=STANDARD_PROCUREMENT_FACE,
            edgecolor=STANDARD_PROCUREMENT_EDGE,
            linewidth=0.9,
            hatch=STANDARD_PROCUREMENT_HATCH,
            alpha=0.95,
        ),
        Rectangle((0.0, 0.0), 1.0, 1.0, facecolor="#777777", edgecolor="#555555", linewidth=0.9, hatch="..", alpha=0.18),
        Rectangle((0.0, 0.0), 1.0, 1.0, facecolor="#777777", edgecolor="#555555", linewidth=0.9, hatch="////", alpha=0.18),
        Rectangle((0.0, 0.0), 1.0, 1.0, facecolor="#777777", edgecolor="#111111", linewidth=1.2, alpha=0.95),
    ]
    legend_labels = [
        "Settled DA energy revenue after aging",
        "Standard tariff / energy revenue",
        "Imbalance random profit",
        "Imbalance penalty",
        "Net revenue",
    ]

    for idx, (scenario, label, color, energy_sum, pos_sum, neg_sum, real_sum) in enumerate(
        zip(
            scenarios,
            labels,
            colors,
            energy_revenue_sum,
            rebap_pos,
            rebap_neg,
            real_net_sum,
        )
    ):
        center = cluster_centers[idx]
        x_energy, x_pos, x_neg, x_net = center + offsets
        top_after_energy = energy_sum
        top_after_profit = top_after_energy + pos_sum
        top_after_penalty = real_sum
        is_standard = scenario == "standard"

        energy_facecolor = STANDARD_PROCUREMENT_FACE if is_standard else color
        energy_edgecolor = STANDARD_PROCUREMENT_EDGE if is_standard else color
        energy_hatch = STANDARD_PROCUREMENT_HATCH if is_standard else None
        energy_alpha = 0.95 if is_standard else 0.30
        energy_linewidth = 0.9 if is_standard else 0.7

        ax.bar(
            x_energy,
            energy_sum,
            width=bar_width,
            color=energy_facecolor,
            edgecolor=energy_edgecolor,
            linewidth=energy_linewidth,
            hatch=energy_hatch,
            alpha=energy_alpha,
            zorder=3,
        )
        ax.bar(x_pos, pos_sum, bottom=top_after_energy, width=bar_width, color=color, edgecolor=color, linewidth=0.9, hatch="..", alpha=0.18, zorder=4)
        ax.bar(x_neg, neg_sum, bottom=top_after_profit, width=bar_width, color=color, edgecolor=color, linewidth=0.9, hatch="////", alpha=0.18, zorder=4)
        ax.bar(x_net, real_sum, width=bar_width, color=color, edgecolor="#111111", linewidth=1.3, alpha=1.0, zorder=5)

        connector_color = "#555555"
        ax.plot([x_pos + bar_width / 2.0, x_neg - bar_width / 2.0], [top_after_profit, top_after_profit], color=connector_color, linewidth=0.9, alpha=0.75, zorder=2)
        ax.scatter([x_pos, x_neg], [top_after_profit, top_after_penalty], s=12, color=color, edgecolors="white", linewidths=0.5, zorder=6)

        _annotate_value(ax, x_energy, energy_sum, energy_sum)
        _annotate_value(ax, x_pos, top_after_profit, pos_sum, signed=True)
        _annotate_value(ax, x_neg, top_after_penalty, neg_sum, signed=True)
        _annotate_value(ax, x_net, real_sum, real_sum, signed=True)

    ax.set_xlim(cluster_centers[0] - 0.75, cluster_centers[-1] + 0.75)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(cluster_centers)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Total [{EURO_SYMBOL}]", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.legend(
        legend_handles,
        legend_labels,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005, 1.0, 0.10),
        mode="expand",
        ncol=3,
        fontsize=FONT_SIZE_LEGEND,
        frameon=True,
        handlelength=1.5,
        borderaxespad=0.0,
        columnspacing=1.3,
        borderpad=0.6,
        labelspacing=0.5,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out_path


def main() -> None:
    """Load the scenario sums CSV and export the revenue waterfall plot."""
    table = load_sum_table(INPUT_CSV)
    out_path = export_waterfall_png(table, OUT_PNG)
    print(f"Saved PNG: {out_path}")
    print(f"Rows: {len(table)}")
    print(f"Columns: {len(table.columns)}")


if __name__ == "__main__":
    main()
