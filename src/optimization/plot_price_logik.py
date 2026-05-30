"""
Erzeugt die PNG-Tabellen fuer die Preislogik-Summary.

Dieses Skript ist bewusst vom eigentlichen CSV-Aufbau getrennt. Es liest die
fertige Summary entweder direkt als DataFrame oder ueber die vorhandene CSV-Datei
und exportiert daraus die acht PNGs:

- ein Overview-PNG
- je ein PNG fuer `physical`, `source`, `target` und `transfer`
- ein Vergleichs-PNG fuer alle Per-step-KPIs
- ein Vergleichs-PNG fuer alle KPIs in Euro pro kWh
- ein Vergleichs-PNG fuer alle Impacts in Euro
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPTIMIZATION_DIR = PROJECT_ROOT / "data" / "optimization"
REPORTS_DIR = PROJECT_ROOT / "reports" / "optimization"

MODELS = ("physical", "source", "target", "transfer")

SUMMARY_CSV = OPTIMIZATION_DIR / "dispatch_combination_summary.csv"
SUMMARY_OVERVIEW_PNG = REPORTS_DIR / "dispatch_combination_summary_overview.png"
PER_STEP_COMPARISON_PNG = REPORTS_DIR / "dispatch_combination_per_step_comparison.png"
PER_KWH_COMPARISON_PNG = REPORTS_DIR / "dispatch_combination_per_kwh_comparison.png"
IMPACT_COMPARISON_PNG = REPORTS_DIR / "dispatch_combination_impact_comparison.png"
SUMMARY_MODEL_PNGS = {
    "physical": REPORTS_DIR / "dispatch_combination_summary_physical.png",
    "source": REPORTS_DIR / "dispatch_combination_summary_source.png",
    "target": REPORTS_DIR / "dispatch_combination_summary_target.png",
    "transfer": REPORTS_DIR / "dispatch_combination_summary_transfer.png",
}

TABLE_DPI = 300
EMPTY_NOTICE_FONTSIZE = 21.0
TABLE_HEADER_FONTSIZE = 18.2
TABLE_CELL_FONTSIZE = 17.0
TABLE_CMAP = LinearSegmentedColormap.from_list(
    "summary_rwg",
    ["#b00020", "#ffffff", "#1f9e5a"],
)
GRAY_CMAP = LinearSegmentedColormap.from_list(
    "summary_gray",
    ["#ffffff", "#b8b8b8"],
)

TABLE_COL_WIDTHS = {
    "classification": 0.92,
    "forecast_state": 0.88,
    "da_price_sign": 0.58,
    "imbalance_price_sign": 0.58,
    "da_vs_imb": 1.12,
    "planned_grid_state": 0.78,
    "realized_grid_state": 0.78,
    "step_count": 0.86,
    "energy_kwh": 0.92,
    "kwh_per_step": 1.06,
    "eur_per_step": 1.18,
    "eur_per_kwh": 1.18,
    "impact_total_eur": 1.18,
    "physical_eur_per_step": 0.88,
    "source_eur_per_step": 0.88,
    "target_eur_per_step": 0.88,
    "transfer_eur_per_step": 0.88,
    "physical_impact_total_eur": 0.92,
    "source_impact_total_eur": 0.92,
    "target_impact_total_eur": 0.92,
    "transfer_impact_total_eur": 0.92,
    "impact_delta_grid_fee_eur": 1.28,
    "impact_rebap_settlement_eur": 1.20,
    "impact_imbalance_da_reference_eur": 1.20,
    "impact_imbalance_spread_eur": 1.20,
}

DISPLAY_TEXT_MAPS = {
    "forecast_state": {"over": "Over", "under": "Under"},
    "da_price_sign": {"neg": "-", "pos": "+"},
    "imbalance_price_sign": {"neg": "-", "pos": "+"},
    "da_vs_imb": {"DA_gt_imb": "DA > Imb", "DA_lt_imb": "DA < Imb"},
    "planned_grid_state": {
        "export": "Exp",
        "import": "Imp",
        "no_flow": "None",
    },
    "realized_grid_state": {
        "export": "Exp",
        "import": "Imp",
        "no_flow": "None",
    },
}

SUMMARY_PLOT_BASE_COLUMNS = [
    "classification",
    "forecast_state",
    "da_price_sign",
    "imbalance_price_sign",
    "da_vs_imb",
    "planned_grid_state",
    "realized_grid_state",
]


def load_summary_table(path: Path = SUMMARY_CSV) -> pd.DataFrame:
    """Liest die fertige Preislogik-Summary aus der CSV-Datei."""

    return pd.read_csv(path, sep=";", decimal=",")


def build_diverging_norm(values: np.ndarray) -> TwoSlopeNorm:
    """Erzeugt eine symmetrische Rot-Weiss-Gruen-Norm um den Nullpunkt."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    span = float(np.max(np.abs(finite)))
    span = max(span, 1e-9)
    return TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)


def build_sequential_norm(values: np.ndarray) -> Normalize:
    """Erzeugt eine Weiss-zu-Grau-Norm fuer rein nichtnegative Kennzahlen."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    vmax = max(float(np.max(finite)), 1e-9)
    return Normalize(vmin=0.0, vmax=vmax)


def format_decimal_comma(value: float, decimals: int) -> str:
    """Formatiert eine Zahl mit deutschem Dezimalkomma fuer die Plotbeschriftung."""

    if not np.isfinite(value):
        return ""
    return f"{value:.{decimals}f}".replace(".", ",")


def format_plot_text(column: str, value: object) -> str:
    """Formatiert nicht-numerische Tabellenzellen fuer die PNG-Ausgabe."""

    if column == "classification":
        return str(value).replace(".", "")
    mapping = DISPLAY_TEXT_MAPS.get(column)
    if mapping is None:
        return str(value)
    return mapping.get(str(value), str(value).replace("_", " "))


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    """Divides two numeric series and returns NaN when the denominator is zero."""

    numerator_values = pd.to_numeric(numerator, errors="coerce").to_numpy(dtype=float)
    denominator_values = pd.to_numeric(denominator, errors="coerce").to_numpy(dtype=float)
    return np.divide(
        numerator_values,
        denominator_values,
        out=np.full(len(numerator_values), np.nan, dtype=float),
        where=denominator_values > 0.0,
    )


def get_plot_header_labels(
    count_col: str,
    energy_col: str,
    kwh_per_step_col: str,
    eur_per_step_col: str,
    ratio_label: str,
    impact_total_col: str,
    impact_delta_grid_fee_col: str,
    impact_imbalance_cash_col: str,
    impact_imbalance_da_reference_col: str,
    impact_imbalance_spread_col: str,
) -> dict[str, str]:
    """Liefert die kurzen sichtbaren Spaltennamen fuer die PNG-Tabelle."""

    return {
        "classification": "Class",
        "forecast_state": "Forecast",
        "da_price_sign": "DA",
        "imbalance_price_sign": "Imb",
        "da_vs_imb": "DA vs Imb",
        "planned_grid_state": "Plan",
        "realized_grid_state": "Real",
        "step_count": "Steps",
        count_col: "Steps",
        energy_col: "kWh",
        kwh_per_step_col: "kWh/step",
        eur_per_step_col: ratio_label,
        impact_total_col: "Impact [€]",
        impact_delta_grid_fee_col: "Grd fee [€]",
        impact_imbalance_cash_col: "Imb cash [€]",
        impact_imbalance_da_reference_col: "DA ref [€]",
        impact_imbalance_spread_col: "Spread [€]",
    }


def build_plot_frame(
    summary: pd.DataFrame,
    count_col: str,
    energy_col: str,
    kwh_per_step_col: str,
    denominator_col: str,
    impact_total_col: str,
    impact_delta_grid_fee_col: str,
    impact_imbalance_cash_col: str,
    impact_imbalance_da_reference_col: str,
    impact_imbalance_spread_col: str,
    eur_per_step_col: str,
    sort_col: str | None = None,
) -> pd.DataFrame:
    """Bereitet die nichtleeren Summary-Zeilen fuer einen einzelnen Plot auf."""

    plot_df = summary.loc[
        summary[count_col] > 0,
        [
            *SUMMARY_PLOT_BASE_COLUMNS,
            count_col,
            energy_col,
            impact_total_col,
            impact_delta_grid_fee_col,
            impact_imbalance_cash_col,
            impact_imbalance_da_reference_col,
            impact_imbalance_spread_col,
        ],
    ].copy()

    plot_df[kwh_per_step_col] = safe_divide(plot_df[energy_col], plot_df[count_col])
    plot_df[eur_per_step_col] = safe_divide(plot_df[impact_total_col], plot_df[denominator_col])

    ordered_columns = [
        *SUMMARY_PLOT_BASE_COLUMNS,
        count_col,
        energy_col,
        kwh_per_step_col,
        eur_per_step_col,
        impact_total_col,
        impact_delta_grid_fee_col,
        impact_imbalance_cash_col,
        impact_imbalance_da_reference_col,
        impact_imbalance_spread_col,
    ]
    plot_df = plot_df[ordered_columns]
    sort_key = sort_col or eur_per_step_col
    plot_df.sort_values(sort_key, ascending=True, kind="mergesort", inplace=True)
    plot_df.reset_index(drop=True, inplace=True)
    return plot_df


def build_per_step_comparison_frame(summary: pd.DataFrame) -> pd.DataFrame:
    """Build a comparison table with overview and per-model KPI values per step."""

    plot_df = summary.loc[summary["step_count"] > 0, SUMMARY_PLOT_BASE_COLUMNS].copy()

    step_count = pd.to_numeric(summary.loc[plot_df.index, "step_count"], errors="coerce")
    total_impact = pd.to_numeric(summary.loc[plot_df.index, "impact_total_eur"], errors="coerce")
    plot_df["step_count"] = step_count
    plot_df["overview_eur_per_step"] = safe_divide(total_impact, step_count)

    for model in MODELS:
        model_count = pd.to_numeric(
            summary.loc[plot_df.index, f"{model}_step_count"],
            errors="coerce",
        )
        model_total = pd.to_numeric(
            summary.loc[plot_df.index, f"{model}_impact_total_eur"],
            errors="coerce",
        )
        plot_df[f"{model}_eur_per_step"] = safe_divide(model_total, model_count)

    ordered_columns = [
        *SUMMARY_PLOT_BASE_COLUMNS,
        "step_count",
        "overview_eur_per_step",
        *[f"{model}_eur_per_step" for model in MODELS],
    ]
    plot_df = plot_df[ordered_columns]
    plot_df.sort_values("overview_eur_per_step", ascending=True, kind="mergesort", inplace=True)
    plot_df.reset_index(drop=True, inplace=True)
    return plot_df


def build_per_kwh_comparison_frame(summary: pd.DataFrame) -> pd.DataFrame:
    """Build a comparison table with overview and per-model KPI values per kWh."""

    plot_df = summary.loc[summary["step_count"] > 0, SUMMARY_PLOT_BASE_COLUMNS].copy()

    plot_df["step_count"] = pd.to_numeric(summary.loc[plot_df.index, "step_count"], errors="coerce")
    total_impact = pd.to_numeric(summary.loc[plot_df.index, "impact_total_eur"], errors="coerce")
    plot_df["energy_kwh"] = pd.to_numeric(summary.loc[plot_df.index, "energy_kwh"], errors="coerce")
    plot_df["kwh_per_step"] = safe_divide(plot_df["energy_kwh"], plot_df["step_count"])
    plot_df["overall_eur_per_kwh"] = safe_divide(total_impact, plot_df["energy_kwh"])

    for model in MODELS:
        model_total = pd.to_numeric(
            summary.loc[plot_df.index, f"{model}_impact_total_eur"],
            errors="coerce",
        )
        model_energy = pd.to_numeric(
            summary.loc[plot_df.index, f"{model}_energy_kwh"],
            errors="coerce",
        )
        plot_df[f"{model}_eur_per_kwh"] = safe_divide(model_total, model_energy)

    ordered_columns = [
        *SUMMARY_PLOT_BASE_COLUMNS,
        "step_count",
        "energy_kwh",
        "kwh_per_step",
        "overall_eur_per_kwh",
        *[f"{model}_eur_per_kwh" for model in MODELS],
    ]
    plot_df = plot_df[ordered_columns]
    plot_df.sort_values("overall_eur_per_kwh", ascending=True, kind="mergesort", inplace=True)
    plot_df.reset_index(drop=True, inplace=True)
    return plot_df


def build_impact_comparison_frame(summary: pd.DataFrame) -> pd.DataFrame:
    """Build a comparison table with overview and per-model impact sums."""

    plot_df = summary.loc[summary["step_count"] > 0, SUMMARY_PLOT_BASE_COLUMNS].copy()
    plot_df["step_count"] = pd.to_numeric(summary.loc[plot_df.index, "step_count"], errors="coerce")
    plot_df["energy_kwh"] = pd.to_numeric(summary.loc[plot_df.index, "energy_kwh"], errors="coerce")
    plot_df["kwh_per_step"] = safe_divide(plot_df["energy_kwh"], plot_df["step_count"])
    plot_df["impact_total_eur"] = pd.to_numeric(summary.loc[plot_df.index, "impact_total_eur"], errors="coerce")
    for model in MODELS:
        plot_df[f"{model}_impact_total_eur"] = pd.to_numeric(
            summary.loc[plot_df.index, f"{model}_impact_total_eur"],
            errors="coerce",
        )

    columns = [
        *SUMMARY_PLOT_BASE_COLUMNS,
        "step_count",
        "energy_kwh",
        "kwh_per_step",
        "impact_total_eur",
        *[f"{model}_impact_total_eur" for model in MODELS],
    ]
    plot_df = plot_df[columns]
    plot_df.sort_values("impact_total_eur", ascending=True, kind="mergesort", inplace=True)
    plot_df.reset_index(drop=True, inplace=True)
    return plot_df


def export_empty_summary_png(output_path: Path) -> None:
    """Schreibt ein Platzhalterbild, falls es fuer einen Plot keine Zeilen gibt."""

    fig, ax = plt.subplots(figsize=(14, 3), dpi=TABLE_DPI)
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        "No nonzero rows available for this export.",
        ha="center",
        va="center",
        fontsize=EMPTY_NOTICE_FONTSIZE,
        color="#333333",
        transform=ax.transAxes,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def export_table_png(
    table: pd.DataFrame,
    output_path: Path,
    columns: list[str],
    header_labels: dict[str, str],
    numeric_decimals: dict[str, int],
    heatmap_columns: list[str],
    count_bar_columns: list[str],
    heatmap_norms: dict[str, Normalize | TwoSlopeNorm] | None = None,
    heatmap_cmaps: dict[str, LinearSegmentedColormap] | None = None,
    bar_fill_colors: dict[str, str] | None = None,
) -> None:
    """Draw an already prepared comparison table as a PNG."""

    if table.empty:
        export_empty_summary_png(output_path)
        return

    widths = [
        TABLE_COL_WIDTHS.get(col, TABLE_COL_WIDTHS.get(col.split("_", 1)[-1], 1.3))
        for col in columns
    ]

    total_width = float(sum(widths))
    n_rows = len(table)
    fig_width = max(22.0, total_width * 1.74)
    fig_height = max(15.8, 2.45 + 0.68 * (n_rows + 1))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=TABLE_DPI)

    ax.set_xlim(0.0, total_width)
    ax.set_ylim(n_rows + 1.3, -0.25)
    ax.axis("off")

    count_maxima: dict[str, float] = {}
    for count_col in count_bar_columns:
        count_values = pd.to_numeric(table[count_col], errors="coerce").to_numpy(dtype=float)
        count_maxima[count_col] = max(1.0, float(np.nanmax(count_values)))

    value_norms = {}
    for column in heatmap_columns:
        if heatmap_norms is not None and column in heatmap_norms:
            value_norms[column] = heatmap_norms[column]
            continue
        value_norms[column] = build_diverging_norm(
            pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        )

    x_positions = [0.0]
    for width in widths[:-1]:
        x_positions.append(x_positions[-1] + width)

    header_face = "#d9e6f5"
    grid_color = "#b8c6d1"
    row_alt_face = "#f6f7f9"
    row_base_face = "#ffffff"
    count_bar_face = "#5a8fd0"
    energy_bar_face = "#d9b300"
    text_color = "#111111"

    for x0, width, column in zip(x_positions, widths, columns):
        ax.add_patch(
            Rectangle(
                (x0, 0.0),
                width,
                1.0,
                facecolor=header_face,
                edgecolor=grid_color,
                linewidth=0.8,
                zorder=1,
            )
        )
        ax.text(
            x0 + width / 2.0,
            0.55,
            header_labels.get(column, column),
            ha="center",
            va="center",
            fontsize=TABLE_HEADER_FONTSIZE,
            fontweight="bold",
            color=text_color,
            zorder=4,
        )

    for row_idx, row in table.iterrows():
        y0 = row_idx + 1.0
        row_face = row_alt_face if row_idx % 2 else row_base_face

        for x0, width, column in zip(x_positions, widths, columns):
            facecolor = row_face
            alpha = 1.0

            if column in value_norms:
                value = float(
                    pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
                )
                if np.isfinite(value):
                    cell_cmap = TABLE_CMAP
                    if heatmap_cmaps is not None:
                        cell_cmap = heatmap_cmaps.get(column, cell_cmap)
                    facecolor = cell_cmap(value_norms[column](value))
                    alpha = 0.35 if abs(value) <= 1e-12 else 0.50
                else:
                    facecolor = "#ffffff"
                    alpha = 1.0

            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    width,
                    1.0,
                    facecolor=facecolor,
                    edgecolor=grid_color,
                    linewidth=0.6,
                    alpha=alpha,
                    zorder=1,
                )
            )

            if column in count_bar_columns:
                count_value = float(row[column])
                bar_width = (width - 0.16) * count_value / count_maxima[column]
                if bar_width > 0.0:
                    bar_face = count_bar_face
                    if bar_fill_colors is not None:
                        bar_face = bar_fill_colors.get(column, bar_face)
                    elif column == "energy_kwh" or column.endswith("_energy_kwh"):
                        bar_face = energy_bar_face
                    ax.add_patch(
                        Rectangle(
                            (x0 + 0.08, y0 + 0.15),
                            bar_width,
                            0.70,
                            facecolor=bar_face,
                            edgecolor="none",
                            alpha=0.92,
                            zorder=2,
                        )
                    )

            value = row[column]
            if column in numeric_decimals:
                text = format_decimal_comma(float(value), numeric_decimals[column])
                ha = "right"
                x_text = x0 + width - 0.08
                fontweight = "normal"
            elif column in count_bar_columns:
                text = f"{int(value)}"
                ha = "right"
                x_text = x0 + width - 0.08
                fontweight = "normal"
            else:
                text = format_plot_text(column, value)
                ha = "left"
                x_text = x0 + 0.08
                fontweight = "bold" if column == "classification" else "normal"

            ax.text(
                x_text,
                y0 + 0.54,
                text,
                ha=ha,
                va="center",
                fontsize=TABLE_CELL_FONTSIZE,
                fontweight=fontweight,
                color=text_color,
                zorder=4,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def export_summary_table_png(
    table: pd.DataFrame,
    output_path: Path,
    count_col: str,
    energy_col: str,
    kwh_per_step_col: str,
    eur_per_step_col: str,
    ratio_label: str,
    impact_total_col: str,
    impact_delta_grid_fee_col: str,
    impact_imbalance_cash_col: str,
    impact_imbalance_da_reference_col: str,
    impact_imbalance_spread_col: str,
    shared_impact_scale: bool = False,
) -> None:
    """Zeichnet eine vorbereitete Summary-Tabelle als PNG."""

    columns = [
        "classification",
        "forecast_state",
        "da_price_sign",
        "imbalance_price_sign",
        "da_vs_imb",
        "planned_grid_state",
        "realized_grid_state",
        count_col,
        energy_col,
        kwh_per_step_col,
        eur_per_step_col,
        impact_total_col,
        impact_delta_grid_fee_col,
        impact_imbalance_cash_col,
        impact_imbalance_da_reference_col,
        impact_imbalance_spread_col,
    ]
    header_labels = get_plot_header_labels(
        count_col=count_col,
        energy_col=energy_col,
        kwh_per_step_col=kwh_per_step_col,
        eur_per_step_col=eur_per_step_col,
        ratio_label=ratio_label,
        impact_total_col=impact_total_col,
        impact_delta_grid_fee_col=impact_delta_grid_fee_col,
        impact_imbalance_cash_col=impact_imbalance_cash_col,
        impact_imbalance_da_reference_col=impact_imbalance_da_reference_col,
        impact_imbalance_spread_col=impact_imbalance_spread_col,
    )
    numeric_decimals = {
        energy_col: 0,
        kwh_per_step_col: 2,
        eur_per_step_col: 2,
        impact_total_col: 2,
        impact_delta_grid_fee_col: 2,
        impact_imbalance_cash_col: 2,
        impact_imbalance_da_reference_col: 2,
        impact_imbalance_spread_col: 2,
    }
    heatmap_columns = [
        kwh_per_step_col,
        eur_per_step_col,
        impact_total_col,
        impact_delta_grid_fee_col,
        impact_imbalance_cash_col,
        impact_imbalance_da_reference_col,
        impact_imbalance_spread_col,
    ]
    heatmap_norms = {
        kwh_per_step_col: build_sequential_norm(
            pd.to_numeric(table[kwh_per_step_col], errors="coerce").to_numpy(dtype=float)
        )
    }
    if shared_impact_scale:
        impact_columns = [
            impact_total_col,
            impact_delta_grid_fee_col,
            impact_imbalance_cash_col,
            impact_imbalance_da_reference_col,
            impact_imbalance_spread_col,
        ]
        impact_values = np.concatenate(
            [
                pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
                for column in impact_columns
            ]
        )
        shared_impact_norm = build_diverging_norm(impact_values)
        heatmap_norms.update({column: shared_impact_norm for column in impact_columns})
    heatmap_cmaps = {
        kwh_per_step_col: GRAY_CMAP,
    }
    export_table_png(
        table=table,
        output_path=output_path,
        columns=columns,
        header_labels=header_labels,
        numeric_decimals=numeric_decimals,
        heatmap_columns=heatmap_columns,
        count_bar_columns=[count_col, energy_col],
        heatmap_norms=heatmap_norms,
        heatmap_cmaps=heatmap_cmaps,
        bar_fill_colors={
            count_col: "#5a8fd0",
            energy_col: "#d9b300",
        },
    )


def export_per_step_comparison_png(summary: pd.DataFrame, output_path: Path = PER_STEP_COMPARISON_PNG) -> Path:
    """Export a comparison PNG with overview and model KPI values per step."""

    table = build_per_step_comparison_frame(summary)
    columns = [
        *SUMMARY_PLOT_BASE_COLUMNS,
        "step_count",
        "overview_eur_per_step",
        *[f"{model}_eur_per_step" for model in MODELS],
    ]
    header_labels = {
        "classification": "Class",
        "forecast_state": "Forecast",
        "da_price_sign": "DA",
        "imbalance_price_sign": "Imb",
        "da_vs_imb": "DA vs Imb",
        "planned_grid_state": "Plan",
        "realized_grid_state": "Real",
        "step_count": "Steps",
        "overview_eur_per_step": "Ov [€/step]",
        "physical_eur_per_step": "Phy",
        "source_eur_per_step": "Src",
        "target_eur_per_step": "Tgt",
        "transfer_eur_per_step": "Trf",
    }
    numeric_decimals = {column: 2 for column in columns if column.endswith("eur_per_step")}
    heatmap_columns = list(numeric_decimals.keys())
    per_step_columns = [column for column in columns if column.endswith("eur_per_step")]
    per_step_values = np.concatenate(
        [
            pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
            for column in per_step_columns
        ]
    )
    shared_norm = build_diverging_norm(per_step_values)
    heatmap_norms = {column: shared_norm for column in per_step_columns}
    export_table_png(
        table=table,
        output_path=output_path,
        columns=columns,
        header_labels=header_labels,
        numeric_decimals=numeric_decimals,
        heatmap_columns=heatmap_columns,
        count_bar_columns=["step_count"],
        heatmap_norms=heatmap_norms,
    )
    return output_path


def export_per_kwh_comparison_png(summary: pd.DataFrame, output_path: Path = PER_KWH_COMPARISON_PNG) -> Path:
    """Export a comparison PNG with overview and model KPI values per kWh."""

    table = build_per_kwh_comparison_frame(summary)
    columns = [
        *SUMMARY_PLOT_BASE_COLUMNS,
        "step_count",
        "energy_kwh",
        "kwh_per_step",
        "overall_eur_per_kwh",
        *[f"{model}_eur_per_kwh" for model in MODELS],
    ]
    header_labels = {
        "classification": "Class",
        "forecast_state": "Forecast",
        "da_price_sign": "DA",
        "imbalance_price_sign": "Imb",
        "da_vs_imb": "DA vs Imb",
        "planned_grid_state": "Plan",
        "realized_grid_state": "Real",
        "step_count": "Steps",
        "energy_kwh": "kWh",
        "kwh_per_step": "kWh/step",
        "overall_eur_per_kwh": "€/kWh",
        "physical_eur_per_kwh": "Physical",
        "source_eur_per_kwh": "Source",
        "target_eur_per_kwh": "Target",
        "transfer_eur_per_kwh": "Transfer",
    }
    numeric_decimals = {
        "energy_kwh": 0,
        "kwh_per_step": 2,
        **{column: 2 for column in columns if column.endswith("eur_per_kwh")},
    }
    heatmap_columns = ["kwh_per_step", *[column for column in columns if column.endswith("eur_per_kwh")]]
    per_kwh_columns = [column for column in columns if column.endswith("eur_per_kwh")]
    per_kwh_values = np.concatenate(
        [
            pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
            for column in per_kwh_columns
        ]
    )
    shared_norm = build_diverging_norm(per_kwh_values)
    heatmap_norms = {column: shared_norm for column in per_kwh_columns}
    heatmap_norms["kwh_per_step"] = build_sequential_norm(
        pd.to_numeric(table["kwh_per_step"], errors="coerce").to_numpy(dtype=float)
    )
    heatmap_cmaps = {
        "kwh_per_step": GRAY_CMAP,
    }
    export_table_png(
        table=table,
        output_path=output_path,
        columns=columns,
        header_labels=header_labels,
        numeric_decimals=numeric_decimals,
        heatmap_columns=heatmap_columns,
        count_bar_columns=["step_count", "energy_kwh"],
        heatmap_norms=heatmap_norms,
        heatmap_cmaps=heatmap_cmaps,
        bar_fill_colors={
            "step_count": "#5a8fd0",
            "energy_kwh": "#d9b300",
        },
    )
    return output_path


def export_impact_comparison_png(summary: pd.DataFrame, output_path: Path = IMPACT_COMPARISON_PNG) -> Path:
    """Export a comparison PNG with overview and per-model impact sums in EUR."""

    table = build_impact_comparison_frame(summary)
    columns = [
        *SUMMARY_PLOT_BASE_COLUMNS,
        "step_count",
        "energy_kwh",
        "kwh_per_step",
        "impact_total_eur",
        *[f"{model}_impact_total_eur" for model in MODELS],
    ]
    header_labels = {
        "classification": "Class",
        "forecast_state": "Forecast",
        "da_price_sign": "DA",
        "imbalance_price_sign": "Imb",
        "da_vs_imb": "DA vs Imb",
        "planned_grid_state": "Plan",
        "realized_grid_state": "Real",
        "step_count": "Steps",
        "energy_kwh": "kWh",
        "kwh_per_step": "kWh/step",
        "impact_total_eur": "Overall [€]",
        "physical_impact_total_eur": "Phy",
        "source_impact_total_eur": "Src",
        "target_impact_total_eur": "Tgt",
        "transfer_impact_total_eur": "Trf",
    }
    numeric_decimals = {
        "energy_kwh": 0,
        "kwh_per_step": 2,
        **{
            column: 2
            for column in columns
            if column.endswith("impact_total_eur")
        },
    }
    heatmap_columns = [
        "kwh_per_step",
        *[column for column in columns if column.endswith("impact_total_eur")],
    ]
    model_impact_columns = [f"{model}_impact_total_eur" for model in MODELS]
    model_impact_values = np.concatenate(
        [
            pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
            for column in model_impact_columns
        ]
    )
    heatmap_norms = {
        "kwh_per_step": build_sequential_norm(
            pd.to_numeric(table["kwh_per_step"], errors="coerce").to_numpy(dtype=float)
        ),
        "impact_total_eur": build_diverging_norm(
            pd.to_numeric(table["impact_total_eur"], errors="coerce").to_numpy(dtype=float)
        )
    }
    model_shared_norm = build_diverging_norm(model_impact_values)
    heatmap_norms.update({column: model_shared_norm for column in model_impact_columns})
    heatmap_cmaps = {
        "kwh_per_step": GRAY_CMAP,
    }
    export_table_png(
        table=table,
        output_path=output_path,
        columns=columns,
        header_labels=header_labels,
        numeric_decimals=numeric_decimals,
        heatmap_columns=heatmap_columns,
        count_bar_columns=["step_count", "energy_kwh"],
        heatmap_norms=heatmap_norms,
        heatmap_cmaps=heatmap_cmaps,
        bar_fill_colors={
            "step_count": "#5a8fd0",
            "energy_kwh": "#d9b300",
        },
    )
    return output_path


def export_summary_pngs(summary: pd.DataFrame) -> list[Path]:
    """Erzeugt das Overview-PNG und die vier modellbezogenen PNGs aus der Summary."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    overview = build_plot_frame(
        summary=summary,
        count_col="step_count",
        energy_col="energy_kwh",
        kwh_per_step_col="kwh_per_step",
        denominator_col="energy_kwh",
        impact_total_col="impact_total_eur",
        impact_delta_grid_fee_col="impact_delta_grid_fee_eur",
        impact_imbalance_cash_col="impact_rebap_settlement_eur",
        impact_imbalance_da_reference_col="impact_imbalance_da_reference_eur",
        impact_imbalance_spread_col="impact_imbalance_spread_eur",
        eur_per_step_col="eur_per_kwh",
        sort_col="impact_total_eur",
    )
    export_summary_table_png(
        table=overview,
        output_path=SUMMARY_OVERVIEW_PNG,
        count_col="step_count",
        energy_col="energy_kwh",
        kwh_per_step_col="kwh_per_step",
        eur_per_step_col="eur_per_kwh",
        ratio_label="€/kWh",
        impact_total_col="impact_total_eur",
        impact_delta_grid_fee_col="impact_delta_grid_fee_eur",
        impact_imbalance_cash_col="impact_rebap_settlement_eur",
        impact_imbalance_da_reference_col="impact_imbalance_da_reference_eur",
        impact_imbalance_spread_col="impact_imbalance_spread_eur",
        shared_impact_scale=False,
    )
    output_paths.append(SUMMARY_OVERVIEW_PNG)
    output_paths.append(export_per_step_comparison_png(summary))
    output_paths.append(export_per_kwh_comparison_png(summary))
    output_paths.append(export_impact_comparison_png(summary))

    for model in MODELS:
        count_col = f"{model}_step_count"
        energy_col = f"{model}_energy_kwh"
        kwh_per_step_col = f"{model}_kwh_per_step"
        eur_per_step_col = f"{model}_eur_per_kwh"
        impact_total_col = f"{model}_impact_total_eur"
        impact_delta_grid_fee_col = f"{model}_impact_delta_grid_fee_eur"
        impact_imbalance_cash_col = f"{model}_impact_rebap_settlement_eur"
        impact_imbalance_da_reference_col = f"{model}_impact_imbalance_da_reference_eur"
        impact_imbalance_spread_col = f"{model}_impact_imbalance_spread_eur"

        plot_df = build_plot_frame(
            summary=summary,
            count_col=count_col,
            energy_col=energy_col,
            kwh_per_step_col=kwh_per_step_col,
            denominator_col=energy_col,
            impact_total_col=impact_total_col,
            impact_delta_grid_fee_col=impact_delta_grid_fee_col,
            impact_imbalance_cash_col=impact_imbalance_cash_col,
            impact_imbalance_da_reference_col=impact_imbalance_da_reference_col,
            impact_imbalance_spread_col=impact_imbalance_spread_col,
            eur_per_step_col=eur_per_step_col,
            sort_col=impact_total_col,
        )
        output_path = SUMMARY_MODEL_PNGS[model]
        export_summary_table_png(
            table=plot_df,
            output_path=output_path,
            count_col=count_col,
            energy_col=energy_col,
            kwh_per_step_col=kwh_per_step_col,
            eur_per_step_col=eur_per_step_col,
            ratio_label="€/kWh",
            impact_total_col=impact_total_col,
            impact_delta_grid_fee_col=impact_delta_grid_fee_col,
            impact_imbalance_cash_col=impact_imbalance_cash_col,
            impact_imbalance_da_reference_col=impact_imbalance_da_reference_col,
            impact_imbalance_spread_col=impact_imbalance_spread_col,
            shared_impact_scale=True,
        )
        output_paths.append(output_path)

    return output_paths


def main() -> None:
    """Liest die vorhandene Summary-CSV und erzeugt daraus alle Preislogik-PNGs."""

    summary = load_summary_table()
    png_paths = export_summary_pngs(summary)
    for png_path in png_paths:
        print(png_path)


if __name__ == "__main__":
    main()
