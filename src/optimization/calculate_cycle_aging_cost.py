"""
Einfaches, oekonomisch kalibriertes 2-Segment-Modell fuer Zyklusalterung.

Ziel:
- Ableitung von zwei konstanten Alterungskostenkoeffizienten (omega_1, omega_2)
- Nutzung ausschliesslich oekonomischer Groessen (keine Zellchemie, kein Arrhenius)

Grundidee:
1) Mittlere Kosten pro Vollzyklus:
   cost_per_full_cycle = replacement_cost_eur / expected_full_cycles
2) Mittlere Kosten pro umgesetzter Energie:
   avg_cost_per_kwh = cost_per_full_cycle / usable_capacity_kwh
3) Zwei Segmente mit gleicher Breite (50% / 50%):
   omega_2 = DEEP_CYCLE_COST_FACTOR * omega_1
   und der Mittelwert beider Segmente entspricht avg_cost_per_kwh.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# GLOBALE PARAMETER
# ============================================================

# Nutzbare Batteriekapazitaet [kWh]
USABLE_CAPACITY_KWH = 12

# Ersatzkosten [EUR]
REPLACEMENT_COST_EUR = 8500.0

# Erwartete Vollzyklen bis Lebensende [-]
EXPECTED_FULL_CYCLES = 12000

# Gewichtung fuer tiefe Zyklen [-]
# omega_2 = DEEP_CYCLE_COST_FACTOR * omega_1
DEEP_CYCLE_COST_FACTOR = 3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "reports" / "optimization"
OUTPUT_PLOT = OUTPUT_DIR / "cycle_aging_cost_piecewise_over_capacity.png"

# A4-freundliches Exportformat
FIGSIZE_A4_INCH = (8.27, 5.0)
EXPORT_DPI = 300



# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def compute_cost_per_full_cycle(replacement_cost_eur: float, expected_full_cycles: int) -> float:
    """Berechnet mittlere Kosten pro Vollzyklus in EUR/Zyklus."""
    return replacement_cost_eur / expected_full_cycles


def compute_average_cost_per_kwh(cost_per_full_cycle_eur: float, usable_capacity_kwh: float) -> float:
    """Berechnet mittlere Alterungskosten in EUR/kWh durchgesetzter Energie."""
    return cost_per_full_cycle_eur / usable_capacity_kwh


def compute_segment_omegas_per_kwh(avg_cost_per_kwh: float, deep_cycle_cost_factor: float) -> tuple[float, float]:
    """
    Berechnet Segmentkoeffizienten in EUR/kWh fuer gleich breite Segmente.

    Herleitung:
    - omega_2 = f * omega_1
    - (omega_1 + omega_2) / 2 = avg_cost_per_kwh
    - => omega_1 = 2 * avg / (1 + f)
    """
    omega_1_per_kwh = (2.0 * avg_cost_per_kwh) / (1.0 + deep_cycle_cost_factor)
    omega_2_per_kwh = deep_cycle_cost_factor * omega_1_per_kwh
    return omega_1_per_kwh, omega_2_per_kwh


def to_eur_per_mwh(value_eur_per_kwh: float) -> float:
    """Rechnet EUR/kWh in EUR/MWh um (x1000)."""
    return value_eur_per_kwh * 1000.0


def plot_piecewise_cost_over_capacity(
    usable_capacity_kwh: float,
    avg_cost_per_kwh: float,
    omega_1_per_kwh: float,
    omega_2_per_kwh: float,
) -> Path:
    """Plottet die lineare avg-Kostenfunktion und die geknickte 2-Segment-Funktion ueber 0..Kapazitaet."""
    x_kwh = np.linspace(0.0, usable_capacity_kwh, 600)
    split_kwh = 0.5 * usable_capacity_kwh

    # Referenz: konstante Steigung avg_cost_per_kwh.
    y_avg_eur = avg_cost_per_kwh * x_kwh

    # 2-Segment-Funktion mit Knick bei 50%:
    # - 0..split_kwh: oberes Segment (keine Tiefentladung) -> omega_1 (guenstiger/flacher)
    # - split_kwh..usable_capacity_kwh: tiefes Segment -> omega_2 (teurer/steiler)
    y_piecewise_eur = np.where(
        x_kwh <= split_kwh,
        omega_1_per_kwh * x_kwh,
        (omega_1_per_kwh * split_kwh) + omega_2_per_kwh * (x_kwh - split_kwh),
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_A4_INCH, dpi=EXPORT_DPI)
    ax.axvline(split_kwh, color="#555555", linestyle="--", linewidth=1.0, alpha=0.85)

    ax.plot(x_kwh, y_avg_eur, color="#1f77b4", linewidth=1.8, label="Average slope")
    ax.plot(x_kwh, y_piecewise_eur, color="#111111", linewidth=1.8, label="Piecewise (omega_1 / omega_2)")

    y_split = omega_1_per_kwh * split_kwh
    y_end = (omega_1_per_kwh * split_kwh) + omega_2_per_kwh * (usable_capacity_kwh - split_kwh)
    ax.scatter([split_kwh, usable_capacity_kwh], [y_split, y_end], color="#111111", s=16, zorder=4)

    ax.text(
        split_kwh * 0.5,
        y_end * 0.96,
        f"0-{split_kwh:.1f} kWh discharged: Upper segment (omega_1)",
        ha="center",
        va="top",
        fontsize=8,
        color="#222222",
    )
    ax.text(
        split_kwh + (usable_capacity_kwh - split_kwh) * 0.5,
        y_end * 0.96,
        f"{split_kwh:.1f}-{usable_capacity_kwh:.1f} kWh discharged: Deep segment (omega_2)",
        ha="center",
        va="top",
        fontsize=8,
        color="#222222",
    )

    ax.set_title(
        f"Cycle Aging Cost vs Discharge Depth ({usable_capacity_kwh:.0f} kWh usable) - Average vs Piecewise",
        fontsize=10,
    )
    ax.set_xlabel("Discharge Depth [kWh]", fontsize=9)
    ax.set_ylabel("Cumulative Aging Cost [EUR]", fontsize=9)
    ax.grid(True, axis="both", linestyle="-", linewidth=0.5, alpha=0.35)
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PLOT)
    plt.close(fig)
    return OUTPUT_PLOT


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main() -> None:
    """Fuehrt die Kalibrierung aus und gibt alle relevanten Kennzahlen aus."""

    # 1) Mittlere Kosten pro Vollzyklus [EUR/Zyklus].
    cost_per_full_cycle_eur = compute_cost_per_full_cycle(
        replacement_cost_eur=REPLACEMENT_COST_EUR,
        expected_full_cycles=EXPECTED_FULL_CYCLES,
    )

    # 2) Mittlere Kosten pro durchgesetzter Energie [EUR/kWh].
    avg_cost_per_kwh = compute_average_cost_per_kwh(
        cost_per_full_cycle_eur=cost_per_full_cycle_eur,
        usable_capacity_kwh=USABLE_CAPACITY_KWH,
    )

    # 3) Segmentkosten in EUR/kWh (flaches Segment und tiefes Segment).
    omega_1_per_kwh, omega_2_per_kwh = compute_segment_omegas_per_kwh(
        avg_cost_per_kwh=avg_cost_per_kwh,
        deep_cycle_cost_factor=DEEP_CYCLE_COST_FACTOR,
    )

    # 4) Fuer die Optimierung werden EUR/MWh benoetigt.
    omega_1_per_mwh = to_eur_per_mwh(omega_1_per_kwh)
    omega_2_per_mwh = to_eur_per_mwh(omega_2_per_kwh)

    # 5) Plausibilisierung: Segmentkosten fuer einen Vollzyklus.
    # Segment 1 = oberes Halbsegment (kein Deep-Discharge, omega_1).
    # Segment 2 = unteres Halbsegment (Deep-Discharge, omega_2).
    segment_1_energy_kwh = USABLE_CAPACITY_KWH * 0.5
    segment_2_energy_kwh = USABLE_CAPACITY_KWH * 0.5
    segment_1_cost_eur = omega_1_per_kwh * segment_1_energy_kwh
    segment_2_cost_eur = omega_2_per_kwh * segment_2_energy_kwh
    full_cycle_cost_from_segments_eur = segment_1_cost_eur + segment_2_cost_eur

    # 6) Ausgabe.
    print("=" * 72)
    print("Einfaches 2-Segment-Modell fuer Zyklusalterung (oekonomisch kalibriert)")
    print("=" * 72)

    print("\nEingangsparameter")
    print(f"usable capacity [kWh]              = {USABLE_CAPACITY_KWH:.4f}")
    print(f"replacement cost [EUR]             = {REPLACEMENT_COST_EUR:.2f}")
    print(f"expected full cycles [-]           = {EXPECTED_FULL_CYCLES}")
    print(f"deep cycle cost factor [-]         = {DEEP_CYCLE_COST_FACTOR:.4f}")

    print("\nAbgeleitete Basisgroessen")
    print(f"cost per full cycle [EUR/cycle]    = {cost_per_full_cycle_eur:.8f}")
    print(f"average degradation [EUR/kWh]      = {avg_cost_per_kwh:.8f}")

    print("\nSegmentkoeffizienten")
    print(f"omega_1 [EUR/kWh]                  = {omega_1_per_kwh:.8f}")
    print(f"omega_1 [EUR/MWh]                  = {omega_1_per_mwh:.6f}")
    print(f"omega_2 [EUR/kWh]                  = {omega_2_per_kwh:.8f}")
    print(f"omega_2 [EUR/MWh]                  = {omega_2_per_mwh:.6f}")

    print("\nPlausibilisierung (Vollzyklus aus Segmenten)")
    print(f"segment 1 energy [kWh]             = {segment_1_energy_kwh:.4f}")
    print(f"segment 2 energy [kWh]             = {segment_2_energy_kwh:.4f}")
    print(f"segment 1 cost [EUR] (omega_1)     = {segment_1_cost_eur:.8f}")
    print(f"segment 2 cost [EUR] (omega_2)     = {segment_2_cost_eur:.8f}")
    print(f"full cycle cost (seg1+seg2) [EUR]  = {full_cycle_cost_from_segments_eur:.8f}")
    print(f"reference full cycle cost [EUR]    = {cost_per_full_cycle_eur:.8f}")

    plot_path = plot_piecewise_cost_over_capacity(
        usable_capacity_kwh=USABLE_CAPACITY_KWH,
        avg_cost_per_kwh=avg_cost_per_kwh,
        omega_1_per_kwh=omega_1_per_kwh,
        omega_2_per_kwh=omega_2_per_kwh,
    )
    print(f"\nPlot saved: {plot_path}")


if __name__ == "__main__":
    main()
