"""
Einfache überschlagsrechnung der c_grid zusatzkosten. Eigentlich werden die zusätze nicht prozentual berechnet sondern individuell nach bezugsmenge und leistung.
Hier werden durchschnittswerte der energierechnung nach https://www.bundesnetzagentur.de/DE/Beschlusskammern/BK08/BK8_06_Netzentgelte/BK8_NetzE.html angenommen.
"""

from __future__ import annotations


# ============================================================
# GLOBALE PARAMETER
# ============================================================

# Strompreis ohne vpp [EUR/kWh]
ELECTRICITY_COST_CT_PER_KWH = 36.06

# Hauptbestandteile (%)
NET_GRID_FEE = 0.269                # Nettonetzentgelt
METERING_AND_OPERATION = 0.01       # Messung und Messstellenbetrieb
CONCESSION_FEE = 0.039              # Konzessionsabgabe
ELECTRICITY_TAX = 0.0               # Stromsteuer
VAT = 0.0                          # Umsatzsteuer (nicht berücksichtigt)

# Umlagen (%)
KWKG_LEVY = 0.007                   # KWKG-Umlage
SECTION19_LEVY = 0.015              # §19 StromNEV Umlage
OFFSHORE_LEVY = 0.016               # Offshore-Netzumlage


# ============================================================
# BERECHNUNG
# ============================================================

def calculate_c_grid(
    electricity_cost_ct_per_kwh: float,
    net_grid_fee: float,
    metering_and_operation: float,
    concession_fee: float,
    electricity_tax: float,
    vat: float,
    kwkg_levy: float,
    section19_levy: float,
    offshore_levy: float
) -> float:
    """Berechnet die zusätzlichen Grid-Kosten IN euro pro MWh"""
    
    c_grid_ct_per_kwh = electricity_cost_ct_per_kwh * (
        net_grid_fee + metering_and_operation + concession_fee + 
        electricity_tax + vat + kwkg_levy + section19_levy + offshore_levy
    )
    
    c_grid_euro_per_mwh = c_grid_ct_per_kwh * 10.0  # Umrechnung von Cent/kWh auf Euro/MWh
    
    return c_grid_euro_per_mwh


def main() -> None:
    c_grid_cost = calculate_c_grid(
        electricity_cost_ct_per_kwh=ELECTRICITY_COST_CT_PER_KWH,
        net_grid_fee=NET_GRID_FEE,
        metering_and_operation=METERING_AND_OPERATION,
        concession_fee=CONCESSION_FEE,
        electricity_tax=ELECTRICITY_TAX,
        vat=VAT,
        kwkg_levy=KWKG_LEVY,
        section19_levy=SECTION19_LEVY,
        offshore_levy=OFFSHORE_LEVY
    )
    
    print(f"Die zusätzlichen Grid-Kosten betragen: {c_grid_cost:.2f} EUR/MWh")
    
    
if __name__ == "__main__":
    main()
    