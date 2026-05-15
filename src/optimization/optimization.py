"""
Pyomo-basierter Optimierungs- und Postprocessing-Kern fuer den Dispatch.

Dieses Modul enthaelt:
- den mathematischen Optimierungskern (MILP + Solver-Aufruf),
- geplante KPI-Nachrechnungen,
- Settlement-/Realisierungs-Postprocessing (DA + reBAP),
- optionale run-basierte Aggregation von Realized-Kennzahlen.

Inhaltlich wird pro Horizont ein lineares Mischganzzahlmodell (MILP) geloest:
- Energiesystem-Bilanz pro Zeitschritt (PV, Last, Netz, Batterie, Curtailment)
- Batteriedynamik mit Lade-/Entladegrenzen und Wirkungsgraden
- Exklusive Lade-/Entladeentscheidung ueber eine Binärvariable
- Zyklusalterungskosten mit zwei Segmenten (niedrig/hoch bewertete Entladung)

Zielfunktion:
- Maximierung des Marktwerts (DA-Erlos minus DA-Bezug inkl. Netzentgelt)
- abzueglich segmentbasierter Zyklusalterungskosten.

Schnittstelle:
- Input: `horizon_df` mit Zeit, Last, PV-Prognose und DA-Preis
- Input: `cfg` mit Batterie- und Kostenparametern
- Output: geloeste 15-min Dispatch-Tabelle plus Solver-Metadaten.

Die Orchestrierung (Publikationszeiten, Fallback-Logik, Szenario-Run-Selektion,
CSV-I/O) liegt in `optimization_scenario_comparison.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyomo.environ as pyo

FIXED_TIMESTEP_MINUTES = 15
FIXED_CYCLE_AGING_SEGMENT_1_SHARE = 0.5
FIXED_SOLVER_NAME = "gurobi"
FIXED_SOLVER_EXECUTABLE = None
FIXED_SOLVER_TEE = False
FIXED_SEGMENT_EMPTY_EPS_WH = 1e-6
FIXED_DATETIME_COL = "datetime"
FIXED_LOAD_COL = "Last [W]"
FIXED_PV_FORECAST_COL = "P_pv_forecast_W"
FIXED_PRICE_COL = "DA Preis [EUR/MWh]"


def solve_milp_for_horizon(
    horizon_df: pd.DataFrame,
    cfg: dict,
    initial_soc_wh: float,
    enforce_terminal_soc: bool = True,
) -> dict:
    """Loest ein MILP fuer genau einen gegebenen Optimierungshorizont.

    Args:
        horizon_df: Zeitreihe fuer den zu optimierenden Horizont mit den
            Spalten `datetime`, `Last [W]`, `P_pv_forecast_W`,
            `DA Preis [EUR/MWh]`.
        cfg: Normalisierte Config mit Teilbloecken `optimization` und `battery`.
        initial_soc_wh: Start-SoC am Horizontanfang in Wh.
        enforce_terminal_soc: Wenn `True`, wird der End-SoC auf den
            konfigurierten Zielwert fixiert.

    Returns:
        Dictionary mit:
        - `df`: geloeste Dispatch-Zeitreihe (pro Zeitschritt)
        - `objective_eur`: Zielfunktionswert in EUR
        - `status`: Solver-Status als String
        - `termination`: Termination-Condition als String

    Raises:
        RuntimeError: Wenn Solver nicht verfuegbar ist oder nicht optimal loest.
    """

    # 1) Parameter aus Config lesen und in Modell-Einheiten bringen

    opt_cfg = cfg["optimization"]
    bat_cfg = cfg["battery"]

    dt_h = FIXED_TIMESTEP_MINUTES / 60.0    #1/4 Stunde in Stunden
    n_steps = len(horizon_df) # Anzahl Zeitschritte im Horizont, mindest 96 fuer 24h mit 15min Schritten, meist mehr da restzeit mitoptimiert wird
    terminal_soc_wh = opt_cfg["terminal_soc_fraction"] * bat_cfg["e_max_kwh"] * 1000.0  # Ziel-SoC am Horizontende in Wh
    e_min_wh = bat_cfg["e_min_kwh"] * 1000.0
    e_max_wh = bat_cfg["e_max_kwh"] * 1000.0
    p_ch_max_w = bat_cfg["p_ch_max_kw"] * 1000.0
    p_dis_max_w = bat_cfg["p_dis_max_kw"] * 1000.0
    eta_ch = bat_cfg["eta_ch"]
    eta_dis = bat_cfg["eta_dis"]
    grid_fee = opt_cfg["grid_fee_eur_per_mwh"]
    omega_1 = float(opt_cfg["cycle_aging_omega_1_eur_per_mwh"])
    omega_2 = float(opt_cfg["cycle_aging_omega_2_eur_per_mwh"])
    segment_1_share = FIXED_CYCLE_AGING_SEGMENT_1_SHARE

    # Nutzbarer Energieinhalt (zwischen e_min und e_max) wird in zwei
    # Alterungssegmente aufgeteilt. Segment 1 steht fuer "guenstigere" Entladung, Segment 2 fuer "teurere" Entladung.
    
    usable_wh = e_max_wh - e_min_wh             # Gesamtnutzbarer Energieinhalt der Batterie in Wh zb 9000 Wh bei 10 kWh Batterie mit 1 kWh Puffer
    seg1_cap_wh = usable_wh * segment_1_share   # Kapazitaet von Segment 1 in Wh, z.B. 4500 Wh bei 50% Segment-1-Anteil
    seg2_cap_wh = usable_wh - seg1_cap_wh       # Kapazitaet von Segment 2 in Wh, z.B. 4500 Wh bei 50% Segment-1-Anteil

    # ------------------------------------------------------------
    # 2) Horizontdaten als Arrays vorbereiten

    dt_col = FIXED_DATETIME_COL
    time_values = pd.to_datetime(horizon_df[dt_col], errors="coerce").to_numpy()

    load = horizon_df[FIXED_LOAD_COL].to_numpy(dtype=float)
    pv = horizon_df[FIXED_PV_FORECAST_COL].to_numpy(dtype=float)
    price = horizon_df[FIXED_PRICE_COL].to_numpy(dtype=float)
    
    # Curtailment-Kappe je Schritt (nur Grenzwert, keine Aenderung von `pv`):
    # - Obergrenze = verfuegbare PV-Leistung
    # - bei nicht-negativem Preis ist Curtailment gesperrt (Kappe = 0)
    curt_cap = [
        (max(float(pv[t]), 0.0) if float(price[t]) < 0.0 else 0.0)
        for t in range(n_steps)
    ]

    # ------------------------------------------------------------
    # 3) Pyomo-Modell, Variablen 

    # Erstellen eines ConcreteModel, da die Anzahl der Zeitschritte (und damit die Anzahl der Variablen/Constraints) bereits bekannt ist.
    model = pyo.ConcreteModel()
    
    #create time index sets, letzter wert inkludiert
    model.T = pyo.RangeSet(0, n_steps - 1) #typically used for decision variables that represent actions or states during each interval
    model.Te = pyo.RangeSet(0, n_steps) #This expanded set is useful for representing values at time boundaries, such as the state of charge at the beginning and end of the horizon, which includes one additional time point compared to the number of intervals.

    # Leistungsgroessen je Zeitschritt
    model.p_grid_in = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_grid_out = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_ch = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_dis = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_curt = pyo.Var(model.T, within=pyo.NonNegativeReals)
    # Netzmodus: 1 = Importmodus (kein Export), 0 = Exportmodus (kein Import)
    model.z_grid_import = pyo.Var(model.T, within=pyo.Binary)
    # Segmentmodus Entladung: 1 = Segment 2 aktiv, 0 = Segment 1 aktiv
    model.z_dis_seg2 = pyo.Var(model.T, within=pyo.Binary)
    # Segmentmodus Ladung: 1 = Segment 1 aktiv, 0 = Segment 2 aktiv
    model.z_ch_seg1 = pyo.Var(model.T, within=pyo.Binary)
    # Binärvariable: 1 = Laden erlaubt, 0 = Entladen erlaubt
    model.u = pyo.Var(model.T, within=pyo.Binary)
    # Energiezustaende (inklusive Start- und Endzustand)
    model.e = pyo.Var(model.Te, within=pyo.NonNegativeReals)
    # Segmentierte Lade-/Entladefluesse und Segmentenergien fuer Alterung
    model.p_ch_seg1 = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_ch_seg2 = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_dis_seg1 = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_dis_seg2 = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.e_seg1 = pyo.Var(model.Te, within=pyo.NonNegativeReals)
    model.e_seg2 = pyo.Var(model.Te, within=pyo.NonNegativeReals)


    # ------------------------------------------------------------
    # 4) Zielfunktion: Maximierung des Marktwerts minus Zyklusalterungskosten

    market_revenue_expr = sum(
        ((price[t] * model.p_grid_out[t]) - ((price[t] + grid_fee) * model.p_grid_in[t])) * dt_h / 1e6  #Einheiten-Umrechnung von Leistung in Arbeit und dann in Euro.
        for t in model.T
    )
    cycle_aging_cost_expr = sum(
        ((omega_1 * model.p_dis_seg1[t]) + (omega_2 * model.p_dis_seg2[t])) * dt_h / 1e6
        for t in model.T
    )
    model.obj = pyo.Objective(
        expr=market_revenue_expr - cycle_aging_cost_expr,
        sense=pyo.maximize,
    )

    # ------------------------------------------------------------
    # 5) Nebenbedingungen

    # Leistungsbilanz im AC-Knoten
    def power_balance_rule(m, t):
        return pv[t] + m.p_grid_in[t] + m.p_dis[t] == load[t] + m.p_grid_out[t] + m.p_ch[t] + m.p_curt[t]

    model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)

    # Curtailment darf nur aus PV stammen, nur in negativen Preisstunden auftreten
    # und nur im Exportmodus aktiv sein.
    def p_curt_cap_rule(m, t):
        return m.p_curt[t] <= (1 - m.z_grid_import[t]) * curt_cap[t]

    model.p_curt_cap = pyo.Constraint(model.T, rule=p_curt_cap_rule)

    # Segmentzerlegung der Lade-/Entladeleistung
    def p_dis_split_rule(m, t):
        return m.p_dis[t] == m.p_dis_seg1[t] + m.p_dis_seg2[t]

    def p_ch_split_rule(m, t):
        return m.p_ch[t] == m.p_ch_seg1[t] + m.p_ch_seg2[t]

    model.p_dis_split = pyo.Constraint(model.T, rule=p_dis_split_rule)
    model.p_ch_split = pyo.Constraint(model.T, rule=p_ch_split_rule)

    # Segment-Nutzungslogik (harte Reihenfolge beim Entladen):
    # 1) Segment 2 darf nur aktiv sein, wenn Segment 1 am Schrittanfang leer ist.
    # 2) Entladung aus Segment 1 und 2 darf nicht gleichzeitig stattfinden.
    def p_dis_seg2_activation_rule(m, t):
        return m.p_dis_seg2[t] <= m.z_dis_seg2[t] * p_dis_max_w

    def p_dis_seg1_activation_rule(m, t):
        return m.p_dis_seg1[t] <= (1 - m.z_dis_seg2[t]) * p_dis_max_w

    def dis_seg2_only_if_seg1_empty_rule(m, t):
        return m.e_seg1[t] <= FIXED_SEGMENT_EMPTY_EPS_WH + (1 - m.z_dis_seg2[t]) * seg1_cap_wh

    model.p_dis_seg2_activation = pyo.Constraint(model.T, rule=p_dis_seg2_activation_rule)
    model.p_dis_seg1_activation = pyo.Constraint(model.T, rule=p_dis_seg1_activation_rule)
    model.dis_seg2_only_if_seg1_empty = pyo.Constraint(model.T, rule=dis_seg2_only_if_seg1_empty_rule)

    # Segment-Nutzungslogik (harte Reihenfolge beim Laden):
    # 1) Segment 2 wird zuerst geladen.
    # 2) Segment 1 darf erst laden, wenn Segment 2 voll ist.
    # 3) Gleichzeitiges Laden beider Segmente ist ausgeschlossen.
    def p_ch_seg1_activation_rule(m, t):
        return m.p_ch_seg1[t] <= m.z_ch_seg1[t] * p_ch_max_w

    def p_ch_seg2_activation_rule(m, t):
        return m.p_ch_seg2[t] <= (1 - m.z_ch_seg1[t]) * p_ch_max_w

    def ch_seg1_only_if_seg2_full_rule(m, t):
        return m.e_seg2[t] >= (seg2_cap_wh - FIXED_SEGMENT_EMPTY_EPS_WH) * m.z_ch_seg1[t]

    model.p_ch_seg1_activation = pyo.Constraint(model.T, rule=p_ch_seg1_activation_rule)
    model.p_ch_seg2_activation = pyo.Constraint(model.T, rule=p_ch_seg2_activation_rule)
    model.ch_seg1_only_if_seg2_full = pyo.Constraint(model.T, rule=ch_seg1_only_if_seg2_full_rule)

    # Gesamtenergie = e_min + Segmentenergien
    def e_split_rule(m, t):
        return m.e[t] == e_min_wh + m.e_seg1[t] + m.e_seg2[t]

    model.e_split = pyo.Constraint(model.Te, rule=e_split_rule)

    # Segmentdynamik mit Wirkungsgraden
    def segment_dyn_1_rule(m, t):
        return m.e_seg1[t + 1] == m.e_seg1[t] + (eta_ch * m.p_ch_seg1[t] * dt_h) - ((1.0 / eta_dis) * m.p_dis_seg1[t] * dt_h)

    def segment_dyn_2_rule(m, t):
        return m.e_seg2[t + 1] == m.e_seg2[t] + (eta_ch * m.p_ch_seg2[t] * dt_h) - ((1.0 / eta_dis) * m.p_dis_seg2[t] * dt_h)

    model.segment_dyn_1 = pyo.Constraint(model.T, rule=segment_dyn_1_rule)
    model.segment_dyn_2 = pyo.Constraint(model.T, rule=segment_dyn_2_rule)

    # Grenzen fuer Gesamt- und Segmentenergien sowie Leistungsgrenzen
    def e_bounds_rule(m, t):
        return pyo.inequality(e_min_wh, m.e[t], e_max_wh)

    def e_seg1_bounds_rule(m, t):
        return pyo.inequality(0.0, m.e_seg1[t], seg1_cap_wh)

    def e_seg2_bounds_rule(m, t):
        return pyo.inequality(0.0, m.e_seg2[t], seg2_cap_wh)

    def p_ch_lim_rule(m, t):
        return m.p_ch[t] <= m.u[t] * p_ch_max_w

    def p_dis_lim_rule(m, t):
        return m.p_dis[t] <= (1 - m.u[t]) * p_dis_max_w

    model.e_bounds = pyo.Constraint(model.Te, rule=e_bounds_rule)
    model.e_seg1_bounds = pyo.Constraint(model.Te, rule=e_seg1_bounds_rule)
    model.e_seg2_bounds = pyo.Constraint(model.Te, rule=e_seg2_bounds_rule)
    model.p_ch_lim = pyo.Constraint(model.T, rule=p_ch_lim_rule)
    model.p_dis_lim = pyo.Constraint(model.T, rule=p_dis_lim_rule)

    # Import/Export-Exklusivitaet ueber zeitschrittspezifische Big-M-Grenzen.
    m_in = [max(float(load[t]), 0.0) + p_ch_max_w for t in range(n_steps)]
    m_out = [max(float(pv[t]), 0.0) + p_dis_max_w for t in range(n_steps)]

    def grid_in_mode_rule(m, t):
        return m.p_grid_in[t] <= m.z_grid_import[t] * m_in[t]

    def grid_out_mode_rule(m, t):
        return m.p_grid_out[t] <= (1 - m.z_grid_import[t]) * m_out[t]

    model.grid_in_mode = pyo.Constraint(model.T, rule=grid_in_mode_rule)
    model.grid_out_mode = pyo.Constraint(model.T, rule=grid_out_mode_rule)


    # ------------------------------------------------------------
    # 6) Initial-/Terminalbedingungen

    # Anfangs-SoC wird in Grenzen geclippt, dann auf Segmente verteilt.
    # Verteilregel: zuerst Segment 2 auffuellen, Rest in Segment 1.
    initial_soc_clipped_wh = float(min(max(initial_soc_wh, e_min_wh), e_max_wh))
    initial_usable_wh = initial_soc_clipped_wh - e_min_wh
    e_seg2_init_wh = min(initial_usable_wh, seg2_cap_wh)
    e_seg1_init_wh = max(initial_usable_wh - seg2_cap_wh, 0.0)

    model.e_init = pyo.Constraint(expr=model.e[0] == initial_soc_clipped_wh)
    model.e_seg1_init = pyo.Constraint(expr=model.e_seg1[0] == e_seg1_init_wh)
    model.e_seg2_init = pyo.Constraint(expr=model.e_seg2[0] == e_seg2_init_wh)
    if enforce_terminal_soc:
        model.e_terminal = pyo.Constraint(expr=model.e[n_steps] == terminal_soc_wh)



    # ------------------------------------------------------------
    # 7) Solver-Aufruf und Optimalitaetscheck

    solver = pyo.SolverFactory(FIXED_SOLVER_NAME, executable=FIXED_SOLVER_EXECUTABLE)
    if solver is None or not solver.available():
        raise RuntimeError(f"Solver nicht verfuegbar: {FIXED_SOLVER_NAME}")

    
    # hier startet optimierung
    # Nach solve(...) schreibt Pyomo die optimalen Werte in die Modellvariablen -> pyo.value(model.variable) gibt den optimalen Wert der Variable/zeitreihenwerte zurueck
    res = solver.solve(model, tee=bool(FIXED_SOLVER_TEE))
    
    status = str(res.solver.status).lower()
    termination = str(res.solver.termination_condition).lower()
    if "ok" not in status or "optimal" not in termination:
        raise RuntimeError(f"Solver-Ergebnis nicht optimal: status={status}, termination={termination}")

    # ------------------------------------------------------------
    # 8) Loesung in DataFrame mappen und KPI-Spalten berechnen

    solved = pd.DataFrame(
        {
            "datetime": time_values,
            "input_P_load_W": load,
            "input_P_pv_forecast_W": pv,
            "input_price_EUR_MWh": price,
            "opt_P_grid_in_W": [pyo.value(model.p_grid_in[t]) for t in range(n_steps)],
            "opt_P_grid_out_W": [pyo.value(model.p_grid_out[t]) for t in range(n_steps)],
            "opt_P_ch_W": [pyo.value(model.p_ch[t]) for t in range(n_steps)],
            "opt_P_dis_W": [pyo.value(model.p_dis[t]) for t in range(n_steps)],
            "opt_P_ch_seg1_W": [pyo.value(model.p_ch_seg1[t]) for t in range(n_steps)],
            "opt_P_ch_seg2_W": [pyo.value(model.p_ch_seg2[t]) for t in range(n_steps)],
            "opt_P_dis_seg1_W": [pyo.value(model.p_dis_seg1[t]) for t in range(n_steps)],
            "opt_P_dis_seg2_W": [pyo.value(model.p_dis_seg2[t]) for t in range(n_steps)],
            "opt_P_curt_W": [pyo.value(model.p_curt[t]) for t in range(n_steps)],
            "opt_u_charging": [int(round(pyo.value(model.u[t]))) for t in range(n_steps)],
            "opt_E_start_Wh": [pyo.value(model.e[t]) for t in range(n_steps)],
            "opt_E_end_Wh": [pyo.value(model.e[t + 1]) for t in range(n_steps)],
            "opt_E_seg1_start_Wh": [pyo.value(model.e_seg1[t]) for t in range(n_steps)],
            "opt_E_seg2_start_Wh": [pyo.value(model.e_seg2[t]) for t in range(n_steps)],
            "opt_E_seg1_end_Wh": [pyo.value(model.e_seg1[t + 1]) for t in range(n_steps)],
            "opt_E_seg2_end_Wh": [pyo.value(model.e_seg2[t + 1]) for t in range(n_steps)],
        }
    )
    
    # ------------------------------------------------------------
    # 8) Wiederholte Nachrechnung der Zielfunktionsbestandteile pro Zeitschritt fuer
    # transparentes Downstream-Reporting in den Szenario-Skripten.
    
    solved["opt_revenue_step_eur"] = (
        (solved["input_price_EUR_MWh"] * solved["opt_P_grid_out_W"])
        - ((solved["input_price_EUR_MWh"] + grid_fee) * solved["opt_P_grid_in_W"])
    ) * dt_h / 1e6
    
    solved["opt_cycle_aging_cost_step_eur"] = (
        (omega_1 * solved["opt_P_dis_seg1_W"]) + (omega_2 * solved["opt_P_dis_seg2_W"])
    ) * dt_h / 1e6
    
    solved["opt_revenue_step_after_aging_eur"] = solved["opt_revenue_step_eur"] - solved["opt_cycle_aging_cost_step_eur"]

    return {
        "df": solved,
        "objective_eur": float(pyo.value(model.obj)),
        "status": status,
        "termination": termination,
    }



#------------------------------------------------------------
# Postprocessing-Funktionen fuer Settlement-/Realisierungskennzahlen

def compute_dispatch_settlement(
    dispatch_df: pd.DataFrame,
    real_pv_series: pd.Series,
    cfg: dict,
    rebap_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Ergaenzt Dispatch-Zeilen um Realized- und reBAP-Settlement-Kennzahlen.

    Args:
        dispatch_df: Optimierter Dispatch (15-min Schritte).
        real_pv_series: Reale PV-Zeitreihe mit DatetimeIndex.
        cfg: Normalisierte Config; benoetigt `optimization.grid_fee_eur_per_mwh`.
        rebap_prices: Tabelle mit Spalte `reBAP_unterdeckt_eur_mwh`
            als signierter reBAP-Preis je Zeitschritt (DatetimeIndex erwartet).

    Returns:
        Dispatch-DataFrame inklusive Realized-/Settlement-Spalten.
        Dabei gilt:
        - `real_revenue_step_da_eur`: reiner Energieanteil auf realisierter Menge
          (ohne Grid Fee)
        - `real_grid_fee_step_eur`: Grid Fee nur auf realem Netzbezug
        - `real_revenue_step_eur`: DA-Fahrplanerloes mit Grid Fee auf realem Import
        - falls bereits ein intradayfaehiger Realpfad vorliegt, wird dieser
          direkt verwendet; sonst greift die bisherige fixed-battery-Nachrechnung

    Raises:
        RuntimeError: Wenn reale PV oder reBAP nicht vollstaendig auf
            Dispatch-Zeitpunkte gemappt werden koennen.
    """
    if dispatch_df.empty:
        return dispatch_df

    out = dispatch_df.copy()
    dt_h = FIXED_TIMESTEP_MINUTES / 60.0
    grid_fee = float(cfg["optimization"]["grid_fee_eur_per_mwh"])

    if "real_P_pv_W" not in out.columns:
        out["real_P_pv_W"] = real_pv_series.reindex(pd.to_datetime(out["datetime"])).to_numpy(dtype=float)

    if out["real_P_pv_W"].isna().any():
        raise RuntimeError("Reale PV konnte nicht vollstaendig auf Dispatch-Zeiten gemappt werden.")

    has_precomputed_real_path = all(
        col in out.columns
        for col in (
            "real_P_curt_W",
            "real_P_grid_in_W",
            "real_P_grid_out_W",
        )
    )
    aging_cost_col = "real_cycle_aging_cost_step_eur" if "real_cycle_aging_cost_step_eur" in out.columns else "opt_cycle_aging_cost_step_eur"

    if not has_precomputed_real_path:
        # Realisierte Curtailment-Leistung:
        # - niemals > geplanter Curtailment-Setpoint
        # - nur bei negativem DA-Preis
        # - nur aus tatsaechlichem PV-Ueberschuss (keine "Curtailment+Import"-Situationen)
        net_without_curt_real = (
            out["input_P_load_W"]
            + out["opt_P_ch_W"]
            - out["real_P_pv_W"]
            - out["opt_P_dis_W"]
        )
        # positiv: realer Netzbezug (Import) = Last + Ladung - PV - Entladung > 0
        # negativ: realer Ueberschuss vor Curtailment => Export
        # davon nur den realen Ueberschuss
        export_surplus_real_w = (-net_without_curt_real).clip(lower=0.0)
        # Zeitpunkte mit negativem DA-Preis identifizieren (nur dort ist Curtailment erlaubt)
        price_is_negative = out["input_price_EUR_MWh"] < 0.0

        out["real_P_curt_W"] = np.where(
            price_is_negative.to_numpy(dtype=bool),
            np.minimum(
                np.minimum(
                    out["opt_P_curt_W"].to_numpy(dtype=float),
                    export_surplus_real_w.to_numpy(dtype=float),
                ),
                out["real_P_pv_W"].clip(lower=0.0).to_numpy(dtype=float),
            ),
            0.0,
        )
        net_grid_real = net_without_curt_real + out["real_P_curt_W"]
        # Realen Netzbezug und reale Einspeisung aus dem saldierten Leistungsfehler berechnen
        out["real_P_grid_in_W"] = net_grid_real.clip(lower=0.0)
        out["real_P_grid_out_W"] = (-net_grid_real).clip(lower=0.0)

    # Nur als Diagnose:
    # - DA-Cashflow auf realisierter Menge (ohne Grid Fee)
    # - echte Grid Fee auf realem Netzbezug
    out["real_revenue_step_da_eur"] = (
        (out["input_price_EUR_MWh"] * out["real_P_grid_out_W"])
        - (out["input_price_EUR_MWh"] * out["real_P_grid_in_W"])
    ) * dt_h / 1e6
    out["real_grid_fee_step_eur"] = (
        grid_fee * out["real_P_grid_in_W"] * dt_h / 1e6
    )

    out["opt_P_grid_net_W"] = out["opt_P_grid_in_W"] - out["opt_P_grid_out_W"]  # nettoer Netzbezug laut Optimierung (positiv = Import, negativ = Export)
    out["real_P_grid_net_W"] = out["real_P_grid_in_W"] - out["real_P_grid_out_W"]  # nettoer Netzbezug in Realitaet (positiv = Import, negativ = Export)


    # reBAP-Preis auf Dispatch-Zeiten mappen und auf vollständigkeit validierem
    dispatch_idx = pd.to_datetime(out["datetime"])
    rebap_signed = rebap_prices["reBAP_unterdeckt_eur_mwh"].reindex(dispatch_idx)
    if rebap_signed.isna().any():
        missing_count = int(rebap_signed.isna().sum())
        raise RuntimeError(
            f"reBAP konnte nicht vollstaendig auf Dispatch-Zeiten gemappt werden (fehlend: {missing_count})."
        )
    
    
    #imbalance leistungen und energie berechnen
    imbalance_delta_w = out["real_P_grid_net_W"] - out["opt_P_grid_net_W"]
    # Signierte Imbalance-Leistung:
    # > 0: Unterdeckung (mehr Import als geplant), < 0: Ueberdeckung.
    out["real_imbalance_W"] = imbalance_delta_w
    out["real_imbalance_price_eur_mwh"] = rebap_signed.to_numpy(dtype=float)

    out["real_imbalance_energy_mwh"] = imbalance_delta_w * dt_h / 1e6

    # Signierte Settlement-Logik gemaess Vorzeichenkonvention:
    # settlement = - (Imbalance-Energie) * reBAP
    # Interpretation:
    # - payment (Kosten)  -> negativ
    # - revenue (Erloes)  -> positiv
    out["real_rebap_settlement_step_eur"] = -out["real_imbalance_energy_mwh"] * out["real_imbalance_price_eur_mwh"]

    # Fachlogik:
    # - DA-Energie bleibt auf dem geplanten Fahrplan
    # - Grid Fee faellt nur auf realen Netzbezug an
    planned_da_energy_step_eur = (
        (out["input_price_EUR_MWh"] * out["opt_P_grid_out_W"])
        - (out["input_price_EUR_MWh"] * out["opt_P_grid_in_W"])
    ) * dt_h / 1e6
    out["real_revenue_step_eur"] = (
        planned_da_energy_step_eur - out["real_grid_fee_step_eur"]
    )
    out["real_revenue_step_net_eur"] = out["real_revenue_step_eur"] + out["real_rebap_settlement_step_eur"]
    out["real_revenue_step_net_after_aging_eur"] = (
        out["real_revenue_step_net_eur"] - out[aging_cost_col]
    )

    return out


def aggregate_run_realized_metrics(dispatch_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregiert Realized-/Imbalance-Kennzahlen auf Run-Ebene.

    Args:
        dispatch_df: Dispatch-DataFrame mit Spalten
            `run_id`, `real_revenue_step_eur`,
            `real_rebap_settlement_step_eur`,
            `real_revenue_step_net_eur`,
            `real_revenue_step_net_after_aging_eur`.

    Returns:
        DataFrame mit:
        - `run_id`
        - `real_revenue_dispatch_day_eur`
        - `real_rebap_settlement_dispatch_day_eur`
        - `real_revenue_dispatch_day_net_eur`
        - `real_revenue_dispatch_day_net_after_aging_eur`
    """
    
    cols = [
        "run_id",
        "real_revenue_step_eur",
        "real_rebap_settlement_step_eur",
        "real_revenue_step_net_eur",
        "real_revenue_step_net_after_aging_eur",
    ]
    if dispatch_df.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "real_revenue_dispatch_day_eur",
                "real_rebap_settlement_dispatch_day_eur",
                "real_revenue_dispatch_day_net_eur",
                "real_revenue_dispatch_day_net_after_aging_eur",
            ]
        )

    realized_by_run = (
        dispatch_df.groupby("run_id", as_index=False)[cols[1:]]
        .sum()
        .rename(
            columns={
                "real_revenue_step_eur": "real_revenue_dispatch_day_eur",                       #DA-Fahrplanerloes pro Tag mit Grid Fee auf realem Import
                "real_rebap_settlement_step_eur": "real_rebap_settlement_dispatch_day_eur",     #imbalance erloes/ kosten pro Tag
                "real_revenue_step_net_eur": "real_revenue_dispatch_day_net_eur",               #netto Erloes pro Tag (DA-Erloes + reBAP-Settlement)
                "real_revenue_step_net_after_aging_eur": "real_revenue_dispatch_day_net_after_aging_eur",  # netto inkl. reBAP und Alterungskosten
            }
        )
    )
    return realized_by_run


__all__ = [
    "solve_milp_for_horizon",
    "compute_dispatch_settlement",
    "aggregate_run_realized_metrics",
]
