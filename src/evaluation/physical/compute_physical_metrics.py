import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Installierte PV-Leistung in Watt (Normalisierung)
INSTALLED_CAPACITY_W = 9720.0  
DROP_INITIAL_DAYS = 4
DROP_FINAL_DAYS = 2

# Farben
PLOT_COLORS_SINGLE = {
    "face": "#a6cee3",
    "edge": "#1f78b4",
    "median": "#e31a1c",
    "line": "#ff7f00",
}
PLOT_COLORS_COMBINED = {
    "all": {"face": "#a6cee3", "edge": "#000000"},
    "intra_day": {"face": "#b2df8a", "edge": "#000000"},
    "day_ahead": {"face": "#fb9a99", "edge": "#000000"},
}



# ============================================================
# Helper
# ============================================================

def get_run_date_from_filename(path):
    """
    Liest Datum (YYYYMMDD) aus dem Dateinamen.
    Rückgabe: Timestamp auf Mitternacht oder NaT
    """
    base = os.path.basename(path)                       # nur Dateiname ohne Pfad
    digits = "".join(ch for ch in base if ch.isdigit()) # alle Ziffern aus dem Namen ziehen
    if len(digits) >= 8:                                # mindestens 8 Ziffern für YYYYMMDD nötig
        return pd.to_datetime(digits[:8],               # erste 8 Ziffern als Datum interpretieren
                              format="%Y%m%d",
                              errors="coerce")          # ungültig -> NaT
    return pd.NaT                                       # nichts Passendes gefunden


def collect_available_days(csv_paths):
    """
    Ermittelt alle verfügbaren Tage direkt aus den valid_time-Spalten
    der vorhandenen CSV-Dateien (normalisiert auf Mitternacht).
    """
    days = set()
    for path in csv_paths:
        try:
            df = pd.read_csv(
                path,
                usecols=["valid_time"],
                encoding="utf-8",
            )
        except Exception:
            continue

        vt = pd.to_datetime(df["valid_time"], errors="coerce")
        days.update(vt.dropna().dt.normalize().unique().tolist())

    if not days:
        return pd.DatetimeIndex([])

    return pd.DatetimeIndex(sorted(days))


def init_agg():
    """
    Erzeugt einfache Zähler für MAE/RMSE pro Gruppe.
    sum_abs: Summe der Absolutfehler
    sum_sq:  Summe der quadrierten Fehler
    count:   Anzahl Werte
    """
    return {
        "all": {"sum_abs": 0.0, "sum_sq": 0.0, "count": 0.0},        # alle Samples
        "intra_day": {"sum_abs": 0.0, "sum_sq": 0.0, "count": 0.0},  # gleicher Tag wie Lauf
        "day_ahead": {"sum_abs": 0.0, "sum_sq": 0.0, "count": 0.0},  # Folgetage nach Lauf
    }


def add_samples(bucket, errors):
    """
    Fügt Fehlerwerte zu einem Zähler hinzu (für MAE/RMSE).
    """
    errs = np.asarray(list(errors), dtype=float)       # wandelt errors in ein NumPy-Array um (falls es z. B. noch eine Pandas-Serie ist)
    if errs.size == 0:                                 # keine Werte -> nichts zu tun
        return
    bucket["sum_abs"] += float(np.sum(np.abs(errs)))  # Summe |e| Summiert alle Absolutfehler → Grundlage für MAE
    bucket["sum_sq"]  += float(np.sum(errs ** 2))     # Summe e^2 Summiert alle quadrierten Fehler → Grundlage für RMSE
    bucket["count"]   += float(errs.size)             # Anzahl N


def time_labels_15min(n):
    """
    Erzeugt HH:MM-Labels in 15-Minuten-Schritten ohne Tageswechsel.
    """
    labels = []                                           # Ergebnisliste
    for i in range(n):                                    # für jeden Schritt
        total_minutes = i * 15                            # Minuten seit 00:00
        h = total_minutes // 60                           # ganze Stunden
        m = total_minutes % 60                            # Restminuten
        labels.append(f"{h:02d}:{m:02d}")                # Format HH:MM
    return labels


def ensure_array(v):
    """
    Gibt bei leerer Liste ein [NaN] zurück, damit Boxplot nicht laggt.
    """
    if len(v) == 0:                                       # leere Liste?
        return np.array([np.nan], dtype=float)            # -> [NaN]
    return np.array(v, dtype=float)                       # sonst als float-Array


def as_float_series(s):
    """
    Wandelt eine Series robust in float um:
    - Leerzeichen entfernen
    - Komma zu Punkt
    - Fehler zu NaN
    """
    s = s.astype(str)                                     # alles zu String
    s = s.str.replace(" ", "", regex=False)               # Leerzeichen raus
    s = s.str.replace(",", ".", regex=False)              # deutsches Komma -> Punkt
    return pd.to_numeric(s, errors="coerce")              # in float, Fehler -> NaN







# ============================================================
# PLOT-HILFE
# ============================================================

def style_single_boxplot(bp):
    # Boxen einfärben (Fläche + Rand)
    for box in bp["boxes"]:
        box.set(facecolor=PLOT_COLORS_SINGLE["face"],
                alpha=0.5,
                edgecolor=PLOT_COLORS_SINGLE["edge"])
    # Whisker einfärben
    for w in bp["whiskers"]:
        w.set(color=PLOT_COLORS_SINGLE["edge"])
    # Kappen einfärben
    for c in bp["caps"]:
        c.set(color=PLOT_COLORS_SINGLE["edge"])
    # Medianlinie einfärben
    for m in bp["medians"]:
        m.set(color=PLOT_COLORS_SINGLE["median"])


def style_combined_boxplot(bp, group):
    face = PLOT_COLORS_COMBINED[group]["face"]            # Gruppenfarbe (Fläche)
    edge = PLOT_COLORS_COMBINED[group]["edge"]            # Gruppenfarbe (Rand)
    for box in bp["boxes"]:
        box.set(facecolor=face, edgecolor=edge)           # Box einfärben
    for w in bp["whiskers"]:
        w.set(color=edge)                                 # Whisker einfärben
    for c in bp["caps"]:
        c.set(color=edge)                                 # Kappen einfärben
    for m in bp["medians"]:
        m.set(color=edge)                                 # Median einfärben




# ===== Einzeltag, nMAE =====

def plot_day_group_boxplots(day_str, group_name, per_step_values, out_dir):
    """
    Boxplots pro 15-Min-Schritt für einen Tag und eine Gruppe (nMAE).
    """
    n_steps = 96                                          # 96 Viertelstunden pro Tag
    data = [ensure_array(vs) for vs in per_step_values]   # leere Listen zu [NaN] machen
    means = np.array([                                     # Mittel pro Schritt (NaN-sicher)
        np.nanmean(d) if np.any(~np.isnan(d)) else np.nan
        for d in data
    ])

    fig, ax = plt.subplots(figsize=(24, 8), dpi=150)      # Figure/Achse anlegen
    bp = ax.boxplot(                                      # Boxplot zeichnen
        data,
        positions=np.arange(n_steps),                     # 0..95 Positionen
        widths=0.6,                                       # Boxbreite
        showfliers=False,                                 # Ausreißer ausblenden
        patch_artist=True,                                # Flächen einfärbbar
    )
    style_single_boxplot(bp)                              # Boxplot-Stil anwenden
    ax.plot(                                              # Linienplot der Mittelwerte
        np.arange(n_steps),
        means,
        color=PLOT_COLORS_SINGLE["line"],
        marker="o",
        markersize=3,
        linewidth=1,
        label="Mean per step",
    )
    ax.set_title(f"{day_str} - {group_name} (normalized to 9.72 kWp)") # Titel
    ax.set_ylabel("nMAE")                                 # y-Label
    ax.set_xlabel("Time of day (15-min steps)")           # x-Label
    ax.set_xticks(np.arange(n_steps))                     # x-Ticks setzen
    ax.set_xticklabels(time_labels_15min(n_steps),        # Tick-Labels 00:00..23:45
                       rotation=90)
    ax.grid(axis="y", linestyle="--", alpha=0.4)          # horizontales Grid
    ax.legend()                                           # Legende anzeigen
    plt.tight_layout()                                    # Layout optimieren

    os.makedirs(out_dir, exist_ok=True)                   # Zielordner anlegen (falls nötig)
    out_path = os.path.join(out_dir,                      # Dateipfad erzeugen
                            f"boxplot_{day_str}_{group_name}.png")
    plt.savefig(out_path)                                 # Plot speichern
    plt.close(fig)                                        # Figure schließen (Speicher frei)
    return out_path                                       # Pfad zurückgeben

# ===== Über mehrere Tage, nMAE =====

def plot_combined_daily_boxes_across_days(day_labels, values_all, values_intra, values_da, out_dir):
    """
    Kombinierter Tages-Boxplot: pro Tag drei Boxen (All, Intra-day, Day-ahead) der nMAE-Samples.
    """
    n_days = len(day_labels)                               # Anzahl Tage
    centers = np.arange(n_days) * 3.0                      # Gruppenzentren (0,3,6,...)
    pos_all = centers - 0.8                                # Position Box "All"
    pos_intra = centers                                    # Position Box "Intra-day"
    pos_da = centers + 0.8                                 # Position Box "Day-ahead"

    fig, ax = plt.subplots(figsize=(max(16, n_days), 8), dpi=150) # dynamische Breite
    data_all = [ensure_array(v) for v in values_all]       # leere Listen absichern
    data_intra = [ensure_array(v) for v in values_intra]
    data_da = [ensure_array(v) for v in values_da]

    bp_all = ax.boxplot(data_all, positions=pos_all,       # Boxen „All“
                        widths=0.6, showfliers=False,
                        patch_artist=True)
    bp_intra = ax.boxplot(data_intra, positions=pos_intra, # Boxen „Intra-day“
                          widths=0.6, showfliers=False,
                          patch_artist=True)
    bp_da = ax.boxplot(data_da, positions=pos_da,          # Boxen „Day-ahead“
                       widths=0.6, showfliers=False,
                       patch_artist=True)

    style_combined_boxplot(bp_all, "all")                  # Farben anwenden
    style_combined_boxplot(bp_intra, "intra_day")
    style_combined_boxplot(bp_da, "day_ahead")

    ax.set_title("Daily error distributions (nMAE): All vs Intra-day vs Day-ahead") # Titel
    ax.set_ylabel("nMAE")                                 # y-Label
    ax.set_xlabel("Day")                                   # x-Label
    ax.set_xticks(centers)                                 # Ticks auf Gruppenzentren
    ax.set_xticklabels(day_labels, rotation=90)            # Tageslabels
    ax.grid(axis="y", linestyle="--", alpha=0.4)           # horizontales Grid
    ax.legend([bp_all["boxes"][0],                         # Legende mit Boxhandles
               bp_intra["boxes"][0],
               bp_da["boxes"][0]],
              ["All", "Intra-day", "Day-ahead"],
              loc="upper right")
    plt.tight_layout()                                     # Layout optimieren

    os.makedirs(out_dir, exist_ok=True)                    # Zielordner anlegen
    out_path = os.path.join(out_dir,                       # Dateipfad
                            "boxplot_days_combined_nMAE.png")
    plt.savefig(out_path)                                  # Plot speichern
    plt.close(fig)                                         # Figure schließen
    return out_path                                        # Pfad zurückgeben

# ===== Einzeltag, nRMSE =====

def plot_day_group_boxplots_nrmse(day_str, group_name, per_step_values, out_dir):
    """
    Boxplots pro 15-Min-Schritt für einen Tag und eine Gruppe (nRMSE).
    """
    n_steps = 96                                          # 96 Viertelstunden
    data = [ensure_array(vs) for vs in per_step_values]   # leere Listen absichern
    means = np.array([                                     # Mittel pro Schritt
        np.nanmean(d) if np.any(~np.isnan(d)) else np.nan
        for d in data
    ])

    fig, ax = plt.subplots(figsize=(24, 8), dpi=150)      # Figure/Achse
    bp = ax.boxplot(                                      # Boxplot zeichnen
        data,
        positions=np.arange(n_steps),
        widths=0.6,
        showfliers=False,
        patch_artist=True,
    )
    style_single_boxplot(bp)                              # Stil anwenden
    ax.plot(                                              # Mittelwerte als Linie
        np.arange(n_steps),
        means,
        color=PLOT_COLORS_SINGLE["line"],
        marker="o",
        markersize=3,
        linewidth=1,
        label="Mean per step",
    )
    ax.set_title(f"{day_str} - {group_name} (normalized to 9.72 kWp) - nRMSE") # Titel
    ax.set_ylabel("nRMSE")                               # y-Label
    ax.set_xlabel("Time of day (15-min steps)")          # x-Label
    ax.set_xticks(np.arange(n_steps))                    # x-Ticks
    ax.set_xticklabels(time_labels_15min(n_steps),       # Tick-Labels
                       rotation=90)
    ax.grid(axis="y", linestyle="--", alpha=0.4)         # Grid
    ax.legend()                                          # Legende
    plt.tight_layout()                                   # Layout

    os.makedirs(out_dir, exist_ok=True)                  # Ordner anlegen
    out_path = os.path.join(out_dir,                     # Dateipfad
                            f"boxplot_{day_str}_{group_name}_nRMSE.png")
    plt.savefig(out_path)                                # speichern
    plt.close(fig)                                       # schließen
    return out_path                                      # Pfad zurück


# ===== Über mehrere Tage, nRMSE =====

def plot_combined_daily_boxes_across_days_nrmse(day_labels, values_all, values_intra, values_da, out_dir):
    """
    Kombinierter Tages-Boxplot: pro Tag drei Boxen (All, Intra-day, Day-ahead) der nRMSE-Samples.
    """
    n_days = len(day_labels)                               # Anzahl Tage
    centers = np.arange(n_days) * 3.0                      # Gruppenzentren
    pos_all = centers - 0.8                                # Positionen
    pos_intra = centers
    pos_da = centers + 0.8

    fig, ax = plt.subplots(figsize=(max(16, n_days), 8), dpi=150) # Figure
    data_all = [ensure_array(v) for v in values_all]       # Daten absichern
    data_intra = [ensure_array(v) for v in values_intra]
    data_da = [ensure_array(v) for v in values_da]

    bp_all = ax.boxplot(data_all, positions=pos_all,       # Boxen zeichnen
                        widths=0.6, showfliers=False,
                        patch_artist=True)
    bp_intra = ax.boxplot(data_intra, positions=pos_intra,
                          widths=0.6, showfliers=False,
                          patch_artist=True)
    bp_da = ax.boxplot(data_da, positions=pos_da,
                       widths=0.6, showfliers=False,
                       patch_artist=True)

    style_combined_boxplot(bp_all, "all")                  # Farben setzen
    style_combined_boxplot(bp_intra, "intra_day")
    style_combined_boxplot(bp_da, "day_ahead")

    ax.set_title("Daily error distributions (nRMSE): All vs Intra-day vs Day-ahead") # Titel
    ax.set_ylabel("nRMSE")                                 # y-Label
    ax.set_xlabel("Day")                                   # x-Label
    ax.set_xticks(centers)                                 # Ticks
    ax.set_xticklabels(day_labels, rotation=90)            # Labels
    ax.grid(axis="y", linestyle="--", alpha=0.4)           # Grid
    ax.legend([bp_all["boxes"][0], bp_intra["boxes"][0], bp_da["boxes"][0]], # Legende
              ["All", "Intra-day", "Day-ahead"],
              loc="upper right")
    plt.tight_layout()                                     # Layout

    os.makedirs(out_dir, exist_ok=True)                    # Ordner anlegen
    out_path = os.path.join(out_dir,                       # Dateipfad
                            "boxplot_days_combined_nRMSE.png")
    plt.savefig(out_path)                                  # speichern
    plt.close(fig)                                         # schließen
    return out_path                                        # Pfad zurückgeben





# ============================================================
# HAUPTLOGIK
# ============================================================

def main():
    
    # ----- Inputs -------
    
    # Ordner anlegen/finden
    project_root = Path(__file__).resolve().parents[3]
    shared_dir = str(project_root / "data" / "results" / "physical_output")   # CSV-Quelle (PV_Calculator Exporte)
    export_dir = str(project_root / "reports" / "physical")  # Ergebnisziel (Plots + CSVs/Excel)
    os.makedirs(export_dir, exist_ok=True)                            # Zielordner sicherstellen

    # Alle CSV-Dateien laden (alphabetisch sortiert)
    csv_paths = sorted(glob.glob(os.path.join(shared_dir, "*.csv")))
    if not csv_paths:                                      # keine Dateien gefunden
        raise FileNotFoundError(f"No CSV files found in {shared_dir}.")

    days = collect_available_days(csv_paths)               # alle verfügbaren Tage aus den Daten laden
    if days.empty:
        raise ValueError("No valid days found in physical output CSVs.")
    min_required_days = DROP_INITIAL_DAYS + DROP_FINAL_DAYS + 1
    if len(days) < min_required_days:
        raise ValueError(
            f"Need at least {min_required_days} valid days, found only {len(days)}."
        )
    days = days[DROP_INITIAL_DAYS: len(days) - DROP_FINAL_DAYS]  # Tag 5 bis vorletzte 2 Tage

    # Speicherstrukturen vorbereiten
    by_day = {d: init_agg() for d in days}                 # Zähler pro Tag und Gruppe


    # ----- Sammelstrukturen anlegen -------

    # Sammelstrukturen pro 15-Minuten-Schritt (nMAE)
    # Erstellt für jeden Zieltag d ein eigenes Wörterbuch.
    # Dieses Wörterbuch hat 3 Schlüssel: Jeder dieser Schlüssel enthält eine Liste mit 96 leeren Listen da Ein Tag 24 h × 4 = 96 Viertelstunden
    # Jede der 96 Listen sammelt später die Fehlerwerte für genau diesen 15-Minuten-Zeitschritt über alle verfügbaren Modelle
    
    per_step_collectors = {
        d: {
            "all": [list() for _ in range(96)],            # 96 leere Listen pro Gruppe
            "intra_day": [list() for _ in range(96)],
            "day_ahead": [list() for _ in range(96)],
        }
        for d in days
    }
    
    # Sammelstrukturen pro 15-Minuten-Schritt nRMSE
    per_step_collectors_rmse = {
        d: {
            "all": [list() for _ in range(96)],
            "intra_day": [list() for _ in range(96)],
            "day_ahead": [list() for _ in range(96)],
        }
        for d in days
    }
    
    # Per-Tag-Samples (für Tages-Boxplots über Tage, nMAE)
    # nicht nach Viertelstunden getrennt, sondern pro Tag nur eine Liste geführt. Diese Struktur sammelt alle Fehlerwerte eines Tages
    per_day_values = {d: {"all": [], "intra_day": [], "day_ahead": []} for d in days}

    # Per-Tag-Samples (nRMSE)
    per_day_values_rmse = {d: {"all": [], "intra_day": [], "day_ahead": []} for d in days}
    
    
    # ----- Datei Läufer -------

    skipped = 0                                            # Zähler für übersprungene CSVs

    # Jede CSV-Datei verarbeiten
    for path in csv_paths:
        try:
            df = pd.read_csv(                              # nur benötigte Spalten lesen
                path,
                usecols=["valid_time", "pv_power_W", "mittelwertleistung [w]", "solar_elevation_deg"],
                encoding="utf-8",
            )
        except Exception:                                  # Datei unlesbar/Spalten fehlen
            skipped += 1                                   # übersprungen zählen
            continue

        run_date = get_run_date_from_filename(path)        # Laufdatum aus Dateiname

        # Spalten robust parsen/konvertieren
        vt = pd.to_datetime(df["valid_time"], errors="coerce")          # Zeitstempel
        y_pred = as_float_series(df["pv_power_W"]).to_numpy()           # Prognoseleistung
        y_ref  = as_float_series(df["mittelwertleistung [w]"]).to_numpy()# Referenzleistung
        solar  = as_float_series(df["solar_elevation_deg"]).to_numpy()  # Sonnenhöhe

        e = (y_pred - y_ref).astype(float)                # Fehler e = Prognose - Referenz

        valid_mask = (~np.isnan(e)) & (solar > 0)         # gültig & Tageslicht
        vt_dates = vt.dt.normalize().values               # nur das Datum (00:00)

        # Über alle Zieltage iterieren (days sind Zieltage Liste)
        for d in days:
            mask_day = valid_mask & (vt_dates == d.to_datetime64())  # Gibt es in dieser Datei Zeilen, deren Datum == d ist?
            if not np.any(mask_day):                       # keine Werte für diesen Tag
                continue

            add_samples(by_day[d]["all"], e[mask_day])     # Zähler „all“ --> summiert die Fehler dieses Tages für Metriken (MAE/RMSE) auf


            # --------- NMAE PLOTS ---------

            # nAE berechnen = |e| / P_inst (für Plots)
            if INSTALLED_CAPACITY_W > 0:
                nAE_vals = np.abs(e[mask_day]) / INSTALLED_CAPACITY_W
            else:
                nAE_vals = np.full(np.count_nonzero(mask_day), np.nan)

            vt_d = pd.to_datetime(vt.values[mask_day])     # Zeitstempel für die nAE-Werte

            # Einsortieren je 15-Min-Slot (00:00–23:45 → 96 Slots)
            for ts, val in zip(vt_d, nAE_vals):
                if np.isnan(val):                          # NaN überspringen
                    continue
                step = int(ts.hour) * 4 + int(ts.minute) // 15  # berechnet den Viertelstunden-Index im Tag 00:00 → 0, 00:15 → 1, …, 23:45 → 95.
                per_step_collectors[d]["all"][step].append(float(val))  # pro Schritt sammeln
                per_day_values[d]["all"].append(float(val))             # pro Tag sammeln

            # Intra-day/Day-ahead trennen (nur wenn run_date vorhanden)
            if pd.notna(run_date):
                run_day = run_date.normalize().to_datetime64()          # Laufdatum (nur Tag)

                intra_mask = mask_day & (vt_dates == run_day)           # intraday: gleiches Datum
                day_ahead_mask = mask_day & (vt_dates > run_day)        # day-ahead: späteres Datum

                # Intra-day sammeln
                if np.any(intra_mask):
                    add_samples(by_day[d]["intra_day"], e[intra_mask])  # Zähler „intra_day“
                    if INSTALLED_CAPACITY_W > 0:
                        nAE_intra = np.abs(e[intra_mask]) / INSTALLED_CAPACITY_W
                    else:
                        nAE_intra = np.full(np.count_nonzero(intra_mask), np.nan)
                    vt_i = pd.to_datetime(vt.values[intra_mask])        # Zeitstempel intra
                    for ts, val in zip(vt_i, nAE_intra):
                        if np.isnan(val):
                            continue
                        step = int(ts.hour) * 4 + int(ts.minute) // 15
                        per_step_collectors[d]["intra_day"][step].append(float(val))
                        per_day_values[d]["intra_day"].append(float(val))

                # Day-ahead sammeln
                if np.any(day_ahead_mask):
                    add_samples(by_day[d]["day_ahead"], e[day_ahead_mask]) # Zähler „day_ahead“
                    if INSTALLED_CAPACITY_W > 0:
                        nAE_da = np.abs(e[day_ahead_mask]) / INSTALLED_CAPACITY_W
                    else:
                        nAE_da = np.full(np.count_nonzero(day_ahead_mask), np.nan)
                    vt_da = pd.to_datetime(vt.values[day_ahead_mask])     # Zeitstempel day-ahead
                    for ts, val in zip(vt_da, nAE_da):
                        if np.isnan(val):
                            continue
                        step = int(ts.hour) * 4 + int(ts.minute) // 15
                        per_step_collectors[d]["day_ahead"][step].append(float(val))
                        per_day_values[d]["day_ahead"].append(float(val))
                        
                        
                        

            # --------- NRMSE PLOTS ---------

            # nRMSE = sqrt(e^2)/P_inst (betragsgleich, aber als RMSE-Form)
            if INSTALLED_CAPACITY_W > 0:
                nRMSE_vals = (np.sqrt(e[mask_day] ** 2)) / INSTALLED_CAPACITY_W
            else:
                nRMSE_vals = np.full(np.count_nonzero(mask_day), np.nan)

            # nRMSE für „all“ einsortieren
            for ts, val in zip(vt_d, nRMSE_vals):
                if np.isnan(val):
                    continue
                step = int(ts.hour) * 4 + int(ts.minute) // 15
                per_step_collectors_rmse[d]["all"][step].append(float(val))
                per_day_values_rmse[d]["all"].append(float(val))

            # nRMSE für intra/day-ahead (nur wenn run_date vorhanden)
            if pd.notna(run_date):
                if np.any(intra_mask):
                    if INSTALLED_CAPACITY_W > 0:
                        nRMSE_intra = (np.sqrt(e[intra_mask] ** 2)) / INSTALLED_CAPACITY_W
                    else:
                        nRMSE_intra = np.full(np.count_nonzero(intra_mask), np.nan)
                    for ts, val in zip(vt_i, nRMSE_intra):
                        if np.isnan(val):
                            continue
                        step = int(ts.hour) * 4 + int(ts.minute) // 15
                        per_step_collectors_rmse[d]["intra_day"][step].append(float(val))
                        per_day_values_rmse[d]["intra_day"].append(float(val))

                if np.any(day_ahead_mask):
                    if INSTALLED_CAPACITY_W > 0:
                        nRMSE_da = (np.sqrt(e[day_ahead_mask] ** 2)) / INSTALLED_CAPACITY_W
                    else:
                        nRMSE_da = np.full(np.count_nonzero(day_ahead_mask), np.nan)
                    for ts, val in zip(vt_da, nRMSE_da):
                        if np.isnan(val):
                            continue
                        step = int(ts.hour) * 4 + int(ts.minute) // 15
                        per_step_collectors_rmse[d]["day_ahead"][step].append(float(val))
                        per_day_values_rmse[d]["day_ahead"].append(float(val))



    # Zusammenfassung als CSV (pro Tag/Gruppe: nMAE, nRMSE)
    rows = []                                              # Zeilenliste für DataFrame
    for d, groups in by_day.items():                       # über Tage
        for gname, stats in groups.items():                # über Gruppen
            cnt = stats["count"]                           # Anzahl N
            if cnt > 0 and INSTALLED_CAPACITY_W > 0:       # nur wenn Daten vorhanden
                mae = stats["sum_abs"] / cnt               # MAE = Sum|e| / N
                rmse = float(np.sqrt(stats["sum_sq"] / cnt)) # RMSE = sqrt(Sum e^2 / N)
                nmae = mae / INSTALLED_CAPACITY_W          # nMAE normalisieren
                nrmse = rmse / INSTALLED_CAPACITY_W        # nRMSE normalisieren
            else:
                nmae = np.nan                              # keine Daten -> NaN
                nrmse = np.nan
            rows.append({                                  # eine Zeile anhängen
                "day": d.strftime("%Y-%m-%d"),
                "group": gname,
                "nMAE": nmae,
                "nRMSE": nrmse,
                "count": int(cnt),
            })

    summary = pd.DataFrame(rows)                           # DataFrame bauen
    summary = summary.rename(columns={"day": "date"})
    summary = summary.sort_values(["date", "group"])       # sortieren
    summary = summary.reset_index(drop=True)               # Index neu
    artifacts_dir = project_root / "data" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_summary = artifacts_dir / "physical_model_testday_metrics.csv"
    summary.to_csv(out_summary, index=False, decimal=",")  # CSV schreiben

    # Plots nach Tag und Gruppe (nMAE)
    box_dir = os.path.join(export_dir, "boxplots_by_day")  # Zielordner
    for d in days:
        day_str = d.strftime("%Y-%m-%d")                   # YYYY-MM-DD
        for group in ("all", "intra_day", "day_ahead"):    # drei Gruppen
            plot_day_group_boxplots(day_str, group,        # Plot erzeugen
                                    per_step_collectors[d][group],
                                    box_dir)

    # Kombinierte Tages-Boxplots (nMAE)
    day_labels = [d.strftime("%Y-%m-%d") for d in days]    # Labels für x-Achse
    across_dir = os.path.join(export_dir, "boxplots_days_across")  # Zielordner
    values_all  = [per_day_values[d]["all"] for d in days]         # Daten je Tag
    values_intra= [per_day_values[d]["intra_day"] for d in days]
    values_da   = [per_day_values[d]["day_ahead"] for d in days]
    plot_combined_daily_boxes_across_days(day_labels,      # Plot erzeugen
                                          values_all, values_intra, values_da,
                                          across_dir)

    # Kombinierte Tages-Boxplots (nRMSE)
    values_all_r  = [per_day_values_rmse[d]["all"] for d in days]
    values_intra_r= [per_day_values_rmse[d]["intra_day"] for d in days]
    values_da_r   = [per_day_values_rmse[d]["day_ahead"] for d in days]
    plot_combined_daily_boxes_across_days_nrmse(day_labels,# Plot erzeugen
                                                values_all_r, values_intra_r, values_da_r,
                                                across_dir)

    # Plots nach Tag und Gruppe (nRMSE)
    box_dir_rmse = os.path.join(export_dir, "boxplots_by_day_nrmse")
    for d in days:
        day_str = d.strftime("%Y-%m-%d")
        for group in ("all", "intra_day", "day_ahead"):
            plot_day_group_boxplots_nrmse(day_str, group, # Plot erzeugen
                                          per_step_collectors_rmse[d][group],
                                          box_dir_rmse)

    # Kurzer Abschluss
    print(f"Done. Files processed: {len(csv_paths)} (skipped: {skipped}).")  # Status
    print(f"Exported CSV: {out_summary}")                                     # Pfad CSV
    print(f"Per-day plots: {box_dir}")                                        # Ordner nMAE Tag
    print(f"Combined daily plot (nMAE): {os.path.join(across_dir, 'boxplot_days_combined_nMAE.png')}") # Kombiniert
    print(f"Combined daily plot (nRMSE): {os.path.join(across_dir, 'boxplot_days_combined_nRMSE.png')}")# Kombiniert



if __name__ == "__main__":
    main()
