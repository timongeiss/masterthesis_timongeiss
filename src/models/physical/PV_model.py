import pvlib
#from pvlib.pvsystem import PVSystem
#from pvlib.modelchain import ModelChain
import pandas as pd
import numpy as np
import logging
import warnings
#from datetime import datetime

_DESOTO_CACHE = {}
DNI_PHYSICAL_CAP_FACTOR = 1.2


#-------------------------------------
# PVLIB Objects und Rechenschritte
#-------------------------------------

# marked
def create_location_pvlib(inverter):

    PVLIBlocation = pvlib.location.Location(latitude = inverter["Lat"],
                                       longitude = inverter["Lon"],
                                       tz = 'Europe/Berlin',                        
                                       altitude = None,                             
                                       name = inverter["Name"])
    return PVLIBlocation


# marked
def calculate_solarposition(df_weather: pd.DataFrame, PVLIBlocation: pvlib.location.Location):      

    # Solarposition df berechnen
    PVLIBsolpos = pvlib.solarposition.get_solarposition(time = df_weather.index, 
                                                  latitude = PVLIBlocation.latitude, 
                                                  longitude = PVLIBlocation.longitude, 
                                                  altitude = None,                                   
                                                  pressure = None, 
                                                  method = 'nrel_numpy', 
                                                  temperature = df_weather["temp_air"])
    
    return PVLIBsolpos


# marked
def create_pvsystem_pvlib(pvsite):
    """
    Erstellt ein pvlib.PVSystem-Objekt mit den wichtigsten PV-spezifischen Konfigurationsparametern.

    Diese Funktion dient dazu, ein PVSystem-Objekt für die spätere Skalierung der Modulleistung
    zu erstellen, basierend auf den Standort- und Systemdaten eines PV-Arrays.
    In diesem konkreten Fall wird das Objekt nicht für Irradiance- oder Temperaturmodelle verwendet, 
    sondern lediglich für Skalierung von Spannung, Strom und Leistung über `.scale_voltage_current_power(mpp)`

    Daher werden nur die Parameter gesetzt, die für diese Skalierung notwendig sind:
    
    Rückgabe:
    PVLIBsystem : pvlib.pvsystem.PVSystem: PVSystem-Objekt mit festgelegter Modulanzahl pro String und einem String pro Wechselrichter.

    - Die Angabe von `surface_tilt` und `surface_azimuth` ist in diesem Fall optional, 
      da diese Werte nicht verwendet werden, solange das PVSystem nicht für Transpositions- 
      oder Temperaturmodelle verwendet wird.
    - Sie können aber sinnvoll sein zur Dokumentation oder für zukünftige Nutzung mit z. B. 
      `ModelChain`.
    """

    PVLIBsystem = pvlib.pvsystem.PVSystem(arrays=None,
                                            surface_tilt = pvsite["Tilt"],
                                            surface_azimuth = pvsite["Azimuth"],
                                            albedo = None,
                                            surface_type = None,
                                            module = None,
                                            module_type = None,
                                            module_parameters = None,
                                            temperature_model_parameters = None,
                                            modules_per_string = pvsite["quantity"],
                                            strings_per_inverter = 1,
                                            inverter = None,
                                            inverter_parameters = None,
                                            racking_model = None,
                                            losses_parameters = None,
                                            name = pvsite["Name"])
    return PVLIBsystem





#-------------------------------------
# PREPROCESSING WEATHER
#-------------------------------------


# marked
def calculate_temperature(df_weather: pd.DataFrame):
    df_weather["temp_air"]  = df_weather["t_2m"] - 273.15
    return df_weather


# marked
def calculate_irradiation(df_weather: pd.DataFrame, PVLIBsolpos):
    '''
    Diese Funktion berechnet die Strahlungen die PVLIB erwartet aus den Werten des DWD je Standort.

    Dafür muss aus dem PvLib Objekt die solarposition, also die position der sonne je nach Tag Uhrzeit genutzt werden.
    Der Zenith Winkel entspricht dem rein rechnerischen Winkel von der Senkrechten zur erde nach unten gemessen. 0° ist Sonnenzenith und 90° ist sie quasi schon am horizont.
    Nun sehen wir die sonne je nach Druck und Temperatur oftmals länger. Das heisst atmosphärische Refraktion (brechung des Lichts über den Horizont). In der Variable apparent_zenith
    wird dies berücksichtigt. Dh rechnerisch ist die sonne bei 90° aber durch refraktion nur bei zb. 88°.
    
    Der DWD berechnet die direkte Strahlung (dni) bereits inkl. dem Solaren Einfallswinkel (dem cos_zenith) je Standort. Nun erwartet PVlib allerdings die Strahlung aus dem Zenith.
    Das liegt daran dass wir später nocht auf die horizontale Fläche rechnen möchten, sondern auf die geneigte. Für unser Transpositionsmodell brauchen wir daher die bereinigte Strahlung aus dem Zenith.
    Daher rechnen wir die aswdir durch den cos_zenith. Alle Winkel kleiner als cos(89) können zu numerischen rechenfehlern (da cos(90)=0 -> teilen durch 0) führen.
    PVlib erwartet außerdem noch den ghi. Normalerweise müsste man nun hier den zenithwinkel auf die direktestrahlung multiplizieren, allerdings ist das eben schon erledeigt vom DWD. Daher nur die addition der Winkel.

    '''
    # Solargeometrie berechnen
    zen = PVLIBsolpos["apparent_zenith"]
    cos_zenith = np.cos(np.radians(zen.clip(upper=89)))
    df_weather["cos_zenith"] = cos_zenith
    df_weather["zenith_deg"] = zen
    daylight = zen < 90

    # Strahlungen berechnen – robuste Behandlung an der Tag/Nacht-Grenze
    dni = pd.Series(0.0, index=df_weather.index)
    dni.loc[daylight] = (df_weather.loc[daylight, "aswdir_s"] / cos_zenith[daylight])
    df_weather["dni"] = dni.clip(lower=0)

    # Negative numerische Artefakte in den horizontalen Flüssen vermeiden
    df_weather["ghi"] = (df_weather["aswdir_s"] + df_weather["aswdifd_s"]).clip(lower=0)
    df_weather["dhi"] = df_weather["aswdifd_s"].clip(lower=0)

    return df_weather


# unsure
def calculate_dni_extra(df_weather: pd.DataFrame):
    '''
    Ghi, dni, dhi sind die tatsächlich ankommenden Werte an einem Standort mit berücksitigung Atmosphäre.
    get_extra_radiation liefert die theoretisch ankommende Strahlung außerhalb der Atmosphäre, identisch für alle Standorte, nur vom Datum abhängig.
    Das ist die Grundlage, um Modelle zu normalisieren oder zu verstehen, wie stark die Sonne maximal strahlt.

    Der Wert für die Solarkonstante wurde 2015 erneuert und löste den von 1982 festgelegten wert (1367 W/m2) ab.
    Als Solarkonstante E0 wird die langjährig gemittelte extraterrestrische Bestrahlungsstärke (Intensität) bezeichnet,
    die von der Sonne bei mittlerem Abstand Erde–Sonne ohne den Einfluss der Atmosphäre senkrecht zur Strahlrichtung auf die Erde auftrifft

    Methoden:
        spencer     (Standard) Formel von J.W. Spencer (1971). Eine häufig verwendete Näherungsformel, die mit Sinus/Kosinus-Terms arbeitet
        pyephem     Nutzt das pyephem-Paket, welches sehr genaue astronomische Berechnungen macht
        asce        American Society of Civil Engineers. Ähnlich präzise wie Spencer
        nrel        NREL (National Renewable Energy Laboratory
    '''

    df_weather["dni_extra"] = pvlib.irradiance.get_extra_radiation(datetime_or_doy = df_weather.index,
                                                                   solar_constant = 1361,                                
                                                                   method  = 'spencer')                                  # pyephem TESTEN
    return df_weather


def apply_physical_dni_cap(df_weather: pd.DataFrame, cap_factor: float = DNI_PHYSICAL_CAP_FACTOR):
    """
    Begrenzt DNI auf einen physikalisch plausiblen Bereich relativ zu dni_extra.
    Das vermeidet Horizon-Ausreisser durch aswdir_s / cos(zenith) nahe 90°.
    """
    dni = pd.to_numeric(df_weather["dni"], errors="coerce").to_numpy(dtype=float)
    dni_cap = (cap_factor * pd.to_numeric(df_weather["dni_extra"], errors="coerce")).to_numpy(dtype=float)

    valid_cap = np.isfinite(dni_cap)
    clipped_high = valid_cap & (dni > dni_cap)
    if np.any(clipped_high):
        logging.info(
            f"Clipped {int(np.sum(clipped_high))} DNI points above {cap_factor:.2f} * dni_extra."
        )
        dni = np.where(clipped_high, dni_cap, dni)

    df_weather["dni"] = pd.Series(dni, index=df_weather.index).clip(lower=0).fillna(0.0)
    return df_weather






def preprocess_weather(df_weather: pd.DataFrame, PVLIBlocation: pvlib.location.Location):

    # Index Datetime
    df_weather["timestamp"] = pd.to_datetime(df_weather["timestamp"]) 
    df_weather.set_index("timestamp", inplace=True)

    # Temperatur von Kelvin in Celsius
    df_weather = calculate_temperature(df_weather)

    # Solarpposition berechnen
    PVLIBsolpos = calculate_solarposition(df_weather, PVLIBlocation)

    # Solargeometrie korrigieren
    df_weather = calculate_irradiation(df_weather, PVLIBsolpos)

    # Extraterrestrial dni berechnen
    df_weather = calculate_dni_extra(df_weather)

    # Numerische Horizon-Ausreisser in der DNI-Reprojektion begrenzen
    df_weather = apply_physical_dni_cap(df_weather)

    return df_weather, PVLIBsolpos




#-------------------------------------
# TRANSPOSITIONSMODELL
#-------------------------------------


#marked
def calculate_POA(df_weather: pd.DataFrame, PVLIBsolpos: pd.DataFrame, pvsite):
    """
    Berechnet die Strahlung auf der geneigten Modulfläche (POA = Plane of Array Irradiance).

    POA beschreibt die gesamte solare Einstrahlung, die tatsächlich auf die Modulfläche trifft.
    Sie setzt sich zusammen aus:
      - direkter Strahlung (poa_direct),
      - diffuser Himmelsstrahlung (poa_sky_diffuse),
      - diffuser Bodenreflexion (poa_ground_diffuse).

    Returns
    PVLIBpoa : pd.DataFrame
        Enthält folgende Spalten:
        - 'poa_global'       : gesamte Einstrahlung auf der Modulfläche [W/m²]
        - 'poa_direct'       : direkter Anteil [W/m²]
        - 'poa_diffuse'      : diffuser Gesamtanteil [W/m²]
        - 'poa_sky_diffuse'  : diffuser Himmelanteil [W/m²]
        - 'poa_ground_diffuse': diffuser Bodenanteil [W/m²]

    """

    PVLIBpoa = pvlib.irradiance.get_total_irradiance(surface_tilt = pvsite["Tilt"],
                                                     surface_azimuth = pvsite["Azimuth"],
                                                     solar_zenith = PVLIBsolpos["apparent_zenith"],
                                                     solar_azimuth = PVLIBsolpos["azimuth"],
                                                     dni = df_weather["dni"],
                                                     ghi = df_weather["ghi"],
                                                     dhi = df_weather["dhi"],
                                                     dni_extra = df_weather["dni_extra"],
                                                     #airmass = None,
                                                     #albedo=  0.25,
                                                     #surface_type = None,
                                                     model = 'haydavies',
                                                     #model_perez = 'allsitescomposite1990'
                                                     )

    return PVLIBpoa


# marked
def calculate_AOI(PVLIBsolpos: pd.DataFrame, pvsite):
    """
    Berechnet den AOI (Angle of Incidence) für eine gegebene PV-Modulfläche.
    Der AOI ist der Winkel zwischen dem Sonnenstrahlvektor und der Senkrechten 
    (Oberflächennormalen) auf die Modulfläche. Er beschreibt, wie direkt die 
    Sonneneinstrahlung auf das Modul trifft.

    Rückgabe
    aoi : pd.Series oder np.ndarray
        Winkel des Einfalls in Grad für jeden Zeitschritt:
        - 0° = Sonne steht senkrecht auf der Modulfläche (maximale Einstrahlung)
        - 90° = Sonne streift die Modulfläche (keine wirksame Einstrahlung)

    Der AOI wird u. a. für optische Verluste oder Effizienzkorrekturen verwendet.
    Kleinere AOI-Werte führen zu höheren wirksamen Einstrahlungsanteilen.
    """

    PVLIBaoi = pvlib.irradiance.aoi(surface_tilt = pvsite["Tilt"],
                                    surface_azimuth = pvsite["Azimuth"],
                                    solar_zenith = PVLIBsolpos["apparent_zenith"],
                                    solar_azimuth = PVLIBsolpos["azimuth"])

    return PVLIBaoi


# marked
def calculate_IAM(PVLIBaoi):
    """
    Incidence Angle Modifier
    Ein Korrekturfaktor (zwischen 0 und 1), der berücksichtigt, dass eine Solarzelle je nach Einfallswinkel des Lichts nicht die volle Leistung liefert.
    Bei kleinen Einfallswinkeln (Sonne fast senkrecht) → kaum Verluste → IAM ≈ 1.
    Bei großen Einfallswinkeln (Sonne sehr flach) → Reflexionsverluste werden größer → IAM < 1.
    pvlib.iam.ashrae(aoi) wendet die ASHRAE‑Modellformel an, eine weit verbreitete empirische Funktion, um diesen Einfluss abzubilden.
    The incidence angle modifier is calculated as IAM=1−b(sec(aoi)−1)
    """

    PVLIBiam = pvlib.iam.ashrae(aoi = PVLIBaoi,
                                #b=0.05
                                )

    # Robustheit: IAM in physikalisch sinnvollen Bereich zwingen, Index erhalten
    return PVLIBiam.fillna(0).clip(lower=0, upper=1)


# marked
def transposition_model(df_weather: pd.DataFrame, PVLIBsolpos: pd.DataFrame, pvsite: pd.DataFrame):

    PVLIBpoa =  calculate_POA(df_weather, PVLIBsolpos, pvsite)

    PVLIBaoi = calculate_AOI(PVLIBsolpos, pvsite)

    PVLIBiam = calculate_IAM(PVLIBaoi)
    
    transposed_irradiance = (PVLIBpoa['poa_direct']*PVLIBiam
                            + PVLIBpoa['poa_sky_diffuse']
                            + PVLIBpoa['poa_ground_diffuse'])

    # Verhindere negative effektive Einstrahlung (z. B. bei extremen AOI)
    transposed_irradiance = transposed_irradiance.clip(lower=0)

    return transposed_irradiance, PVLIBpoa, PVLIBiam




#-------------------------------------
# TEMPERATURMODELL
#-------------------------------------

#marked
def temperature_model(PVLIBpoa: pd.DataFrame, df_weather: pd.DataFrame):
    """
    Berechnet die Modultemperatur mithilfe des Faiman-Modells gemäß IEC 61853.
    Das Faiman-Modell beschreibt den Temperaturanstieg eines PV-Moduls relativ zur 
    Umgebungstemperatur in Abhängigkeit von Einstrahlung und Windgeschwindigkeit.
    Das Modell nutzt die Standardparameter u0=25.0 W/m²K und u1=6.84 W/m²K/(m/s), empirisch ermittelt für kristalline Silizium-Module.
    Eine höhere Windgeschwindigkeit führt zu besserer Kühlung und damit geringerer Modultemperatur.
    Die Modultemperatur beeinflusst direkt den Wirkungsgrad der PV-Zellen und ist daher wichtig für die DC-Leistungsberechnung.

    Rückgabe     module_temperature : pd.Series oder np.ndarray: Modellierte Modultemperatur in Grad Celsius.
    """

    PVLIBcelltemperature = pvlib.temperature.faiman(poa_global = PVLIBpoa['poa_global'],
                                                    temp_air = df_weather['temp_air'],
                                                    wind_speed = df_weather['wind_speed'],
                                                    u0=35.7,
                                                    u1=6.84
                                                    )

    return PVLIBcelltemperature




#-------------------------------------
# ZELLMODELL
#-------------------------------------


def calculate_stc(modultyp:pd.DataFrame):
    """
    Diese Werte geben an, wie stark sich der Strom bzw. die Spannung bei Temperaturänderung verändern.
    Hier wird von den relativen (aus dem Datenblatt) auf die absoluten Werte gerechnet.
    stc_alpha_sc: die absolute Stromänderung pro °C in Ampere/°C
    stc_beta_voc: die absolute Spannungsänderung pro °C in Volt/°C -> Das Minuszeichen ist nötig, weil die Spannung mit steigender Temperatur abnimmt
    """

    stc_alpha_sc = modultyp["alpha_sc"] * modultyp["i_sc"] 
    stc_beta_voc = (-abs(modultyp["beta_voc"])) * modultyp["v_oc"]

    return stc_alpha_sc, stc_beta_voc



def _is_physical_desoto_fit(params: dict) -> bool:
    """
    Basale Plausibilitätsprüfung der De Soto Parameter.
    """
    required = ["I_L_ref", "I_o_ref", "R_s", "R_sh_ref", "a_ref"]
    values = [params.get(k, np.nan) for k in required]
    if not np.all(np.isfinite(values)):
        return False
    return (
        params["I_L_ref"] > 0
        and params["I_o_ref"] > 0
        and params["R_s"] >= 0
        and params["R_sh_ref"] > 0
        and params["a_ref"] > 0
    )


def calculate_desotocell(modultyp:pd.DataFrame, stc_alpha_sc: float, stc_beta_voc: float):
    """
    Ermittelt die elektrischen Modellparameter eines PV-Moduls anhand des De Soto Single-Diode-Modells.
    Diese Funktion nutzt typische Datenblattwerte eines PV-Moduls bei STC, um die Parameter
    fuer ein physikalisch fundiertes I-V-Kennlinienmodell (Ein-Dioden-Modell) zu berechnen.
    Die Methode folgt dem De Soto-Verfahren (IEC-konform) und basiert auf der Loesung
    nichtlinearer Gleichungssysteme mit `scipy.optimize.root`.
    """

    cache_key = (
        float(modultyp["v_mp"]),
        float(modultyp["i_mp"]),
        float(modultyp["v_oc"]),
        float(modultyp["i_sc"]),
        float(stc_alpha_sc),
        float(stc_beta_voc),
    )
    if cache_key in _DESOTO_CACHE:
        return _DESOTO_CACHE[cache_key]

    max_cells = 120
    best_nonphysical = None

    for cells in range(max_cells, 0, -1):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                result = pvlib.ivtools.sdm.fit_desoto(
                    v_mp=modultyp["v_mp"],
                    i_mp=modultyp["i_mp"],
                    v_oc=modultyp["v_oc"],
                    i_sc=modultyp["i_sc"],
                    alpha_sc=stc_alpha_sc,
                    beta_voc=stc_beta_voc,
                    cells_in_series=cells,
                    EgRef=1.121,
                    dEgdT=-0.0002677,
                    temp_ref=25,
                    irrad_ref=1000,
                )
            params = result[0]

            if _is_physical_desoto_fit(params):
                logging.info(f"Using De Soto fit with cells_in_series = {cells}")
                _DESOTO_CACHE[cache_key] = params
                return params

            if best_nonphysical is None:
                best_nonphysical = (cells, params)
            logging.info(
                f"Discarding non-physical De Soto fit at cells_in_series = {cells} "
                f"(R_s={params.get('R_s')}, R_sh_ref={params.get('R_sh_ref')}, a_ref={params.get('a_ref')})."
            )

        except RuntimeError as e:
            logging.info(f"Error: {e}")
            logging.info(f"Retrying with cells_in_series = {cells - 1}")

    if best_nonphysical is not None:
        cells, params = best_nonphysical
        logging.warning(
            "No physical De Soto fit found. Falling back to non-physical fit "
            f"with cells_in_series = {cells}."
        )
        _DESOTO_CACHE[cache_key] = params
        return params

    raise RuntimeError("Parameter estimation failed for all cells_in_series candidates.")


# marked
def calculate_desotomodule(transposed_irradiance: pd.DataFrame, PVLIBcelltemperature: pd.DataFrame, PVLIBdesotocell: dict):
    """
    Berechnet die elektrischen Modellparameter eines PV-Moduls für gegebene Einstrahlung und Zelltemperatur
    basierend auf dem De Soto Single-Diode-Modell. Die Funktion liefert die fünf Parameter der Ein-Dioden-Gleichung,
    die anschließend mit `pvlib.pvsystem.singlediode()` zu einem vollständigen I/V-Modell vervollständigt werden können.

    Typischer Einsatz:
    - Nach einer vorhergehenden Berechnung der STC-Modellparameter mit `fit_desoto()`.
    - Im PV-Leistungsmodell zur Bestimmung des Modulverhaltens unter realen Einstrahlungs- und Temperaturbedingungen.

    Eingaben:
    - `effective_irradiance`: Effektive Einstrahlung auf die Modulfläche [W/m²] (inkl. IAM-Korrektur).
    - `temp_cell`: Zelltemperatur [°C].
    - `alpha_sc`: Temperaturkoeffizient des Kurzschlussstroms [A/K].
    - `a_ref`: Modifizierter Idealfaktor bei STC (n·Ns·V_th) [V].
    - `I_L_ref`: Lichtstrom bei STC [A].
    - `I_o_ref`: Sättigungsstrom der Diode bei STC [A].
    - `R_sh_ref`: Shuntwiderstand bei STC [Ω].
    - `R_s`: Serienwiderstand bei STC [Ω].
    - `EgRef`: Bandlücke des Halbleitermaterials bei 25 °C [eV] (Standard: 1.121 eV für Silizium).
    - `dEgdT`: Temperaturabhängigkeit der Bandlücke [1/K] (Standard: –0.0002677).
    - `irrad_ref`: Referenz-Einstrahlung [W/m²] (Standard: 1000).
    - `temp_ref`: Referenz-Temperatur [°C] (Standard: 25).

    Rückgabe:
    Tuple mit den fünf parametrisierten Werten für das Ein-Dioden-Modell:
    - `photocurrent` (I_L)     : Lichtstrom bei gegebenen Bedingungen [A]
    - `saturation_current` (I_o): Sättigungsstrom bei gegebenen Bedingungen [A]
    - `resistance_series` (R_s) : Serienwiderstand [Ω]
    - `resistance_shunt` (R_sh) : Shuntwiderstand [Ω]
    - `nNsVth`                  : Idealfaktor-Term n·Ns·V_th [V]

    Hinweis:
    Die berechneten Werte sind Eingangsgrößen für `pvlib.pvsystem.singlediode()` zur Simulation der I/V-Kennlinie
    unter realen Bedingungen. Das Modell berücksichtigt Temperatur- und Einstrahlungseinflüsse detaillierter als
    vereinfachte Modelle (z. B. PVWatts).
    """

    PVLIBdesotomodule = pvlib.pvsystem.calcparams_desoto(effective_irradiance = transposed_irradiance,
                                                         temp_cell = PVLIBcelltemperature,
                                                         alpha_sc = PVLIBdesotocell['alpha_sc'], 
                                                         a_ref = PVLIBdesotocell['a_ref'], 
                                                         I_L_ref = PVLIBdesotocell['I_L_ref'], 
                                                         I_o_ref = PVLIBdesotocell['I_o_ref'], 
                                                         R_sh_ref = PVLIBdesotocell['R_sh_ref'], 
                                                         R_s = PVLIBdesotocell['R_s'], 
                                                         EgRef = 1.121,
                                                         dEgdT = -0.0002677,
                                                         irrad_ref = 1000,
                                                         temp_ref = 25)
    return PVLIBdesotomodule


# marked
def calculate_mpp(PVLIBdesotomodule: pd.DataFrame):
    """
    Berechnet den maximalen Leistungspunkt (MPP) eines PV-Moduls anhand der Ein-Dioden-Modellparameter.
    Im Gegensatz zu `singlediode()` liefert diese Funktion nur den MPP (d. h. Strom, Spannung und Leistung am Arbeitspunkt),
    nicht aber die gesamte I/V-Kurve.

    Typische Anwendung:
    - Nach `calcparams_desoto()` zur Bestimmung des aktuellen MPP bei gegebenen Einstrahlungs- und Temperaturwerten.
    - Für schnelle Leistungsprognosen ohne detaillierte Kurvenanalyse.

    Optionale Parameter (v. a. für CdTe oder a-Si):
    - `d2mutau`             : Rekombinationsparameter für Dünnschichtmodule [V] (Standard: 0)
    - `NsVbi`               : Built-in Voltage × Zellanzahl für Dünnschichtmodule [V] (Standard: ∞)
    - `method`              : Rechenmethode – 'brentq' (robust) oder 'newton' (schneller, aber potenziell instabil)

    Rückgabe:
    - `OrderedDict` mit:
    - `i_mp`: Strom am MPP [A]
    - `v_mp`: Spannung am MPP [V]
    - `p_mp`: Leistung am MPP [W]

    Hinweis:
    - `brentq` ist der Standard, da er konvergiert, auch wenn die Ableitung schlecht ist.
    - `newton` ist schneller, aber empfindlicher gegenüber schlechten Startwerten.
    """

    # pvlib.pvsystem.calcparams_desoto returns a tuple:
    # (photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth)
    phot, sat, rs, rsh, nns_vth = PVLIBdesotomodule

    def _solve(method: str) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            return pvlib.pvsystem.singlediode(
                photocurrent=phot,
                saturation_current=sat,
                resistance_series=rs,
                resistance_shunt=rsh,
                nNsVth=nns_vth,
                method=method,
            )[["i_mp", "v_mp", "p_mp"]]

    try:
        PVLIBmpp = _solve("lambertw")
    except ValueError as e:
        if "upper >= lower is required" not in str(e):
            raise
        logging.warning("MPP lambertw failed with 'upper >= lower'; applying brentq fallback.")
        PVLIBmpp = _solve("brentq")

    # Physically, MPP values cannot be negative or non-finite.
    PVLIBmpp = PVLIBmpp.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    PVLIBmpp["i_mp"] = PVLIBmpp["i_mp"].clip(lower=0)
    PVLIBmpp["v_mp"] = PVLIBmpp["v_mp"].clip(lower=0)
    PVLIBmpp["p_mp"] = PVLIBmpp["p_mp"].clip(lower=0)

    return PVLIBmpp


# marked
def cell_model(modultyp:pd.DataFrame, PVLIBcelltemperature:pd.DataFrame, transposed_irradiance: pd.DataFrame):

    stc_alpha_sc, stc_beta_voc= calculate_stc(modultyp)

    PVLIBdesotocell = calculate_desotocell(modultyp, stc_alpha_sc, stc_beta_voc)

    PVLIBdesotomodule = calculate_desotomodule(transposed_irradiance, PVLIBcelltemperature, PVLIBdesotocell)

    PVLIBmpp = calculate_mpp(PVLIBdesotomodule)

    return PVLIBmpp





#-------------------------------------
# LEISTUNGSMODELL
#-------------------------------------

# marked
def scale_module_to_system_pdc(PVLIBmpp: pd.DataFrame, pvsite: dict):
    """
    Skaliert die MPP-Leistung eines Moduls auf Systemebene (String × Module) und
    gibt die resultierende DC-Leistung als Zeitreihe zurück.

    Verwendet `PVSystem.scale_voltage_current_power(mpp)` mit
    `modules_per_string = pvsite["quantity"]` und `strings_per_inverter = 1`.

    Rückgabe
    pdc : pd.Series
        Skaliertes `p_mp` (DC-Leistung) je Zeitstempel.
    """

    PVLIBsystem = create_pvsystem_pvlib(pvsite)
    DCscaled = PVLIBsystem.scale_voltage_current_power(PVLIBmpp)
    return DCscaled["p_mp"]


# marked
def calculate_acpower_from_pdc(pdc: pd.Series, inverter: dict):
    """
    Berechnet die AC-Leistung aus der gesamten DC-Leistung eines Inverters
    unter Verwendung des PVWatts-InvertermodeIls.

    Erwartet die bereits aufsummierte DC-Leistung aller Strings (pdc).
    """

    ACpower = pvlib.inverter.pvwatts(
        pdc=pdc,
        pdc0=inverter["dc_max"],
        eta_inv_nom=inverter["eta_max"],
    )
    return ACpower


