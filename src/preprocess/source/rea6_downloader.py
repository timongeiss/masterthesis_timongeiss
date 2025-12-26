# check 29.11.2025

"""
Einfacher Downloader fuer DWD COSMO REA6 (stuendlich, 2D).
Laedt ASWDIR_S, ASWDIFD_S und T_2M fuer alle Monate 2014-2017,
entpackt .bz2 zu .grb und legt alles im Zielordner ab.
"""

import bz2
import os
import shutil
from pathlib import Path
import requests

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]


# Basis-URL des DWD-OpenData-Portals
BASE_URL = "https://opendata.dwd.de/climate_environment/REA/COSMO_REA6/hourly/2D"

# Gewuenschte Parameter und Jahre
PARAMETERS = ["ASWDIR_S", "ASWDIFD_S", "T_2M"]

YEARS = range(2014, 2018)


# Zielordner (Standard: projectroot/data/raw/source_feature_space/rea6)
DEFAULT_TARGET = PROJECT_ROOT / "data" / "raw" / "source_feature_space" / "rea6"
TARGET_ROOT = Path(os.environ.get("REA6_TARGET_DIR", DEFAULT_TARGET))

def download_and_extract(parameter: str) -> None:
    """
    Laedt alle Monatsdateien eines Parameters und entpackt sie.
    """
    
    for year in YEARS:
        for month in range(1, 13):
            month_tag = f"{year}{month:02d}"
            filename = f"{parameter}.2D.{month_tag}.grb.bz2"
            url = f"{BASE_URL}/{parameter}/{filename}"

            out_dir = TARGET_ROOT / parameter / str(year)
            bz2_path = out_dir / filename
            grb_path = bz2_path.with_suffix("")  # .bz2 entfernen -> .grb

            if grb_path.exists():
                print(f"Ueberspringe {grb_path} (bereits vorhanden)")
                continue

            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                # Download mit Stream, um RAM zu sparen
                print(f"Lade {url}")
                with requests.get(url, stream=True, timeout=60) as resp:
                    resp.raise_for_status() # OK-Status (200) weiter
                    with open(bz2_path, "wb") as f: #wb write binary
                        for chunk in resp.iter_content(chunk_size=1_048_576): # Datei stückweise herunterladen 1048576 = 1024 × 1024 = 1 MiB
                            if chunk:
                                f.write(chunk)

                # Entpacken von .bz2 nach .grb
                with bz2.open(bz2_path, "rb") as src, open(grb_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                bz2_path.unlink(missing_ok=True)
                print(f"Fertig: {grb_path}")
                
            except Exception as exc:  # pragma: no cover - Laufzeitfehler nur loggen
                print(f"Fehler bei {url}: {exc}")
                bz2_path.unlink(missing_ok=True)


def main() -> None:
    print(f"Zielordner: {TARGET_ROOT.resolve()}")
    for parameter in PARAMETERS:
        download_and_extract(parameter)


if __name__ == "__main__":
    main()
