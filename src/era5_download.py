"""Télécharge une série temporelle ERA5-Land (point en Tunisie) via l'API Copernicus CDS.

Pré-requis (une seule fois) :
  1) compte gratuit sur https://cds.climate.copernicus.eu
  2) page du jeu de données 'ERA5-Land hourly time-series data from 1950 to present' -> accepter la licence
  3) fichier ~/.cdsapirc :
        url: https://cds.climate.copernicus.eu/api
        key: <VOTRE-JETON-PERSONNEL>
  pip install "cdsapi>=0.7.2"
Usage : python -m src.era5_download
Si CDS refuse la requête, copiez la requête affichée par le bouton 'Show API request' de la page du dataset.
"""
import zipfile

import cdsapi

from . import config as C

DATASET = "reanalysis-era5-land-timeseries"
REQUEST = {
    "variable": [
        "2m_temperature",
        "total_precipitation",
        "surface_solar_radiation_downwards",
        "volumetric_soil_water_layer_1",
        "volumetric_soil_water_layer_2",
        "volumetric_soil_water_layer_3",
        "volumetric_soil_water_layer_4",
    ],
    "location": {"longitude": C.ERA5_LON, "latitude": C.ERA5_LAT},
    "date": [f"{C.ERA5_START}/{C.ERA5_END}"],
    "data_format": "csv",
}


def main():
    out = C.DATA_RAW / "era5land_point.zip"
    print("Requête envoyée (peut attendre plusieurs minutes dans la file CDS)...")
    cdsapi.Client().retrieve(DATASET, REQUEST, str(out))
    if zipfile.is_zipfile(out):
        with zipfile.ZipFile(out) as z:
            z.extractall(C.DATA_RAW / "era5land_point")
            print("Fichiers extraits :", z.namelist())
    print("OK ->", out)


if __name__ == "__main__":
    main()
