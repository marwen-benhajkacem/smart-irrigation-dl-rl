"""Transforme le CSV horaire ERA5-Land en séries JOURNALIÈRES (comme gym-DSSAT) :
  rain (mm), tmax/tmin (°C), srad (MJ/m2/j), sw_top (couche 1), sw_root (moyenne pondérée 0-100 cm), saisonnalité.
Usage : python -m src.era5_prepare     ->  data/processed/era5_daily.csv
"""
import glob

import numpy as np
import pandas as pd

from . import config as C


def deaccumulate(s):
    """ERA5-Land : les cumuls repartent à 01:00 UTC. On reconstruit l'incrément horaire."""
    inc = s.diff()
    hour = s.index.hour
    inc[hour == 1] = s[hour == 1]
    inc.iloc[0] = s.iloc[0]
    return inc.clip(lower=0)


def find_col(cols, *keys):
    for k in keys:
        for c in cols:
            if c.lower() == k or c.lower().startswith(k + "_") or k in c.lower():
                return c
    raise KeyError(f"colonne {keys} introuvable parmi {list(cols)}")


def main():
    files = glob.glob(str(C.DATA_RAW / "era5land_point" / "*.csv")) + glob.glob(str(C.DATA_RAW / "*.csv"))
    if not files:
        raise FileNotFoundError("Aucun CSV ERA5 : lancez python -m src.era5_download")
    df = pd.read_csv(files[0])
    print("Colonnes du CSV :", list(df.columns))
    tcol = next(c for c in df.columns if c.lower() in ("valid_time", "time", "date", "datetime"))
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.sort_values(tcol).set_index(tcol)
    cols = df.columns
    t2m, tp, ssrd = find_col(cols, "t2m"), find_col(cols, "tp"), find_col(cols, "ssrd")
    sw1, sw2, sw3 = find_col(cols, "swvl1"), find_col(cols, "swvl2"), find_col(cols, "swvl3")

    temp = df[t2m] - 273.15 if df[t2m].mean() > 100 else df[t2m]          # K -> °C
    rain_h = deaccumulate(df[tp]) * 1000.0                                # m -> mm
    srad_h = deaccumulate(df[ssrd]) / 1e6                                 # J/m2 -> MJ/m2
    # L'incrément daté t couvre (t-1h, t] : on le range dans le jour de (t-1h)
    shift = pd.Timedelta(hours=1)
    rain_d = rain_h.groupby((rain_h.index - shift).date).sum()
    srad_d = srad_h.groupby((srad_h.index - shift).date).sum()
    daily = pd.DataFrame({
        "tmax": temp.groupby(temp.index.date).max(),
        "tmin": temp.groupby(temp.index.date).min(),
        "rain": rain_d, "srad": srad_d,
        "sw_top": df[sw1].groupby(df.index.date).mean(),
        "sw_root": ((df[sw1] * 7 + df[sw2] * 21 + df[sw3] * 72) / 100).groupby(df.index.date).mean(),
    })
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    daily = daily.dropna()
    doy = daily.index.dayofyear.values
    daily["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    daily["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out = C.DATA_PROC / "era5_daily.csv"
    daily.reset_index().to_csv(out, index=False)
    print(daily.describe().round(3))
    print(f"{len(daily)} jours ({daily.index.min().date()} -> {daily.index.max().date()}) -> {out}")


if __name__ == "__main__":
    main()
