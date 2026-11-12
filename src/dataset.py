"""Construction des fenêtres glissantes pour la prévision.
Entrée du modèle : WINDOW jours d'historique [K, F].  Sortie : humidité J+1..J+H.
Cible apprise = ÉCART à l'humidité actuelle (plus stable), normalisé ; on reconstruit ensuite le niveau.
Découpage SPLIT PAR GRAINE / PAR ANNÉE : jamais de mélange temporel.
"""
import json

import numpy as np
import pandas as pd

from . import config as C


def make_windows(df, features, target, group_cols, K, H):
    Xs, Ys, S0 = [], [], []
    for _, g in df.groupby(group_cols, sort=False):
        g = g.sort_values("t")
        X = g[features].to_numpy(np.float32)
        y = g[target].to_numpy(np.float32)
        T = len(g)
        if T < K + H:
            continue
        idx = np.arange(K - 1, T - H)
        Xs.append(np.stack([X[i - K + 1: i + 1] for i in idx]))
        Ys.append(np.stack([y[i + 1: i + 1 + H] for i in idx]))
        S0.append(y[idx])
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(S0)


def _pack(df_by_split, features, target, group_cols, K, H):
    train = df_by_split["train"]
    mean = train[features].mean().to_numpy(np.float32)
    std = train[features].std().replace(0, 1).to_numpy(np.float32) + 1e-8
    sw_std = float(train[target].std())
    out = {"features": features, "mean": mean.tolist(), "std": std.tolist(), "sw_std": sw_std, "K": K, "H": H}
    for split, df in df_by_split.items():
        X, Y, S0 = make_windows(df, features, target, group_cols, K, H)
        out[split] = {
            "X": (X - mean) / std,                          # [N,K,F] normalisé
            "Y_delta": (Y - S0[:, None]) / sw_std,          # cible apprise
            "sw_now": S0, "Y_raw": Y,                       # pour reconstruire / évaluer
        }
    return out


def load_sim(K=C.WINDOW, H=C.HORIZON):
    meta = json.loads((C.DATA_PROC / "sim_meta.json").read_text())
    frames = {s: pd.read_csv(C.DATA_PROC / f"sim_{s}.csv.gz") for s in ("train", "val", "test")}
    return _pack(frames, meta["features"], "sw_root", ["seed", "episode"], K, H)


def load_era5(K=C.WINDOW, H=C.HORIZON):
    df = pd.read_csv(C.DATA_PROC / "era5_daily.csv", parse_dates=["date"])
    df["t"] = np.arange(len(df))
    df["year"] = df["date"].dt.year
    feats = ["sw_root", "sw_top", "rain", "tmax", "tmin", "srad", "doy_sin", "doy_cos"]
    frames = {
        "train": df[df.year <= 2015].assign(g=0),
        "val": df[(df.year >= 2016) & (df.year <= 2019)].assign(g=0),
        "test": df[df.year >= 2020].assign(g=0),
    }
    return _pack(frames, feats, "sw_root", ["g"], K, H)


def load_source(source, K=C.WINDOW, H=C.HORIZON):
    return load_sim(K, H) if source == "sim" else load_era5(K, H)
