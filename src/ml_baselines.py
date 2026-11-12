"""Baselines de prévision SANS deep learning : persistance + Random Forest (+ XGBoost si installé).
  python -m src.ml_baselines --source sim
Sortie : results/forecast_<source>_ml.csv   (même format que les modèles DL)
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from . import config as C
from .dataset import load_source
from .metrics import horizon_table


def engineer(X, features):
    """X [N,K,F] normalisé -> variables tabulaires (dernier jour, moyennes 3 et 7 j, tendances)."""
    last = X[:, -1, :]
    m3 = X[:, -3:, :].mean(axis=1)
    m7 = X[:, -7:, :].mean(axis=1)
    j = features.index("sw_root")
    d3 = (X[:, -1, j] - X[:, -4, j])[:, None]
    d7 = (X[:, -1, j] - X[:, -8, j])[:, None]
    return np.concatenate([last, m3, m7, d3, d7], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sim", choices=["sim", "era5"])
    ap.add_argument("--max_train", type=int, default=150_000)
    args = ap.parse_args()
    data = load_source(args.source)
    feats = data["features"]

    tabs = []
    te = data["test"]
    persist = np.repeat(te["sw_now"][:, None], data["H"], axis=1)
    tabs.append(horizon_table(te["Y_raw"], persist, "persistance"))

    Xtr, Ytr = data["train"]["X"], data["train"]["Y_delta"]
    if len(Xtr) > args.max_train:
        idx = np.random.default_rng(0).choice(len(Xtr), args.max_train, replace=False)
        Xtr, Ytr = Xtr[idx], Ytr[idx]
    Ftr, Fte = engineer(Xtr, feats), engineer(te["X"], feats)

    rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=0)
    rf.fit(Ftr, Ytr)
    pred = te["sw_now"][:, None] + rf.predict(Fte) * data["sw_std"]
    tabs.append(horizon_table(te["Y_raw"], pred, "random_forest"))

    try:
        from xgboost import XGBRegressor
        from sklearn.multioutput import MultiOutputRegressor
        xgb = MultiOutputRegressor(XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, n_jobs=-1))
        xgb.fit(Ftr, Ytr)
        pred = te["sw_now"][:, None] + xgb.predict(Fte) * data["sw_std"]
        tabs.append(horizon_table(te["Y_raw"], pred, "xgboost"))
    except ImportError:
        print("xgboost non installé : ignoré (pip install xgboost)")

    out = pd.concat(tabs)
    out.to_csv(C.RESULTS / f"forecast_{args.source}_ml.csv", index=False)
    print(out.pivot(index="horizon", columns="model", values="rmse").round(5))


if __name__ == "__main__":
    main()
