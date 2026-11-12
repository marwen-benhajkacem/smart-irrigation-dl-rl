"""Graphiques du rapport.
  python -m src.plots pareto  results/eval_baselines_test.csv results/eval_ppo_v1_s0_test.csv
  python -m src.plots learning models/ppo_v1_s0 models/dqn_s0
  python -m src.plots forecast results/forecast_sim_lstm.csv results/forecast_sim_ml.csv
  python -m src.plots season results/daily_ppo_v1_s0_test.csv
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config as C


def pareto(files):
    df = pd.concat([pd.read_csv(f) for f in files])
    g = df.groupby("method")[["irrig_mm", "yield_kg_ha"]].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, r in g.iterrows():
        ax.errorbar(r[("irrig_mm", "mean")], r[("yield_kg_ha", "mean")], xerr=r[("irrig_mm", "std")] / 4,
                    yerr=r[("yield_kg_ha", "std")] / 4, fmt="o", capsize=3, label=m)
    ax.set_xlabel("Eau d'irrigation par saison (mm)")
    ax.set_ylabel("Rendement (kg/ha)")
    ax.set_title("Compromis rendement / eau (haut-gauche = mieux)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    out = C.RESULTS / "fig_pareto.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("->", out)


def learning(folders):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for d in folders:
        m = pd.read_csv(f"{d}/train.monitor.csv", skiprows=1)
        ax.plot(m["l"].cumsum(), m["r"].rolling(20, min_periods=1).mean(), label=d.rstrip("/").split("/")[-1])
    ax.set_xlabel("Pas d'entraînement")
    ax.set_ylabel("Récompense par saison (moyenne glissante 20)")
    ax.legend()
    ax.grid(alpha=0.3)
    out = C.RESULTS / "fig_learning.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("->", out)


def forecast(files):
    df = pd.concat([pd.read_csv(f) for f in files])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, g in df.groupby("model"):
        s = g.groupby("horizon").rmse.mean()
        ax.plot(s.index, s.values, marker="o", label=m)
    ax.set_xlabel("Horizon (jours)")
    ax.set_ylabel("RMSE humidité (cm3/cm3)")
    ax.legend()
    ax.grid(alpha=0.3)
    out = C.RESULTS / "fig_forecast_rmse.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("->", out)


def season(csv):
    d = pd.read_csv(csv)
    d = d[(d.seed == d.seed.iloc[0]) & (d.episode == d.episode.iloc[0])]
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(d.t, d.sw_root)
    ax[0].set_ylabel("Humidité sol")
    ax[1].bar(d.t, d.rain, color="steelblue", label="pluie")
    ax[1].bar(d.t, d.irrig, color="orange", label="irrigation")
    ax[1].legend()
    ax[1].set_ylabel("mm")
    ax[2].plot(d.t, d.swfac)
    ax[2].set_ylabel("swfac")
    ax[2].set_xlabel("Jour")
    out = C.RESULTS / "fig_season.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("->", out)


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    {"pareto": pareto, "learning": learning, "forecast": forecast, "season": lambda r: season(r[0])}[cmd](rest)
