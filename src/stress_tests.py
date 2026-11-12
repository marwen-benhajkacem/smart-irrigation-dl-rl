"""Tests de robustesse : saisons normales / sèches / chaudes (sélection par quantiles) + bruit capteur.

  python -m src.stress_tests --rl models/ppo_v1_s0,models/ppo_v2_s0 --n_seeds 20
Sorties : results/stress_scenarios.csv , results/stress_noise.csv
"""
import argparse

import numpy as np
import pandas as pd

from . import config as C
from .evaluate import best_baselines, evaluate_policy
from .sb3_policy import load_rl_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl", default="", help="dossiers séparés par des virgules")
    ap.add_argument("--n_seeds", type=int, default=20)
    ap.add_argument("--noise_seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=C.EPISODES_PER_SEED)
    args = ap.parse_args()

    methods = [(p, {}, p.name) for p in best_baselines()]
    for d in [x for x in args.rl.split(",") if x]:
        pol, kw = load_rl_policy(d)
        methods.append((pol, kw, pol.name))

    seeds = C.SEEDS["test"][: args.n_seeds]
    frames = [evaluate_policy(p, seeds, args.episodes, label=lab, env_kwargs=dict(kw)) for p, kw, lab in methods]
    df = pd.concat(frames, ignore_index=True)

    # Scénarios définis par la météo (identique pour toutes les méthodes -> on la lit sur 'no_irrigation')
    ref = df[df.method == "no_irrigation"][["seed", "episode", "rain_mm", "tmax_mean"]]
    q_rain, q_hot = ref.rain_mm.quantile(0.2), ref.tmax_mean.quantile(0.8)
    ref = ref.assign(drought=ref.rain_mm <= q_rain, heat=ref.tmax_mean >= q_hot)
    chk = df.groupby(["seed", "episode"]).rain_mm.nunique().max()
    if chk > 1:
        print("ATTENTION : la pluie diffère entre méthodes pour une même graine -> météo non appariée !")
    df = df.merge(ref[["seed", "episode", "drought", "heat"]], on=["seed", "episode"])
    rows = []
    for name, mask in [("normal", ~(df.drought | df.heat)), ("secheresse", df.drought), ("canicule", df.heat),
                       ("secheresse+canicule", df.drought & df.heat), ("toutes", df.seed >= 0)]:
        sub = df[mask]
        g = sub.groupby("method")[["yield_kg_ha", "irrig_mm", "stress_days", "reward_total", "wue_kg_per_mm"]].mean()
        g["n_saisons"] = sub.groupby("method").size()
        g.insert(0, "scenario", name)
        rows.append(g.reset_index())
    out = pd.concat(rows)
    out.to_csv(C.RESULTS / "stress_scenarios.csv", index=False)
    print(out.round(2).to_string(index=False))

    # Bruit capteur
    noise_rows = []
    for sigma in [0.0, 0.01, 0.02, 0.04]:
        for p, kw, lab in methods:
            kw2 = dict(kw)
            kw2["obs_noise_std"] = sigma
            d = evaluate_policy(p, C.SEEDS["test"][: args.noise_seeds], args.episodes, label=lab, env_kwargs=kw2)
            noise_rows.append(dict(sigma=sigma, method=lab, yield_kg_ha=d.yield_kg_ha.mean(), irrig_mm=d.irrig_mm.mean(),
                                   reward_total=d.reward_total.mean()))
    nz = pd.DataFrame(noise_rows)
    nz.to_csv(C.RESULTS / "stress_noise.csv", index=False)
    print(nz.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
