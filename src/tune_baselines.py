"""Règle les deux baselines (calendrier fixe, seuil) sur le split VALIDATION
en maximisant la MÊME récompense que l'agent RL (comparaison équitable).
Sortie : results/best_baselines.json  + results/tuning_baselines.csv
"""
import argparse
import json
import itertools

import pandas as pd

from . import config as C
from .evaluate import evaluate_policy
from .policies import FixedCalendar, Threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=4)
    args = ap.parse_args()
    seeds = C.SEEDS["val"][: args.n_seeds]

    rows = []
    for period, amount in itertools.product([3, 5, 7, 10, 14], [10, 15, 20]):
        df = evaluate_policy(FixedCalendar(period, amount), seeds, args.episodes)
        rows.append(dict(kind="calendar", period=period, amount=amount, thr=None, reward=df.reward_total.mean(),
                         yield_kg_ha=df.yield_kg_ha.mean(), irrig_mm=df.irrig_mm.mean()))
    for thr, amount in itertools.product([0.2, 0.3, 0.4, 0.5, 0.6], [5, 10, 15]):
        df = evaluate_policy(Threshold(thr, amount), seeds, args.episodes)
        rows.append(dict(kind="threshold", period=None, amount=amount, thr=thr, reward=df.reward_total.mean(),
                         yield_kg_ha=df.yield_kg_ha.mean(), irrig_mm=df.irrig_mm.mean()))
    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS / "tuning_baselines.csv", index=False)

    bc = res[res.kind == "calendar"].sort_values("reward", ascending=False).iloc[0]
    bt = res[res.kind == "threshold"].sort_values("reward", ascending=False).iloc[0]
    best = {"calendar": {"period": int(bc.period), "amount": float(bc.amount)},
            "threshold": {"thr": float(bt.thr), "amount": float(bt.amount)}}
    (C.RESULTS / "best_baselines.json").write_text(json.dumps(best, indent=2))
    print("\nMeilleures baselines (validation) :", best)


if __name__ == "__main__":
    main()
