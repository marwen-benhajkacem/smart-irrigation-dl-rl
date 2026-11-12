"""Tableau final comparatif + différences appariées (IC 95 % bootstrap) par rapport à une méthode de référence.

  python -m src.compare --files results/eval_baselines_test.csv results/eval_ppo_v1_s0_test.csv --ref threshold
Sortie : results/summary.csv  (+ tableau Markdown affiché)
"""
import argparse

import pandas as pd

from . import config as C
from .metrics import paired_bootstrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--ref", default="threshold", help="début du nom de la méthode de référence")
    args = ap.parse_args()
    df = pd.concat([pd.read_csv(f) for f in args.files], ignore_index=True)
    cols = ["yield_kg_ha", "irrig_mm", "stress_days", "reward_total", "wue_kg_per_mm"]
    summ = df.groupby("method")[cols].agg(["mean", "std"]).round(2)
    refname = next(m for m in df.method.unique() if m.startswith(args.ref))
    ref = df[df.method == refname].set_index(["seed", "episode"])
    rows = []
    for m in df.method.unique():
        cur = df[df.method == m].set_index(["seed", "episode"])
        common = cur.index.intersection(ref.index)
        dy = paired_bootstrap(cur.loc[common, "yield_kg_ha"] - ref.loc[common, "yield_kg_ha"])
        dw = paired_bootstrap(cur.loc[common, "irrig_mm"] - ref.loc[common, "irrig_mm"])
        dr = paired_bootstrap(cur.loc[common, "reward_total"] - ref.loc[common, "reward_total"])
        rows.append(dict(method=m, vs=refname, d_yield=dy[0], d_yield_lo=dy[1], d_yield_hi=dy[2],
                         d_water=dw[0], d_water_lo=dw[1], d_water_hi=dw[2],
                         d_reward=dr[0], d_reward_lo=dr[1], d_reward_hi=dr[2], n=len(common)))
    paired = pd.DataFrame(rows).round(2)
    # Résumé par FAMILLE (ex: ppo_v1_s0, ppo_v1_s1, ppo_v1_s2 -> ppo_v1) : moyenne et écart entre graines d'entraînement
    df["family"] = df.method.str.replace(r"_s\d+$", "", regex=True)
    per_model = df.groupby(["family", "method"])[cols].mean().reset_index()
    fam = per_model.groupby("family")[cols].agg(["mean", "std"]).round(2)
    fam.to_csv(C.RESULTS / "summary_family.csv")
    print("Résumé par famille (moyenne et écart-type ENTRE graines d'entraînement) :")
    print(fam.to_string())
    print()
    summ.to_csv(C.RESULTS / "summary.csv")
    paired.to_csv(C.RESULTS / "summary_paired.csv", index=False)
    print(summ.to_string())
    print("\nDifférences appariées vs", refname, "(IC95 % bootstrap) :")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
