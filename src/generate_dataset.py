"""Génère des saisons simulées avec un MÉLANGE de politiques (données variées) pour entraîner la prévision.

  python -m src.generate_dataset --split train --workers 1
  python -m src.generate_dataset --split val
  python -m src.generate_dataset --split test
Sortie : data/processed/sim_<split>.csv.gz  (une ligne = un jour d'une saison)
Astuce : testez d'abord  --n_seeds 2 --episodes 3  (quelques secondes).
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pandas as pd

from . import config as C
from .env_wrapper import IrrigationEnv
from .policies import sample_data_policy


def run_chunk(args):
    split, seed, n_episodes = args
    rng = np.random.default_rng(seed + 5000)
    env = IrrigationEnv(seed=seed, action_mode="mm")
    frames = []
    for ep in range(n_episodes):
        pol = sample_data_policy(rng)
        env.reset()
        pol.reset()
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(np.array([pol.act(env.context())], dtype=np.float32))
            done = term or trunc
        lg = env.get_log()
        lg.insert(0, "policy", pol.name)
        lg.insert(0, "episode", ep)
        lg.insert(0, "seed", seed)
        lg.insert(0, "split", split)
        frames.append(lg)
    env.close()
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--n_seeds", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=C.EPISODES_PER_SEED)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    seeds = C.SEEDS[args.split][: args.n_seeds] if args.n_seeds else C.SEEDS[args.split]
    jobs = [(args.split, s, args.episodes) for s in seeds]
    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as ex:
            frames = list(ex.map(run_chunk, jobs))
    else:
        frames = []
        for i, j in enumerate(jobs):
            frames.append(run_chunk(j))
            print(f"  graine {j[1]} ({i + 1}/{len(jobs)})  {time.time() - t0:.0f}s", flush=True)
    df = pd.concat(frames, ignore_index=True)
    out = C.DATA_PROC / f"sim_{args.split}.csv.gz"
    df.to_csv(out, index=False)
    n_seasons = df.groupby(["seed", "episode"]).ngroups
    print(f"{len(df)} lignes, {n_seasons} saisons -> {out}  ({time.time() - t0:.0f}s)")
    print("Colonnes :", list(df.columns))
    print("Valeurs manquantes :", int(df.isna().sum().sum()))
    meta = {"features": [c for c in ["sw_root", "sw_top", "rain", "tmax", "tmin", "srad", "dap", "irrig_prev"] if c in df.columns]}
    (C.DATA_PROC / "sim_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
