"""Boucle d'évaluation : méthodes x graines x saisons -> CSV (une ligne par saison).

Exemples :
  python -m src.evaluate --split val  --methods none,calendar_7d,threshold_0.4
  python -m src.evaluate --split test --methods best                 # baselines réglées
  python -m src.evaluate --split test --rl models/ppo_v1_s0 --label ppo_v1_s0
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from . import config as C
from .env_wrapper import IrrigationEnv
from .policies import NoIrrigation, FixedCalendar, Threshold


def run_episode(env, policy):
    env.reset()
    policy.reset()
    done = False
    info = {}
    while not done:
        mm = policy.act(env.context())
        _, _, term, trunc, info = env.step(np.array([mm], dtype=np.float32))
        done = term or trunc
    return info["episode_stats"]


def evaluate_policy(policy, seeds, episodes_per_seed, label=None, env_kwargs=None, keep_logs=False):
    """Même seeds => même météo pour toutes les méthodes (comparaison appariée)."""
    env_kwargs = dict(env_kwargs or {})
    env_kwargs.setdefault("action_mode", "mm")
    rows, logs = [], []
    t0 = time.time()
    for seed in seeds:
        env = IrrigationEnv(seed=seed, **env_kwargs)
        for ep in range(episodes_per_seed):
            stats = run_episode(env, policy)
            stats.update(method=label or policy.name, seed=seed, episode=ep)
            rows.append(stats)
            if keep_logs:
                lg = env.get_log()
                lg["method"], lg["seed"], lg["episode"] = label or policy.name, seed, ep
                logs.append(lg)
        env.close()
    print(f"  {label or policy.name}: {len(rows)} saisons en {time.time() - t0:.0f}s "
          f"| rendement moyen {np.mean([r['yield_kg_ha'] for r in rows]):.0f} kg/ha "
          f"| eau {np.mean([r['irrig_mm'] for r in rows]):.0f} mm")
    df = pd.DataFrame(rows)
    return (df, pd.concat(logs)) if keep_logs else df


def parse_method(token):
    """'none' | 'calendar_7d_15' | 'threshold_0.4_10'  (period/amount ou thr/amount)."""
    parts = token.split("_")
    if parts[0] == "none":
        return NoIrrigation()
    if parts[0] == "calendar":
        return FixedCalendar(period=int(parts[1].replace("d", "")), amount=float(parts[2]) if len(parts) > 2 else 15.0)
    if parts[0] == "threshold":
        return Threshold(thr=float(parts[1]), amount=float(parts[2]) if len(parts) > 2 else 10.0)
    raise ValueError(f"méthode inconnue : {token}")


def best_baselines():
    f = C.RESULTS / "best_baselines.json"
    if not f.exists():
        raise FileNotFoundError("Lancez d'abord : python -m src.tune_baselines")
    b = json.loads(f.read_text())
    return [NoIrrigation(),
            FixedCalendar(b["calendar"]["period"], b["calendar"]["amount"]),
            Threshold(b["threshold"]["thr"], b["threshold"]["amount"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--methods", default="")
    ap.add_argument("--rl", default="", help="dossier d'un modèle RL (models/xxx)")
    ap.add_argument("--label", default="")
    ap.add_argument("--n_seeds", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=C.EPISODES_PER_SEED)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--keep_logs", action="store_true")
    args = ap.parse_args()

    seeds = C.SEEDS[args.split][: args.n_seeds] if args.n_seeds else C.SEEDS[args.split]
    all_df = []
    if args.methods:
        pols = best_baselines() if args.methods == "best" else [parse_method(m) for m in args.methods.split(",")]
        for p in pols:
            all_df.append(evaluate_policy(p, seeds, args.episodes, env_kwargs={"obs_noise_std": args.noise}))
    if args.rl:
        from .sb3_policy import load_rl_policy
        pol, env_kwargs = load_rl_policy(args.rl)
        env_kwargs["obs_noise_std"] = args.noise
        label = args.label or args.rl.rstrip("/").split("/")[-1]
        res = evaluate_policy(pol, seeds, args.episodes, label=label, env_kwargs=env_kwargs, keep_logs=args.keep_logs)
        if args.keep_logs:
            res[1].to_csv(C.RESULTS / f"daily_{label}_{args.split}.csv", index=False)
            res = res[0]
        all_df.append(res)

    out = pd.concat(all_df, ignore_index=True)
    name = args.label or ("baselines" if args.methods else "rl")
    path = C.RESULTS / f"eval_{name}_{args.split}{'_noise' + str(args.noise) if args.noise else ''}.csv"
    out.to_csv(path, index=False)
    print(out.groupby("method")[["yield_kg_ha", "irrig_mm", "stress_days", "reward_total"]].mean().round(2))
    print("->", path)


if __name__ == "__main__":
    main()
