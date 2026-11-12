"""ÉTAPE 2 du plan : comprendre l'environnement gym-DSSAT.
Affiche les espaces d'observation/d'action, lance 1 saison sans irrigation, mesure la vitesse
et enregistre results/discovery.json.   Usage :  python scripts/00_discover_env.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                   # noqa: E402
from src.dssat_factory import make_dssat, raw_reset, raw_step  # noqa: E402


def describe(v):
    a = np.asarray(v)
    return f"type={type(v).__name__:<8} shape={a.shape} valeur={np.round(a.ravel()[:4], 4).tolist()}"


env = make_dssat(seed=123, random_weather=True)
print("=" * 70)
print("observation_space :", getattr(env, "observation_space", "?"))
print("action_space      :", getattr(env, "action_space", "?"))
print("=" * 70)

obs = raw_reset(env)
print("CLES DE L'OBSERVATION INITIALE")
for k, v in obs.items():
    print(f"  {k:<10} {describe(v)}")

history = []
t0 = time.time()
done, steps = False, 0
while not done:
    obs, r, done, info = raw_step(env, {"amir": 0.0})   # saison SANS irrigation
    steps += 1
    if obs is not None:
        history.append({k: np.asarray(v, dtype=float).ravel().tolist() for k, v in obs.items()})
dt = time.time() - t0
env.close()

print("=" * 70)
print(f"Saison sans irrigation : {steps} jours simulés en {dt:.1f}s  ({steps / max(dt, 1e-9):.0f} pas/s)")
print("Dernière observation :")
last = history[-1]
for k, v in last.items():
    print(f"  {k:<10} {np.round(v[:4], 3).tolist()}")

ranges = {}
for k in history[0].keys():
    arr = np.array([h[k] for h in history], dtype=float)
    ranges[k] = {"min": float(np.nanmin(arr)), "max": float(np.nanmax(arr))}
print("=" * 70)
print("PLAGES DE VALEURS (utile pour vérifier unités et signification) :")
for k, v in ranges.items():
    print(f"  {k:<10} min={v['min']:.4g}  max={v['max']:.4g}")

(C.RESULTS / "discovery.json").write_text(json.dumps({
    "keys": list(history[0].keys()), "season_days": steps, "steps_per_second": steps / max(dt, 1e-9),
    "ranges": ranges, "last_obs": last}, indent=2))
print("\n-> results/discovery.json écrit. Copiez-moi cette sortie si une clé attendue manque.")
needed = ["sw", "rain", "tmax", "tmin", "dap", "swfac", "grnwt"]
missing = [k for k in needed if k not in history[0]]
print("Clés essentielles manquantes :", missing if missing else "aucune  OK")
