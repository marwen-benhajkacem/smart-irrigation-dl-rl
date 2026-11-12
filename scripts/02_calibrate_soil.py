"""Calibre les bornes 'sec' et 'humide' de l'humidité sondée.
  dry = 2e percentile des saisons SANS irrigation ; wet = 98e percentile des saisons très irriguées.
Sortie : data/processed/soil_bounds.json   (utilisé par les politiques seuil)
Usage : python scripts/02_calibrate_soil.py [--n 10]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                   # noqa: E402
from src.dssat_factory import make_dssat, raw_reset, raw_step  # noqa: E402
from src.features import sw_root                               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10, help="saisons par régime")
n = ap.parse_args().n


def collect(amount_mm, seed):
    env = make_dssat(seed=seed, random_weather=True)
    vals = []
    for _ in range(n):
        obs, done = raw_reset(env), False
        while not done:
            obs, _, done, _ = raw_step(env, {"amir": float(amount_mm)})
            if obs is not None:
                vals.append(sw_root(obs))
    env.close()
    return np.array(vals)


dry_vals = collect(0.0, seed=9001)
wet_vals = collect(C.MAX_IRRIG_MM, seed=9002)
bounds = {"dry": float(np.percentile(dry_vals, 2)), "wet": float(np.percentile(wet_vals, 98))}
C.SOIL_BOUNDS_FILE.write_text(json.dumps(bounds, indent=2))
print("Bornes calibrées :", bounds)
print("  (sans irrigation : médiane", round(float(np.median(dry_vals)), 4),
      "| très irrigué : médiane", round(float(np.median(wet_vals)), 4), ")")
if bounds["wet"] - bounds["dry"] < 0.02:
    print("ATTENTION : écart très faible entre sec et humide -> vérifiez SW_SENSOR_IDX dans config.py")
