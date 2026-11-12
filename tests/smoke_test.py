"""Test de fumée du pipeline Python avec le simulateur JOUET (sans DSSAT).
  USE_FAKE_DSSAT=1 python -m tests.smoke_test
"""
import os
os.environ["USE_FAKE_DSSAT"] = "1"

import json
import numpy as np
from src import config as C
from src.env_wrapper import IrrigationEnv
from src.evaluate import evaluate_policy
from src.policies import NoIrrigation, FixedCalendar, Threshold

# 1) bornes du sol (mini-calibration)
from src.dssat_factory import make_dssat, raw_reset, raw_step
from src.features import sw_root
vals = {}
for name, a in (("dry", 0.0), ("wet", 20.0)):
    env = make_dssat(seed=1); v = []
    for _ in range(3):
        o, d = raw_reset(env), False
        while not d:
            o, _, d, _ = raw_step(env, {"amir": a}); v.append(sw_root(o))
    vals[name] = float(np.percentile(v, 2 if name == "dry" else 98))
C.SOIL_BOUNDS_FILE.write_text(json.dumps(vals))
print("bornes", vals)

# 2) baselines
for pol in (NoIrrigation(), FixedCalendar(7, 15), Threshold(0.4, 10)):
    df = evaluate_policy(pol, [300, 301], 2)
assert len(df) == 4
# 3) action discrète + obs
env = IrrigationEnv(seed=5, action_mode="discrete")
o, _ = env.reset(); print("obs dim", o.shape)
done = False
while not done:
    o, r, te, tr, info = env.step(env.action_space.sample()); done = te or tr
print("stats", {k: round(v, 2) for k, v in info["episode_stats"].items()})
print("SMOKE TEST OK")
