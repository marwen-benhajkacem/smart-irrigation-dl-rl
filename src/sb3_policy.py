"""Charge un agent entraîné (models/<tag>/) et l'expose comme Policy pour l'évaluation."""
import json
import pickle
from pathlib import Path

import numpy as np

from . import config as C
from .policies import Policy


class SB3Policy(Policy):
    def __init__(self, model, obs_rms, clip_obs, meta, name):
        self.model, self.meta, self.name = model, meta, name
        self.obs_rms, self.clip_obs = obs_rms, clip_obs
        self.recurrent = meta["algo"] == "rppo"
        self.levels = meta.get("levels", C.ACTION_LEVELS_MM)
        self.max_mm = C.MAX_IRRIG_MM
        self.reset()

    def reset(self):
        self.state, self.first = None, True

    def act(self, ctx):
        obs = ctx["obs"][None, :].astype(np.float32)
        if self.obs_rms is not None:
            obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -self.clip_obs, self.clip_obs)
        if self.recurrent:
            a, self.state = self.model.predict(obs, state=self.state, episode_start=np.array([self.first]), deterministic=True)
        else:
            a, _ = self.model.predict(obs, deterministic=True)
        self.first = False
        if self.meta["action"] == "discrete":
            return float(min(self.levels[int(np.asarray(a).ravel()[0])], self.max_mm))
        return float(np.clip(float(np.asarray(a).ravel()[0]), 0, 1) * self.max_mm)


def load_rl_policy(folder):
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    if meta["algo"] == "ppo":
        from stable_baselines3 import PPO as Algo
    elif meta["algo"] == "dqn":
        from stable_baselines3 import DQN as Algo
    else:
        from sb3_contrib import RecurrentPPO as Algo
    model = Algo.load(str(folder / "model.zip"), device="cpu")
    vn = pickle.load(open(folder / "vecnormalize.pkl", "rb"))
    forecaster = None
    if meta.get("forecaster"):
        from .forecaster_runtime import ForecasterRuntime
        forecaster = ForecasterRuntime(meta["forecaster"])
    env_kwargs = {"profile": meta["profile"], "forecaster": forecaster, "action_mode": "mm"}
    return SB3Policy(model, vn.obs_rms, vn.clip_obs, meta, folder.name), env_kwargs
