"""Enveloppe Gymnasium autour de gym-DSSAT (mode irrigation).

Rôles :
  * traduit l'action de l'agent (indice discret / continu / mm) en {'amir': mm}
  * construit le vecteur d'observation selon un profil ('sensor' ou 'full')
  * calcule NOTRE récompense : d_rendement - LAMBDA*eau - MU*stress
  * ajoute du bruit capteur (tests de robustesse)
  * ajoute optionnellement la prévision DL à l'état (agent v2)
  * garde un journal journalier (pour les graphiques) et des statistiques d'épisode
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from . import config as C
from .dssat_factory import make_dssat, raw_reset, raw_step
from .features import get_sw, daily_features, sw_root


class IrrigationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=0, random_weather=True, action_mode="discrete", profile="sensor",
                 lam=C.LAMBDA_WATER, mu=C.MU_STRESS, obs_noise_std=0.0, forecaster=None,
                 mode=C.ENV_MODE, levels=None):
        super().__init__()
        assert action_mode in ("discrete", "continuous", "mm")
        self.seed_value = int(seed)
        self.action_mode = action_mode
        self.profile_name = profile
        self.profile = C.OBS_PROFILES[profile]
        self.lam, self.mu = float(lam), float(mu)
        self.obs_noise_std = float(obs_noise_std)
        self.forecaster = forecaster
        self.mode = mode
        self.rng = np.random.default_rng(self.seed_value + 777)

        self.env = make_dssat(seed=self.seed_value, mode=mode, random_weather=random_weather)
        self.max_mm = self._detect_max_mm()
        lv = list(levels if levels is not None else C.ACTION_LEVELS_MM)
        self.levels = [min(float(x), self.max_mm) for x in lv]

        # Un premier reset sert à inspecter la structure ; il est réutilisé pour l'épisode 1
        self._first_obs = raw_reset(self.env)
        self._first_pending = True
        self._init_episode(self._first_obs)

        if action_mode == "discrete":
            self.action_space = spaces.Discrete(len(self.levels))
        elif action_mode == "continuous":
            self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(0.0, self.max_mm, shape=(1,), dtype=np.float32)
        n = self._last_obs.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)

    # ------------------------------------------------------------ utilitaires
    def _detect_max_mm(self):
        try:
            hi = float(np.asarray(self.env.action_space["amir"].high).ravel()[0])
            return min(hi, C.MAX_IRRIG_MM)
        except Exception:
            return C.MAX_IRRIG_MM

    def _sense(self, raw):
        """Observation 'capteur' : ajoute du bruit gaussien sur l'humidité du sol."""
        if self.obs_noise_std <= 0:
            return raw
        sensed = copy.copy(raw)
        sw = get_sw(raw)
        sensed["sw"] = np.clip(sw + self.rng.normal(0.0, self.obs_noise_std, sw.shape), 0.0, 1.0)
        return sensed

    def _build_obs(self):
        s = self._sensed
        feats = []
        sw = get_sw(s)
        for i in self.profile["sw_idx"]:
            if i < sw.size:
                feats.append(sw[i])
        for k in self.profile["scalar"]:
            if k in s:
                feats.append(float(s[k]))
        feats.append(float(self._irrig_prev))
        if self.forecaster is not None:
            feats.extend((self.fc_pred - sw_root(s)).tolist())   # écart prévu par rapport à aujourd'hui
        return np.asarray(feats, dtype=np.float32)

    def _update_forecast(self):
        if self.forecaster is not None:
            self.forecaster.update(daily_features(self._sensed, self._irrig_prev))
            self.fc_pred = self.forecaster.predict()

    def _init_episode(self, raw):
        self._raw = raw
        self._sensed = self._sense(raw)
        self._irrig_prev = 0.0
        self._prev_grn = float(raw.get("grnwt", 0.0) or 0.0)
        self.t = 0
        self.irrig_total = 0.0
        self.n_events = 0
        self.rain_total = float(raw.get("rain", 0.0) or 0.0)
        self.stress_days = 0
        self.swfac_sum = 0.0
        self.tmax_sum = float(raw.get("tmax", 0.0) or 0.0)
        self.reward_total = 0.0
        self.log = []
        if self.forecaster is not None:
            self.forecaster.reset()
            self.fc_pred = np.zeros(C.HORIZON, dtype=float)
            self._update_forecast()
        self._last_obs = self._build_obs()

    def _to_mm(self, action):
        if self.action_mode == "discrete":
            return float(self.levels[int(action)])
        a = float(np.asarray(action, dtype=float).ravel()[0])
        if self.action_mode == "continuous":
            return float(np.clip(a, 0.0, 1.0) * self.max_mm)
        return float(np.clip(a, 0.0, self.max_mm))

    def _fmt_action(self, mm):
        a = {"amir": float(mm)}
        if self.mode == "all":
            a["anfer"] = 0.0
        return a

    # -------------------------------------------------------------- API gym
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._first_pending:
            raw = self._first_obs
            self._first_pending = False
        else:
            raw = raw_reset(self.env)
        self._init_episode(raw)
        return self._last_obs, {}

    def context(self):
        """Informations données aux politiques (baselines ou agent RL)."""
        return {"obs": self._last_obs, "sw_root": sw_root(self._sensed),
                "raw": self._sensed, "t": self.t}

    def step(self, action):
        mm = self._to_mm(action)
        row = daily_features(self._raw, self._irrig_prev)       # état à l'instant t
        row.update(t=self.t, irrig=mm, swfac=float(self._raw.get("swfac", 0.0)),
                   grnwt=float(self._raw.get("grnwt", 0.0) or 0.0),
                   xlai=float(self._raw.get("xlai", 0.0) or 0.0))
        self.log.append(row)

        raw, _, done, _ = raw_step(self.env, self._fmt_action(mm))
        if raw is None:                 # sécurité : certaines versions renvoient None à la fin
            raw = self._raw
        grn = float(raw.get("grnwt", self._prev_grn) or 0.0)
        swf = float(raw.get("swfac", 0.0))

        reward = (grn - self._prev_grn) / C.YIELD_SCALE - self.lam * mm - self.mu * swf
        self._prev_grn = grn
        self.irrig_total += mm
        self.n_events += int(mm > 0)
        self.rain_total += float(raw.get("rain", 0.0) or 0.0)
        self.stress_days += int(swf >= C.STRESS_THRESHOLD)
        self.swfac_sum += swf
        self.tmax_sum += float(raw.get("tmax", 0.0) or 0.0)
        self.reward_total += reward
        self.t += 1

        self._raw = raw
        self._sensed = self._sense(raw)
        self._irrig_prev = mm
        self._update_forecast()
        self._last_obs = self._build_obs()

        truncated = (not done) and self.t >= 400
        info = {}
        if done or truncated:
            info["episode_stats"] = self.episode_stats()
        return self._last_obs, float(reward), bool(done), bool(truncated), info

    def episode_stats(self):
        n = max(self.t, 1)
        irrig = self.irrig_total
        return {
            "yield_kg_ha": self._prev_grn,
            "irrig_mm": irrig,
            "rain_mm": self.rain_total,
            "n_irrig_events": self.n_events,
            "stress_days": self.stress_days,
            "mean_swfac": self.swfac_sum / n,
            "tmax_mean": self.tmax_sum / (n + 1),
            "season_days": self.t,
            "reward_total": self.reward_total,
            "iwue_kg_per_mm": (self._prev_grn / irrig) if irrig > 0 else float("nan"),
            "wue_kg_per_mm": self._prev_grn / max(irrig + self.rain_total, 1e-6),
        }

    def get_log(self):
        return pd.DataFrame(self.log)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
