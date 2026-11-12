"""Simulateur JOUET qui imite l'API de gym-DSSAT (obs = dict, step({'amir': mm})).

But : tester tout le pipeline Python (baselines, dataset, prévision, RL) sans DSSAT,
par exemple sur Colab ou sur votre PC.   >>> NE SERT PAS pour les résultats finaux. <<<
Usage :  USE_FAKE_DSSAT=1 python -m tests.smoke_test
"""
import numpy as np


class _Box:
    def __init__(self, low, high):
        self.low = np.array([low], dtype=float)
        self.high = np.array([high], dtype=float)


class FakeDssatEnv:
    def __init__(self, seed=0, random_weather=True, mode="irrigation", season_days=160):
        self.rng = np.random.default_rng(seed)
        self.random_weather = random_weather
        self.T = season_days
        self.mode = mode
        self.action_space = {"amir": _Box(0.0, 50.0)}
        self.cap = 100.0   # capacité du réservoir racinaire (mm)
        self.t = 0

    # ---------------------------------------------------------------- météo
    def _make_weather(self):
        d = np.arange(self.T + 2)
        scale = self.rng.uniform(0.3, 1.5) if self.random_weather else 1.0
        heat = self.rng.normal(0.0, 2.0) if self.random_weather else 0.0
        self.tmax = 26 + 7 * np.sin(2 * np.pi * (d - 20) / 365 * 2) + heat + self.rng.normal(0, 2, d.size)
        self.tmin = self.tmax - 10 + self.rng.normal(0, 1, d.size)
        self.srad = 18 + 6 * np.sin(2 * np.pi * (d - 20) / 365 * 2) + self.rng.normal(0, 2, d.size)
        wet = self.rng.random(d.size) < 0.18 * scale
        self.rain = wet * self.rng.exponential(12.0, d.size)

    def _sw_layers(self):
        frac = self.S / self.cap
        base = 0.08 + 0.22 * frac
        offs = np.linspace(0.02, -0.02, 9)
        return np.clip(base + offs, 0.02, 0.45)

    def _obs(self):
        stress = float(min(1.0, self.S / (0.5 * self.cap)))
        self.swfac_now = stress
        return {
            "sw": self._sw_layers(),
            "rain": float(self.rain[self.t]),
            "tmax": float(self.tmax[self.t]),
            "tmin": float(self.tmin[self.t]),
            "srad": float(self.srad[self.t]),
            "dap": int(self.t),
            "swfac": stress,
            "xlai": float(min(5.0, self.t / 30.0)),
            "istage": int(min(9, 1 + self.t // 20)),
            "vstage": float(min(18, self.t / 8.0)),
            "grnwt": float(self.grn),
            "totir": float(self.totir),
        }

    # ------------------------------------------------------------ API gym
    def reset(self):
        self._make_weather()
        self.S = 55.0
        self.t = 0
        self.grn = 0.0
        self.totir = 0.0
        return self._obs()

    def step(self, action):
        a = float(action["amir"]) if isinstance(action, dict) else float(action)
        a = float(np.clip(a, 0.0, 50.0))
        self.totir += a
        et = max(0.0, 0.18 * (self.tmax[self.t] - 8.0)) * (0.4 + 0.6 * self.S / self.cap)
        self.S = float(np.clip(self.S + self.rain[self.t] + a - et - 0.02 * self.S, 0.0, self.cap))
        self.t += 1
        stress = float(min(1.0, self.S / (0.5 * self.cap)))
        if self.t > 60:   # remplissage du grain
            self.grn += 85.0 * stress * (self.srad[self.t] / 18.0) * (1.0 if self.t < 150 else 0.2)
        done = self.t >= self.T
        return self._obs(), 0.0, done, {}

    def close(self):
        pass
