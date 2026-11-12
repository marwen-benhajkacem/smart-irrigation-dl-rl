"""Politiques d'irrigation de référence (baselines). Chaque politique renvoie des mm d'eau."""
import numpy as np
from . import config as C
from .features import rel_moisture


class Policy:
    name = "policy"

    def reset(self):
        pass

    def act(self, ctx):          # ctx = env.context()
        raise NotImplementedError


class NoIrrigation(Policy):
    name = "no_irrigation"

    def act(self, ctx):
        return 0.0


class FixedCalendar(Policy):
    """Irrigue 'amount' mm tous les 'period' jours, quelle que soit la météo ou le sol."""

    def __init__(self, period=7, amount=15.0, start_day=10):
        self.period, self.amount, self.start_day = int(period), float(amount), int(start_day)
        self.name = f"calendar_{self.period}d_{self.amount:g}mm"

    def act(self, ctx):
        t = ctx["t"]
        return self.amount if (t >= self.start_day and (t - self.start_day) % self.period == 0) else 0.0


class Threshold(Policy):
    """Irrigue 'amount' mm si l'humidité relative sondée < seuil."""

    def __init__(self, thr=0.4, amount=10.0):
        self.thr, self.amount = float(thr), float(amount)
        self.name = f"threshold_{self.thr:.2f}_{self.amount:g}mm"

    def act(self, ctx):
        return self.amount if rel_moisture(ctx["sw_root"]) < self.thr else 0.0


class RandomPolicy(Policy):
    """Utilisée seulement pour GÉNÉRER des données variées (jamais comme baseline)."""

    def __init__(self, p=0.2, levels=(5.0, 10.0, 15.0, 20.0), seed=0):
        self.p, self.levels = float(p), list(levels)
        self.rng = np.random.default_rng(seed)
        self.name = f"random_p{self.p:.2f}"

    def act(self, ctx):
        return float(self.rng.choice(self.levels)) if self.rng.random() < self.p else 0.0


def sample_data_policy(rng):
    """Mélange de politiques pour que le modèle de prévision voie des situations variées."""
    u = rng.random()
    amount = float(rng.choice([5, 10, 15, 20]))
    if u < 0.15:
        return NoIrrigation()
    if u < 0.45:
        return FixedCalendar(period=int(rng.integers(3, 15)), amount=amount)
    if u < 0.80:
        return Threshold(thr=float(rng.uniform(0.15, 0.7)), amount=amount)
    return RandomPolicy(p=float(rng.uniform(0.05, 0.4)), seed=int(rng.integers(1_000_000)))
