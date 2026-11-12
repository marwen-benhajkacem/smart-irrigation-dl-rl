"""Fonctions de features PARTAGÉES (génération de données, prévision, environnement RL).
Un seul endroit => aucune incohérence entre l'entraînement et l'utilisation en ligne.
"""
import json
import numpy as np
from . import config as C


def get_sw(obs):
    """Vecteur d'humidité volumétrique par couche (cm3/cm3)."""
    return np.asarray(obs.get("sw", []), dtype=float).ravel()


def sw_root(obs):
    """Humidité 'sondée' = moyenne des couches de C.SW_SENSOR_IDX."""
    sw = get_sw(obs)
    if sw.size == 0:
        return float("nan")
    idx = [i for i in C.SW_SENSOR_IDX if i < sw.size]
    return float(sw[idx].mean())


def daily_features(obs, irrig_prev):
    """Ligne de features journalière utilisée par le modèle de prévision."""
    sw = get_sw(obs)
    row = {
        "sw_root": sw_root(obs),
        "sw_top": float(sw[0]) if sw.size else float("nan"),
    }
    for k in C.FORECAST_SCALARS:
        if k in obs:
            row[k] = float(obs[k])
    row["irrig_prev"] = float(irrig_prev)
    return row


_BOUNDS = None


def soil_bounds():
    """Bornes sec/humide calibrées par scripts/02_calibrate_soil.py."""
    global _BOUNDS
    if _BOUNDS is None:
        if not C.SOIL_BOUNDS_FILE.exists():
            raise FileNotFoundError(
                "soil_bounds.json absent : lancez d'abord  python scripts/02_calibrate_soil.py")
        _BOUNDS = json.loads(C.SOIL_BOUNDS_FILE.read_text())
    return _BOUNDS


def rel_moisture(sw_root_value):
    """Humidité relative dans [0,1] (0 = sec, 1 = humide)."""
    b = soil_bounds()
    r = (sw_root_value - b["dry"]) / max(b["wet"] - b["dry"], 1e-6)
    return float(np.clip(r, 0.0, 1.0))
