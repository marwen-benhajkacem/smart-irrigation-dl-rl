"""Création de l'environnement gym-DSSAT (ou du simulateur jouet) + compatibilité d'API."""
from . import config as C


def make_dssat(seed=0, mode=None, random_weather=True, log_path=None):
    mode = mode or C.ENV_MODE
    if C.USE_FAKE:
        from tests.fake_dssat import FakeDssatEnv
        return FakeDssatEnv(seed=seed, random_weather=random_weather, mode=mode)

    log_path = log_path or str(C.LOGS / "dssat-pdi.log")
    args = {
        "log_saving_path": log_path,
        "mode": mode,
        "seed": int(seed),
        "random_weather": bool(random_weather),
    }
    try:
        import gym  # gym-DSSAT utilise l'ancienne librairie 'gym'
        return gym.make(C.ENV_ID, **args)
    except Exception as first_error:   # repli : import direct de la classe
        try:
            from gym_dssat_pdi.envs.dssat_pdi import DssatPdi
            return DssatPdi(**args)
        except Exception as second_error:
            raise RuntimeError(
                "Impossible de créer gym-DSSAT.\n"
                f"  gym.make        -> {first_error!r}\n"
                f"  import direct   -> {second_error!r}\n"
                "Etes-vous bien DANS le conteneur Docker ? (voir guide, Semaine 1)") from second_error


def raw_reset(env):
    """reset() compatible ancien gym (obs) et nouveau (obs, info)."""
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def raw_step(env, action_dict):
    """step() compatible 4-tuple (gym) et 5-tuple (gymnasium)."""
    out = env.step(action_dict)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = out
        done = bool(done)
    return obs, reward, done, info
