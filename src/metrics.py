"""Métriques de prévision et comparaisons statistiques."""
import numpy as np


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def kge(y, p):
    """Kling-Gupta Efficiency (2009) : 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)."""
    y, p = np.asarray(y), np.asarray(p)
    if y.std() == 0 or p.std() == 0:
        return float("nan")
    r = np.corrcoef(y, p)[0, 1]
    alpha = p.std() / y.std()
    beta = p.mean() / y.mean()
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def horizon_table(Y_true, Y_pred, label):
    """Métriques par horizon J+1..J+H. Y_* de forme [N, H]."""
    import pandas as pd
    rows = []
    for h in range(Y_true.shape[1]):
        rows.append(dict(model=label, horizon=h + 1,
                         rmse=rmse(Y_true[:, h], Y_pred[:, h]),
                         mae=mae(Y_true[:, h], Y_pred[:, h]),
                         r2=r2(Y_true[:, h], Y_pred[:, h]),
                         kge=kge(Y_true[:, h], Y_pred[:, h])))
    return pd.DataFrame(rows)


def paired_bootstrap(diff, n_boot=5000, seed=0):
    """Moyenne et IC 95 % (bootstrap) d'une différence appariée (même météo)."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    rng = np.random.default_rng(seed)
    means = [rng.choice(diff, size=diff.size, replace=True).mean() for _ in range(n_boot)]
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
