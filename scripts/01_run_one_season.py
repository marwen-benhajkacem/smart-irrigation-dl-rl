"""Joue 1 saison avec 3 politiques sur la MEME météo et trace les courbes.
Usage :  python scripts/01_run_one_season.py
(nécessite soil_bounds.json pour la politique seuil -> lancer d'abord 02_calibrate_soil.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
from src import config as C            # noqa: E402
from src.evaluate import run_episode   # noqa: E402
from src.env_wrapper import IrrigationEnv   # noqa: E402
from src.policies import NoIrrigation, FixedCalendar, Threshold   # noqa: E402

policies = [NoIrrigation(), FixedCalendar(7, 15), Threshold(0.4, 10)]
fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
for pol in policies:
    env = IrrigationEnv(seed=42, action_mode="mm")      # même graine => même météo
    stats = run_episode(env, pol)
    lg = env.get_log()
    lg = {c: lg[c].to_numpy() for c in lg.columns}   # compat matplotlib 3.3 / pandas 2
    print(f"{pol.name:<28} rendement={stats['yield_kg_ha']:.0f} kg/ha  eau={stats['irrig_mm']:.0f} mm  "
          f"jours de stress={stats['stress_days']}  récompense={stats['reward_total']:.2f}")
    axes[0].plot(lg["t"], lg["sw_root"], label=pol.name)
    axes[1].plot(lg["t"], lg["swfac"])
    axes[2].plot(lg["t"], lg["irrig"].cumsum())
    axes[3].plot(lg["t"], lg["grnwt"])
    rain = lg["rain"]
    env.close()
axes[0].set_ylabel("Humidité sol (cm3/cm3)"); axes[0].legend(fontsize=8)
axes[1].set_ylabel("swfac (0 = pas de stress, 1 = stress max)")
axes[2].set_ylabel("Eau cumulée (mm)")
axes[3].set_ylabel("Grain (kg/ha)"); axes[3].set_xlabel("Jours après semis")
ax2 = axes[0].twinx(); ax2.bar(lg["t"], rain, color="steelblue", alpha=0.3); ax2.set_ylabel("Pluie (mm)")
plt.tight_layout()
out = C.RESULTS / "one_season.png"
plt.savefig(out, dpi=130)
print("->", out)
