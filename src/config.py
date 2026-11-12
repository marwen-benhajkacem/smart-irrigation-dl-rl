"""Configuration centrale du projet.
Modifiez les valeurs ICI (jamais en dur dans les autres fichiers).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
for _d in (DATA_RAW, DATA_PROC, RESULTS, MODELS, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Environnement gym-DSSAT
# --------------------------------------------------------------------------
# USE_FAKE_DSSAT=1 -> simulateur JOUET (tests/fake_dssat.py) pour tester le code
# sans DSSAT. A NE JAMAIS utiliser pour les résultats finaux.
USE_FAKE = os.environ.get("USE_FAKE_DSSAT", "0") == "1"
ENV_ID = "gym_dssat_pdi:GymDssatPdi-v0"
ENV_MODE = "irrigation"          # 'irrigation' | 'fertilization' | 'all'

# --------------------------------------------------------------------------
# Graines (= séquences météo). Aucun chevauchement entre splits !
# Chaque graine génère EPISODES_PER_SEED saisons météo indépendantes.
# --------------------------------------------------------------------------
SEEDS = {
    "train": list(range(100, 200)),   # 100 graines x 10 saisons = 1000 saisons
    "val": list(range(200, 220)),     # 20 x 10 = 200 saisons
    "test": list(range(300, 320)),    # 20 x 10 = 200 saisons
}
EPISODES_PER_SEED = 10

# --------------------------------------------------------------------------
# Actions (mm d'eau par jour)
# --------------------------------------------------------------------------
ACTION_LEVELS_MM = [0.0, 5.0, 10.0, 15.0, 20.0]
MAX_IRRIG_MM = 20.0

# --------------------------------------------------------------------------
# Récompense : d_rendement/YIELD_SCALE - LAMBDA*eau_mm - MU*swfac
# --------------------------------------------------------------------------
YIELD_SCALE = 1000.0     # kg/ha -> t/ha
LAMBDA_WATER = 0.004     # coût d'1 mm d'eau (en t/ha équivalent) -> 4 kg/ha par mm
MU_STRESS = 0.02         # pénalité proportionnelle à l'indice de stress
STRESS_THRESHOLD = 0.9   # swfac >= 0.9 => jour de stress sévère

# --------------------------------------------------------------------------
# Capteurs / observation
# --------------------------------------------------------------------------
# Indices des couches de sol "mesurées" par la sonde (ex: 3 profondeurs)
SW_SENSOR_IDX = [0, 2, 4]

OBS_PROFILES = {
    # Réaliste: ce qu'un agriculteur peut mesurer (sonde + station météo + jour)
    "sensor": {
        "scalar": ["dap", "rain", "tmax", "tmin", "srad"],
        "sw_idx": [0, 2, 4],
    },
    # "Oracle": état agronomique complet (borne supérieure, non déployable)
    "full": {
        "scalar": ["dap", "rain", "tmax", "tmin", "srad", "swfac", "xlai",
                   "istage", "vstage", "grnwt", "totir"],
        "sw_idx": list(range(9)),
    },
}

# --------------------------------------------------------------------------
# Prévision (Deep Learning)
# --------------------------------------------------------------------------
WINDOW = 14     # jours d'historique en entrée
HORIZON = 7     # jours prédits (J+1 ... J+7)
FORECAST_SCALARS = ["rain", "tmax", "tmin", "srad", "dap"]   # + sw_root, sw_top, irrig_prev

# --------------------------------------------------------------------------
# ERA5-Land (validation externe) : point en Tunisie centrale (modifiable)
# --------------------------------------------------------------------------
ERA5_LAT = 35.68
ERA5_LON = 10.10
ERA5_START = "1981-01-01"
ERA5_END = "2023-12-31"

SOIL_BOUNDS_FILE = DATA_PROC / "soil_bounds.json"
