# Agent d'irrigation intelligente (DL + RL) — code du projet

Voir le guide Word `Guide_Projet_Irrigation_Intelligente.docx` (étapes, explications, résultats attendus).

## Arborescence
- `src/config.py`            : TOUS les paramètres (graines, actions, récompense, capteurs, ERA5)
- `src/env_wrapper.py`       : enveloppe Gymnasium de gym-DSSAT (récompense, bruit, prévision)
- `src/policies.py`          : baselines (aucune irrigation, calendrier, seuil)
- `src/evaluate.py`, `tune_baselines.py`, `compare.py`, `stress_tests.py` : évaluation
- `src/generate_dataset.py`, `dataset.py` : données simulées + fenêtres glissantes
- `src/models.py`, `train_forecaster.py`, `ml_baselines.py` : prévision (LSTM/GRU/Transformer, RF/XGB)
- `src/era5_download.py`, `era5_prepare.py` : validation externe ERA5-Land
- `src/train_rl.py`, `sb3_policy.py`, `forecaster_runtime.py` : agents PPO/DQN/RecurrentPPO (v1, v2)
- `src/explain.py`           : explications (gabarit + LLM optionnel)
- `scripts/00_discover_env.py`, `01_run_one_season.py`, `02_calibrate_soil.py` : prise en main
- `tests/fake_dssat.py`, `tests/smoke_test.py` : simulateur JOUET pour tester le code (pas de résultats finaux !)

## Ordre d'exécution (depuis la racine du projet, DANS le conteneur Docker gym-DSSAT)
    python scripts/00_discover_env.py
    python scripts/02_calibrate_soil.py
    python scripts/01_run_one_season.py
    python -m src.tune_baselines && python -m src.evaluate --split test --methods best
    python -m src.generate_dataset --split train && ... val && ... test
    python -m src.ml_baselines --source sim
    python -m src.train_forecaster --source sim --model lstm        # sur Colab (GPU) conseillé
    python -m src.train_rl --algo ppo --tag ppo_v1_s0
    python -m src.train_rl --algo ppo --forecaster models/forecaster_sim_lstm_s0.pt --tag ppo_v2_s0
    python -m src.evaluate --split test --rl models/ppo_v1_s0
    python -m src.stress_tests --rl models/ppo_v1_s0,models/ppo_v2_s0
