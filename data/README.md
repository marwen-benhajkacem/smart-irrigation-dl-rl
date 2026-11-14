# Data

No dataset is stored in this repository; everything is regenerated.

| Source | Use | How to get it |
|---|---|---|
| **gym-DSSAT** (DSSAT crop simulator, maize) | Simulated seasons: weather, soil moisture, yield | Run inside the Docker image built from `Dockerfile.project` |
| **ERA5-Land** (Copernicus) | External validation of the forecaster (planned) | `python -m src.era5_download` (free CDS account, `~/.cdsapirc`, never commit it) |

Splits use disjoint seeds (train 100-199, val 200-219, test 300-319), 10 weather seasons per seed, so no weather sequence is shared between splits. Reference outputs (soil bounds, tuned baselines, test evaluation) are versioned in `references/`.
