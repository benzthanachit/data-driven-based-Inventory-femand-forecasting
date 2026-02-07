# wilcoxon_test.py

import json
from scipy.stats import wilcoxon

# Load CV results
with open("experiments/results/step3_cv_results.json") as f:
    cv = json.load(f)

# Compute per-fold MAE for ensemble vs LightGBM
ens_mae = [r["mae"] for r in cv["cv_results"]]

# We need LightGBM per-fold MAE: recompute or store in JSON.
# Here assume we rerun CV to collect lightgbm mae:
# for demonstration, load from experiments/results/lgbm_cv.json
with open("experiments/results/lgbm_cv_results.json") as f:
    lgbm_cv = json.load(f)
lgbm_mae = [r["mae"] for r in lgbm_cv["cv_results"]]

stat, p = wilcoxon(lgbm_mae, ens_mae)
print(f"Wilcoxon signed-rank test: stat={stat:.4f}, p-value={p:.4f}")
if p < 0.05:
    print("Ensemble MAE significantly different from LightGBM (p<0.05)")
else:
    print("No significant difference (p>=0.05)")