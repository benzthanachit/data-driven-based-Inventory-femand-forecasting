# run_step3.py
"""
Step 3: Time‐Series Cross‐Validation for Weighted Ensemble
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

# Make sure src and experiments are on path
root = Path(__file__).parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "experiments"))

from utils.data_utils import load_m5_data
from models.base.sarimax_model import SARIMAXModel
from models.base.lstm_model import LSTMModel
from models.base.lightgbm_model import LightGBMModel
from models.ensemble.weighted_ensemble import WeightedAverageEnsemble
from experiments.configs.ensemble_configs import get_ensemble_config

def setup_logging():
    Path('logs').mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler("logs/step3_execution.log"), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def main(item_id="HOBBIES_1_001", store_id="CA_1", n_splits=5):
    logger = setup_logging()
    logger.info("STEP 3: Cross-Validation for Weighted Ensemble")

    # Load full series
    full = load_m5_data(item_id, store_id)
    y = full['demand'].values
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_results = []
    fold_weights = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(full)):
        logger.info(f"Fold {fold+1}/{n_splits}")
        train, val = full.iloc[train_idx], full.iloc[val_idx]
        y_train, y_val = train['demand'].values, val['demand'].values

        # Prepare features & predictions for this fold
        # SARIMAX
        sar = SARIMAXModel()
        sar.load_model(f"models/saved/sarimax_model.pkl")
        p_val_sar = sar.predict(steps=len(val))
        # LSTM
        lstm = LSTMModel()
        lstm.load_model("models/saved/lstm_model")
        p_val_lstm = lstm.predict(y_train, steps=len(val))
        # LightGBM
        lgbm = LightGBMModel()
        lgbm.load_model("models/saved/lightgbm_model")
        combined = pd.concat([train, val], ignore_index=True)
        feats = lgbm.create_features(combined, 'demand')
        p_val_lgbm = lgbm.predict(feats.iloc[len(train):].drop(columns=['date','demand']))

        preds_val = {
            'sarimax': p_val_sar,
            'lstm': p_val_lstm,
            'lightgbm': p_val_lgbm
        }

        # Fit ensemble on val
        cfg = get_ensemble_config()
        we_cfg = cfg['weighted_ensemble'].copy()
        we_cfg.pop('max_iterations', None)
        ens = WeightedAverageEnsemble(
            optimization_method=we_cfg['optimization_method'],
            loss_function=we_cfg['loss_function'],
            weights_bounds=we_cfg['weights_bounds'],
            normalize_weights=we_cfg['normalize_weights'],
            random_state=we_cfg['random_state']
        )
        res = ens.fit(preds_val, y_val)
        weights = res['weights']
        fold_weights.append(weights)

        # Evaluate on val
        val_pred = ens.predict(preds_val)
        mae = mean_absolute_error(y_val, val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        cv_results.append({'fold': fold+1, 'mae': mae, 'rmse': rmse})
        logger.info(f"Fold {fold+1} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, weights: {weights}")

    # Summarize CV
    maes = [r['mae'] for r in cv_results]
    rmses = [r['rmse'] for r in cv_results]
    summary = {
        'cv_results': cv_results,
        'mean_mae': float(np.mean(maes)),
        'std_mae': float(np.std(maes)),
        'mean_rmse': float(np.mean(rmses)),
        'std_rmse': float(np.std(rmses)),
        'weights_per_fold': fold_weights
    }
    Path("experiments/results").mkdir(exist_ok=True)
    with open("experiments/results/step3_cv_results.json","w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"CV Summary: MAE {summary['mean_mae']:.4f}±{summary['std_mae']:.4f}")

if __name__=="__main__":
    main()