# run_step2.py - Updated for validation/test split, CLI params, and weight visualization

import sys
import os
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add src and experiments to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))
sys.path.insert(0, str(current_dir / "experiments"))

from models.base.sarimax_model import SARIMAXModel
from models.base.lstm_model import LSTMModel
from models.base.lightgbm_model import LightGBMModel
from models.ensemble.weighted_ensemble import WeightedAverageEnsemble
from utils.data_utils import prepare_m5_data_for_models
from experiments.configs.ensemble_configs import get_ensemble_config

def setup_logging():
    Path('logs').mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('logs/step2_execution.log'), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def plot_weights(weights: dict, output_path: str):
    names, vals = zip(*weights.items())
    plt.figure(figsize=(6,4))
    plt.bar(names, vals, color=['#1f77b4','#ff7f0e','#2ca02c'])
    plt.ylabel('Weight')
    plt.title('Optimized Ensemble Weights')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def main(item, store, opt_method):
    logger = setup_logging()
    print(f"🚀 STEP 2: Weighted Ensemble for {item} @ {store}")

    # Load data splits
    train_df, val_df, test_df = prepare_m5_data_for_models(item, store)
    y_val = val_df['demand'].values
    y_test = test_df['demand'].values

    # Generate predictions on VAL and TEST
    preds_val = {}
    preds_test = {}

    # Load SARIMAX
    sar = SARIMAXModel()
    sar.load_model('models/saved/sarimax_model.pkl')
    preds_val['sarimax'] = sar.predict(steps=len(val_df))
    preds_test['sarimax'] = sar.predict(steps=len(test_df))

    # LSTM
    lstm = LSTMModel()
    lstm.load_model('models/saved/lstm_model')
    p_val = lstm.predict(train_df['demand'].values, steps=len(val_df))
    p_test = lstm.predict(np.concatenate([train_df['demand'].values, val_df['demand'].values]), steps=len(test_df))
    preds_val['lstm'] = p_val
    preds_test['lstm'] = p_test

    # LightGBM
    lgbm = LightGBMModel()
    lgbm.load_model('models/saved/lightgbm_model')
    # create_features returns full combined DF
    full = pd.concat([train_df, val_df, test_df], ignore_index=True)
    feat = lgbm.create_features(full, 'demand')
    val_feat = feat.iloc[len(train_df):len(train_df)+len(val_df)].drop(columns=['date','demand'])
    test_feat = feat.iloc[len(train_df)+len(val_df):].drop(columns=['date','demand'])
    preds_val['lightgbm'] = lgbm.predict(val_feat)
    preds_test['lightgbm'] = lgbm.predict(test_feat)

    # Ensure same lengths
    for name in preds_val:
        assert len(preds_val[name]) == len(y_val), f"Len mismatch on val {name}"
        assert len(preds_test[name]) == len(y_test), f"Len mismatch on test {name}"

    # Optimize on VAL
    config = get_ensemble_config()
    ensemble_cfg = config['weighted_ensemble'].copy()
    ensemble_cfg.pop('max_iterations', None)
    ens = WeightedAverageEnsemble(
        optimization_method=ensemble_cfg['optimization_method'],
        loss_function=ensemble_cfg['loss_function'],
        weights_bounds=ensemble_cfg['weights_bounds'],
        normalize_weights=ensemble_cfg['normalize_weights'],
        random_state=ensemble_cfg['random_state']
    )
    fit_res = ens.fit(preds_val, y_val)
    weights = fit_res['weights']
    print("Optimized weights:", weights)

    # Evaluate on TEST
    ens_pred = ens.predict(preds_test)
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_test, ens_pred)
    rmse = np.sqrt(mean_squared_error(y_test, ens_pred))
    print(f"Ensemble TEST Performance: MAE={mae:.4f}, RMSE={rmse:.4f}")

    # Save weights plot
    plot_weights(weights, 'figures/step2_weights.png')
    print("Weight plot saved to figures/step2_weights.png")

    # Save report
    report = {
        'item': item, 'store': store,
        'test_mae': mae, 'test_rmse': rmse,
        'weights': weights, 'opt_method': opt_method,
        'timestamp': datetime.now().isoformat()
    }
    os.makedirs('experiments/results', exist_ok=True)
    with open('experiments/results/step2_ensemble_report.json','w') as f:
        json.dump(report, f, indent=2)
    print("Report saved to experiments/results/step2_ensemble_report.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", default="HOBBIES_1_001")
    parser.add_argument("--store", default="CA_1")
    parser.add_argument("--opt-method", default="scipy", choices=["scipy","optuna"])
    args = parser.parse_args()
    main(args.item, args.store, args.opt_method)