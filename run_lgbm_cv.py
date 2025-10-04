# run_lgbm_cv.py - Fixed imports

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ensure src and experiments on path
root = Path(__file__).parent
sys.path.insert(0, str(root / 'src'))

from utils.data_utils import load_m5_data
from models.base.lightgbm_model import LightGBMModel


def main(item='HOBBIES_1_001', store='CA_1', n_splits=5):
    df = load_m5_data(item, store)
    y = df['demand'].values
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df), 1):
        train, val = df.iloc[train_idx], df.iloc[val_idx]
        y_val = y[val_idx]
        model = LightGBMModel()
        model.load_model('models/saved/lightgbm_model')
        combined = pd.concat([train, val], ignore_index=True)
        feats = model.create_features(combined, 'demand')
        X_val = feats.iloc[len(train):].drop(columns=['date','demand'])
        p_val = model.predict(X_val)
        mae = mean_absolute_error(y_val, p_val)
        rmse = np.sqrt(mean_squared_error(y_val, p_val))
        cv_results.append({'fold':fold,'mae':mae,'rmse':rmse})
        print(f"Fold {fold}: MAE={mae:.4f}, RMSE={rmse:.4f}")
    summary = {
        'cv_results': cv_results,
        'mean_mae': float(np.mean([r['mae'] for r in cv_results])),  
        'std_mae': float(np.std([r['mae'] for r in cv_results])),  
        'mean_rmse': float(np.mean([r['rmse'] for r in cv_results])),  
        'std_rmse': float(np.std([r['rmse'] for r in cv_results]))  
    }
    Path('experiments/results').mkdir(exist_ok=True)
    with open('experiments/results/lgbm_cv_results.json','w') as f:
        json.dump(summary,f,indent=2)
    print('Saved LightGBM CV results to experiments/results/lgbm_cv_results.json')

if __name__=='__main__':
    main()