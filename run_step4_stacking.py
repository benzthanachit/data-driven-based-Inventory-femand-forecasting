# run_step4_stacking.py
"""
Step 4: Stacking Ensemble Implementation
This script generates out-of-fold (OOF) predictions from base models
and uses them to train a meta-learner for final predictions.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import json
import warnings

# --- 1. SETUP AND IMPORTS ---
warnings.filterwarnings('ignore')

# Add src and experiments to Python path
root = Path(__file__).parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "experiments"))

# Import your existing modules
from utils.data_utils import prepare_m5_data_for_models, load_m5_data
from models.base.sarimax_model import SARIMAXModel
from models.base.lstm_model import LSTMModel
from models.base.lightgbm_model import LightGBMModel
from experiments.configs.model_configs import get_config

# Import Meta-learner and evaluation metrics
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

def setup_logging():
    """Setup logging configuration"""
    Path('logs').mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('logs/step4_stacking.log'), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def create_lgbm_features_for_fold(train_df, val_df, model):
    """
    Helper to create features for a specific fold for LightGBM.
    This updated version handles NaNs and infinities gracefully.
    """
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    feature_df = model.create_features(combined_df, 'demand')

    # --- FIX STARTS HERE ---
    # 1. Replace infinities created by pct_change with NaN
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    feature_cols = [col for col in feature_df.columns if col not in ['date', 'demand']]

    # 2. Instead of dropping NaNs, fill them with 0. This is a robust way
    #    to handle the start of the series where lags/rolling features are NaN.
    feature_df[feature_cols] = feature_df[feature_cols].fillna(0)
    # --- FIX ENDS HERE ---

    # Now, create the train/val splits from the cleaned feature dataframe
    X_train = feature_df.iloc[:len(train_df)][feature_cols]
    y_train = feature_df.loc[X_train.index, 'demand']

    X_val = feature_df.iloc[len(train_df):][feature_cols]
    y_val_aligned = feature_df.loc[X_val.index, 'demand'].values # Ensure alignment

    return X_train, y_train, X_val, y_val_aligned

def main(item_id="HOBBIES_1_001", store_id="CA_1", n_splits=5):
    """Main function to run the stacking ensemble experiment"""
    logger = setup_logging()
    logger.info("🚀 STEP 4: Stacking Ensemble Implementation")

    categorical_cols = [
        'year', 'month', 'day', 'day_of_week', 'day_of_year', 
        'week_of_year', 'quarter', 'is_weekend', 'is_month_start', 
        'is_month_end', 'is_quarter_start', 'is_quarter_end',
        'event_name_1', 'event_type_1', 'promotional_event', 'holiday_event'
    ]

    # --- 2. DATA PREPARATION (OUTER SPLIT) ---
    logger.info("Preparing data with an outer holdout test split...")
    full_df = load_m5_data(item_id, store_id)
    
    # We will use the original train+val split as our development set
    # and the original test split as our final holdout set.
    train_ratio = get_config().DATA_CONFIG['train_ratio']
    val_ratio = get_config().DATA_CONFIG['validation_ratio']
    
    dev_size = int((train_ratio + val_ratio) * len(full_df))
    
    development_set = full_df.iloc[:dev_size].copy()
    holdout_test_set = full_df.iloc[dev_size:].copy()

    y_dev = development_set['demand'].values
    y_test_holdout = holdout_test_set['demand'].values

    logger.info(f"Development set size: {len(development_set)}")
    logger.info(f"Holdout test set size: {len(holdout_test_set)}")

    # --- 3. GENERATE OUT-OF-FOLD (OOF) PREDICTIONS ---
    logger.info("Generating Out-of-Fold (OOF) predictions for the meta-learner...")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Arrays to store OOF predictions
    oof_preds = np.zeros((len(development_set), 3)) # 3 base models
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(development_set)):
        logger.info(f"--- Processing Fold {fold+1}/{n_splits} ---")
        
        train_fold_df, val_fold_df = development_set.iloc[train_idx], development_set.iloc[val_idx]
        y_train_fold, y_val_fold = y_dev[train_idx], y_dev[val_idx]

        # A. SARIMAX
        logger.info("Training SARIMAX for OOF...")
        sar_model = SARIMAXModel(order=(1,1,1), seasonal_order=(1,1,1,12))
        sar_model.train(train_fold_df['demand'])
        sar_preds = sar_model.predict(steps=len(val_fold_df))
        oof_preds[val_idx, 0] = sar_preds

        # B. LSTM
        logger.info("Training LSTM for OOF...")
        lstm_model = LSTMModel(lstm_units=[50, 50])
        lstm_model.train(y_train_fold, epochs=20, verbose=0)
        lstm_preds = lstm_model.predict(y_train_fold, steps=len(val_fold_df))
        oof_preds[val_idx, 1] = lstm_preds

        # C. LightGBM
        logger.info("Training LightGBM for OOF...")
        lgbm_model = LightGBMModel(feature_engineering=True)
        X_train_lgbm, y_train_lgbm, X_val_lgbm, _ = create_lgbm_features_for_fold(train_fold_df, val_fold_df, lgbm_model)
        # Identify categorical features that exist in the training columns
        existing_categorical_cols = [col for col in categorical_cols if col in X_train_lgbm.columns]
        lgbm_model.train(
            X_train_lgbm, 
            y_train_lgbm, 
            categorical_features=existing_categorical_cols, # Pass the list here
            verbose_eval=False
        )
        lgbm_preds = lgbm_model.predict(X_val_lgbm)
        oof_preds[val_idx, 2] = lgbm_preds
    
    logger.info("✅ OOF prediction generation complete.")

    # --- 4. TRAIN META-LEARNER ---
    logger.info("Training the meta-learner (Ridge Regression)...")
    
    X_meta = oof_preds
    y_meta = y_dev
    
    meta_learner = Ridge(alpha=1.0)
    meta_learner.fit(X_meta, y_meta)
    
    logger.info(f"Meta-learner trained. Coefficients: {meta_learner.coef_}")

    # --- 5. EVALUATE ON HOLDOUT TEST SET ---
    logger.info("Evaluating Stacking Ensemble on the holdout test set...")

    # A. Train base models on the full development set
    logger.info("Retraining base models on full development set...")
    
    # SARIMAX
    sar_full = SARIMAXModel(order=(1,1,1), seasonal_order=(1,1,1,12))
    sar_full.train(development_set['demand'])
    
    # LSTM
    lstm_full = LSTMModel(lstm_units=[50, 50])
    lstm_full.train(y_dev, epochs=20, verbose=0)
    
    # LightGBM
    lgbm_full = LightGBMModel(feature_engineering=True)
    X_dev_lgbm, y_dev_lgbm, X_test_lgbm, _ = create_lgbm_features_for_fold(development_set, holdout_test_set, lgbm_full)
    # Identify categorical features that exist in the development columns
    existing_categorical_cols_full = [col for col in categorical_cols if col in X_dev_lgbm.columns]

    lgbm_full.train(
        X_dev_lgbm, 
        y_dev_lgbm, 
        categorical_features=existing_categorical_cols_full, # Pass the list here
        verbose_eval=False
    )

    # B. Generate predictions on the holdout test set to create meta-features
    logger.info("Generating base predictions for the holdout test set...")
    
    test_sar_preds = sar_full.predict(steps=len(holdout_test_set))
    test_lstm_preds = lstm_full.predict(y_dev, steps=len(holdout_test_set))
    test_lgbm_preds = lgbm_full.predict(X_test_lgbm)
    
    X_test_meta = np.column_stack([
        test_sar_preds,
        test_lstm_preds,
        test_lgbm_preds
    ])
    
    # C. Make final predictions with the meta-learner
    final_stacking_preds = meta_learner.predict(X_test_meta)
    
    # D. Calculate final performance metrics
    stacking_mae = mean_absolute_error(y_test_holdout, final_stacking_preds)
    stacking_rmse = np.sqrt(mean_squared_error(y_test_holdout, final_stacking_preds))

    logger.info("📊 Stacking Ensemble FINAL Performance:")
    logger.info(f"   - Test MAE: {stacking_mae:.4f}")
    logger.info(f"   - Test RMSE: {stacking_rmse:.4f}")
    
    # --- 6. SAVE REPORT ---
    report = {
        'timestamp': datetime.now().isoformat(),
        'item': item_id,
        'store': store_id,
        'ensemble_type': 'stacking',
        'meta_learner': 'Ridge',
        'test_mae': stacking_mae,
        'test_rmse': stacking_rmse,
        'meta_learner_coef': meta_learner.coef_.tolist(),
        'status': 'completed'
    }
    
    report_path = root / 'experiments/results/step4_stacking_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"💾 Report saved to {report_path}")

if __name__ == "__main__":
    main()