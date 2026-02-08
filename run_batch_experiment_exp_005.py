# run_batch_experiment.py
"""
Batch Experiment Script for Multi-Item Inventory Forecasting
Target: Master's Thesis Defense (Final Sprint)
Experiment ID: EXP-003
Change Log: 
- Base Models: LightGBM + LSTM (SARIMAX skipped)
- Meta Learner: LinearRegression (NNLS) instead of Ridge
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.m5_data_processor import process_multiple_items
from src.models.model_wrappers import train_sarimax_wrapper, train_lstm_wrapper, train_lightgbm_wrapper
from src.models.ensemble.stacking_model_exp_004 import StackingEnsemble
from src.utils.data_utils import load_data
from experiments.configs.model_configs import get_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def run_experiment_for_item(item_file: str, item_id: str, n_folds: int = 5) -> list:
    """
    Run 5-Fold Nested Cross-Validation for a single item
    """
    results_list = []
    
    # Load data
    df = pd.read_csv(item_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Configuration
    config = get_config('development')
    test_horizon = config.DATA_CONFIG['forecast_horizon'] 
    
    total_len = len(df)
    
    logger.info(f"Processing Item: {item_id} (Length: {total_len})")
    
    for fold in range(n_folds):
        try:
            # Rolling Origin Split
            test_end_idx = total_len - (fold * test_horizon)
            test_start_idx = test_end_idx - test_horizon
            
            if test_start_idx < 100: 
                break
                
            train_val_df = df.iloc[:test_start_idx].copy()
            test_df = df.iloc[test_start_idx:test_end_idx].copy()
            
            val_start_idx = len(train_val_df) - test_horizon
            
            train_df = train_val_df.iloc[:val_start_idx].copy()
            val_df = train_val_df.iloc[val_start_idx:].copy()
            
            logger.info(f"  Fold {fold+1}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            
            # 1. Train Base Models
            
            # --- SARIMAX:
            sarimax_res = train_sarimax_wrapper(train_df, val_df, test_df, config)
            # sarimax_res = {'status': 'skipped'}
            
            # LSTM
            lstm_res = train_lstm_wrapper(train_df, val_df, test_df, config)
            # LightGBM
            lgbm_res = train_lightgbm_wrapper(train_df, val_df, test_df, config)
            
            # Store Base Results
            models_map = {
                'SARIMAX': sarimax_res, 
                'LSTM': lstm_res, 
                'LightGBM': lgbm_res
            }
            
            for name, res in models_map.items():
                if res['status'] == 'success':
                    results_list.append({
                        'item_id': item_id,
                        'fold': fold + 1,
                        'model': name,
                        'mae': res['metrics']['mae'],
                        'rmse': res['metrics']['rmse'],
                        'status': 'success'
                    })
                else:
                     results_list.append({
                        'item_id': item_id,
                        'fold': fold + 1,
                        'model': name,
                        'mae': None, 'rmse': None,
                        'status': 'failed',
                        'error': res.get('error', 'unknown')
                    })

            # 2. Train Stacking Ensemble
            valid_preds = {}
            test_preds = {}
            valid_y = val_df['demand'].values
            
            available_models = []
            
            # SARIMAX (Skipped logic)
            if sarimax_res.get('status') == 'success':
                try:
                    m = sarimax_res['model']
                    valid_preds['SARIMAX'] = m.predict(steps=len(val_df)).values
                    test_preds['SARIMAX'] = sarimax_res['predictions'].values
                    available_models.append('SARIMAX')
                except Exception as e: logger.warning(f"SARIMAX Val Pred failed: {e}")

            if lstm_res['status'] == 'success':
                try:
                    m = lstm_res['model']
                    input_data = train_df['demand'].values
                    valid_preds['LSTM'] = m.predict(input_data, steps=len(val_df))
                    test_preds['LSTM'] = lstm_res['predictions']
                    available_models.append('LSTM')
                except Exception as e: logger.warning(f"LSTM Val Pred failed: {e}")

            if lgbm_res['status'] == 'success':
                try:
                    m = lgbm_res['model']
                    val_feat = m.create_features(val_df, 'demand')
                    feat_cols = [c for c in val_feat.columns if c not in ['date', 'demand']]
                    X_val = val_feat[feat_cols]
                    valid_preds['LightGBM'] = m.predict(X_val)
                    test_preds['LightGBM'] = lgbm_res['predictions']
                    available_models.append('LightGBM')
                except Exception as e: logger.warning(f"LightGBM Val Pred failed: {e}")

            # 3. Execute Stacking
            if len(available_models) >= 2:
                try:
                    # --- EXP-004 CHANGE: Use 'mae' optimizer ---
                    stacker = StackingEnsemble(meta_learner_type='mae')
                    # -------------------------------------------
                    
                    stacker.train(valid_preds, valid_y)
                    
                    stacking_metrics = stacker.evaluate(test_preds, test_df['demand'].values)
                    
                    results_list.append({
                        'item_id': item_id,
                        'fold': fold + 1,
                        'model': 'StackingEnsemble',
                        'mae': stacking_metrics['mae'],
                        'rmse': stacking_metrics['rmse'],
                        'status': 'success'
                    })
                    logger.info(f"  Stacking MAE: {stacking_metrics['mae']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Stacking failed: {e}")
                    results_list.append({
                        'item_id': item_id, 'fold': fold + 1, 'model': 'StackingEnsemble',
                        'status': 'failed', 'error': str(e)
                    })
            
        except Exception as e:
            logger.error(f"Fold {fold+1} failed: {e}")
            
    return results_list

def main():
    print("🚀 Starting Batch Experiment (Multi-Item) - EXP-003 (NNLS)")
    print("==========================================================")
    
    n_items = 5 
    print(f"preparing data for {n_items} items...")
    
    processed_files = process_multiple_items(
        category='HOBBIES',
        n_items=n_items,
        strategy='top_volume'
    )
    
    if not processed_files:
        print("❌ No items processed. Exiting.")
        return

    all_results = []
    
    for i, file_path in enumerate(processed_files):
        filename = Path(file_path).name
        parts = filename.split('_')
        item_id = "_".join(parts[1:-3]) 
        
        print(f"\nProcessing {i+1}/{len(processed_files)}: {item_id}")
        
        try:
            results = run_experiment_for_item(file_path, item_id)
            all_results.extend(results)
            
            pd.DataFrame(all_results).to_csv('experiments/results/results_multi_item_temp.csv', index=False)
            
        except Exception as e:
            print(f"❌ Failed processing {item_id}: {e}")
            import traceback
            traceback.print_exc()

    final_df = pd.DataFrame(all_results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'experiments/results/results_multi_item_{timestamp}.csv'
    
    final_df.to_csv(output_filename, index=False)
    
    print("\n✅ Experiment Completed!")
    print(f"Results saved to {output_filename}")
    
    if not final_df.empty and 'mae' in final_df.columns:
        summary = final_df[final_df['status']=='success'].groupby('model')[['mae', 'rmse']].mean()
        print("\n📊 Average Performance:")
        print(summary)


if __name__ == "__main__":
    main()