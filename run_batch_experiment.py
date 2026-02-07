# run_batch_experiment.py
"""
Batch Experiment Script for Multi-Item Inventory Forecasting
Target: Master's Thesis Defense (Final Sprint)
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
from src.models.ensemble.stacking_model import StackingEnsemble
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
    test_horizon = config.DATA_CONFIG['forecast_horizon'] # e.g. 30 days
    
    # Create folds (Rolling Origin)
    # We want 5 test periods at the end of the series
    total_len = len(df)
    
    logger.info(f"Processing Item: {item_id} (Length: {total_len})")
    
    for fold in range(n_folds):
        try:
            # Define indices
            # Fold 0: Test is the last block
            # Fold 1: Test is the 2nd to last block ...
            # Actually, standard TimeSeriesSplit expands forward.
            # But "Rolling Origin" usually means we test on T+1, T+2...
            # Let's take the last 5 months as 5 folds.
            
            test_end_idx = total_len - (fold * test_horizon)
            test_start_idx = test_end_idx - test_horizon
            
            if test_start_idx < 100: # Ensure enough training data
                break
                
            # Split: Train+Val | Test
            train_val_df = df.iloc[:test_start_idx].copy()
            test_df = df.iloc[test_start_idx:test_end_idx].copy()
            
            # Nested Split for Stacking: Train | Val
            # Use last horizon of train_val for validation (stacking training)
            val_start_idx = len(train_val_df) - test_horizon
            
            train_df = train_val_df.iloc[:val_start_idx].copy()
            val_df = train_val_df.iloc[val_start_idx:].copy()
            
            logger.info(f"  Fold {fold+1}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            
            # 1. Train Base Models
            # SARIMAX
            sarimax_res = train_sarimax_wrapper(train_df, val_df, test_df, config)
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
            # We need predictions on Validation Set to train Stacking
            # Get valid predictions from trained base models
            # Note: Wrapper returns Test predictions. We need a way to get Val predictions.
            # Inspecting wrappers: they accept val_df but use it for early stopping.
            # We need to explicitly predict on val_df using the trained models.
            
            valid_preds = {}
            test_preds = {}
            valid_y = val_df['demand'].values
            
            # Helper to get predictions
            available_models = []
            
            if sarimax_res['status'] == 'success':
                # SARIMAX predict val (need to handle steps/exog carefully, or used fitted values if in-sample)
                # For simplicity in this sprint, we re-predict steps ahead from train end
                # Model is trained on train_df. Val follows immediately.
                try:
                    m = sarimax_res['model']
                    # Forecast steps=len(val)
                    valid_preds['SARIMAX'] = m.predict(steps=len(val_df)).values
                    test_preds['SARIMAX'] = sarimax_res['predictions'].values # Wrapper returns test preds
                    available_models.append('SARIMAX')
                except Exception as e: logger.warning(f"SARIMAX Val Pred failed: {e}")

            if lstm_res['status'] == 'success':
                try:
                    m = lstm_res['model']
                    # Predict Val
                    # LSTM needs context. train_df end is context for val_df.
                    # Model.predict wrapper logic:
                    # predictions = model.predict(full_data, steps=len(test_df))
                    # We replicate this for val
                    input_data = train_df['demand'].values
                    valid_preds['LSTM'] = m.predict(input_data, steps=len(val_df))
                    test_preds['LSTM'] = lstm_res['predictions']
                    available_models.append('LSTM')
                except Exception as e: logger.warning(f"LSTM Val Pred failed: {e}")

            if lgbm_res['status'] == 'success':
                try:
                    m = lgbm_res['model']
                    # Re-create features for Val (already done conceptually in wrapper but we need access)
                    # We can use the model to predict on X_val if we can construct it.
                    # LightGBMModel.predict takes X.
                    # We can re-use the create_features method.
                    val_feat = m.create_features(val_df, 'demand')
                    feat_cols = [c for c in val_feat.columns if c not in ['date', 'demand']]
                    X_val = val_feat[feat_cols]
                    valid_preds['LightGBM'] = m.predict(X_val)
                    test_preds['LightGBM'] = lgbm_res['predictions']
                    available_models.append('LightGBM')
                except Exception as e: logger.warning(f"LightGBM Val Pred failed: {e}")

            # 3. Execute Stacking if we have at least 2 models
            if len(available_models) >= 2:
                try:
                    stacker = StackingEnsemble(meta_learner_type='ridge')
                    
                    # Filter valid_preds to ensure same length (should be)
                    # Train Stacker
                    stacker.train(valid_preds, valid_y)
                    
                    # Evaluate Stacker on Test
                    # Use test_preds from base models
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
    print("🚀 Starting Batch Experiment (Multi-Item)")
    print("=========================================")
    
    # 1. Select and Process Items
    n_items = 5 # Start with 5 for testing, user asked for 20-50
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
    
    # 2. Main Loop
    for i, file_path in enumerate(processed_files):
        # Extract item_id from filename (m5_{item_id}_{store_id}_data.csv)
        filename = Path(file_path).name
        parts = filename.split('_')
        # HOBBIES_1_001 is parts[1] + '_' + parts[2] + '_' + parts[3] 
        # Actually logic inside process_multiple_items naming: f"m5_{item_id}_{store_id}_data.csv"
        # Item ID could contain underscores.
        # Safe way: Load DF and check columns? Or just rely on string parsing if we know format.
        # M5 IDs like HOBBIES_1_001.
        item_id = "_".join(parts[1:-3]) # approximation
        
        print(f"\nProcessing {i+1}/{len(processed_files)}: {item_id}")
        
        try:
            results = run_experiment_for_item(file_path, item_id)
            all_results.extend(results)
            
            # Intermediate Save
            pd.DataFrame(all_results).to_csv('experiments/results/results_multi_item_temp.csv', index=False)
            
        except Exception as e:
            print(f"❌ Failed processing {item_id}: {e}")
            import traceback
            traceback.print_exc()

    # 3. Final Save and specific analysis
    final_df = pd.DataFrame(all_results)
    
    # สร้างชื่อไฟล์พร้อม Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # ตัวอย่าง: 20251014_153000
    output_filename = f'experiments/results/results_multi_item_{timestamp}.csv'
    
    # บันทึกด้วยชื่อใหม่
    final_df.to_csv(output_filename, index=False)
    
    print("\n✅ Experiment Completed!")
    print(f"Results saved to {output_filename}")
    
    # Show Summary
    if not final_df.empty and 'mae' in final_df.columns:
        summary = final_df[final_df['status']=='success'].groupby('model')[['mae', 'rmse']].mean()
        print("\n📊 Average Performance:")
        print(summary)


if __name__ == "__main__":
    main()
