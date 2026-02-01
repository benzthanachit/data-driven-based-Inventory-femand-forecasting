
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd()))

from src.models.model_wrappers import train_lstm_wrapper
from experiments.configs.model_configs import get_config

def test_lstm():
    # Load data
    df = pd.read_csv('data/processed/m5_HOBBIES_1_348_CA_1_data.csv')
    
    # Split exactly like batch experiment
    test_horizon = 30
    total_len = len(df)
    test_end_idx = total_len
    test_start_idx = test_end_idx - test_horizon
    
    train_val_df = df.iloc[:test_start_idx].copy()
    test_df = df.iloc[test_start_idx:test_end_idx].copy()
    
    val_start_idx = len(train_val_df) - test_horizon
    train_df = train_val_df.iloc[:val_start_idx].copy()
    val_df = train_val_df.iloc[val_start_idx:].copy()
    
    print(f"Train len: {len(train_df)}")
    print(f"Val len: {len(val_df)}")
    
    # Run Wrapper
    config = get_config('development')
    # Faster training for test
    # config.LSTM_CONFIG['epochs'] = 2 
    
    print("Running LSTM Wrapper...")
    res = train_lstm_wrapper(train_df, val_df, test_df, config)
    
    if res['status'] == 'success':
        print("✅ LSTM Success!")
        print(f"MAE: {res['metrics']['mae']}")
    else:
        print(f"❌ LSTM Failed: {res['error']}")

if __name__ == "__main__":
    test_lstm()
