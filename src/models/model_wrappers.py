# src/models/model_wrappers.py
"""
Wrapper functions for training models to support batch processing
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional

from src.models.base.sarimax_model import SARIMAXModel
from src.models.base.lstm_model import LSTMModel
from src.models.base.lightgbm_model import LightGBMModel
from experiments.configs.model_configs import get_config

logger = logging.getLogger(__name__)

def train_sarimax_wrapper(train_df: pd.DataFrame, 
                         val_df: pd.DataFrame, 
                         test_df: pd.DataFrame,
                         config=None) -> Dict[str, Any]:
    """
    Wrapper to train SARIMAX model
    """
    try:
        if config is None:
            config = get_config('development')
            
        sarimax_config = config.SARIMAX_CONFIG
        
        # Initialize model
        model = SARIMAXModel(
            order=sarimax_config['order'],
            seasonal_order=sarimax_config['seasonal_order']
        )
        
        # Train
        train_results = model.train(train_df['demand'])
        
        # Predict on Test
        test_pred = model.predict(steps=len(test_df))
        
        # Calculate Metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(test_df['demand'], test_pred)
        rmse = np.sqrt(mean_squared_error(test_df['demand'], test_pred))
        
        return {
            'model': model,
            'predictions': test_pred,
            'metrics': {'mae': mae, 'rmse': rmse},
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"SARIMAX Wrapper failed: {str(e)}")
        return {'status': 'failed', 'error': str(e)}

def train_lstm_wrapper(train_df: pd.DataFrame, 
                      val_df: pd.DataFrame, 
                      test_df: pd.DataFrame,
                      config=None) -> Dict[str, Any]:
    """
    Wrapper to train LSTM model
    """
    try:
        if config is None:
            config = get_config('development')
            
        lstm_config = config.LSTM_CONFIG
        
        # Initialize model
        model = LSTMModel(
            sequence_length=lstm_config['sequence_length'],
            lstm_units=lstm_config['lstm_units'],
            dropout_rate=lstm_config['dropout_rate'],
            learning_rate=lstm_config['learning_rate']
        )
        
        # Train
        # Train
        train_data = train_df['demand'].values
        
        # Prepare Validation Data with Context
        val_data = None
        if val_df is not None:
            # Prepend last sequence_length from train to val to provide context
            # This is critical because LSTMModel._create_sequences needs previous window to predict the first validation point
            seq_len = lstm_config['sequence_length']
            if len(train_data) >= seq_len:
                val_context = train_data[-seq_len:]
                val_data = np.concatenate([val_context, val_df['demand'].values])
            else:
                # Fallback if train is too short (unlikely in M5 but needed for robustness)
                val_data = val_df['demand'].values
        
        model.train(
            train_data=train_data,
            validation_data=val_data,
            epochs=lstm_config['epochs'],
            batch_size=lstm_config['batch_size'],
            verbose=0
        )
        
        # Predict on Test
        # Note: LSTM needs sequence context from previous data
        # We concatenate last part of val (or train) to start predicting test
        full_data = pd.concat([train_df, val_df])['demand'].values
        predictions = model.predict(full_data, steps=len(test_df))
        
        # Calculate Metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(test_df['demand'], predictions)
        rmse = np.sqrt(mean_squared_error(test_df['demand'], predictions))
        
        return {
            'model': model,
            'predictions': predictions,
            'metrics': {'mae': mae, 'rmse': rmse},
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"LSTM Wrapper failed: {str(e)}")
        return {'status': 'failed', 'error': str(e)}

def train_lightgbm_wrapper(train_df: pd.DataFrame, 
                          val_df: pd.DataFrame, 
                          test_df: pd.DataFrame,
                          config=None) -> Dict[str, Any]:
    """
    Wrapper to train LightGBM model
    """
    try:
        if config is None:
            config = get_config('development')
            
        lgbm_config = config.LIGHTGBM_CONFIG
        
        # Initialize model
        model = LightGBMModel(
            params=lgbm_config['params'],
            feature_engineering=lgbm_config['feature_engineering']
        )
        
        # Create Features
        train_feat = model.create_features(train_df, 'demand')
        val_feat = model.create_features(val_df, 'demand')
        test_feat = model.create_features(test_df, 'demand')
        
        # Prepare Data
        feat_cols = [c for c in train_feat.columns if c not in ['date', 'demand']]
        
        X_train = train_feat[feat_cols]
        y_train = train_feat['demand']
        
        X_val = val_feat[feat_cols]
        y_val = val_feat['demand']
        
        X_test = test_feat[feat_cols]
        
        # Train
        model.train(
            X_train, y_train,
            X_val, y_val,
            num_boost_round=lgbm_config['num_boost_round'],
            early_stopping_rounds=lgbm_config['early_stopping_rounds'],
            verbose_eval=-1
        )
        
        # Predict
        predictions = model.predict(X_test)
        
        # Calculate Metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(test_df['demand'], predictions)
        rmse = np.sqrt(mean_squared_error(test_df['demand'], predictions))
        
        return {
            'model': model,
            'predictions': predictions,
            'metrics': {'mae': mae, 'rmse': rmse},
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"LightGBM Wrapper failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'error': str(e)}
