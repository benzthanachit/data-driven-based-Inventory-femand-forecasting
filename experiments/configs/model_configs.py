# experiments/configs/model_configs.py
"""
Configuration management for all models and experiments
"""

import os
from typing import Dict, Any


class ModelConfig:
    """Configuration class for all models and pipeline settings"""
    
    # Data settings
    DATA_CONFIG = {
        'target_column': 'demand',
        'date_column': 'date',
        'train_ratio': 0.7,
        'validation_ratio': 0.15,
        'test_ratio': 0.15,
        'sequence_length': 30,
        'forecast_horizon': 30
    }
    
    # SARIMAX model configuration
    SARIMAX_CONFIG = {
        'order': (1, 1, 1),
        'seasonal_order': (1, 1, 1, 12),
        'enforce_stationarity': False,
        'enforce_invertibility': False,
        'maxiter': 200,
        'method': 'lbfgs',
        'trend': 'c',
        'auto_order': False  # Set to True for automatic order selection
    }
    
    # LSTM model configuration  
    LSTM_CONFIG = {
        'sequence_length': 30,
        'lstm_units': [64, 32],
        'dropout_rate': 0.1,
        'lstm_units': [64, 32],
        'dropout_rate': 0.1,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'patience': 15,
        'validation_split': 0.2,
        'optimizer': 'adam',
        'batch_norm': False,
        'l2_reg': 0.001,
        'scaler_type': 'standard'
    }
    
    # LightGBM configuration
    LIGHTGBM_CONFIG = {
        'params': {
            'objective': 'regression',
            'metric': 'l1',
            'boosting_type': 'gbdt',
            'device_type': 'gpu', # Enable GPU
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': -1,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'seed': 42
        },
        'num_boost_round': 1000,
        'early_stopping_rounds': 100,
        'feature_engineering': True,
        'auto_feature_selection': False,
        'scaler_type': None
    }
    
    # Feature engineering settings
    FEATURE_CONFIG = {
        'lag_features': [1, 2, 3, 7, 14, 30],
        'rolling_windows': [7, 14, 30],
        'ewm_spans': [7, 30],
        'create_time_features': True,
        'create_difference_features': True
    }
    
    # Ensemble configuration (for future use)
    ENSEMBLE_CONFIG = {
        'method': 'weighted_average',
        'optimization_method': 'scipy',
        'cv_folds': 5,
        'test_size': 30,
        'weights_bounds': (0.0, 1.0),
        'normalize_weights': True
    }
    
    # Evaluation metrics
    METRICS_CONFIG = {
        'primary_metric': 'MAE',
        'metrics': ['MAE', 'RMSE', 'MAPE', 'SMAPE'],
        'significance_level': 0.05
    }
    
    # File paths and directories
    PATHS_CONFIG = {
        'data_dir': 'data',
        'raw_data_dir': 'data/raw',
        'processed_data_dir': 'data/processed',
        'synthetic_data_dir': 'data/synthetic',
        'models_dir': 'models/saved',
        'results_dir': 'experiments/results',
        'figures_dir': 'figures',
        'logs_dir': 'logs'
    }
    
    # Logging configuration
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_handler': True,
        'console_handler': True
    }
    
    @classmethod
    def get_full_config(cls) -> Dict[str, Any]:
        """Get complete configuration dictionary"""
        return {
            'data': cls.DATA_CONFIG,
            'sarimax': cls.SARIMAX_CONFIG,
            'lstm': cls.LSTM_CONFIG,
            'lightgbm': cls.LIGHTGBM_CONFIG,
            'features': cls.FEATURE_CONFIG,
            'ensemble': cls.ENSEMBLE_CONFIG,
            'metrics': cls.METRICS_CONFIG,
            'paths': cls.PATHS_CONFIG,
            'logging': cls.LOGGING_CONFIG
        }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for dir_name in cls.PATHS_CONFIG.values():
            os.makedirs(dir_name, exist_ok=True)
    
    @classmethod
    def update_config(cls, config_updates: Dict[str, Any]):
        """Update configuration with new values"""
        for section, updates in config_updates.items():
            if hasattr(cls, f"{section.upper()}_CONFIG"):
                config_attr = getattr(cls, f"{section.upper()}_CONFIG")
                config_attr.update(updates)


# Environment-specific configurations
class DevelopmentConfig(ModelConfig):
    """Development environment configuration"""
    
    LSTM_CONFIG = ModelConfig.LSTM_CONFIG.copy()
    LSTM_CONFIG.update({
        'epochs': 50,  # Increased for better convergence
        'patience': 10
    })
    
    LIGHTGBM_CONFIG = ModelConfig.LIGHTGBM_CONFIG.copy()
    LIGHTGBM_CONFIG['num_boost_round'] = 200


class ProductionConfig(ModelConfig):
    """Production environment configuration"""
    
    LSTM_CONFIG = ModelConfig.LSTM_CONFIG.copy()
    LSTM_CONFIG.update({
        'epochs': 200,  # More thorough training
        'patience': 20
    })
    
    LIGHTGBM_CONFIG = ModelConfig.LIGHTGBM_CONFIG.copy()
    LIGHTGBM_CONFIG['num_boost_round'] = 2000


# Configuration factory
def get_config(environment: str = 'development') -> ModelConfig:
    """Get configuration based on environment"""
    if environment.lower() == 'production':
        return ProductionConfig()
    elif environment.lower() == 'development':
        return DevelopmentConfig()
    else:
        return ModelConfig()


# Experiment-specific configurations
EXPERIMENT_CONFIGS = {
    'quick_test': {
        'lstm': {'epochs': 10, 'patience': 5},
        'lightgbm': {'num_boost_round': 50},
        'data': {'train_ratio': 0.8, 'validation_ratio': 0.1, 'test_ratio': 0.1}
    },
    
    'full_experiment': {
        'lstm': {'epochs': 200, 'patience': 20},
        'lightgbm': {'num_boost_round': 1000},
        'data': {'train_ratio': 0.7, 'validation_ratio': 0.15, 'test_ratio': 0.15}
    },
    
    'hyperparameter_tuning': {
        'sarimax': {'auto_order': True},
        'lstm': {'epochs': 100, 'patience': 15},
        'lightgbm': {'auto_feature_selection': True, 'num_boost_round': 500}
    }
}


def get_experiment_config(experiment_name: str) -> Dict[str, Any]:
    """Get configuration for specific experiment"""
    base_config = get_config('development').get_full_config()
    
    if experiment_name in EXPERIMENT_CONFIGS:
        experiment_updates = EXPERIMENT_CONFIGS[experiment_name]
        
        # Update base config with experiment-specific settings
        for section, updates in experiment_updates.items():
            if section in base_config:
                base_config[section].update(updates)
    
    return base_config


# Export for easy import
__all__ = ['ModelConfig', 'DevelopmentConfig', 'ProductionConfig', 
           'get_config', 'get_experiment_config', 'EXPERIMENT_CONFIGS']