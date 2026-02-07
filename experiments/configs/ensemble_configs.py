# experiments/configs/ensemble_configs.py
"""
Configuration for ensemble methods and Step 2 experiments
"""

import os
from typing import Dict, Any
from .model_configs import ModelConfig


class EnsembleConfig:
    """Configuration class for ensemble methods"""
    
    # Weighted Average Ensemble Configuration
    WEIGHTED_ENSEMBLE_CONFIG = {
        'optimization_method': 'scipy',    # 'scipy', 'optuna', 'grid_search'
        'loss_function': 'mae',           # 'mae', 'mse', 'rmse'
        'weights_bounds': (0.0, 1.0),    # Bounds for individual weights
        'normalize_weights': True,        # Normalize weights to sum to 1
        'random_state': 42,
        'max_iterations': 1000
    }
    
    # Cross-validation Configuration
    CV_CONFIG = {
        'method': 'time_series',          # 'time_series', 'standard'
        'n_splits': 5,                    # Number of CV folds
        'test_size': 30,                  # Test size for time series CV
        'scoring': 'mae',                 # Scoring metric
        'shuffle': False                  # Don't shuffle for time series
    }
    
    # Ensemble Evaluation Configuration
    EVALUATION_CONFIG = {
        'metrics': ['mae', 'rmse', 'mape', 'smape'],
        'confidence_level': 0.95,
        'bootstrap_samples': 1000,
        'significance_test': 'wilcoxon',  # 'wilcoxon', 'ttest'
        'plot_results': True
    }
    
    # Model Loading Configuration
    MODEL_LOADING_CONFIG = {
        'model_paths': {
            'sarimax': 'models/saved/sarimax_model.pkl',
            'lstm': 'models/saved/lstm_model',
            'lightgbm': 'models/saved/lightgbm_model'
        },
        'data_paths': {
            'train': 'data/processed/train_data.csv',
            'val': 'data/processed/val_data.csv', 
            'test': 'data/processed/test_data.csv',
            'full': 'data/synthetic/demand_data.csv'
        },
        'feature_data_available': True
    }
    
    # Ensemble Optimization Settings
    OPTIMIZATION_CONFIG = {
        'scipy': {
            'method': 'SLSQP',
            'maxiter': 1000,
            'ftol': 1e-9,
            'constraints': True
        },
        'optuna': {
            'n_trials': 100,
            'timeout': 300,  # seconds
            'n_jobs': 1
        },
        'grid_search': {
            'resolution': 0.05,
            'max_combinations': 10000
        }
    }
    
    # Results Storage Configuration
    RESULTS_CONFIG = {
        'save_directory': 'experiments/results',
        'ensemble_model_path': 'models/saved/ensemble_model',
        'report_filename': 'step2_ensemble_report.json',
        'detailed_results_filename': 'step2_detailed_results.json',
        'plots_directory': 'figures/step2_ensemble',
        'save_predictions': True
    }
    
    @classmethod
    def get_full_config(cls) -> Dict[str, Any]:
        """Get complete ensemble configuration"""
        return {
            'weighted_ensemble': cls.WEIGHTED_ENSEMBLE_CONFIG,
            'cv': cls.CV_CONFIG,
            'evaluation': cls.EVALUATION_CONFIG,
            'model_loading': cls.MODEL_LOADING_CONFIG,
            'optimization': cls.OPTIMIZATION_CONFIG,
            'results': cls.RESULTS_CONFIG
        }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories for ensemble results"""
        directories = [
            cls.RESULTS_CONFIG['save_directory'],
            cls.RESULTS_CONFIG['plots_directory'],
            'models/saved'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)


# Experiment-specific ensemble configurations
ENSEMBLE_EXPERIMENTS = {
    'quick_test': {
        'weighted_ensemble': {
            'optimization_method': 'scipy',
            'max_iterations': 100
        },
        'cv': {
            'n_splits': 3,
            'test_size': 20
        }
    },
    
    'full_optimization': {
        'weighted_ensemble': {
            'optimization_method': 'scipy',
            'max_iterations': 1000
        },
        'cv': {
            'n_splits': 5,
            'test_size': 30
        }
    },
    
    'comprehensive': {
        'weighted_ensemble': {
            'optimization_method': 'optuna',
            'max_iterations': 200
        },
        'cv': {
            'n_splits': 10,
            'test_size': 30
        },
        'evaluation': {
            'bootstrap_samples': 2000
        }
    }
}


def get_ensemble_config(experiment_name: str = 'full_optimization') -> Dict[str, Any]:
    """Get configuration for specific ensemble experiment"""
    base_config = EnsembleConfig.get_full_config()
    
    if experiment_name in ENSEMBLE_EXPERIMENTS:
        experiment_updates = ENSEMBLE_EXPERIMENTS[experiment_name]
        
        # Update base config with experiment-specific settings
        for section, updates in experiment_updates.items():
            if section in base_config:
                base_config[section].update(updates)
    
    return base_config


# Ensemble model combinations to test
ENSEMBLE_COMBINATIONS = {
    'all_models': ['sarimax', 'lstm', 'lightgbm'],
    'best_two': ['lstm', 'lightgbm'],
    'ml_models': ['lstm', 'lightgbm'],
    'statistical_ml': ['sarimax', 'lightgbm']
}


# Export for easy import
__all__ = [
    'EnsembleConfig', 
    'get_ensemble_config', 
    'ENSEMBLE_EXPERIMENTS',
    'ENSEMBLE_COMBINATIONS'
]