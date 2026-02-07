# src/models/ensemble/weighted_ensemble.py
"""
Weighted Average Ensemble - Moved from previous implementation
Fixed and optimized version for Step 2
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from abc import ABC, abstractmethod

# Optimization libraries
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Try to import optuna (optional)
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)


class BaseEnsemble(ABC):
    """Abstract base class for ensemble methods"""
    
    @abstractmethod
    def fit(self, predictions: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, Any]:
        """Fit the ensemble method"""
        pass
    
    @abstractmethod
    def predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate ensemble predictions"""
        pass


class WeightedAverageEnsemble(BaseEnsemble):
    """
    Weighted Average Ensemble with optimized weights
    """
    
    def __init__(self, 
                 optimization_method: str = 'scipy',
                 loss_function: str = 'mae',
                 weights_bounds: Tuple[float, float] = (0.0, 1.0),
                 normalize_weights: bool = True,
                 random_state: int = 42):
        """
        Initialize Weighted Average Ensemble
        
        Args:
            optimization_method: Method for weight optimization ('scipy', 'optuna', 'grid_search')
            loss_function: Loss function to minimize ('mae', 'mse', 'rmse')
            weights_bounds: Bounds for individual weights
            normalize_weights: Whether to normalize weights to sum to 1
            random_state: Random state for reproducibility
        """
        self.optimization_method = optimization_method
        self.loss_function = loss_function
        self.weights_bounds = weights_bounds
        self.normalize_weights = normalize_weights
        self.random_state = random_state
        
        # Ensemble state
        self.weights_ = None
        self.model_names_ = None
        self.optimization_result_ = None
        self.is_fitted = False
        
        np.random.seed(random_state)
    
    def _get_loss_function(self) -> Callable:
        """Get loss function"""
        if self.loss_function == 'mae':
            return mean_absolute_error
        elif self.loss_function == 'mse':
            return mean_squared_error
        elif self.loss_function == 'rmse':
            return lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")
    
    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize weights to sum to 1"""
        if self.normalize_weights:
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                return weights / weight_sum
            else:
                return np.ones_like(weights) / len(weights)
        return weights
    
    def _ensemble_predict(self, predictions: Dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions using given weights"""
        weights = self._normalize_weights(weights)
        
        ensemble_pred = np.zeros(len(next(iter(predictions.values()))))
        
        for i, (model_name, pred) in enumerate(predictions.items()):
            ensemble_pred += weights[i] * pred
        
        return ensemble_pred
    
    def _objective_function(self, weights: np.ndarray, 
                           predictions: Dict[str, np.ndarray], 
                           y_true: np.ndarray) -> float:
        """Objective function for weight optimization"""
        try:
            ensemble_pred = self._ensemble_predict(predictions, weights)
            loss_fn = self._get_loss_function()
            return loss_fn(y_true, ensemble_pred)
        except Exception as e:
            logger.warning(f"Error in objective function: {str(e)}")
            return np.inf
    
    def _scipy_optimize(self, predictions: Dict[str, np.ndarray], y_true: np.ndarray) -> Tuple[np.ndarray, float]:
        """Optimize weights using scipy.minimize"""
        n_models = len(predictions)
        
        # Initial weights (equal weights)
        initial_weights = np.ones(n_models) / n_models
        
        # Bounds for each weight
        bounds = [self.weights_bounds for _ in range(n_models)]
        
        # Constraint: weights sum to 1 (if normalization is enabled)
        constraints = []
        if self.normalize_weights:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1
            })
        
        # Optimize
        result = minimize(
            fun=self._objective_function,
            x0=initial_weights,
            args=(predictions, y_true),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            return result.x, result.fun
        else:
            logger.warning(f"Scipy optimization failed: {result.message}")
            return initial_weights, self._objective_function(initial_weights, predictions, y_true)
    
    def fit(self, predictions: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, Any]:
        """Fit the weighted ensemble"""
        logger.info(f"Fitting weighted ensemble with {self.optimization_method} optimization")
        
        # Validate inputs
        if not predictions:
            raise ValueError("Predictions dictionary cannot be empty")
        
        # Store model names
        self.model_names_ = list(predictions.keys())
        
        # Check prediction shapes
        pred_lengths = [len(pred) for pred in predictions.values()]
        if len(set(pred_lengths)) > 1:
            raise ValueError("All predictions must have the same length")
        
        if len(y_true) != pred_lengths[0]:
            raise ValueError("y_true length must match predictions length")
        
        # Optimize weights
        if self.optimization_method == 'scipy':
            weights, best_score = self._scipy_optimize(predictions, y_true)
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")
        
        # Store results
        self.weights_ = self._normalize_weights(weights)
        self.optimization_result_ = {
            'best_score': best_score,
            'method': self.optimization_method,
            'model_names': self.model_names_,
            'weights': self.weights_.copy()
        }
        
        self.is_fitted = True
        
        # Create results summary
        results = {
            'weights': dict(zip(self.model_names_, self.weights_)),
            'best_score': best_score,
            'optimization_method': self.optimization_method,
            'loss_function': self.loss_function
        }
        
        logger.info(f"Ensemble fitted. Best {self.loss_function}: {best_score:.6f}")
        logger.info(f"Optimal weights: {results['weights']}")
        
        return results
    
    def predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Validate input
        if set(predictions.keys()) != set(self.model_names_):
            raise ValueError("Prediction keys must match fitted model names")
        
        # Reorder predictions to match fitted order
        ordered_predictions = {name: predictions[name] for name in self.model_names_}
        
        # Generate ensemble predictions
        return self._ensemble_predict(ordered_predictions, self.weights_)
    
    def get_weights(self) -> Dict[str, float]:
        """Get the fitted weights"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before getting weights")
        
        return dict(zip(self.model_names_, self.weights_))
    
    def save_ensemble(self, filepath: str):
        """Save ensemble model"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted ensemble")
        
        import pickle
        ensemble_data = {
            'weights': self.weights_,
            'model_names': self.model_names_,
            'optimization_result': self.optimization_result_,
            'params': {
                'optimization_method': self.optimization_method,
                'loss_function': self.loss_function,
                'weights_bounds': self.weights_bounds,
                'normalize_weights': self.normalize_weights,
                'random_state': self.random_state
            }
        }
        
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        logger.info(f"Ensemble model saved to {filepath}.pkl")
    
    def load_ensemble(self, filepath: str):
        """Load ensemble model"""
        import pickle
        with open(f"{filepath}.pkl", 'rb') as f:
            ensemble_data = pickle.load(f)
        
        # Restore ensemble state
        self.weights_ = ensemble_data['weights']
        self.model_names_ = ensemble_data['model_names']
        self.optimization_result_ = ensemble_data['optimization_result']
        
        # Restore parameters
        params = ensemble_data['params']
        self.optimization_method = params['optimization_method']
        self.loss_function = params['loss_function']
        self.weights_bounds = params['weights_bounds']
        self.normalize_weights = params['normalize_weights']
        self.random_state = params['random_state']
        
        self.is_fitted = True
        
        logger.info(f"Ensemble model loaded from {filepath}.pkl")


# Example usage and testing
if __name__ == "__main__":
    # Create sample predictions from multiple models
    np.random.seed(42)
    n_samples = 200
    
    # Generate true values
    t = np.arange(n_samples)
    y_true = 50 + 0.1 * t + 10 * np.sin(2 * np.pi * t / 50) + np.random.normal(0, 2, n_samples)
    
    # Generate model predictions (with different error patterns)
    predictions = {
        'sarimax': y_true + np.random.normal(0, 3, n_samples),    # Higher variance
        'lstm': y_true + 2 + np.random.normal(0, 1, n_samples),  # Biased but lower variance
        'lightgbm': y_true + np.random.normal(0, 0.5, n_samples) # Best individual model
    }
    
    # Test weighted ensemble
    print("Testing Weighted Average Ensemble...")
    ensemble = WeightedAverageEnsemble(optimization_method='scipy')
    results = ensemble.fit(predictions, y_true)
    print("Fitting results:", results)
    
    # Test predictions
    ensemble_pred = ensemble.predict(predictions)
    ensemble_mae = mean_absolute_error(y_true, ensemble_pred)
    print(f"Ensemble MAE: {ensemble_mae:.4f}")
    
    # Compare with individual models
    for name, pred in predictions.items():
        mae = mean_absolute_error(y_true, pred)
        print(f"{name} MAE: {mae:.4f}")
    
    print(f"\nOptimal weights: {ensemble.get_weights()}")