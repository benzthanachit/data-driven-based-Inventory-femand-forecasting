# src/models/ensemble/stacking_model.py
"""
Stacking Ensemble Model Implementation
Updated for EXP-004: Added MAE Optimization
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

class MAEWeightOptimizer:
    """
    Custom Meta-Learner that optimizes Mean Absolute Error (MAE)
    using SciPy's SLSQP solver.
    Constraints: Weights sum to 1, Weights >= 0
    """
    def __init__(self):
        self.weights = None
        self.coef_ = None # for interface compatibility

    def fit(self, X, y):
        # X can be DataFrame or ndarray
        x_values = X.values if hasattr(X, 'values') else X
        n_models = x_values.shape[1]
        
        # Objective Function: Minimize MAE
        def objective(weights):
            preds = np.dot(x_values, weights)
            return np.mean(np.abs(y - preds))
        
        # Constraint: Sum of weights = 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        # Bounds: 0 <= w <= 1
        bounds = tuple((0.0, 1.0) for _ in range(n_models))
        
        # Initial guess: Equal weights
        initial_guess = np.ones(n_models) / n_models
        
        # Run Optimization
        try:
            result = minimize(
                objective, 
                initial_guess, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                tol=1e-6
            )
            self.weights = result.x
        except Exception as e:
            logger.error(f"Optimization failed: {e}. Fallback to equal weights.")
            self.weights = initial_guess
            
        self.coef_ = self.weights
        return self

    def predict(self, X):
        x_values = X.values if hasattr(X, 'values') else X
        return np.dot(x_values, self.weights)

class StackingEnsemble:
    """
    Stacking Ensemble that uses a meta-learner to combine base model predictions
    """
    
    def __init__(self, meta_learner_type: str = 'ridge', alpha: float = 1.0):
        self.meta_learner_type = meta_learner_type
        self.meta_learner = self._get_meta_learner(meta_learner_type, alpha)
        self.is_trained = False
        self.weights = None
        
    def _get_meta_learner(self, learner_type: str, alpha: float):
        if learner_type == 'ridge':
            return Ridge(alpha=alpha, positive=True, fit_intercept=False)
        elif learner_type == 'linear':
            return LinearRegression(positive=True, fit_intercept=False)
        elif learner_type == 'mae':
            return MAEWeightOptimizer() # New Custom Optimizer
        else:
            raise ValueError(f"Unsupported meta-learner: {learner_type}")
            
    def prepare_meta_features(self, predictions_dict: Dict[str, Union[np.ndarray, pd.Series]]) -> pd.DataFrame:
        """
        Convert dictionary of predictions into a DataFrame for the meta-learner
        """
        data = {}
        for name, preds in predictions_dict.items():
            if isinstance(preds, pd.Series):
                data[name] = preds.values
            else:
                data[name] = preds
                
        return pd.DataFrame(data)
        
    def train(self, 
              base_predictions: Dict[str, Union[np.ndarray, pd.Series]], 
              y_true: Union[np.ndarray, pd.Series]):
        """
        Train the meta-learner on OOF predictions from base models
        """
        X_meta = self.prepare_meta_features(base_predictions)
        
        # Train meta-learner
        self.meta_learner.fit(X_meta, y_true)
        self.is_trained = True
        
        # Store weights (coefficients)
        # Handle different meta-learner attributes
        if hasattr(self.meta_learner, 'coef_'):
            coefs = self.meta_learner.coef_
            if coefs.ndim > 1: coefs = coefs.flatten()
            self.weights = dict(zip(X_meta.columns, coefs))
        
        # Normalize weights to sum to 1 (optional, but good for interpretation)
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.normalized_weights = {k: v/total_weight for k, v in self.weights.items()}
        else:
            self.normalized_weights = self.weights
            
        logger.info(f"Stacking Ensemble trained ({self.meta_learner_type}). Weights: {self.weights}")
        return self.weights
        
    def predict(self, base_predictions: Dict[str, Union[np.ndarray, pd.Series]]) -> np.ndarray:
        """
        Generate predictions using the meta-learner
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained first")
            
        X_meta = self.prepare_meta_features(base_predictions)
        return self.meta_learner.predict(X_meta)
        
    def evaluate(self, 
                 base_predictions: Dict[str, Union[np.ndarray, pd.Series]], 
                 y_true: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Evaluate ensemble performance
        """
        preds = self.predict(base_predictions)
        
        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        
        return {
            'mae': mae,
            'rmse': rmse
        }