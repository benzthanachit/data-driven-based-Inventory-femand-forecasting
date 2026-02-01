# src/models/ensemble/stacking_model.py
"""
Stacking Ensemble Model Implementation
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

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
        else:
            raise ValueError(f"Unsupported meta-learner: {learner_type}")
            
    def prepare_meta_features(self, predictions_dict: Dict[str, Union[np.ndarray, pd.Series]]) -> pd.DataFrame:
        """
        Convert dictionary of predictions into a DataFrame for the meta-learner
        """
        # Ensure all are numpy arrays
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
        self.weights = dict(zip(X_meta.columns, self.meta_learner.coef_))
        
        # Normalize weights to sum to 1 (optional, but good for interpretation)
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.normalized_weights = {k: v/total_weight for k, v in self.weights.items()}
        else:
            self.normalized_weights = self.weights
            
        logger.info(f"Stacking Ensemble trained. Weights: {self.weights}")
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
