# Individual Model Implementations - SARIMAX Model

"""
src/models/base/sarimax_model.py
SARIMAX model implementation for time series forecasting
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Any, Optional, Tuple
import logging

# Statistical models
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class SARIMAXModel:
    """
    SARIMAX Model wrapper with hyperparameter tuning and validation
    """
    
    def __init__(self, 
                 order: Tuple[int, int, int] = (1, 1, 1), 
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
                 trend: str = 'c',
                 enforce_stationarity: bool = False,
                 enforce_invertibility: bool = False):
        """
        Initialize SARIMAX model
        
        Args:
            order: (p, d, q) non-seasonal parameters
            seasonal_order: (P, D, Q, s) seasonal parameters  
            trend: Trend component ('c' for constant, 't' for trend, etc.)
            enforce_stationarity: Whether to enforce stationarity
            enforce_invertibility: Whether to enforce invertibility
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        
        self.model = None
        self.fitted_model = None
        self.is_trained = False
        self.training_data = None
        self.exog_data = None
        
    def _validate_data(self, data: pd.Series, exog: Optional[pd.DataFrame] = None) -> bool:
        """Validate input data"""
        if not isinstance(data, pd.Series):
            raise ValueError("Data must be a pandas Series")
        
        if data.isnull().sum() > 0:
            logger.warning(f"Found {data.isnull().sum()} missing values in data")
        
        if len(data) < 50:
            logger.warning("Data length is less than 50 observations. Results may be unreliable.")
        
        return True
    
    def _check_stationarity(self, data: pd.Series) -> Dict[str, Any]:
        """Check stationarity using ADF test"""
        from statsmodels.tsa.stattools import adfuller
        
        result = adfuller(data.dropna())
        
        stationarity_result = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
        
        return stationarity_result
    
    def analyze_seasonality(self, data: pd.Series, period: int = 12) -> Dict[str, Any]:
        """Analyze seasonal components"""
        try:
            if len(data) < 2 * period:
                return {'error': 'Insufficient data for seasonal decomposition'}
            
            decomposition = seasonal_decompose(
                data.dropna(), 
                model='additive', 
                period=period, 
                extrapolate_trend='freq'
            )
            
            # Calculate seasonal strength
            seasonal_var = np.var(decomposition.seasonal)
            remainder_var = np.var(decomposition.resid[~np.isnan(decomposition.resid)])
            seasonal_strength = seasonal_var / (seasonal_var + remainder_var)
            
            return {
                'seasonal_strength': seasonal_strength,
                'trend_component': decomposition.trend,
                'seasonal_component': decomposition.seasonal,
                'residual_component': decomposition.resid
            }
            
        except Exception as e:
            logger.error(f"Seasonal analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def auto_arima_order(self, 
                        data: pd.Series, 
                        max_p: int = 3, 
                        max_d: int = 2, 
                        max_q: int = 3,
                        max_P: int = 2, 
                        max_D: int = 1, 
                        max_Q: int = 2,
                        seasonal: bool = True) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
        """
        Automatically determine optimal ARIMA orders using information criteria
        """
        logger.info("Running auto ARIMA order selection...")
        
        best_aic = np.inf
        best_order = None
        best_seasonal_order = None
        
        # Define ranges
        p_range = range(0, max_p + 1)
        d_range = range(0, max_d + 1)  
        q_range = range(0, max_q + 1)
        
        if seasonal:
            P_range = range(0, max_P + 1)
            D_range = range(0, max_D + 1)
            Q_range = range(0, max_Q + 1)
        else:
            P_range = [0]
            D_range = [0]
            Q_range = [0]
        
        total_combinations = len(p_range) * len(d_range) * len(q_range) * len(P_range) * len(D_range) * len(Q_range)
        logger.info(f"Testing {total_combinations} combinations...")
        
        tested = 0
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    for P in P_range:
                        for D in D_range:
                            for Q in Q_range:
                                try:
                                    order = (p, d, q)
                                    seasonal_order = (P, D, Q, self.seasonal_order[3])
                                    
                                    model = SARIMAX(
                                        data,
                                        order=order,
                                        seasonal_order=seasonal_order,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False
                                    )
                                    
                                    fitted = model.fit(disp=False, maxiter=50)
                                    
                                    if fitted.aic < best_aic:
                                        best_aic = fitted.aic
                                        best_order = order
                                        best_seasonal_order = seasonal_order
                                    
                                    tested += 1
                                    
                                except Exception:
                                    continue
        
        logger.info(f"Tested {tested} valid combinations")
        logger.info(f"Best order: {best_order}, Best seasonal order: {best_seasonal_order}, AIC: {best_aic:.4f}")
        
        if best_order is None:
            logger.warning("Auto ARIMA failed, using default orders")
            return self.order, self.seasonal_order
        
        return best_order, best_seasonal_order
    
    def train(self, 
              train_data: pd.Series, 
              exog: Optional[pd.DataFrame] = None,
              auto_order: bool = False,
              validate: bool = True,
              maxiter: int = 200) -> Dict[str, Any]:
        """
        Train SARIMAX model with optional validation
        """
        logger.info(f"Training SARIMAX model with order {self.order} and seasonal_order {self.seasonal_order}")
        
        # Validate data
        self._validate_data(train_data, exog)
        
        # Store training data
        self.training_data = train_data.copy()
        self.exog_data = exog.copy() if exog is not None else None
        
        # Auto order selection if requested
        if auto_order:
            self.order, self.seasonal_order = self.auto_arima_order(train_data)
            logger.info(f"Auto-selected order: {self.order}, seasonal_order: {self.seasonal_order}")
        
        try:
            # Initialize model
            self.model = SARIMAX(
                train_data,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility
            )
            
            # Fit model
            self.fitted_model = self.model.fit(disp=False, maxiter=maxiter)
            self.is_trained = True
            
            # Collect results
            results = {
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'log_likelihood': self.fitted_model.llf,
                'order_used': self.order,
                'seasonal_order_used': self.seasonal_order
            }
            
            if validate:
                # Model validation
                validation_results = self._validate_model(train_data)
                results.update(validation_results)
            
            logger.info(f"SARIMAX training completed. AIC: {results['aic']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"SARIMAX training failed: {str(e)}")
            raise
    
    def _validate_model(self, train_data: pd.Series) -> Dict[str, Any]:
        """Validate fitted model"""
        validation_results = {}
        
        try:
            # Residual analysis
            residuals = self.fitted_model.resid
            validation_results['residual_mean'] = np.mean(residuals)
            validation_results['residual_std'] = np.std(residuals)
            
            # Ljung-Box test for residual autocorrelation
            ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
            validation_results['ljung_box_pvalue'] = ljung_box['lb_pvalue'].iloc[-1]
            validation_results['residuals_autocorrelated'] = ljung_box['lb_pvalue'].iloc[-1] < 0.05
            
            # Normality test of residuals
            from scipy.stats import normaltest
            _, normality_pvalue = normaltest(residuals.dropna())
            validation_results['residual_normality_pvalue'] = normality_pvalue
            validation_results['residuals_normal'] = normality_pvalue > 0.05
            
        except Exception as e:
            logger.warning(f"Model validation failed: {str(e)}")
            validation_results['validation_error'] = str(e)
        
        return validation_results
    
    def predict(self, 
                steps: int, 
                exog: Optional[pd.DataFrame] = None,
                return_conf_int: bool = False,
                alpha: float = 0.05) -> pd.Series:
        """
        Generate forecasts
        
        Args:
            steps: Number of steps to forecast
            exog: Exogenous variables for forecast period
            return_conf_int: Whether to return confidence intervals
            alpha: Significance level for confidence intervals
            
        Returns:
            Forecast values (and confidence intervals if requested)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            if return_conf_int:
                forecast = self.fitted_model.get_forecast(steps=steps, exog=exog)
                predictions = forecast.predicted_mean
                conf_int = forecast.conf_int(alpha=alpha)
                
                return {
                    'forecast': predictions,
                    'lower_ci': conf_int.iloc[:, 0],
                    'upper_ci': conf_int.iloc[:, 1]
                }
            else:
                forecast = self.fitted_model.forecast(steps=steps, exog=exog)
                return forecast
                
        except Exception as e:
            logger.error(f"SARIMAX prediction failed: {str(e)}")
            raise
    
    def predict_in_sample(self) -> pd.Series:
        """Get in-sample predictions (fitted values)"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting fitted values")
        
        return self.fitted_model.fittedvalues
    
    def get_residuals(self) -> pd.Series:
        """Get model residuals"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting residuals")
        
        return self.fitted_model.resid
    
    def get_model_summary(self) -> str:
        """Get detailed model summary"""
        if not self.is_trained:
            return "Model not trained yet"
        
        return str(self.fitted_model.summary())
    
    def plot_diagnostics(self, figsize: Tuple[int, int] = (12, 8)):
        """Plot model diagnostics"""
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting diagnostics")
        
        try:
            import matplotlib.pyplot as plt
            
            fig = self.fitted_model.plot_diagnostics(figsize=figsize)
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.error("Matplotlib not available for plotting")
            return None
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'trend': self.trend,
            'enforce_stationarity': self.enforce_stationarity,
            'enforce_invertibility': self.enforce_invertibility,
            'is_trained': self.is_trained
        }
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        import pickle
        
        model_data = {
            'fitted_model': self.fitted_model,
            'params': self.get_params(),
            'training_data': self.training_data,
            'exog_data': self.exog_data
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"SARIMAX model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.fitted_model = model_data['fitted_model']
        self.training_data = model_data['training_data']
        self.exog_data = model_data['exog_data']
        
        # Restore parameters
        params = model_data['params']
        self.order = params['order']
        self.seasonal_order = params['seasonal_order']
        self.trend = params['trend']
        self.enforce_stationarity = params['enforce_stationarity']
        self.enforce_invertibility = params['enforce_invertibility']
        self.is_trained = params['is_trained']
        
        logger.info(f"SARIMAX model loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    trend = np.linspace(50, 100, 365)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(365) / 365.25)
    noise = np.random.normal(0, 5, 365)
    data = pd.Series(trend + seasonal + noise, index=dates)
    
    # Initialize and train model
    model = SARIMAXModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    
    # Train with auto order selection
    results = model.train(data, auto_order=True)
    print("Training Results:", results)
    
    # Make predictions
    predictions = model.predict(30)
    print("Predictions shape:", predictions.shape)
    
    # Get model summary
    print("Model Summary:")
    print(model.get_model_summary())