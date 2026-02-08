# Fixed LightGBM Model - Complete corrected version

"""
src/models/base/lightgbm_model.py - Fixed for KeyError: 'mae'
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

# Tree-based models
import lightgbm as lgb

# Data preprocessing and evaluation
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class LightGBMModel:
    """
    Advanced LightGBM Model implementation for time series forecasting
    """
    
    def __init__(self, 
                 params: Optional[Dict] = None,
                 feature_engineering: bool = True,
                 auto_feature_selection: bool = False,
                 scaler_type: Optional[str] = None):
        """
        Initialize LightGBM model
        
        Args:
            params: LightGBM parameters
            feature_engineering: Whether to perform automatic feature engineering
            auto_feature_selection: Whether to perform automatic feature selection
            scaler_type: Type of scaler ('standard', 'robust', None)
        """
        self.params = params or self._get_default_params()
        self.feature_engineering = feature_engineering
        self.auto_feature_selection = auto_feature_selection
        self.scaler_type = scaler_type
        
        # Model components
        self.model = None
        self.feature_importance_ = None
        self.feature_names_ = None
        self.scaler = self._create_scaler() if scaler_type else None
        self.is_trained = False
        
        # Training information
        self.best_iteration = None
        self.training_log = []
        self.feature_selection_results = None
        
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default LightGBM parameters optimized for time series"""
        return {
            'objective': 'regression',
            'metric': 'l1',  # FIXED: Changed from 'mae' to 'l1'
            'boosting_type': 'gbdt',
            'device': 'cpu',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': -1,
            'min_data_in_leaf': 20,
            'min_child_weight': 0.001,
            'min_child_samples': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'categorical_feature': 'auto',
            'verbose': -1,
            'seed': 42,
            'deterministic': True
        }
    
    def _create_scaler(self):
        """Create appropriate scaler"""
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            return None
    
    def _create_time_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """
        Create time-based features
        
        Args:
            df: DataFrame with date column
            date_col: Name of date column
            
        Returns:
            DataFrame with time features
        """
        df = df.copy()
        
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Basic time features
            df['year'] = df[date_col].dt.year
            df['month'] = df[date_col].dt.month
            df['day'] = df[date_col].dt.day
            df['day_of_week'] = df[date_col].dt.dayofweek
            df['day_of_year'] = df[date_col].dt.dayofyear
            df['week_of_year'] = df[date_col].dt.isocalendar().week
            df['quarter'] = df[date_col].dt.quarter
            
            # Cyclical features
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
            df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Binary features
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
            df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
            df['is_quarter_start'] = df[date_col].dt.is_quarter_start.astype(int)
            df['is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame, target_col: str, 
                            lags: List[int] = [1, 2, 3, 7, 14, 30]) -> pd.DataFrame:
        """
        Create lag features
        
        Args:
            df: DataFrame with target column
            target_col: Name of target column
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame, target_col: str,
                                windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            df: DataFrame with target column
            target_col: Name of target column
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        for window in windows:
            # Rolling statistics
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
            df[f'{target_col}_rolling_median_{window}'] = df[target_col].rolling(window=window).median()
            
            # Rolling quantiles
            df[f'{target_col}_rolling_q25_{window}'] = df[target_col].rolling(window=window).quantile(0.25)
            df[f'{target_col}_rolling_q75_{window}'] = df[target_col].rolling(window=window).quantile(0.75)
            
        return df
    
    def _create_exponential_features(self, df: pd.DataFrame, target_col: str,
                                   spans: List[int] = [7, 30]) -> pd.DataFrame:
        """
        Create exponential weighted moving average features
        
        Args:
            df: DataFrame with target column
            target_col: Name of target column
            spans: List of span periods
            
        Returns:
            DataFrame with exponential features
        """
        df = df.copy()
        
        for span in spans:
            df[f'{target_col}_ewm_{span}'] = df[target_col].ewm(span=span).mean()
            df[f'{target_col}_ewm_std_{span}'] = df[target_col].ewm(span=span).std()
        
        return df
    
    def _create_difference_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Create difference features
        
        Args:
            df: DataFrame with target column
            target_col: Name of target column
            
        Returns:
            DataFrame with difference features
        """
        df = df.copy()
        
        # First difference
        df[f'{target_col}_diff_1'] = df[target_col].diff(1)
        df[f'{target_col}_diff_7'] = df[target_col].diff(7)
        df[f'{target_col}_diff_30'] = df[target_col].diff(30)
        
        # Percentage change
        df[f'{target_col}_pct_change_1'] = df[target_col].pct_change(1)
        df[f'{target_col}_pct_change_7'] = df[target_col].pct_change(7)
        
        return df
    
    def create_features(self, df: pd.DataFrame, target_col: str, 
                       date_col: str = 'date') -> pd.DataFrame:
        """
        Create comprehensive feature set
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            date_col: Name of date column
            
        Returns:
            DataFrame with engineered features
        """
        if not self.feature_engineering:
            return df
        
        logger.info("Creating engineered features...")
        
        # Start with original data
        feature_df = df.copy()
        
        # Time-based features
        if date_col in df.columns:
            feature_df = self._create_time_features(feature_df, date_col)
        
        # Lag features
        feature_df = self._create_lag_features(feature_df, target_col)
        
        # Rolling window features
        feature_df = self._create_rolling_features(feature_df, target_col)
        
        # Exponential weighted features
        feature_df = self._create_exponential_features(feature_df, target_col)
        
        # Difference features
        feature_df = self._create_difference_features(feature_df, target_col)
        
        logger.info(f"Created {len(feature_df.columns) - len(df.columns)} new features")
        
        # แปลง object columns เป็น category หรือ numeric สำหรับ LightGBM
        object_cols = feature_df.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            logger.info(f"Converting object columns to categorical: {object_cols}")
            for col in object_cols:
                # ลองแปลงเป็น numeric ก่อน ถ้าไม่ได้ใช้ category
                try:
                    feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
                    # ถ้าแปลงแล้วมี NaN เยอะ แสดงว่าเป็น text ให้ใช้ category
                    if feature_df[col].isna().sum() > len(feature_df) * 0.5:
                        feature_df[col] = df[col].astype('category').cat.codes
                    else:
                        feature_df[col] = feature_df[col].fillna(0)
                except:
                    # ถ้าแปลง numeric ไม่ได้ ให้ใช้ category codes
                    feature_df[col] = feature_df[col].astype('category').cat.codes

        return feature_df
    
    def _select_features(self, X_train: pd.DataFrame, y_train: pd.Series,
                        method: str = 'importance',
                        n_features: Optional[int] = None) -> List[str]:
        """
        Automatic feature selection
        
        Args:
            X_train: Training features
            y_train: Training target
            method: Feature selection method
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Performing feature selection using {method} method...")
        
        if method == 'importance':
            # Train a simple model to get feature importance
            temp_model = lgb.LGBMRegressor(**self.params, n_estimators=100)
            temp_model.fit(X_train, y_train)
            
            # Get feature importance
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': temp_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Select top n features
            if n_features is None:
                n_features = min(50, len(X_train.columns))  # Default to top 50
            
            selected_features = importance_df.head(n_features)['feature'].tolist()
            
            self.feature_selection_results = {
                'method': method,
                'n_selected': len(selected_features),
                'importance_scores': importance_df
            }
            
            logger.info(f"Selected {len(selected_features)} features using importance-based selection")
            
        else:
            selected_features = X_train.columns.tolist()
        
        return selected_features
    
    def train(self, 
              X_train: pd.DataFrame, 
              y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              num_boost_round: int = 1000,
              early_stopping_rounds: int = 100,
              categorical_features: Optional[List[str]] = None,
              sample_weight: Optional[np.ndarray] = None,
              verbose_eval: int = 100) -> Dict[str, Any]:
        """
        Train LightGBM model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            num_boost_round: Number of boosting rounds
            early_stopping_rounds: Early stopping rounds
            categorical_features: List of categorical feature names
            sample_weight: Sample weights
            verbose_eval: Verbose evaluation frequency
            
        Returns:
            Training results dictionary
        """
        logger.info("Training LightGBM model...")
        
        # Feature selection if enabled
        if self.auto_feature_selection:
            selected_features = self._select_features(X_train, y_train)
            X_train = X_train[selected_features]
            if X_val is not None:
                X_val = X_val[selected_features]
        
        # Store feature names
        self.feature_names_ = X_train.columns.tolist()
        
        # Scale features if specified
        if self.scaler:
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            
            if X_val is not None:
                X_val_scaled = pd.DataFrame(
                    self.scaler.transform(X_val),
                    columns=X_val.columns,
                    index=X_val.index
                )
            else:
                X_val_scaled = None
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        # ตรวจสอบและแปลง object columns ก่อน train
        X_train_clean = X_train_scaled.copy()
        X_val_clean = X_val_scaled.copy() if X_val_scaled is not None else None

        # แปลง object columns ที่เหลือเป็น numeric
        object_cols = X_train_clean.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            logger.warning(f"Found object columns in training data: {object_cols}")
            for col in object_cols:
                X_train_clean[col] = pd.Categorical(X_train_clean[col]).codes
                if X_val_clean is not None:
                    X_val_clean[col] = pd.Categorical(X_val_clean[col]).codes

        # ตรวจสอบว่าไม่มี object columns เหลือ
        remaining_objects = X_train_clean.select_dtypes(include=['object']).columns.tolist()
        if remaining_objects:
            raise ValueError(f"Still have object columns: {remaining_objects}")

        # Prepare datasets
        train_data = lgb.Dataset(
            X_train_clean, 
            label=y_train,
            weight=sample_weight,
            categorical_feature=categorical_features or []
        )
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val_clean is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val_clean, 
                label=y_val,
                categorical_feature=categorical_features or []
            )
            valid_sets.append(val_data)
            valid_names.append('valid')

        # Callbacks
        callbacks = []
        if early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))
        
        if verbose_eval > 0:
            callbacks.append(lgb.log_evaluation(verbose_eval))
        
        # Train model
        try:
            # Force CPU mode by removing potential GPU parameters
            if 'device_type' in self.params:
                self.params.pop('device_type')
            
            self.params['device'] = 'cpu'
            self.params['gpu_platform_id'] = -1
            self.params['gpu_device_id'] = -1
            
            print("⚡ FORCE CPU MODE ACTIVATED: params['device'] set to 'cpu'")
            
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks
            )
            
            self.is_trained = True
            self.best_iteration = self.model.best_iteration
            self.feature_importance_ = self.model.feature_importance()
            
            # Prepare results
            results = {
                'best_iteration': self.best_iteration,
                'n_features': len(self.feature_names_),
                'feature_names': self.feature_names_
            }
            
            # FIXED: Add validation score with proper error handling
            if hasattr(self.model, 'best_score') and 'valid' in self.model.best_score:
                valid_metrics = self.model.best_score['valid']
                
                # Try to find the metric that was actually used
                metric_to_find = self.params.get('metric', 'l1')
                
                if metric_to_find in valid_metrics:
                    results['best_score'] = valid_metrics[metric_to_find]
                elif 'l1' in valid_metrics:  # MAE = l1 in LightGBM
                    results['best_score'] = valid_metrics['l1']
                elif 'l2' in valid_metrics:  # MSE = l2 in LightGBM
                    results['best_score'] = valid_metrics['l2']
                elif 'rmse' in valid_metrics:
                    results['best_score'] = valid_metrics['rmse']
                else:
                    # Use the first available metric
                    results['best_score'] = list(valid_metrics.values())[0]
                    logger.warning(f"Using first available metric: {list(valid_metrics.keys())[0]}")
            else:
                results['best_score'] = None
                logger.warning("No validation score available")
            
            # Add feature selection results
            if self.feature_selection_results:
                results['feature_selection'] = self.feature_selection_results
            
            logger.info(f"LightGBM training completed. Best iteration: {self.best_iteration}")
            
            return results
            
        except Exception as e:
            logger.error(f"LightGBM training failed: {str(e)}")
            raise
    
    def predict(self, 
                X: pd.DataFrame,
                num_iteration: Optional[int] = None) -> np.ndarray:
        """
        Generate predictions
        
        Args:
            X: Input features
            num_iteration: Number of iterations to use for prediction
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Select features if feature selection was used
        if self.auto_feature_selection and self.feature_names_:
            X = X[self.feature_names_]
        
        # Scale features if scaler was used
        if self.scaler:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        # แปลง object columns ถ้ามี (เหมือนใน train)
        X_predict = X_scaled.copy()
        object_cols = X_predict.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            for col in object_cols:
                X_predict[col] = pd.Categorical(X_predict[col]).codes

        # Make predictions
        predictions = self.model.predict(
            X_predict, 
            num_iteration=num_iteration or self.best_iteration
        )
        
        return predictions
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance
        
        Args:
            importance_type: Type of importance ('gain', 'split')
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_importance(self, max_num_features: int = 20, figsize: Tuple[int, int] = (10, 6)):
        """Plot feature importance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting importance")
        
        try:
            import matplotlib.pyplot as plt
            
            importance_df = self.get_feature_importance().head(max_num_features)
            
            plt.figure(figsize=figsize)
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Importance')
            plt.title('Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            return plt.gcf()
            
        except ImportError:
            logger.error("Matplotlib not available for plotting")
            return None
    
    def cross_validate(self, 
                      X: pd.DataFrame, 
                      y: pd.Series,
                      cv_folds: int = 5,
                      scoring: str = 'mae') -> Dict[str, Any]:
        """
        Perform time series cross-validation
        
        Args:
            X: Features
            y: Target
            cv_folds: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {cv_folds}-fold time series cross-validation...")
        
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Training fold {fold + 1}/{cv_folds}")
            
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create temporary model
            temp_model = LightGBMModel(
                params=self.params.copy(),
                feature_engineering=False  # Features already engineered
            )
            
            # Train on fold
            temp_model.train(
                X_fold_train, y_fold_train,
                X_fold_val, y_fold_val,
                verbose_eval=0
            )
            
            # Predict on validation set
            fold_predictions = temp_model.predict(X_fold_val)
            
            # Calculate score
            if scoring == 'mae':
                score = mean_absolute_error(y_fold_val, fold_predictions)
            elif scoring == 'mse':
                score = mean_squared_error(y_fold_val, fold_predictions)
            else:
                raise ValueError(f"Unsupported scoring method: {scoring}")
            
            cv_scores.append(score)
        
        return {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'scoring': scoring
        }
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return {
            'params': self.params,
            'feature_engineering': self.feature_engineering,
            'auto_feature_selection': self.auto_feature_selection,
            'scaler_type': self.scaler_type,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names_
        }
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        # Save LightGBM model
        self.model.save_model(f"{filepath}.txt")
        
        # Save additional components
        import pickle
        model_data = {
            'params': self.get_params(),
            'scaler': self.scaler,
            'feature_selection_results': self.feature_selection_results,
            'best_iteration': self.best_iteration
        }
        
        with open(f"{filepath}_components.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"LightGBM model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        # Load LightGBM model
        self.model = lgb.Booster(model_file=f"{filepath}.txt")
        
        # Load additional components
        import pickle
        with open(f"{filepath}_components.pkl", 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore components
        params = model_data['params']
        self.params = params['params']
        self.feature_engineering = params['feature_engineering']
        self.auto_feature_selection = params['auto_feature_selection']
        self.scaler_type = params['scaler_type']
        self.is_trained = params['is_trained']
        self.feature_names_ = params['feature_names']
        
        self.scaler = model_data['scaler']
        self.feature_selection_results = model_data['feature_selection_results']
        self.best_iteration = model_data['best_iteration']
        
        logger.info(f"LightGBM model loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Create sample data with features
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    trend = np.linspace(50, 100, n_samples)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
    noise = np.random.normal(0, 5, n_samples)
    
    df = pd.DataFrame({
        'date': dates,
        'demand': trend + seasonal + noise,
        'promotional_event': np.random.binomial(1, 0.05, n_samples),
        'holiday_event': np.random.binomial(1, 0.02, n_samples)
    })
    
    # Initialize model
    model = LightGBMModel(
        feature_engineering=True,
        auto_feature_selection=True
    )
    
    # Create features
    feature_df = model.create_features(df, 'demand', 'date')
    
    # Prepare data
    feature_cols = [col for col in feature_df.columns if col not in ['date', 'demand']]
    X = feature_df[feature_cols].dropna()
    y = feature_df.loc[X.index, 'demand']
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train model
    results = model.train(X_train, y_train, X_test, y_test)
    print("Training Results:", results)
    
    # Make predictions
    predictions = model.predict(X_test)
    print("Predictions shape:", predictions.shape)
    
    # Feature importance
    importance = model.get_feature_importance()
    print("Top 10 features:")
    print(importance.head(10))