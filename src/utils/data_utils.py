# src/utils/data_utils.py
"""
Data utility functions for the hybrid inventory forecasting project
"""
import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def create_sample_data(n_samples: int = 1000, 
                      seasonal: bool = True,
                      trend_strength: float = 0.05,
                      seasonal_strength: float = 20.0,
                      noise_level: float = 5.0,
                      random_seed: int = 42) -> pd.DataFrame:
    """
    Create sample synthetic data for testing and development
    
    Args:
        n_samples: Number of samples to generate
        seasonal: Whether to include seasonal patterns
        trend_strength: Strength of trend component
        seasonal_strength: Strength of seasonal component
        noise_level: Standard deviation of noise
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic demand data
    """
    np.random.seed(random_seed)
    
    # Create date range
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Create base trend
    trend = 50 + trend_strength * np.arange(n_samples)
    
    # Add seasonality if requested
    if seasonal:
        # Annual seasonality
        annual_seasonal = seasonal_strength * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
        # Weekly seasonality
        weekly_seasonal = (seasonal_strength * 0.3) * np.sin(2 * np.pi * np.arange(n_samples) / 7)
        seasonal_component = annual_seasonal + weekly_seasonal
    else:
        seasonal_component = np.zeros(n_samples)
    
    # Add noise
    noise = np.random.normal(0, noise_level, n_samples)
    
    # Create events (promotional and holiday)
    promotional_event = np.random.binomial(1, 0.05, n_samples)
    holiday_event = np.random.binomial(1, 0.02, n_samples)
    
    # Combine components
    demand = trend + seasonal_component + noise
    demand += promotional_event * 15  # Promotion boost
    demand += holiday_event * 10      # Holiday boost
    demand = np.maximum(demand, 1)    # Ensure positive values
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'demand': demand,
        'promotional_event': promotional_event,
        'holiday_event': holiday_event
    })
    
    logger.info(f"Created synthetic dataset with {n_samples} samples")
    logger.info(f"Demand range: {demand.min():.2f} to {demand.max():.2f}")
    
    return df


def split_time_series_data(df: pd.DataFrame, 
                          target_col: str,
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.15,
                          test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train, validation, and test sets
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Validate ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Ratios must sum to 1.0")
    
    n_samples = len(df)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    # Split data chronologically
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size+val_size].copy()
    test_df = df.iloc[train_size+val_size:].copy()
    
    logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def check_data_quality(df: pd.DataFrame, target_col: str) -> dict:
    """
    Check data quality and return summary statistics
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        
    Returns:
        Dictionary with data quality metrics
    """
    quality_report = {}
    
    # Basic statistics
    quality_report['n_samples'] = len(df)
    quality_report['date_range'] = (df['date'].min(), df['date'].max()) if 'date' in df.columns else None
    
    # Missing values
    quality_report['missing_values'] = df.isnull().sum().to_dict()
    quality_report['missing_percentage'] = (df.isnull().sum() / len(df) * 100).to_dict()
    
    # Target variable statistics
    if target_col in df.columns:
        target_data = df[target_col]
        quality_report['target_stats'] = {
            'mean': target_data.mean(),
            'std': target_data.std(),
            'min': target_data.min(),
            'max': target_data.max(),
            'median': target_data.median(),
            'skewness': target_data.skew(),
            'kurtosis': target_data.kurtosis()
        }
        
        # Check for outliers (using IQR method)
        Q1 = target_data.quantile(0.25)
        Q3 = target_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = target_data[(target_data < lower_bound) | (target_data > upper_bound)]
        quality_report['outliers'] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(target_data) * 100,
            'values': outliers.tolist() if len(outliers) < 20 else 'Too many to list'
        }
    
    # Data types
    quality_report['data_types'] = df.dtypes.to_dict()
    
    return quality_report


def validate_model_inputs(X: pd.DataFrame, y: pd.Series) -> bool:
    """
    Validate inputs for model training
    
    Args:
        X: Feature DataFrame
        y: Target Series
        
    Returns:
        True if valid, raises exception otherwise
    """
    # Check for matching lengths
    if len(X) != len(y):
        raise ValueError(f"Feature and target lengths don't match: {len(X)} vs {len(y)}")
    
    # Check for sufficient data
    if len(X) < 50:
        raise ValueError(f"Insufficient data for training: {len(X)} samples (minimum 50 required)")
    
    # Check for missing values in target
    if y.isnull().sum() > 0:
        raise ValueError(f"Target variable contains {y.isnull().sum()} missing values")
    
    # Check for infinite values
    if np.isinf(y).sum() > 0:
        raise ValueError(f"Target variable contains {np.isinf(y).sum()} infinite values")
    
    # Check feature matrix
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric features found in feature matrix")
    
    logger.info("Model inputs validation passed")
    return True


def save_data(df: pd.DataFrame, filepath: str, format: str = 'csv') -> None:
    """
    Save DataFrame to file
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        format: File format ('csv', 'json', 'pickle')
    """
    if format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'json':
        df.to_json(filepath, orient='records', date_format='iso')
    elif format == 'pickle':
        df.to_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Data saved to {filepath} in {format} format")


def load_data(filepath: str, format: str = 'csv') -> pd.DataFrame:
    """
    Load DataFrame from file
    
    Args:
        filepath: Input file path
        format: File format ('csv', 'json', 'pickle')
        
    Returns:
        Loaded DataFrame
    """
    if format == 'csv':
        df = pd.read_csv(filepath)
    elif format == 'json':
        df = pd.read_json(filepath, orient='records')
    elif format == 'pickle':
        df = pd.read_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Convert date column if exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"Data loaded from {filepath}")
    return df


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    df = create_sample_data(1000, seasonal=True)
    print("Sample data created:")
    print(df.head())
    print(f"Shape: {df.shape}")
    
    # Check data quality
    quality = check_data_quality(df, 'demand')
    print("\nData quality report:")
    for key, value in quality.items():
        print(f"{key}: {value}")
    
    # Split data
    train_df, val_df, test_df = split_time_series_data(df, 'demand')
    print(f"\nData split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Save sample data
    save_data(df, 'data/synthetic/sample_data.csv')
    print("Sample data saved to data/synthetic/sample_data.csv")

def load_m5_data(item_id: str = "HOBBIES_1_001", 
                store_id: str = "CA_1",
                force_reprocess: bool = False) -> pd.DataFrame:
    """
    Load and process M5 data for specified item and store
    """
    from .m5_data_processor import process_m5_to_standard_format
    
    processed_file = f"data/processed/m5_{item_id}_{store_id}_data.csv"
    
    if not os.path.exists(processed_file) or force_reprocess:
        logger.info(f"Processing M5 data for {item_id} in {store_id}")
        process_m5_to_standard_format(item_id, store_id, "data/processed")
    
    # Load processed data
    df = pd.read_csv(processed_file)
    df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"Loaded M5 data: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
    
    return df


def prepare_m5_data_for_models(item_id: str = "HOBBIES_1_001", 
                              store_id: str = "CA_1",
                              train_ratio: float = 0.7,
                              val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare M5 data splits for training
    """
    
    # Load M5 data
    df = load_m5_data(item_id, store_id)
    
    # Split data chronologically
    n_samples = len(df)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size + val_size].copy()
    test_df = df.iloc[train_size + val_size:].copy()
    
    # Save splits
    train_df.to_csv('data/processed/train_data.csv', index=False)
    val_df.to_csv('data/processed/val_data.csv', index=False)  
    test_df.to_csv('data/processed/test_data.csv', index=False)
    
    logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df