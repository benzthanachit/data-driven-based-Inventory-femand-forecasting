# src/utils/m5_data_processor.py
"""
M5 Dataset Preprocessing for Hybrid Ensemble Forecasting
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

class M5DataProcessor:
    """Process M5 Forecasting Competition dataset"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.sales_df = None
        self.calendar_df = None
        self.prices_df = None
        
    def load_m5_data(self):
        """Load all M5 CSV files"""
        logger.info("Loading M5 dataset...")
        
        # Load main datasets
        self.sales_df = pd.read_csv(self.data_dir / "sales_train_validation.csv")
        self.calendar_df = pd.read_csv(self.data_dir / "calendar.csv")
        self.prices_df = pd.read_csv(self.data_dir / "sell_prices.csv")
        
        logger.info(f"Loaded sales: {self.sales_df.shape}")
        logger.info(f"Loaded calendar: {self.calendar_df.shape}")
        logger.info(f"Loaded prices: {self.prices_df.shape}")
        
    def select_item_store(self, item_id: str = "HOBBIES_1_001", store_id: str = "CA_1"):
        """Select specific item and store combination"""
        logger.info(f"Processing item: {item_id}, store: {store_id}")
        
        # Filter sales data
        item_data = self.sales_df[
            (self.sales_df['item_id'] == item_id) & 
            (self.sales_df['store_id'] == store_id)
        ]
        
        if item_data.empty:
            raise ValueError(f"No data found for item {item_id} in store {store_id}")
        
        return item_data.iloc[0]
    
    def create_time_series(self, item_id: str = "HOBBIES_1_001", 
                          store_id: str = "CA_1", 
                          max_days: int = 1913) -> pd.DataFrame:
        """Create time series DataFrame with features"""
        
        # Get item data
        item_row = self.select_item_store(item_id, store_id)
        
        # Extract sales columns (d_1 to d_1913)
        day_cols = [f'd_{i}' for i in range(1, max_days + 1) if f'd_{i}' in item_row.index]
        sales_values = item_row[day_cols].values
        
        # Create date index from calendar
        calendar_subset = self.calendar_df.head(len(day_cols)).copy()
        dates = pd.to_datetime(calendar_subset['date'])
        
        # Create main DataFrame
        df = pd.DataFrame({
            'date': dates,
            'demand': sales_values
        })
        
        # Add calendar features
        df = self._add_calendar_features(df, calendar_subset)
        
        # Add price features
        df = self._add_price_features(df, item_id, store_id)
        
        # Remove rows with missing values in demand
        df = df[df['demand'].notna()].reset_index(drop=True)
        
        logger.info(f"Created time series with {len(df)} days")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Demand range: {df['demand'].min():.2f} to {df['demand'].max():.2f}")
        
        return df
    
    def _add_calendar_features(self, df: pd.DataFrame, calendar_subset: pd.DataFrame) -> pd.DataFrame:
        """Add calendar-based features"""
        
        # Event features
        df['event_name_1'] = calendar_subset['event_name_1'].fillna('').reset_index(drop=True)
        df['event_type_1'] = calendar_subset['event_type_1'].fillna('').reset_index(drop=True)
        
        # Binary event features
        df['has_event'] = (df['event_name_1'] != '').astype(int)
        df['promotional_event'] = df['has_event']  # For compatibility
        df['holiday_event'] = (df['event_type_1'].str.contains('Holiday|Cultural', na=False)).astype(int)
        
        # SNAP benefits (if available)
        if 'snap_CA' in calendar_subset.columns:
            df['snap_CA'] = calendar_subset['snap_CA'].fillna(0).reset_index(drop=True)
        if 'snap_TX' in calendar_subset.columns:
            df['snap_TX'] = calendar_subset['snap_TX'].fillna(0).reset_index(drop=True)
        if 'snap_WI' in calendar_subset.columns:
            df['snap_WI'] = calendar_subset['snap_WI'].fillna(0).reset_index(drop=True)
        
        # Time features
        df['weekday'] = calendar_subset['weekday'].reset_index(drop=True)
        df['wm_yr_wk'] = calendar_subset['wm_yr_wk'].reset_index(drop=True)
        df['month'] = calendar_subset['month'].reset_index(drop=True)
        df['year'] = calendar_subset['year'].reset_index(drop=True)
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame, item_id: str, store_id: str) -> pd.DataFrame:
        """Add price features"""
        
        # Filter price data for this item and store
        item_prices = self.prices_df[
            (self.prices_df['item_id'] == item_id) & 
            (self.prices_df['store_id'] == store_id)
        ].copy()
        
        if item_prices.empty:
            logger.warning(f"No price data found for {item_id} in {store_id}")
            df['sell_price'] = 1.0  # Default price
            return df
        
        # Merge with main DataFrame on wm_yr_wk
        df = df.merge(
            item_prices[['wm_yr_wk', 'sell_price']], 
            on='wm_yr_wk', 
            how='left'
        )
        
        # Forward fill missing prices
        df['sell_price'] = df['sell_price'].fillna(method='ffill')
        df['sell_price'] = df['sell_price'].fillna(1.0)  # Fill remaining with default
        
        return df

def process_m5_to_standard_format(item_id: str = "HOBBIES_1_001", 
                                 store_id: str = "CA_1",
                                 output_dir: str = "data/processed") -> str:
    """Process M5 data to standard format for Step 1"""
    
    # Initialize processor
    processor = M5DataProcessor()
    processor.load_m5_data()
    
    # Create time series
    df = processor.create_time_series(item_id, store_id)
    
    # Save processed data
    output_path = Path(output_dir) / f"m5_{item_id}_{store_id}_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    logger.info(f"M5 data processed and saved to {output_path}")
    
    return str(output_path)