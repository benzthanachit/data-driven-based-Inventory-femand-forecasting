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
        
    def get_items_by_category(self, category: str, store_id: str = "CA_1", n_items: int = 20, strategy: str = 'top_volume') -> list:
        """
        Get list of item_ids based on category and strategy
        strategies: 'top_volume', 'random', 'intermittent'
        """
        # Filter for store and category
        mask = (self.sales_df['store_id'] == store_id) & (self.sales_df['cat_id'] == category)
        subset = self.sales_df[mask].copy()
        
        if subset.empty:
            logger.warning(f"No items found for category {category} in {store_id}")
            return []
            
        # Calculate total volume for ranking
        day_cols = [c for c in subset.columns if c.startswith('d_')]
        subset['total_volume'] = subset[day_cols].sum(axis=1)
        
        if strategy == 'top_volume':
            # Select top N items by volume
            selected = subset.nlargest(n_items, 'total_volume')
        elif strategy == 'random':
            # Select random N items
            selected = subset.sample(n=min(n_items, len(subset)), random_state=42)
        elif strategy == 'intermittent':
            # Select items with high zero counts (e.g. > 50% zeros) but still some volume
            # This is a simple heuristic
            zeros_count = (subset[day_cols] == 0).sum(axis=1)
            subset['zero_ratio'] = zeros_count / len(day_cols)
            # Filter for 30% to 70% zeros
            intermittent = subset[(subset['zero_ratio'] > 0.3) & (subset['zero_ratio'] < 0.7)]
            selected = intermittent.nlargest(n_items, 'total_volume') 

        item_ids = selected['item_id'].tolist()
        logger.info(f"Selected {len(item_ids)} items using strategy '{strategy}'")
        return item_ids

    def select_item_store(self, item_id: str, store_id: str):
        """Select specific item and store combination"""
        # logger.info(f"Processing item: {item_id}, store: {store_id}")
        
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
        df['sell_price'] = df['sell_price'].ffill()
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

def process_multiple_items(category: str = "HOBBIES",
                         store_id: str = "CA_1",
                         n_items: int = 20,
                         strategy: str = 'top_volume',
                         output_dir: str = "data/processed") -> list:
    """
    Process multiple items efficiently (load data once)
    """
    # Initialize processor
    processor = M5DataProcessor()
    processor.load_m5_data()
    
    # Get items
    item_ids = processor.get_items_by_category(category, store_id, n_items, strategy)
    
    processed_files = []
    
    for item_id in item_ids:
        try:
            # Create time series
            df = processor.create_time_series(item_id, store_id)
            
            # Save processed data
            output_path = Path(output_dir) / f"m5_{item_id}_{store_id}_data.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(output_path, index=False)
            processed_files.append(str(output_path))
            
            # logger.info(f"Processed {item_id}")
            
        except Exception as e:
            logger.error(f"Failed to process {item_id}: {str(e)}")
            
    logger.info(f"Successfully processed {len(processed_files)}/{len(item_ids)} items")
    return processed_files