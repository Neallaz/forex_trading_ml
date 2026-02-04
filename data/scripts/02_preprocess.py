
"""
Forex Data Preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')
import os
import sys
# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from config.settings import settings

class ForexDataPreprocessor:
    """Forex Data Preprocessor Class"""
    
    def __init__(self):
        self.raw_dir = Path(settings.RAW_DATA_DIR)
        self.processed_dir = Path(settings.PROCESSED_DATA_DIR)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_raw_data(self, symbol):
        """Load raw data"""
        file_pattern = f"{symbol}_*.csv"
        files = list(self.raw_dir.glob(file_pattern))
        
        if not files:
            print(f"No file found for {symbol}")
            return None
        
        # Use the first available file
        file_path = files[0]
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Ensure chronological order
        df.sort_index(inplace=True)
        
        return df
    
    def clean_data(self, df):
        """Clean data"""
        # Remove rows with NaN values
        initial_len = len(df)
        df = df.dropna()
        
        # Remove outliers (unusual prices)
        for col in ['open', 'high', 'low', 'close']:
            q1 = df[col].quantile(0.01)
            q3 = df[col].quantile(0.99)
            df = df[(df[col] >= q1) & (df[col] <= q3)]
        
        # Handle data gaps
        df = self._handle_data_gaps(df)
        
        print(f"Data cleaned from {initial_len} to {len(df)} records")
        return df
    
    def _handle_data_gaps(self, df):
        """Handle time gaps in data"""
        # Create complete time range
        full_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq='1H'
        )
        
        # Reindex to have all hours
        df = df.reindex(full_range)
        
        # Forward fill for missing data
        df = df.ffill()
        
        # Backward fill for beginning of data
        df = df.bfill()
        
        return df
    
    def calculate_returns(self, df):
        """Calculate returns"""
        # Simple return
        df['returns'] = df['close'].pct_change()
        
        # Logarithmic return
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility
        df['volatility_20'] = df['log_returns'].rolling(window=20).std()
        
        return df
    
    def add_time_features(self, df):
        """Add time-based features"""
        # Time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['week_of_year'] = df.index.isocalendar().week
        
        # Trading sessions
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)
        df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(int)
        df['asia_session'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        return df
    
    def create_target_variable(self, df, horizon=5):
        """Create target variable for classification"""
        # Direction of movement in specified time horizon
        future_return = df['close'].shift(-horizon) / df['close'] - 1
        
        # Classification: 1 for upward movement, 0 for downward movement
        df['target'] = (future_return > 0).astype(int)
        
        # Regression target: percentage change
        df['target_return'] = future_return
        
        return df
    
    def process_pair(self, symbol):
        """Complete preprocessing of a currency pair"""
        print(f"Processing {symbol}...")
        
        # Load data
        df = self.load_raw_data(symbol)
        if df is None:
            return None
        
        # Clean
        df = self.clean_data(df)
        
        # Calculate returns
        df = self.calculate_returns(df)
        
        # Add time features
        df = self.add_time_features(df)
        
        # Create target variable
        df = self.create_target_variable(df, settings.PREDICTION_HORIZON)
        
        # Save processed data
        output_path = self.processed_dir / f"{symbol}_processed.csv"
        df.to_csv(output_path)
        
        print(f"Data for {symbol} saved to {output_path}")
        return df
    
    def process_all_pairs(self):
        """Preprocess all currency pairs"""
        processed_data = {}
        
        for pair in settings.FOREX_PAIRS[:3]:
            df = self.process_pair(pair)
            if df is not None:
                processed_data[pair] = df
        
        return processed_data

if __name__ == "__main__":
    preprocessor = ForexDataPreprocessor()
    data = preprocessor.process_all_pairs()
    print(f"{len(data)} currency pairs processed")
