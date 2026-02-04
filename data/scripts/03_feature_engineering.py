"""
Financial Feature Engineering
"""

import os 
import sys
import pandas as pd
import numpy as np
import talib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from config.settings import settings

class FeatureEngineer:
    """Financial Feature Engineering Class"""
    
    def __init__(self):
        self.processed_dir = Path(settings.PROCESSED_DATA_DIR)
        self.scaler = StandardScaler()
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators using TA-Lib"""
        
        # Convert to numpy arrays for TA-Lib
        open_prices = df['open'].values.astype(float)
        high_prices = df['high'].values.astype(float)
        low_prices = df['low'].values.astype(float)
        close_prices = df['close'].values.astype(float)
        volume = df['volume'].values.astype(float) if 'volume' in df.columns else None
        
        # Trend indicators
        df['SMA_20'] = talib.SMA(close_prices, timeperiod=20)
        df['SMA_50'] = talib.SMA(close_prices, timeperiod=50)
        df['EMA_10'] = talib.EMA(close_prices, timeperiod=10)
        df['EMA_20'] = talib.EMA(close_prices, timeperiod=20)
        df['EMA_50'] = talib.EMA(close_prices, timeperiod=50)
        
        # Momentum indicators
        df['RSI'] = talib.RSI(close_prices, timeperiod=14)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
            close_prices, fastperiod=12, slowperiod=26, signalperiod=9
        )
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(
            high_prices, low_prices, close_prices,
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        df['WILLIAMS_R'] = talib.WILLR(
            high_prices, low_prices, close_prices, timeperiod=14
        )
        
        # Volatility indicators
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
            close_prices, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        df['ATR'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        
        # Volume indicators
        if volume is not None:
            df['OBV'] = talib.OBV(close_prices, volume)
            df['ADOSC'] = talib.ADOSC(high_prices, low_prices, close_prices, volume)
        
        # Derived price features
        df['high_low_pct'] = (df['high'] - df['low']) / df['close'] * 100
        df['close_open_pct'] = (df['close'] - df['open']) / df['open'] * 100
        
        # Derived return features
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_10'] = df['close'].pct_change(10)
        df['returns_20'] = df['close'].pct_change(20)
        
        # Various volatility measures
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['log_returns'].rolling(window).std()
        
        return df
    
    def create_lag_features(self, df, columns, lags=[1, 2, 3, 5, 10]):
        """Create lag features"""
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df, columns, windows=[5, 10, 20]):
        """Create rolling features"""
        for col in columns:
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
                df[f'{col}_rolling_min_{window}'] = df[col].rolling(window).min()
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(window).max()
        
        return df
    
    def calculate_pair_correlation(self, df1, df2, column='close'):
        """Calculate correlation between two currency pairs"""
        corr_window = 20
        corr = df1[column].rolling(corr_window).corr(df2[column])
        return corr
    
    def prepare_features_for_ml(self, df, scale_features=True):
        """Final feature preparation for ML models"""
        
        # Remove rows with NaN values
        initial_len = len(df)
        df = df.dropna()
        
        # Separate features and target
        target_cols = ['target', 'target_return']
        feature_cols = [col for col in df.columns if col not in target_cols]
        
        X = df[feature_cols].copy()
        y = df[['target', 'target_return']].copy()
        
        # Scaling features
        if scale_features:
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # Save scaler for later use
            joblib.dump(self.scaler, self.processed_dir / 'scaler.pkl')
        
        return X, y
    
    def engineer_features_for_pair(self, symbol):
        """Feature engineering for a currency pair"""
        print(f"Engineering features for {symbol}...")
        
        # Load processed data
        input_path = self.processed_dir / f"{symbol}_processed.csv"
        if not input_path.exists():
            print(f"File {input_path} does not exist")
            return None
        
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)
        
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Create lag features for price and important indicators
        price_cols = ['close', 'high', 'low', 'open']
        indicator_cols = ['RSI', 'MACD', 'ATR', 'SMA_20', 'EMA_20']
        
        df = self.create_lag_features(df, price_cols + indicator_cols)
        
        # Create rolling features
        df = self.create_rolling_features(df, ['returns', 'log_returns', 'volatility_20'])
        
        # Final preparation
        X, y = self.prepare_features_for_ml(df)
        
        # Save features
        features_df = pd.concat([X, y], axis=1)
        output_path = self.processed_dir / f"{symbol}_features.csv"
        features_df.to_csv(output_path)
        
        print(f"Features for {symbol} saved to {output_path}")
        return features_df
    
    def engineer_all_pairs(self):
        """Feature engineering for all currency pairs"""
        features_data = {}
        
        for pair in settings.FOREX_PAIRS[:3]:
            df = self.engineer_features_for_pair(pair)
            if df is not None:
                features_data[pair] = df
        
        return features_data

if __name__ == "__main__":
    engineer = FeatureEngineer()
    features = engineer.engineer_all_pairs()
    print(f"Feature engineering completed for {len(features)} pairs")