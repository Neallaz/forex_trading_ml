# data/scripts/03_feature_engineering.py
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import feature_config, path_config
from utils.indicators import TechnicalIndicators
from loguru import logger

class FeatureEngineer:
    def __init__(self):
        self.processed_data_dir = Path(path_config.PROCESSED_DATA_DIR)
        self.indicators_calculator = TechnicalIndicators()
        
    def load_processed_data(self, symbol: str) -> pd.DataFrame:
        """بارگذاری داده‌های پردازش شده"""
        filename = f"{symbol.replace('/', '_')}_processed.csv"
        filepath = self.processed_data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """اضافه کردن اندیکاتورهای تکنیکال"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Moving Averages
        for period in feature_config.INDICATORS['sma']:
            df[f'sma_{period}'] = self.indicators_calculator.sma(df['close'], period)
        
        for period in feature_config.INDICATORS['ema']:
            df[f'ema_{period}'] = self.indicators_calculator.ema(df['close'], period)
        
        # RSI
        for period in feature_config.INDICATORS['rsi']:
            df[f'rsi_{period}'] = self.indicators_calculator.rsi(df['close'], period)
        
        # MACD
        for fast, slow, signal in feature_config.INDICATORS['macd']:
            macd_line, signal_line, histogram = self.indicators_calculator.macd(
                df['close'], fast, slow, signal
            )
            df[f'macd_{fast}_{slow}'] = macd_line
            df[f'macd_signal_{signal}'] = signal_line
            df[f'macd_hist_{signal}'] = histogram
        
        # Bollinger Bands
        for period, std in feature_config.INDICATORS['bollinger']:
            upper, middle, lower = self.indicators_calculator.bollinger_bands(
                df['close'], period, std
            )
            df[f'bb_upper_{period}'] = upper
            df[f'bb_middle_{period}'] = middle
            df[f'bb_lower_{period}'] = lower
            df[f'bb_width_{period}'] = (upper - lower) / middle
        
        # ATR
        for period in feature_config.INDICATORS['atr']:
            df[f'atr_{period}'] = self.indicators_calculator.atr(
                df['high'], df['low'], df['close'], period
            )
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """اضافه کردن ویژگی‌های زمانی"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # استخراج ویژگی‌های زمانی
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # سشن‌های معاملاتی
        df['trading_session'] = df['hour'].apply(self._get_trading_session)
        
        # تبدیل به dummy variables
        session_dummies = pd.get_dummies(df['trading_session'], prefix='session')
        df = pd.concat([df, session_dummies], axis=1)
        
        return df
    
    def _get_trading_session(self, hour: int) -> str:
        """تعیین سشن معاملاتی بر اساس ساعت"""
        if 0 <= hour < 8:
            return 'asian'
        elif 8 <= hour < 16:
            return 'european'
        elif 16 <= hour < 24:
            return 'american'
        else:
            return 'other'
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """اضافه کردن ویژگی‌های قیمتی"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # نسبت‌های قیمت
        df['high_low_ratio'] = df['high'] / df['low'] - 1
        df['close_open_ratio'] = df['close'] / df['open'] - 1
        
        # نوسانات
        for window in [10, 20, 30]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
        
        # مومنتوم
        for window in [5, 10, 20]:
            df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
        
        # شکاف قیمت
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        return df
    
    def create_labels(self, df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """ایجاد labels برای classification و regression"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # 1. Classification: پیش‌بینی جهت حرکت (1: بالا، 0: پایین)
        future_return = df['close'].shift(-horizon) / df['close'] - 1
        df['label_direction'] = (future_return > 0).astype(int)
        
        # 2. Regression: پیش‌بینی مقدار بازده
        df['label_return'] = future_return
        
        # 3. Classification سه کلاسه (بالا، ثابت، پایین)
        threshold = 0.001  # 0.1%
        df['label_3class'] = pd.cut(future_return, 
                                   bins=[-np.inf, -threshold, threshold, np.inf],
                                   labels=[0, 1, 2])
        
        return df
    
    def save_features(self, df: pd.DataFrame, symbol: str):
        """ذخیره ویژگی‌های مهندسی شده"""
        if df.empty:
            return
        
        filename = f"{symbol.replace('/', '_')}_features.csv"
        filepath = self.processed_data_dir / filename
        
        df.to_csv(filepath)
        logger.info(f"Saved features to {filepath}")
    
    def engineer_features(self, symbol: str) -> pd.DataFrame:
        """مهندسی ویژگی برای یک نماد"""
        logger.info(f"Engineering features for {symbol}...")
        
        # 1. بارگذاری داده‌های پردازش شده
        df = self.load_processed_data(symbol)
        
        if df.empty:
            logger.warning(f"No data for {symbol}")
            return pd.DataFrame()
        
        # 2. اندیکاتورهای تکنیکال
        df = self.add_technical_indicators(df)
        
        # 3. ویژگی‌های زمانی
        df = self.add_time_features(df)
        
        # 4. ویژگی‌های قیمتی
        df = self.add_price_features(df)
        
        # 5. ایجاد labels
        df = self.create_labels(df, horizon=5)
        
        # 6. حذف NaNها
        df = df.dropna()
        
        # 7. ذخیره
        self.save_features(df, symbol)
        
        logger.info(f"Engineered {len(df.columns)} features for {symbol}")
        return df
    
    def run(self):
        """اجرای فرآیند مهندسی ویژگی"""
        logger.info("Starting feature engineering...")
        
        for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:  # لیست ثابت برای شروع
            self.engineer_features(symbol)
        
        logger.info("Feature engineering completed!")

def main():
    engineer = FeatureEngineer()
    engineer.run()

if __name__ == "__main__":
    main()