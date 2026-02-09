# data/scripts/02_preprocess.py
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import data_config, path_config
from loguru import logger

class DataPreprocessor:
    def __init__(self):
        self.raw_data_dir = Path(path_config.RAW_DATA_DIR)
        self.processed_data_dir = Path(path_config.PROCESSED_DATA_DIR)
        
    def load_raw_data(self, symbol: str) -> pd.DataFrame:
        """بارگذاری داده‌های خام"""
        filename = f"{symbol.replace('/', '_')}_{data_config.TIMEFRAMES['1h']}.csv"
        filepath = self.raw_data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath, index_col='Datetime', parse_dates=True)
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """پاکسازی داده‌ها"""
        if df.empty:
            return df
        
        # حذف ردیف‌های با مقادیر NaN
        df_clean = df.dropna()
        
        # حذف duplicate
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        
        # حذف outliers در قیمت
        for col in ['open', 'high', 'low', 'close']:
            q1 = df_clean[col].quantile(0.01)
            q3 = df_clean[col].quantile(0.99)
            df_clean = df_clean[(df_clean[col] >= q1) & (df_clean[col] <= q3)]
        
        # اطمینان از ترتیب زمانی
        df_clean = df_clean.sort_index()
        
        return df_clean
    
    def resample_data(self, df: pd.DataFrame, timeframe: str = '1h') -> pd.DataFrame:
        """ریسیمپل کردن داده به تایم‌فریم مشخص"""
        if df.empty:
            return df
        
        # ریسیمپل کردن
        resample_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        df_resampled = df.resample(timeframe).agg(resample_rules)
        
        # حذف NaNهای ایجاد شده
        df_resampled = df_resampled.dropna()
        
        return df_resampled
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه بازده‌ها"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # بازده ساده
        df['returns'] = df['close'].pct_change()
        
        # بازده لگاریتمی
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # بازده‌های با تاخیر
        for lag in [1, 2, 3, 5, 10]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'log_returns_lag_{lag}'] = df['log_returns'].shift(lag)
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, symbol: str):
        """ذخیره داده‌های پردازش شده"""
        if df.empty:
            return
        
        filename = f"{symbol.replace('/', '_')}_processed.csv"
        filepath = self.processed_data_dir / filename
        
        df.to_csv(filepath)
        logger.info(f"Saved processed data to {filepath}")
    
    def process_symbol(self, symbol: str) -> pd.DataFrame:
        """پردازش داده‌های یک نماد"""
        logger.info(f"Processing {symbol}...")
        
        # 1. بارگذاری داده‌های خام
        df_raw = self.load_raw_data(symbol)
        
        if df_raw.empty:
            logger.warning(f"No data for {symbol}")
            return pd.DataFrame()
        
        # 2. پاکسازی
        df_clean = self.clean_data(df_raw)
        
        # 3. ریسیمپل
        df_resampled = self.resample_data(df_clean, '1h')
        
        # 4. محاسبه بازده‌ها
        df_with_returns = self.calculate_returns(df_resampled)
        
        # 5. ذخیره
        self.save_processed_data(df_with_returns, symbol)
        
        logger.info(f"Processed {len(df_with_returns)} rows for {symbol}")
        return df_with_returns
    
    def run(self):
        """اجرای فرآیند پیش‌پردازش"""
        logger.info("Starting data preprocessing...")
        
        for symbol in data_config.CURRENCY_PAIRS:
            self.process_symbol(symbol)
        
        logger.info("Data preprocessing completed!")

def main():
    preprocessor = DataPreprocessor()
    preprocessor.run()

if __name__ == "__main__":
    main()