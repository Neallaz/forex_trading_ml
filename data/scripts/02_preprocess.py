"""
پیش‌پردازش داده‌های فارکس
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings

class ForexDataPreprocessor:
    """کلاس پیش‌پردازش داده‌های فارکس"""
    
    def __init__(self):
        self.raw_dir = Path(settings.RAW_DATA_DIR)
        self.processed_dir = Path(settings.PROCESSED_DATA_DIR)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_raw_data(self, symbol):
        """بارگذاری داده‌های خام"""
        file_pattern = f"{symbol}_*.csv"
        files = list(self.raw_dir.glob(file_pattern))
        
        if not files:
            print(f"فایلی برای {symbol} یافت نشد")
            return None
        
        # استفاده از اولین فایل موجود
        file_path = files[0]
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # اطمینان از ترتیب زمانی
        df.sort_index(inplace=True)
        
        return df
    
    def clean_data(self, df):
        """پاکسازی داده‌ها"""
        # حذف سطرهای با مقادیر NaN
        initial_len = len(df)
        df = df.dropna()
        
        # حذف outliers (قیمت‌های غیرمعمول)
        for col in ['open', 'high', 'low', 'close']:
            q1 = df[col].quantile(0.01)
            q3 = df[col].quantile(0.99)
            df = df[(df[col] >= q1) & (df[col] <= q3)]
        
        # بررسی gaps در داده‌ها
        df = self._handle_data_gaps(df)
        
        print(f"داده‌ها از {initial_len} به {len(df)} رکورد پاکسازی شد")
        return df
    
    def _handle_data_gaps(self, df):
        """مدیریت gaps زمانی در داده‌ها"""
        # ایجاد بازه زمانی کامل
        full_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq='1H'
        )
        
        # reindex کردن برای داشتن تمام ساعات
        df = df.reindex(full_range)
        
        # forward fill برای داده‌های گمشده
        df = df.ffill()
        
        # backward fill برای ابتدای داده‌ها
        df = df.bfill()
        
        return df
    
    def calculate_returns(self, df):
        """محاسبه بازده‌ها"""
        # بازده ساده
        df['returns'] = df['close'].pct_change()
        
        # بازده لگاریتمی
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # بازده نوسانی
        df['volatility_20'] = df['log_returns'].rolling(window=20).std()
        
        return df
    
    def add_time_features(self, df):
        """اضافه کردن ویژگی‌های زمانی"""
        # ویژگی‌های زمانی
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['week_of_year'] = df.index.isocalendar().week
        
        # سشن‌های معاملاتی
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)
        df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(int)
        df['asia_session'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        return df
    
    def create_target_variable(self, df, horizon=5):
        """ایجاد متغیر هدف برای classification"""
        # جهت حرکت در افق زمانی مشخص
        future_return = df['close'].shift(-horizon) / df['close'] - 1
        
        # classification: 1 برای بالا رفتن، 0 برای پایین آمدن
        df['target'] = (future_return > 0).astype(int)
        
        # regression target: درصد تغییرات
        df['target_return'] = future_return
        
        return df
    
    def process_pair(self, symbol):
        """پیش‌پردازش کامل یک جفت ارز"""
        print(f"پیش‌پردازش {symbol}...")
        
        # بارگذاری داده‌ها
        df = self.load_raw_data(symbol)
        if df is None:
            return None
        
        # پاکسازی
        df = self.clean_data(df)
        
        # محاسبه بازده‌ها
        df = self.calculate_returns(df)
        
        # اضافه کردن ویژگی‌های زمانی
        df = self.add_time_features(df)
        
        # ایجاد متغیر هدف
        df = self.create_target_variable(df, settings.PREDICTION_HORIZON)
        
        # ذخیره داده‌های پردازش شده
        output_path = self.processed_dir / f"{symbol}_processed.csv"
        df.to_csv(output_path)
        
        print(f"داده‌های {symbol} در {output_path} ذخیره شد")
        return df
    
    def process_all_pairs(self):
        """پیش‌پردازش تمام جفت‌ارزها"""
        processed_data = {}
        
        for pair in settings.FOREX_PAIRS[:3]:
            df = self.process_pair(pair)
            if df is not None:
                processed_data[pair] = df
        
        return processed_data

if __name__ == "__main__":
    preprocessor = ForexDataPreprocessor()
    data = preprocessor.process_all_pairs()
    print(f"{len(data)} جفت ارز پردازش شد")