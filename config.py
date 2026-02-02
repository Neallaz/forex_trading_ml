"""
تنظیمات و کانفیگ‌های پروژه - Forex ML Trading System
"""

import os
from datetime import datetime
from dotenv import load_dotenv

# بارگذاری متغیرهای محیطی
load_dotenv()

class Config:
    """کلاس تنظیمات اصلی پروژه"""
    
    # مسیرها
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # تنظیمات API
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
    OANDA_ACCOUNT_ID = os.getenv('OANDA_ACCOUNT_ID', '')
    OANDA_ACCESS_TOKEN = os.getenv('OANDA_ACCESS_TOKEN', '')
    
    # تنظیمات بازار فارکس
    FOREX_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
    TIMEFRAME = '1h'  # 1h, 4h, 1d
    START_DATE = '2020-01-01'
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    # تنظیمات مدل‌سازی
    SEQUENCE_LENGTH = 60  # طول سری زمانی برای مدل‌ها
    PREDICTION_HORIZON = 5  # پیش‌بینی 5 دوره آینده
    TRAIN_TEST_SPLIT = 0.8  # 80% آموزش، 20% تست
    
    # پارامترهای معاملاتی
    INITIAL_CAPITAL = 10000.0
    COMMISSION = 0.0001  # 0.01% برای فارکس
    POSITION_SIZE = 0.02  # 2% از سرمایه در هر معامله
    STOP_LOSS = 0.02  # 2% حد ضرر
    TAKE_PROFIT = 0.04  # 4% حد سود
    
    # اندیکاتورهای تکنیکال
    INDICATORS = {
        'SMA': [20, 50],
        'EMA': [10, 20],
        'RSI': 14,
        'MACD': (12, 26, 9),
        'BBANDS': 20,
        'ATR': 14
    }
    
    # تنظیمات مدل‌های ML
    ML_MODELS = {
        'random_forest': {'n_estimators': 100, 'max_depth': 10},
        'xgboost': {'n_estimators': 100, 'max_depth': 6},
        'logistic_regression': {'C': 1.0, 'max_iter': 1000}
    }
    
    # تنظیمات مدل‌های DL
    DL_MODELS = {
        'lstm': {'units': [64, 32], 'dropout': 0.3, 'epochs': 50},
        'cnn_lstm': {'filters': [64, 32], 'kernel_size': 3, 'lstm_units': 50},
        'attention': {'attention_units': 64, 'lstm_units': 128}
    }
    
    # معیارهای عملکرد
    PERFORMANCE_METRICS = [
        'sharpe_ratio',
        'sortino_ratio',
        'max_drawdown',
        'win_rate',
        'profit_factor',
        'total_return'
    ]
    
    # تنظیمات Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'logs/trading_system.log'
    
    # تنظیمات ذخیره‌سازی
    SAVE_MODELS = True
    SAVE_PREDICTIONS = True
    SAVE_BACKTEST_RESULTS = True
    
    @classmethod
    def get_data_path(cls):
        """دریافت مسیر داده‌ها"""
        return os.path.join(cls.BASE_DIR, 'data')
    
    @classmethod
    def get_raw_data_path(cls):
        """دریافت مسیر داده‌های خام"""
        return os.path.join(cls.get_data_path(), 'raw')
    
    @classmethod
    def get_processed_data_path(cls):
        """دریافت مسیر داده‌های پردازش شده"""
        return os.path.join(cls.get_data_path(), 'processed')
    
    @classmethod
    def get_models_path(cls):
        """دریافت مسیر مدل‌ها"""
        return os.path.join(cls.BASE_DIR, 'models')
    
    @classmethod
    def get_trading_path(cls):
        """دریافت مسیر معاملات"""
        return os.path.join(cls.BASE_DIR, 'trading')
    
    @classmethod
    def create_directories(cls):
        """ایجاد دایرکتوری‌های مورد نیاز"""
        directories = [
            cls.get_raw_data_path(),
            cls.get_processed_data_path(),
            os.path.join(cls.get_models_path(), 'ml'),
            os.path.join(cls.get_models_path(), 'dl'),
            os.path.join(cls.get_models_path(), 'ensemble'),
            os.path.join(cls.get_trading_path(), 'backtesting'),
            os.path.join(cls.get_trading_path(), 'risk_management'),
            os.path.join(cls.get_trading_path(), 'strategies'),
            os.path.join(cls.BASE_DIR, 'visualization'),
            os.path.join(cls.BASE_DIR, 'utils'),
            os.path.join(cls.BASE_DIR, 'logs'),
            os.path.join(cls.BASE_DIR, 'results')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print("✅ تمام دایرکتوری‌های لازم ایجاد شدند")

# ایجاد instance از کانفیگ
config = Config()

# ایجاد دایرکتوری‌ها هنگام import
config.create_directories()