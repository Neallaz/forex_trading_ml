# config/settings.py
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# بارگذاری متغیرهای محیطی
load_dotenv()

# ==================== تنظیمات داده ====================
class DataConfig:
    # جفت ارزها
    CURRENCY_PAIRS = os.getenv('CURRENCY_PAIRS', 'EURUSD,GBPUSD,USDJPY').split(',')
    
    # تایم‌فریم‌ها
    TIMEFRAMES = {
        '1h': '1h',
        '4h': '4h',
        '1d': '1d'
    }
    
    # تاریخچه داده
    START_DATE = datetime(2020, 1, 1)
    END_DATE = datetime(2023, 12, 31)
    
    # فیلدهای قیمت
    OHLC_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    
    # سورس داده (اولویت‌بندی شده)
    DATA_SOURCES = ['yfinance', 'alpha_vantage', 'oanda']

# ==================== تنظیمات مدل ====================
class ModelConfig:
    # پنجره نگاه به عقب (lookback window)
    LOOKBACK_WINDOW = int(os.getenv('LOOKBACK_WINDOW', 60))
    
    # افق پیش‌بینی
    FORECAST_HORIZON = int(os.getenv('FORECAST_HORIZON', 5))
    
    # تقسیم داده
    TRAIN_TEST_SPLIT = float(os.getenv('TRAIN_TEST_SPLIT', 0.8))
    VALIDATION_SPLIT = 0.15
    
    # پارامترهای مدل‌های ML
    RANDOM_FOREST_PARAMS = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'random_state': 42
    }
    
    XGBOOST_PARAMS = {
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'random_state': 42
    }
    
    # پارامترهای LSTM
    LSTM_PARAMS = {
        'units': [128, 64, 32],
        'dropout': 0.2,
        'recurrent_dropout': 0.2,
        'batch_size': 32,
        'epochs': 100,
        'patience': 10
    }

# ==================== تنظیمات معاملاتی ====================
class TradingConfig:
    # سرمایه اولیه
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 10000))
    
    # مدیریت ریسک
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', 0.1))
    STOP_LOSS_PERCENT = float(os.getenv('STOP_LOSS_PERCENT', 0.02))
    TAKE_PROFIT_PERCENT = float(os.getenv('TAKE_PROFIT_PERCENT', 0.04))
    
    # کارمزدها
    COMMISSION = 0.0005  # 0.05%
    SLIPPAGE = 0.0001    # 1 pip
    
    # ساعت معاملات (UTC)
    TRADING_HOURS = {
        'start': 0,   # 00:00 UTC
        'end': 21     # 21:00 UTC
    }

# ==================== تنظیمات ویژگی‌ها ====================
class FeatureConfig:
    # اندیکاتورهای تکنیکال
    INDICATORS = {
        'sma': [10, 20, 50, 100],
        'ema': [10, 20, 50],
        'rsi': [14],
        'macd': [(12, 26, 9)],
        'bollinger': [(20, 2)],
        'atr': [14],
        'stochastic': [(14, 3, 3)],
        'adx': [14],
        'obv': [],
        'volume_sma': [20]
    }
    
    # ویژگی‌های زمانی
    TIME_FEATURES = [
        'hour', 'day_of_week', 'day_of_month', 
        'month', 'quarter', 'is_weekend',
        'trading_session'  # Asian, European, American
    ]
    
    # ویژگی‌های قیمتی
    PRICE_FEATURES = [
        'returns_1', 'returns_5', 'returns_10',
        'log_returns_1', 'log_returns_5',
        'high_low_ratio', 'close_open_ratio',
        'volatility_10', 'volatility_20'
    ]

# ==================== تنظیمات مسیرها ====================
class PathConfig:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # مسیرهای داده
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    
    # مسیرهای مدل
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    ML_MODELS_DIR = os.path.join(MODELS_DIR, 'ml')
    DL_MODELS_DIR = os.path.join(MODELS_DIR, 'dl')
    ENSEMBLE_MODELS_DIR = os.path.join(MODELS_DIR, 'ensemble')
    
    # مسیرهای نتایج
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    
    @classmethod
    def create_directories(cls):
        """ایجاد تمام پوشه‌های مورد نیاز"""
        directories = [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.ML_MODELS_DIR,
            cls.DL_MODELS_DIR,
            cls.ENSEMBLE_MODELS_DIR,
            cls.RESULTS_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# ==================== نمونه‌سازی ====================
data_config = DataConfig()
model_config = ModelConfig()
trading_config = TradingConfig()
feature_config = FeatureConfig()
path_config = PathConfig()

# ایجاد پوشه‌ها
path_config.create_directories()