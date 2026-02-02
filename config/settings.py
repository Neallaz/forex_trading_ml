"""
تنظیمات اصلی پروژه - Forex ML Trading System
"""

import os
from datetime import datetime
from dotenv import load_dotenv

# بارگذاری متغیرهای محیطی
load_dotenv()

class Settings:
    # مسیرها
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    # تنظیمات بازار فارکس
    FOREX_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
    TIMEFRAME = "1h"  # 1h, 4h, 1d
    START_DATE = "2020-01-01"
    END_DATE = datetime.now().strftime("%Y-%m-%d")
    
    # تنظیمات API
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
    OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
    OANDA_ACCESS_TOKEN = os.getenv("OANDA_ACCESS_TOKEN")
    
    # تنظیمات مدل‌سازی
    FEATURE_WINDOW = 60  # 60 کندل برای ساخت features
    PREDICTION_HORIZON = 5  # پیش‌بینی 5 کندل آینده
    TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test
    RANDOM_SEED = 42
    
    # اندیکاتورهای تکنیکال
    TECHNICAL_INDICATORS = {
        "trend": ["EMA_10", "EMA_20", "EMA_50", "SMA_20", "SMA_50"],
        "momentum": ["RSI", "MACD", "STOCH_K", "STOCH_D", "WILLIAMS_R"],
        "volatility": ["BB_upper", "BB_middle", "BB_lower", "ATR"],
        "volume": ["OBV", "CMF", "MFI"],
        "custom": ["log_return", "volatility_20", "high_low_pct"]
    }
    
    # تنظیمات بکتست
    INITIAL_CAPITAL = 10000.0
    COMMISSION = 0.0001  # 0.01% برای فارکس
    SLIPPAGE = 0.0001  # 1 pip
    POSITION_SIZE_PCT = 0.02  # 2% از سرمایه در هر معامله
    
    # تنظیمات ریسک
    STOP_LOSS_PCT = 0.02  # 2% حد ضرر
    TAKE_PROFIT_PCT = 0.04  # 4% حد سود
    MAX_DRAWDOWN_LIMIT = 0.20  # حداکثر drawdown مجاز
    
    # معیارهای عملکرد
    METRICS = ["sharpe_ratio", "sortino_ratio", "max_drawdown", 
               "win_rate", "profit_factor", "calmar_ratio"]

settings = Settings()