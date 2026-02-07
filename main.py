# main.py
import argparse
import sys
import warnings
from pathlib import Path

# اضافه کردن مسیر پروژه به sys.path
sys.path.append(str(Path(__file__).parent))

from config.settings import path_config
from loguru import logger

# تنظیمات لاگ‌گیری
logger.add(
    path_config.LOGS_DIR + "/forex_trading_{time}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO"
)

def run_data_pipeline():
    """اجرای پایپلاین داده"""
    logger.info("Starting data pipeline...")
    
    from data.scripts import download_data, preprocess, feature_engineering
    
    # 1. دانلود داده‌ها
    download_data.main()
    
    # 2. پیش‌پردازش
    preprocess.main()
    
    # 3. مهندسی ویژگی
    feature_engineering.main()
    
    logger.info("Data pipeline completed!")

def train_models():
    """آموزش مدل‌ها"""
    logger.info("Starting model training...")
    
    from models.ml.train_ml import train_all_ml_models
    from models.dl.train_dl import train_all_dl_models
    from models.ensemble.ensemble_trainer import create_ensemble_model
    
    # 1. آموزش مدل‌های ML
    ml_results = train_all_ml_models()
    
    # 2. آموزش مدل‌های DL
    dl_results = train_all_dl_models()
    
    # 3. ساخت مدل Ensemble
    ensemble_results = create_ensemble_model()
    
    logger.info(f"Model training completed!")
    return {
        'ml': ml_results,
        'dl': dl_results,
        'ensemble': ensemble_results
    }

def run_backtesting():
    """اجرای بکتست"""
    logger.info("Starting backtesting...")
    
    from trading.backtesting.backtester import Backtester
    
    backtester = Backtester()
    results = backtester.run()
    
    logger.info(f"Backtesting completed!")
    return results

def run_live_monitoring():
    """مانیتورینگ زنده"""
    logger.info("Starting live monitoring...")
    
    from trading.live.monitor import LiveMonitor
    
    monitor = LiveMonitor()
    monitor.start()
    
    return monitor

def main():
    """تابع اصلی"""
    parser = argparse.ArgumentParser(description='Forex ML Trading System')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['data', 'train', 'backtest', 'live', 'all'],
                       help='Mode of operation')
    parser.add_argument('--pair', type=str, default='EURUSD',
                       help='Currency pair to process')
    parser.add_argument('--timeframe', type=str, default='1h',
                       help='Timeframe for data')
    
    args = parser.parse_args()
    
    logger.info(f"Starting Forex ML Trading System in {args.mode} mode")
    
    try:
        if args.mode == 'data':
            run_data_pipeline()
        
        elif args.mode == 'train':
            train_models()
        
        elif args.mode == 'backtest':
            run_backtesting()
        
        elif args.mode == 'live':
            run_live_monitoring()
        
        elif args.mode == 'all':
            run_data_pipeline()
            train_models()
            run_backtesting()
        
        logger.info("Process completed successfully!")
        
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()