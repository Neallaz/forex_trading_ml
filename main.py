"""
نقطه ورود اصلی پروژه - Forex ML Trading System
"""

import sys
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# اضافه کردن مسیر پروژه
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.settings import settings

def run_pipeline(symbol=None, steps=None):
    """اجرای کامل pipeline پروژه"""
    
    print("\n" + "="*60)
    print("🏦 Forex ML Trading System - Complete Pipeline")
    print("="*60)
    
    steps = steps or ["all"]
    
    if "all" in steps or "download" in steps:
        print("\n📥 Step 1: Downloading Data...")
        from data.scripts.01_download_data import ForexDataDownloader
        downloader = ForexDataDownloader()
        downloader.download_all_pairs()
    
    if "all" in steps or "preprocess" in steps:
        print("\n🔧 Step 2: Preprocessing Data...")
        from data.scripts.02_preprocess import ForexDataPreprocessor
        preprocessor = ForexDataPreprocessor()
        preprocessor.process_all_pairs()
    
    if "all" in steps or "features" in steps:
        print("\n⚙️ Step 3: Feature Engineering...")
        from data.scripts.03_feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()
        engineer.engineer_all_pairs()
    
    if "all" in steps or "ml" in steps:
        print("\n🤖 Step 4: Training ML Models...")
        from models.ml.train_ml import MLModelTrainer
        ml_trainer = MLModelTrainer()
        ml_trainer.train_for_all_pairs()
    
    if "all" in steps or "dl" in steps:
        print("\n🧠 Step 5: Training DL Models...")
        from models.dl.train_dl import DLModelTrainer
        dl_trainer = DLModelTrainer()
        dl_trainer.train_for_all_pairs()
    
    if "all" in steps or "ensemble" in steps:
        print("\n🎯 Step 6: Training Ensemble Model...")
        from models.ensemble.ensemble_trainer import EnsembleTrainer
        ensemble_trainer = EnsembleTrainer()
        ensemble_trainer.train_all_ensembles()
    
    if "all" in steps or "backtest" in steps:
        print("\n📊 Step 7: Running Backtests...")
        from trading.backtesting.backtester import Backtester
        backtester = Backtester()
        backtester.run_comparative_backtest()
    
    if "all" in steps or "dashboard" in steps:
        print("\n📈 Step 8: Launching Dashboard...")
        print("Dashboard will be available at: http://localhost:8501")
        import subprocess
        subprocess.run(["streamlit", "run", "dashboard/app.py"])
    
    print("\n" + "="*60)
    print("✅ Pipeline completed successfully!")
    print("="*60)

def main():
    """تابع اصلی"""
    parser = argparse.ArgumentParser(description='Forex ML Trading System')
    parser.add_argument('--steps', nargs='+', 
                       choices=['download', 'preprocess', 'features', 
                               'ml', 'dl', 'ensemble', 'backtest', 
                               'dashboard', 'all'],
                       default=['all'],
                       help='Steps to run in the pipeline')
    parser.add_argument('--symbol', type=str,
                       help='Specific currency pair to process')
    
    args = parser.parse_args()
    
    run_pipeline(symbol=args.symbol, steps=args.steps)

if __name__ == "__main__":
    main()