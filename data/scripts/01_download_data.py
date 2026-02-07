# data/scripts/01_download_data.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import data_config, path_config
from loguru import logger

class DataDownloader:
    def __init__(self):
        self.currency_pairs = data_config.CURRENCY_PAIRS
        self.timeframe = '1h'  # Yahoo Finance از 1h پشتیبانی می‌کند
        
    def download_yahoo_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """دانلود داده از Yahoo Finance"""
        try:
            # تبدیل نماد فارکس به فرمت Yahoo
            yahoo_symbol = symbol.replace('/', '') + '=X'
            
            logger.info(f"Downloading {yahoo_symbol} from {start_date} to {end_date}")
            
            # دانلود داده
            df = yf.download(
                yahoo_symbol,
                start=start_date,
                end=end_date,
                interval='1h',
                progress=False
            )
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # تغییر نام ستون‌ها
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # اضافه کردن اطلاعات نماد
            df['symbol'] = symbol
            df.index.name = 'timestamp'
            
            logger.info(f"Downloaded {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading {symbol}: {e}")
            return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, symbol: str):
        """ذخیره داده در فایل CSV"""
        if df.empty:
            return
        
        filename = f"{symbol.replace('/', '_')}_{data_config.TIMEFRAMES['1h']}.csv"
        filepath = Path(path_config.RAW_DATA_DIR) / filename
        
        df.to_csv(filepath)
        logger.info(f"Saved data to {filepath}")
    
    def run(self):
        """اجرای فرآیند دانلود"""
        logger.info("Starting data download process...")
        
        start_date = data_config.START_DATE
        end_date = data_config.END_DATE
        
        for symbol in self.currency_pairs:
            df = self.download_yahoo_data(symbol, start_date, end_date)
            self.save_data(df, symbol)
            time.sleep(1)  # مکث برای جلوگیری از rate limit
        
        logger.info("Data download completed!")

def main():
    downloader = DataDownloader()
    downloader.run()

if __name__ == "__main__":
    main()