"""
اسکریپت دانلود داده‌های فارکس از منابع رایگان
"""

import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
from datetime import datetime, timedelta
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings

class ForexDataDownloader:
    """کلاس دانلود داده‌های فارکس"""
    
    def __init__(self):
        self.data_dir = Path(settings.RAW_DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_from_yfinance(self, symbol, start_date, end_date, interval="1h"):
        """
        دانلود داده از Yahoo Finance
        Yahoo Finance از نمادهای فارکس مثل EURUSD=X پشتیبانی می‌کند
        """
        try:
            # تبدیل نماد فارکس به فرمت Yahoo
            if "USD" in symbol and symbol != "USDJPY":
                yf_symbol = f"{symbol}=X"
            else:
                yf_symbol = f"{symbol}=X"
            
            print(f"دانلود {symbol} از {start_date} تا {end_date}...")
            
            # دانلود داده
            df = yf.download(
                yf_symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False
            )
            
            if df.empty:
                print(f"داده‌ای برای {symbol} یافت نشد")
                return None
            
            # تغییر نام ستون‌ها
            df.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # ذخیره فایل
            file_path = self.data_dir / f"{symbol}_{interval}.csv"
            df.to_csv(file_path)
            print(f"داده‌های {symbol} در {file_path} ذخیره شد")
            
            return df
            
        except Exception as e:
            print(f"خطا در دانلود {symbol}: {e}")
            return None
    
    def download_from_alphavantage(self, symbol, interval="60min"):
        """
        دانلود داده از Alpha Vantage (رایگان با محدودیت)
        """
        import requests
        
        # کلید API (می‌توانید از demo استفاده کنید یا کلید خود را ثبت کنید)
        api_key = settings.ALPHA_VANTAGE_API_KEY
        
        # برای فارکس
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "FX_INTRADAY",
            "from_symbol": symbol[:3],
            "to_symbol": symbol[3:],
            "interval": interval,
            "outputsize": "full",
            "apikey": api_key
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if "Time Series FX (" + interval + ")" in data:
                df = pd.DataFrame.from_dict(
                    data["Time Series FX (" + interval + ")"], 
                    orient='index'
                )
                df = df.astype(float)
                df.index = pd.to_datetime(df.index)
                df.columns = ['open', 'high', 'low', 'close']
                
                # اضافه کردن volume با مقدار 0
                df['volume'] = 0
                
                # ذخیره فایل
                file_path = self.data_dir / f"{symbol}_{interval}.csv"
                df.to_csv(file_path)
                print(f"داده‌های {symbol} از Alpha Vantage ذخیره شد")
                return df
            else:
                print(f"خطا در دریافت داده: {data.get('Note', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"خطا در دانلود از Alpha Vantage: {e}")
            return None
    
    def download_from_ccxt(self, symbol, exchange_name="binance"):
        """
        دانلود داده از صرافی‌های crypto (برای جفت‌ارزهای مرتبط)
        """
        try:
            exchange = getattr(ccxt, exchange_name)()
            
            # تبدیل نماد فارکس به نماد crypto
            crypto_symbol = symbol.replace("USD", "USDT")
            
            # دانلود داده‌های اخیر
            ohlcv = exchange.fetch_ohlcv(
                crypto_symbol, 
                timeframe='1h', 
                limit=1000
            )
            
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # ذخیره فایل
            file_path = self.data_dir / f"{symbol}_crypto_1h.csv"
            df.to_csv(file_path)
            
            return df
            
        except Exception as e:
            print(f"خطا در دانلود از CCXT: {e}")
            return None
    
    def download_all_pairs(self):
        """دانلود تمام جفت‌ارزها"""
        print("شروع دانلود داده‌های فارکس...")
        
        for pair in settings.FOREX_PAIRS[:3]:  # فقط 3 جفت اول برای شروع
            # ابتدا از Yahoo Finance تلاش می‌کنیم
            df = self.download_from_yfinance(
                pair,
                settings.START_DATE,
                settings.END_DATE,
                interval="1h"
            )
            
            # اگر Yahoo کار نکرد، از Alpha Vantage استفاده می‌کنیم
            if df is None or len(df) < 100:
                print(f"تغییر به Alpha Vantage برای {pair}...")
                df = self.download_from_alphavantage(pair)
            
            if df is not None:
                print(f"{pair}: {len(df)} رکورد دانلود شد")
            else:
                print(f"{pair}: دانلود ناموفق")
            
            time.sleep(2)  # جلوگیری از rate limiting
        
        print("دانلود داده‌ها کامل شد!")

if __name__ == "__main__":
    downloader = ForexDataDownloader()
    downloader.download_all_pairs()