# data/scripts/01_download_data_simple.py

"""
Forex Data Download Script - Fixed Version
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ForexDataDownloader:
    """Forex Data Downloader Class"""
    
    def __init__(self):
        # Create folders
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Main forex pairs
        self.forex_pairs = [
            "EURUSD",
            "GBPUSD", 
            "USDJPY",
            "USDCHF",
            "AUDUSD",
            "USDCAD",
            "NZDUSD"
        ]
        
        # Default dates
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=90)  # Last 3 months
    
    def download_from_yfinance(self, symbol, start_date=None, end_date=None, interval="1h"):
        """
        Download data from Yahoo Finance
        """
        try:
            if start_date is None:
                start_date = self.start_date
            if end_date is None:
                end_date = self.end_date
            
            # Convert forex symbol to Yahoo format
            yf_symbol = f"{symbol}=X"
            
            print(f"📥 Downloading {symbol} from {start_date.date()} to {end_date.date()}...")
            
            # Download data
            df = yf.download(
                yf_symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                timeout=30
            )
            
            if df.empty:
                print(f"⚠️ No data found for {symbol}")
                return None
            
            # DEBUG: Show what columns we get
            print(f"   DEBUG: Original columns: {df.columns.tolist()}")
            print(f"   DEBUG: Original shape: {df.shape}")
            
            # Handle column names - yfinance returns different column names based on data
            # Forex data usually has: Open, High, Low, Close
            # We need to standardize column names
            
            # Map column names to standard format
            column_mapping = {
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            }
            
            # Create new dataframe with standardized columns
            standard_df = pd.DataFrame()
            
            if 'Open' in df.columns:
                standard_df['open'] = df['Open']
            if 'High' in df.columns:
                standard_df['high'] = df['High']
            if 'Low' in df.columns:
                standard_df['low'] = df['Low']
            if 'Close' in df.columns:
                standard_df['close'] = df['Close']
            if 'Adj Close' in df.columns:
                standard_df['adj_close'] = df['Adj Close']
            if 'Volume' in df.columns:
                standard_df['volume'] = df['Volume']
            else:
                standard_df['volume'] = 0  # Add volume column if missing
            
            # If we got MultiIndex columns (common with yfinance)
            if isinstance(df.columns, pd.MultiIndex):
                print(f"   DEBUG: MultiIndex columns detected")
                # Flatten the MultiIndex
                df.columns = ['_'.join(col).strip() for col in df.columns.values]
                print(f"   DEBUG: Flattened columns: {df.columns.tolist()}")
                
                # Try to extract standard columns
                for col in df.columns:
                    if 'Open' in col:
                        standard_df['open'] = df[col]
                    elif 'High' in col:
                        standard_df['high'] = df[col]
                    elif 'Low' in col:
                        standard_df['low'] = df[col]
                    elif 'Close' in col:
                        standard_df['close'] = df[col]
                    elif 'Adj Close' in col:
                        standard_df['adj_close'] = df[col]
                    elif 'Volume' in col:
                        standard_df['volume'] = df[col]
            
            # If standard_df is still empty, use the first 5 columns
            if standard_df.empty and len(df.columns) >= 4:
                print(f"   DEBUG: Using first {min(5, len(df.columns))} columns")
                for i, col_name in enumerate(['open', 'high', 'low', 'close', 'volume'][:len(df.columns)]):
                    if i < len(df.columns):
                        standard_df[col_name] = df.iloc[:, i]
            
            # Save file
            file_path = self.data_dir / f"{symbol}_{interval}.csv"
            standard_df.to_csv(file_path)
            print(f"✅ {symbol}: {len(standard_df)} records saved to {file_path}")
            
            # Show sample info
            print(f"   First date: {standard_df.index[0]}")
            print(f"   Last date: {standard_df.index[-1]}")
            print(f"   Last price: {standard_df['close'].iloc[-1]:.5f}")
            print(f"   Columns: {standard_df.columns.tolist()}")
            
            return standard_df
            
        except Exception as e:
            print(f"❌ Error downloading {symbol}: {str(e)}")
            return None
    
    def download_all_pairs(self):
        """Download all currency pairs"""
        print("=" * 50)
        print("🚀 Starting Forex Data Download")
        print("=" * 50)
        
        success_count = 0
        
        for pair in self.forex_pairs:
            df = self.download_from_yfinance(pair)
            
            if df is not None:
                success_count += 1
            
            time.sleep(2)  # Pause to avoid rate limiting
        
        print("=" * 50)
        print(f"📊 Final Result: {success_count} of {len(self.forex_pairs)} pairs downloaded")
        
        if success_count > 0:
            print("✅ Download successful!")
        else:
            print("⚠️ There was a problem downloading data")
        print("=" * 50)
    
    def test_connection(self):
        """Test connection and check packages"""
        print("🔍 Testing connection and packages...")
        
        try:
            import yfinance
            print("✅ yfinance is installed")
        except ImportError:
            print("❌ yfinance is not installed! Run: pip install yfinance")
            return False
        
        try:
            import pandas
            print(f"✅ pandas is installed (version: {pd.__version__})")
        except ImportError:
            print("❌ pandas is not installed! Run: pip install pandas")
            return False
        
        return True

if __name__ == "__main__":
    downloader = ForexDataDownloader()
    
    # Test connection
    if downloader.test_connection():
        # Start download
        downloader.download_all_pairs()
        
        # Show list of downloaded files
        print("\n📁 Downloaded files:")
        print("-" * 30)
        for file in Path("data/raw").glob("*.csv"):
            file_size = file.stat().st_size
            print(f"  📄 {file.name} ({file_size:,} bytes)")
    else:
        print("❌ Required packages are not installed. Please install them first.")

        
# """
# اسکریپت دانلود داده‌های فارکس از منابع رایگان
# """

# import pandas as pd
# import numpy as np
# import yfinance as yf
# import ccxt
# from datetime import datetime, timedelta
# import time
# from pathlib import Path
# import warnings
# warnings.filterwarnings('ignore')

# from config.settings import settings

# class ForexDataDownloader:
#     """کلاس دانلود داده‌های فارکس"""
    
#     def __init__(self):
#         self.data_dir = Path(settings.RAW_DATA_DIR)
#         self.data_dir.mkdir(parents=True, exist_ok=True)
        
#     def download_from_yfinance(self, symbol, start_date, end_date, interval="1h"):
#         """
#         دانلود داده از Yahoo Finance
#         Yahoo Finance از نمادهای فارکس مثل EURUSD=X پشتیبانی می‌کند
#         """
#         try:
#             # تبدیل نماد فارکس به فرمت Yahoo
#             if "USD" in symbol and symbol != "USDJPY":
#                 yf_symbol = f"{symbol}=X"
#             else:
#                 yf_symbol = f"{symbol}=X"
            
#             print(f"دانلود {symbol} از {start_date} تا {end_date}...")
            
#             # دانلود داده
#             df = yf.download(
#                 yf_symbol,
#                 start=start_date,
#                 end=end_date,
#                 interval=interval,
#                 progress=False
#             )
            
#             if df.empty:
#                 print(f"داده‌ای برای {symbol} یافت نشد")
#                 return None
            
#             # تغییر نام ستون‌ها
#             df.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
#             df = df[['open', 'high', 'low', 'close', 'volume']]
            
#             # ذخیره فایل
#             file_path = self.data_dir / f"{symbol}_{interval}.csv"
#             df.to_csv(file_path)
#             print(f"داده‌های {symbol} در {file_path} ذخیره شد")
            
#             return df
            
#         except Exception as e:
#             print(f"خطا در دانلود {symbol}: {e}")
#             return None
    
#     def download_from_alphavantage(self, symbol, interval="60min"):
#         """
#         دانلود داده از Alpha Vantage (رایگان با محدودیت)
#         """
#         import requests
        
#         # کلید API (می‌توانید از demo استفاده کنید یا کلید خود را ثبت کنید)
#         api_key = settings.ALPHA_VANTAGE_API_KEY
        
#         # برای فارکس
#         url = f"https://www.alphavantage.co/query"
#         params = {
#             "function": "FX_INTRADAY",
#             "from_symbol": symbol[:3],
#             "to_symbol": symbol[3:],
#             "interval": interval,
#             "outputsize": "full",
#             "apikey": api_key
#         }
        
#         try:
#             response = requests.get(url, params=params)
#             data = response.json()
            
#             if "Time Series FX (" + interval + ")" in data:
#                 df = pd.DataFrame.from_dict(
#                     data["Time Series FX (" + interval + ")"], 
#                     orient='index'
#                 )
#                 df = df.astype(float)
#                 df.index = pd.to_datetime(df.index)
#                 df.columns = ['open', 'high', 'low', 'close']
                
#                 # اضافه کردن volume با مقدار 0
#                 df['volume'] = 0
                
#                 # ذخیره فایل
#                 file_path = self.data_dir / f"{symbol}_{interval}.csv"
#                 df.to_csv(file_path)
#                 print(f"داده‌های {symbol} از Alpha Vantage ذخیره شد")
#                 return df
#             else:
#                 print(f"خطا در دریافت داده: {data.get('Note', 'Unknown error')}")
#                 return None
                
#         except Exception as e:
#             print(f"خطا در دانلود از Alpha Vantage: {e}")
#             return None
    
#     def download_from_ccxt(self, symbol, exchange_name="binance"):
#         """
#         دانلود داده از صرافی‌های crypto (برای جفت‌ارزهای مرتبط)
#         """
#         try:
#             exchange = getattr(ccxt, exchange_name)()
            
#             # تبدیل نماد فارکس به نماد crypto
#             crypto_symbol = symbol.replace("USD", "USDT")
            
#             # دانلود داده‌های اخیر
#             ohlcv = exchange.fetch_ohlcv(
#                 crypto_symbol, 
#                 timeframe='1h', 
#                 limit=1000
#             )
            
#             df = pd.DataFrame(
#                 ohlcv, 
#                 columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
#             )
#             df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
#             df.set_index('timestamp', inplace=True)
            
#             # ذخیره فایل
#             file_path = self.data_dir / f"{symbol}_crypto_1h.csv"
#             df.to_csv(file_path)
            
#             return df
            
#         except Exception as e:
#             print(f"خطا در دانلود از CCXT: {e}")
#             return None
    
#     def download_all_pairs(self):
#         """دانلود تمام جفت‌ارزها"""
#         print("شروع دانلود داده‌های فارکس...")
        
#         for pair in settings.FOREX_PAIRS[:3]:  # فقط 3 جفت اول برای شروع
#             # ابتدا از Yahoo Finance تلاش می‌کنیم
#             df = self.download_from_yfinance(
#                 pair,
#                 settings.START_DATE,
#                 settings.END_DATE,
#                 interval="1h"
#             )
            
#             # اگر Yahoo کار نکرد، از Alpha Vantage استفاده می‌کنیم
#             if df is None or len(df) < 100:
#                 print(f"تغییر به Alpha Vantage برای {pair}...")
#                 df = self.download_from_alphavantage(pair)
            
#             if df is not None:
#                 print(f"{pair}: {len(df)} رکورد دانلود شد")
#             else:
#                 print(f"{pair}: دانلود ناموفق")
            
#             time.sleep(2)  # جلوگیری از rate limiting
        
#         print("دانلود داده‌ها کامل شد!")

# if __name__ == "__main__":
#     downloader = ForexDataDownloader()
#     downloader.download_all_pairs()