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
    """Forex Data Downloader Class"""
    
    def __init__(self):
        # Create folders
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Main forex pairs
        self.currency_pairs = data_config.CURRENCY_PAIRS
        # Default dates
        # self.end_date = datetime.now()
        # self.start_date = self.end_date - timedelta(days=90)  # Last 3 months
        self.start_date = data_config.START_DATE
        self.end_date = data_config.END_DATE
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
        
        for pair in  self.currency_pairs:
            df = self.download_from_yfinance(pair)
            
            if df is not None:
                success_count += 1
            
            time.sleep(2)  # Pause to avoid rate limiting
        
        print("=" * 50)
        print(f"📊 Final Result: {success_count} of {len(self.currency_pairs)} pairs downloaded")
        
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
    downloader = DataDownloader()
    
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


# import yfinance as yf
# import pandas as pd
# from datetime import datetime, timedelta
# import time
# from pathlib import Path
# import sys
# sys.path.append(str(Path(__file__).parent.parent.parent))

# from config.settings import data_config, path_config
# from loguru import logger

# class DataDownloader:
#     def __init__(self):
#         self.currency_pairs = data_config.CURRENCY_PAIRS
#         self.timeframe = '1h'  # Yahoo Finance only supports 1h for intraday
        
#     def download_yahoo_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
#         """Download data from Yahoo Finance"""
#         try:
#             yahoo_symbol = symbol.replace('/', '') + '=X'
#             logger.info(f"Downloading {yahoo_symbol} from {start_date} to {end_date}")
            
#             df = yf.download(
#                 yahoo_symbol,
#                 start=start_date,
#                 end=end_date,
#                 interval='1h',
#                 progress=False
#             )
            
#             if df.empty:
#                 logger.warning(f"No data found for {symbol}")
#                 return pd.DataFrame()
            
#             # Rename columns to standard
#             df = df.rename(columns={
#                 'Open': 'open',
#                 'High': 'high',
#                 'Low': 'low',
#                 'Close': 'close',
#                 'Volume': 'volume'
#             })
            
#             df['symbol'] = symbol
#             df.index.name = 'timestamp'
            
#             logger.info(f"Downloaded {len(df)} rows for {symbol}")
#             return df
            
#         except Exception as e:
#             logger.error(f"Error downloading {symbol}: {e}")
#             return pd.DataFrame()
    
#     def save_data(self, df: pd.DataFrame, symbol: str):
#         """Save clean data to CSV for preprocessing"""
#         if df.empty:
#             logger.warning(f"No data to save for {symbol}")
#             return
        
#         filename = f"{symbol.replace('/', '_')}_{data_config.TIMEFRAMES['1h']}.csv"
#         filepath = Path(path_config.RAW_DATA_DIR) / filename
        
#         # Reset index to make timestamp a column
#         df_reset = df.reset_index()
        
#         # Keep only necessary columns in correct order
#         columns_to_save = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
        
#         # Filter to only columns that exist
#         available_columns = [col for col in columns_to_save if col in df_reset.columns]
#         df_reset = df_reset[available_columns]
        
#         # Ensure symbol column exists and has correct value
#         if 'symbol' not in df_reset.columns:
#             df_reset['symbol'] = symbol
        
#         # Save with ONLY the header (no extra rows)
#         df_reset.to_csv(filepath, index=False)
#         logger.info(f"Saved {len(df_reset)} rows to {filepath}")

#     def run(self):
#         """Run the full download process"""
#         logger.info("Starting data download process...")
#         start_date = data_config.START_DATE
#         end_date = data_config.END_DATE
        
#         for symbol in self.currency_pairs:
#             df = self.download_yahoo_data(symbol, start_date, end_date)
#             self.save_data(df, symbol)
#             time.sleep(1)  # avoid rate limits
        
#         logger.info("Data download completed!")

# def main():
#     downloader = DataDownloader()
#     downloader.run()

# if __name__ == "__main__":
#     main()
