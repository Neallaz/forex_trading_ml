"""
توابع کمکی و ابزارهای عمومی
"""

import os
import sys
import json
import pickle
import hashlib
import logging
import random
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# تنظیمات logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProjectHelper:
    """
    کلاس کمکی برای مدیریت پروژه
    """
    
    @staticmethod
    def setup_project(project_name: str = "forex-ml-trading") -> Path:
        """
        تنظیمات اولیه پروژه
        """
        # ایجاد دایرکتوری‌های اصلی
        directories = [
            'data/raw',
            'data/processed',
            'data/scripts',
            'models/ml',
            'models/dl',
            'models/ensemble',
            'trading/backtesting/results',
            'trading/risk_management',
            'trading/strategies',
            'utils',
            'config',
            'notebooks',
            'dashboard/static',
            'logs'
        ]
        
        base_path = Path.cwd() / project_name
        
        for directory in directories:
            dir_path = base_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        
        # ایجاد فایل‌های ضروری
        essential_files = {
            'config/settings.py': '',
            'config/__init__.py': '',
            'requirements.txt': '',
            '.env.example': 'ALPHA_VANTAGE_API_KEY=your_key_here\n',
            '.gitignore': __get_gitignore_content(),
            'README.md': f'# {project_name}\n\nProject documentation',
            'main.py': '#!/usr/bin/env python3\n\nprint("Hello Forex ML Trading!")'
        }
        
        for file_path, content in essential_files.items():
            file_full_path = base_path / file_path
            if not file_full_path.exists():
                file_full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_full_path, 'w') as f:
                    f.write(content)
                logger.info(f"Created file: {file_full_path}")
        
        logger.info(f"Project setup completed in: {base_path}")
        return base_path
    
    @staticmethod
    def load_config(config_file: str = 'config/settings.yaml') -> Dict:
        """
        بارگذاری تنظیمات از فایل YAML
        """
        config_path = Path(config_file)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_file}")
            return config
        else:
            logger.warning(f"Config file {config_file} not found, using defaults")
            return {}
    
    @staticmethod
    def save_config(config: Dict, config_file: str = 'config/settings.yaml'):
        """
        ذخیره تنظیمات در فایل YAML
        """
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Saved config to {config_file}")
    
    @staticmethod
    def setup_logging(log_file: str = 'logs/project.log', 
                     log_level: str = 'INFO'):
        """
        تنظیمات logging
        """
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # تنظیم log level
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            numeric_level = logging.INFO
        
        # تنظیم handlers
        handlers = [
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
        
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        
        logger.info(f"Logging setup completed. Log file: {log_file}")
    
    @staticmethod
    def get_project_root() -> Path:
        """
        پیدا کردن root پروژه
        """
        current_path = Path.cwd()
        
        # جستجو برای فایل‌های شناسه پروژه
        project_files = ['.git', 'requirements.txt', 'setup.py', 'pyproject.toml']
        
        while current_path != current_path.parent:
            for project_file in project_files:
                if (current_path / project_file).exists():
                    return current_path
            current_path = current_path.parent
        
        return Path.cwd()

class DataHelper:
    """
    کلاس کمکی برای کار با داده‌ها
    """
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, 
                          required_columns: List[str] = None) -> Tuple[bool, str]:
        """
        اعتبارسنجی DataFrame
        """
        if df.empty:
            return False, "DataFrame is empty"
        
        if df.isnull().all().any():
            return False, "Some columns are completely NaN"
        
        if len(df) < 10:
            return False, f"Too few rows: {len(df)}"
        
        # بررسی ستون‌های ضروری
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return False, f"Missing required columns: {missing_columns}"
        
        # بررسی duplicate index
        if df.index.duplicated().any():
            return False, "Duplicate indices found"
        
        # بررسی monotonic index (برای داده‌های زمانی)
        if hasattr(df.index, 'is_monotonic_increasing'):
            if not df.index.is_monotonic_increasing:
                return False, "Index is not monotonic increasing"
        
        return True, "DataFrame is valid"
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame, 
                       fill_method: str = 'ffill',
                       remove_outliers: bool = True,
                       outlier_threshold: float = 3.0) -> pd.DataFrame:
        """
        پاکسازی DataFrame
        """
        df_clean = df.copy()
        
        # حذف duplicate indices
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        
        # مرتب‌سازی بر اساس index
        df_clean = df_clean.sort_index()
        
        # پر کردن مقادیر NaN
        if fill_method == 'ffill':
            df_clean = df_clean.ffill()
        elif fill_method == 'bfill':
            df_clean = df_clean.bfill()
        elif fill_method == 'interpolate':
            df_clean = df_clean.interpolate(method='linear')
        
        # حذف outliers
        if remove_outliers:
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if df_clean[col].std() > 0:  # فقط اگر variance داشته باشد
                    z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                    df_clean = df_clean[z_scores < outlier_threshold]
        
        # حذف سطرهای با NaN باقی‌مانده
        df_clean = df_clean.dropna()
        
        logger.info(f"Cleaned DataFrame: {len(df)} -> {len(df_clean)} rows")
        return df_clean
    
    @staticmethod
    def split_time_series(df: pd.DataFrame, 
                         train_size: float = 0.7,
                         val_size: float = 0.15,
                         test_size: float = 0.15) -> Tuple[pd.DataFrame, ...]:
        """
        تقسیم داده‌های زمانی
        """
        total_size = train_size + val_size + test_size
        if abs(total_size - 1.0) > 0.01:
            raise ValueError("Sizes must sum to 1.0")
        
        n_samples = len(df)
        train_end = int(n_samples * train_size)
        val_end = train_end + int(n_samples * val_size)
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end] if val_size > 0 else None
        test_df = df.iloc[val_end:] if test_size > 0 else None
        
        logger.info(f"Time series split: Train={len(train_df)}, "
                   f"Val={len(val_df) if val_df is not None else 0}, "
                   f"Test={len(test_df) if test_df is not None else 0}")
        
        if val_df is not None and test_df is not None:
            return train_df, val_df, test_df
        elif val_df is not None:
            return train_df, val_df
        else:
            return train_df, test_df
    
    @staticmethod
    def create_rolling_windows(data: np.ndarray, 
                              window_size: int,
                              step_size: int = 1) -> np.ndarray:
        """
        ایجاد پنجره‌های rolling
        """
        n_samples = len(data)
        windows = []
        
        for i in range(0, n_samples - window_size + 1, step_size):
            windows.append(data[i:i + window_size])
        
        return np.array(windows)
    
    @staticmethod
    def calculate_data_statistics(df: pd.DataFrame) -> Dict:
        """
        محاسبه آمار توصیفی داده‌ها
        """
        stats = {}
        
        # آمار کلی
        stats['shape'] = df.shape
        stats['dtypes'] = df.dtypes.to_dict()
        stats['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024**2
        
        # آمار مقادیر NaN
        nan_stats = df.isnull().sum()
        stats['nan_counts'] = nan_stats.to_dict()
        stats['nan_percentage'] = (nan_stats / len(df) * 100).to_dict()
        
        # آمار ستون‌های عددی
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_stats = df[numeric_cols].describe().to_dict()
            stats['numeric_stats'] = numeric_stats
            
            # skewness و kurtosis
            from scipy import stats as scipy_stats
            for col in numeric_cols:
                numeric_stats[col]['skewness'] = scipy_stats.skew(df[col].dropna())
                numeric_stats[col]['kurtosis'] = scipy_stats.kurtosis(df[col].dropna())
        
        # آمار ستون‌های categorical
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            cat_stats = {}
            for col in cat_cols:
                cat_stats[col] = {
                    'unique_count': df[col].nunique(),
                    'top_values': df[col].value_counts().head(5).to_dict()
                }
            stats['categorical_stats'] = cat_stats
        
        return stats
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, filepath: str, 
                      format: str = 'parquet', **kwargs):
        """
        ذخیره DataFrame در فایل
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'parquet':
            df.to_parquet(filepath, **kwargs)
        elif format == 'csv':
            df.to_csv(filepath, **kwargs)
        elif format == 'pickle':
            df.to_pickle(filepath, **kwargs)
        elif format == 'feather':
            df.to_feather(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"DataFrame saved to {filepath} ({format})")
    
    @staticmethod
    def load_dataframe(filepath: str, format: str = None) -> pd.DataFrame:
        """
        بارگذاری DataFrame از فایل
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # تشخیص format از پسوند فایل
        if format is None:
            suffix = path.suffix.lower()
            if suffix == '.parquet':
                format = 'parquet'
            elif suffix == '.csv':
                format = 'csv'
            elif suffix in ['.pkl', '.pickle']:
                format = 'pickle'
            elif suffix == '.feather':
                format = 'feather'
            else:
                raise ValueError(f"Unknown file format: {suffix}")
        
        if format == 'parquet':
            df = pd.read_parquet(filepath)
        elif format == 'csv':
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        elif format == 'pickle':
            df = pd.read_pickle(filepath)
        elif format == 'feather':
            df = pd.read_feather(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"DataFrame loaded from {filepath}: shape={df.shape}")
        return df

class ModelHelper:
    """
    کلاس کمکی برای کار با مدل‌ها
    """
    
    @staticmethod
    def save_model(model: Any, filepath: str, 
                  format: str = 'pickle', **kwargs):
        """
        ذخیره مدل
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(model, f, **kwargs)
        
        elif format == 'joblib':
            import joblib
            joblib.dump(model, filepath, **kwargs)
        
        elif hasattr(model, 'save') and format == 'keras':
            # مدل Keras/TensorFlow
            model.save(filepath, **kwargs)
        
        elif hasattr(model, 'save_model') and format == 'xgboost':
            # مدل XGBoost
            model.save_model(filepath)
        
        else:
            raise ValueError(f"Unsupported format for model type: {format}")
        
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str, format: str = None) -> Any:
        """
        بارگذاری مدل
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # تشخیص format از پسوند فایل
        if format is None:
            suffix = path.suffix.lower()
            if suffix in ['.pkl', '.pickle', '.joblib']:
                format = 'pickle'
            elif suffix in ['.h5', '.keras', '.tf']:
                format = 'keras'
            elif suffix == '.json':
                format = 'xgboost'
            else:
                format = 'pickle'  # پیش‌فرض
        
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
        
        elif format == 'joblib':
            import joblib
            model = joblib.load(filepath)
        
        elif format == 'keras':
            from tensorflow import keras
            model = keras.models.load_model(filepath)
        
        elif format == 'xgboost':
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(filepath)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Model loaded from {filepath}")
        return model
    
    @staticmethod
    def get_model_summary(model: Any) -> Dict:
        """
        دریافت خلاصه مدل
        """
        summary = {}
        
        # نوع مدل
        summary['model_type'] = type(model).__name__
        
        # پارامترهای مدل
        if hasattr(model, 'get_params'):
            try:
                summary['parameters'] = model.get_params()
            except:
                summary['parameters'] = "Cannot retrieve parameters"
        
        # تعداد پارامترها (برای مدل‌های DL)
        if hasattr(model, 'count_params'):
            summary['total_params'] = model.count_params()
        
        # معماری مدل (برای Keras)
        if hasattr(model, 'summary'):
            import io
            import sys
            
            # گرفتن summary به صورت string
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            
            try:
                model.summary()
                summary['architecture'] = buffer.getvalue()
            except:
                summary['architecture'] = "Cannot get architecture"
            finally:
                sys.stdout = old_stdout
        
        return summary
    
    @staticmethod
    def create_model_checkpoint(filepath: str, 
                               monitor: str = 'val_loss',
                               mode: str = 'min',
                               save_best_only: bool = True,
                               verbose: int = 1):
        """
        ایجاد callback برای checkpoint مدل
        """
        from tensorflow.keras.callbacks import ModelCheckpoint
        
        checkpoint = ModelCheckpoint(
            filepath=filepath,
            monitor=monitor,
            mode=mode,
            save_best_only=save_best_only,
            verbose=verbose
        )
        
        return checkpoint
    
    @staticmethod
    def create_early_stopping(monitor: str = 'val_loss',
                             patience: int = 10,
                             restore_best_weights: bool = True):
        """
        ایجاد callback برای early stopping
        """
        from tensorflow.keras.callbacks import EarlyStopping
        
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=restore_best_weights,
            verbose=1
        )
        
        return early_stopping

class TimeHelper:
    """
    کلاس کمکی برای کار با زمان
    """
    
    @staticmethod
    def convert_timeframe(timeframe: str) -> str:
        """
        تبدیل timeframe به فرمت استاندارد
        """
        timeframe = timeframe.upper()
        
        conversions = {
            '1M': '1min', '5M': '5min', '15M': '15min',
            '30M': '30min', '1H': '1h', '4H': '4h',
            '1D': '1d', '1W': '1wk', '1MO': '1mo'
        }
        
        return conversions.get(timeframe, timeframe)
    
    @staticmethod
    def get_trading_dates(start_date: str, end_date: str, 
                         freq: str = 'B') -> pd.DatetimeIndex:
        """
        دریافت تاریخ‌های معاملاتی
        """
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        return dates
    
    @staticmethod
    def calculate_time_difference(date1: datetime, date2: datetime, 
                                unit: str = 'days') -> float:
        """
        محاسبه اختلاف زمانی
        """
        diff = abs(date2 - date1)
        
        if unit == 'days':
            return diff.days + diff.seconds / 86400
        elif unit == 'hours':
            return diff.days * 24 + diff.seconds / 3600
        elif unit == 'minutes':
            return diff.days * 1440 + diff.seconds / 60
        elif unit == 'seconds':
            return diff.total_seconds()
        else:
            raise ValueError(f"Unknown unit: {unit}")
    
    @staticmethod
    def is_market_open(timezone: str = 'America/New_York',
                      current_time: datetime = None) -> bool:
        """
        بررسی باز بودن بازار
        """
        if current_time is None:
            current_time = datetime.now()
        
        import pytz
        
        # تبدیل به timezone بازار
        market_tz = pytz.timezone(timezone)
        market_time = current_time.astimezone(market_tz)
        
        # ساعت بازار (NYSE)
        market_open = market_time.replace(hour=9, minute=30, second=0)
        market_close = market_time.replace(hour=16, minute=0, second=0)
        
        # روزهای تعطیل (ساده شده)
        weekday = market_time.weekday()
        is_weekend = weekday >= 5
        
        # بررسی باز بودن بازار
        if not is_weekend and market_open <= market_time <= market_close:
            return True
        
        return False
    
    @staticmethod
    def get_next_market_open(timezone: str = 'America/New_York',
                           current_time: datetime = None) -> datetime:
        """
        دریافت زمان باز شدن بعدی بازار
        """
        if current_time is None:
            current_time = datetime.now()
        
        import pytz
        from datetime import timedelta
        
        market_tz = pytz.timezone(timezone)
        market_time = current_time.astimezone(market_tz)
        
        # اگر بازار باز است، زمان فعلی را برگردان
        if TimeHelper.is_market_open(timezone, current_time):
            return market_time
        
        # اگر بعد از ساعت بسته شدن است، روز بعد
        market_close_today = market_time.replace(hour=16, minute=0, second=0)
        
        if market_time > market_close_today:
            next_day = market_time + timedelta(days=1)
        else:
            next_day = market_time
        
        # پیدا کردن روز کاری بعدی
        while next_day.weekday() >= 5:  # شنبه=5, یکشنبه=6
            next_day += timedelta(days=1)
        
        # تنظیم ساعت باز شدن
        next_open = next_day.replace(hour=9, minute=30, second=0)
        
        return next_open

class FileHelper:
    """
    کلاس کمکی برای کار با فایل‌ها
    """
    
    @staticmethod
    def get_file_hash(filepath: str, algorithm: str = 'md5') -> str:
        """
        محاسبه hash فایل
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        hash_func = getattr(hashlib, algorithm)()
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    @staticmethod
    def compare_files(file1: str, file2: str, 
                     compare_hash: bool = True) -> bool:
        """
        مقایسه دو فایل
        """
        path1 = Path(file1)
        path2 = Path(file2)
        
        if not path1.exists() or not path2.exists():
            return False
        
        # مقایسه سایز
        if path1.stat().st_size != path2.stat().st_size:
            return False
        
        # مقایسه hash
        if compare_hash:
            hash1 = FileHelper.get_file_hash(file1)
            hash2 = FileHelper.get_file_hash(file2)
            return hash1 == hash2
        
        return True
    
    @staticmethod
    def find_files(directory: str, pattern: str = '*',
                  recursive: bool = True) -> List[Path]:
        """
        پیدا کردن فایل‌ها با pattern
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return []
        
        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))
        
        return files
    
    @staticmethod
    def compress_file(input_file: str, output_file: str = None,
                     method: str = 'gzip') -> str:
        """
        فشرده‌سازی فایل
        """
        import gzip
        import bz2
        import lzma
        
        if output_file is None:
            output_file = input_file + {
                'gzip': '.gz',
                'bzip2': '.bz2',
                'lzma': '.xz'
            }.get(method, '.gz')
        
        compressors = {
            'gzip': gzip.open,
            'bzip2': bz2.open,
            'lzma': lzma.open
        }
        
        if method not in compressors:
            raise ValueError(f"Unsupported compression method: {method}")
        
        compressor = compressors[method]
        
        with open(input_file, 'rb') as f_in:
            with compressor(output_file, 'wb') as f_out:
                f_out.write(f_in.read())
        
        logger.info(f"Compressed {input_file} to {output_file}")
        return output_file

class SecurityHelper:
    """
    کلاس کمکی برای امنیت
    """
    
    @staticmethod
    def encrypt_string(text: str, key: str = None) -> str:
        """
        رمزنگاری رشته
        """
        if key is None:
            key = os.environ.get('ENCRYPTION_KEY', 'default_key')
        
        import base64
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        
        # تولید key از password
        salt = b'salt_'  # در واقعیت باید random باشد
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key_bytes = base64.urlsafe_b64encode(kdf.derive(key.encode()))
        
        # رمزنگاری
        fernet = Fernet(key_bytes)
        encrypted = fernet.encrypt(text.encode())
        
        return encrypted.decode()
    
    @staticmethod
    def decrypt_string(encrypted_text: str, key: str = None) -> str:
        """
        رمزگشایی رشته
        """
        if key is None:
            key = os.environ.get('ENCRYPTION_KEY', 'default_key')
        
        import base64
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        
        # تولید key از password
        salt = b'salt_'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key_bytes = base64.urlsafe_b64encode(kdf.derive(key.encode()))
        
        # رمزگشایی
        fernet = Fernet(key_bytes)
        decrypted = fernet.decrypt(encrypted_text.encode())
        
        return decrypted.decode()
    
    @staticmethod
    def generate_api_key(length: int = 32) -> str:
        """
        تولید API key
        """
        import secrets
        import string
        
        alphabet = string.ascii_letters + string.digits
        api_key = ''.join(secrets.choice(alphabet) for _ in range(length))
        
        return api_key
    
    @staticmethod
    def validate_api_key(api_key: str, stored_hash: str) -> bool:
        """
        اعتبارسنجی API key
        """
        import hashlib
        
        # در واقعیت باید از روش‌های ایمن‌تر استفاده کرد
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        return api_key_hash == stored_hash

class PerformanceHelper:
    """
    کلاس کمکی برای بررسی performance
    """
    
    @staticmethod
    def measure_time(func):
        """
        دکوراتور برای اندازه‌گیری زمان اجرا
        """
        import time
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            logger.info(f"Function {func.__name__} took {elapsed_time:.2f} seconds")
            
            return result
        
        return wrapper
    
    @staticmethod
    def profile_memory(func):
        """
        دکوراتور برای پروفایل حافظه
        """
        import tracemalloc
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracemalloc.start()
            
            result = func(*args, **kwargs)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            logger.info(f"Function {func.__name__} - "
                       f"Current memory: {current / 10**6:.2f} MB, "
                       f"Peak memory: {peak / 10**6:.2f} MB")
            
            return result
        
        return wrapper
    
    @staticmethod
    def check_gpu_available() -> Dict:
        """
        بررسی دسترسی GPU
        """
        gpu_info = {'available': False, 'devices': []}
        
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            
            if gpus:
                gpu_info['available'] = True
                for gpu in gpus:
                    gpu_info['devices'].append({
                        'name': gpu.name,
                        'device_type': gpu.device_type
                    })
                
                logger.info(f"GPU available: {len(gpus)} devices")
            else:
                logger.info("No GPU available, using CPU")
        
        except ImportError:
            logger.warning("TensorFlow not installed, GPU check skipped")
        
        return gpu_info

class ValidationHelper:
    """
    کلاس کمکی برای اعتبارسنجی
    """
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        اعتبارسنجی email
        """
        import re
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """
        اعتبارسنجی URL
        """
        import re
        
        pattern = r'^https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)$'
        return bool(re.match(pattern, url))
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """
        اعتبارسنجی شماره تلفن
        """
        import re
        
        pattern = r'^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$'
        return bool(re.match(pattern, phone))

# توابع کمکی اضافی
def generate_random_string(length: int = 10) -> str:
    """تولید رشته تصادفی"""
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))

def format_bytes(size: float) -> str:
    """فرمت کردن حجم فایل"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

def retry_on_exception(max_retries: int = 3, delay: float = 1.0):
    """
    دکوراتور برای retry روی exception
    """
    import time
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. "
                                 f"Retrying in {delay} seconds...")
                    time.sleep(delay * (attempt + 1))  # Exponential backoff
            
            raise Exception(f"Failed after {max_retries} attempts")
        
        return wrapper
    
    return decorator

def singleton(cls):
    """
    دکوراتور singleton
    """
    instances = {}
    
    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

def __get_gitignore_content():
    """محتوای فایل .gitignore"""
    return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Data
data/raw/
data/processed/
*.csv
*.parquet
*.feather
*.pkl
*.pickle
*.h5
*.keras

# Models
models/**/*.pkl
models/**/*.joblib
models/**/*.h5
models/**/*.keras

# Logs
logs/
*.log

# Environment Variables
.env
.env.local
.env.*.local

# Jupyter Notebook
.ipynb_checkpoints/

# Backtesting results
trading/backtesting/results/

# Dashboard cache
dashboard/__pycache__/
"""

def setup_environment():
    """تنظیمات محیطی"""
    # بارگذاری متغیرهای محیطی
    load_dotenv()
    
    # تنظیمات numpy
    np.random.seed(42)
    np.set_printoptions(precision=4, suppress=True)
    
    # تنظیمات pandas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    
    # تنظیمات matplotlib
    import matplotlib
    matplotlib.use('Agg')  # برای سرورها
    
    logger.info("Environment setup completed")

if __name__ == "__main__":
    print("ماژول توابع کمکی")
    setup_environment()