# trading/strategies/ml_strategy.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import trading_config, path_config
from loguru import logger

class MLTradingStrategy:
    """استراتژی معاملاتی مبتنی بر ML"""
    
    def __init__(self, symbol: str = 'EURUSD'):
        self.symbol = symbol
        self.models_dir = Path(path_config.ML_MODELS_DIR)
        self.models = {}
        self.scaler = None
        
        # بارگذاری مدل‌ها
        self.load_models()
        
        # وضعیت معاملات
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        
    def load_models(self):
        """بارگذاری مدل‌های آموزش دیده"""
        try:
            # بارگذاری scaler
            scaler_path = self.models_dir / f"{self.symbol}_scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # بارگذاری مدل‌ها
            models_to_load = {
                'random_forest': f"{self.symbol}_random_forest.pkl",
                'xgboost': f"{self.symbol}_xgboost.pkl",
                'lightgbm': f"{self.symbol}_lightgbm.pkl"
            }
            
            for model_name, filename in models_to_load.items():
                model_path = self.models_dir / filename
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                        logger.info(f"Loaded {model_name} model")
            
            if not self.models:
                logger.warning("No ML models found!")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def prepare_features(self, df: pd.DataFrame, lookback: int = 60) -> np.ndarray:
        """آماده‌سازی ویژگی‌ها برای پیش‌بینی"""
        if len(df) < lookback:
            return None
        
        # گرفتن آخرین lookback دوره
        recent_data = df.iloc[-lookback:].copy()
        
        # استخراج features
        feature_cols = [col for col in recent_data.columns 
                       if not col.startswith('label_')]
        
        features = recent_data[feature_cols].values
        
        # Standardization
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # تغییر شکل برای مدل‌های مختلف
        return features.flatten().reshape(1, -1)
    
    def ensemble_prediction(self, features: np.ndarray) -> Tuple[float, float]:
        """پیش‌بینی ensemble با ترکیب مدل‌ها"""
        if not self.models or features is None:
            return 0.5, 0.0  # خنثی
        
        predictions = []
        probabilities = []
        weights = {'random_forest': 0.4, 'xgboost': 0.4, 'lightgbm': 0.2}
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features)[0][1]
                    pred = 1 if proba > 0.5 else 0
                else:
                    pred = model.predict(features)[0]
                    proba = pred if pred in [0, 1] else 0.5
                
                predictions.append(pred * weights.get(model_name, 1/len(self.models)))
                probabilities.append(proba * weights.get(model_name, 1/len(self.models)))
                
            except Exception as e:
                logger.error(f"Error in {model_name} prediction: {e}")
                continue
        
        if not predictions:
            return 0.5, 0.0
        
        # میانگین وزن‌دار
        avg_prediction = np.sum(predictions)
        avg_probability = np.sum(probabilities)
        
        return avg_probability, avg_prediction
    
    def generate_signal(self, df: pd.DataFrame) -> Dict:
        """تولید سیگنال معاملاتی"""
        features = self.prepare_features(df)
        
        if features is None:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'prediction': 0.5,
                'position': self.position
            }
        
        # پیش‌بینی ensemble
        probability, prediction = self.ensemble_prediction(features)
        
        # قوانین سیگنال‌دهی
        signal = 'HOLD'
        confidence = abs(probability - 0.5) * 2  # تبدیل به رنج 0-1
        
        # تصمیم‌گیری با threshold
        buy_threshold = 0.6
        sell_threshold = 0.4
        
        if probability > buy_threshold and self.position <= 0:
            signal = 'BUY'
        elif probability < sell_threshold and self.position >= 0:
            signal = 'SELL'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'prediction': probability,
            'position': self.position,
            'features_shape': features.shape
        }
    
    def calculate_position_size(self, capital: float, risk_per_trade: float = 0.01) -> float:
        """محاسبه اندازه پوزیشن"""
        position_size = capital * risk_per_trade
        
        # محدودیت‌های اضافی
        max_position = capital * trading_config.MAX_POSITION_SIZE
        position_size = min(position_size, max_position)
        
        return position_size
    
    def calculate_stop_loss_take_profit(self, entry_price: float, signal: str) -> Tuple[float, float]:
        """محاسبه حد ضرر و حد سود"""
        if signal == 'BUY':
            stop_loss = entry_price * (1 - trading_config.STOP_LOSS_PERCENT)
            take_profit = entry_price * (1 + trading_config.TAKE_PROFIT_PERCENT)
        elif signal == 'SELL':
            stop_loss = entry_price * (1 + trading_config.STOP_LOSS_PERCENT)
            take_profit = entry_price * (1 - trading_config.TAKE_PROFIT_PERCENT)
        else:
            return 0, 0
        
        return stop_loss, take_profit
    
    def update_position(self, signal: str, current_price: float, capital: float) -> Dict:
        """به‌روزرسانی پوزیشن"""
        trade_info = {
            'action': 'HOLD',
            'position_size': 0,
            'entry_price': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'pnl': 0
        }
        
        # محاسبه سود/زیان پوزیشن فعلی
        if self.position != 0:
            if self.position > 0:  # Long position
                trade_info['pnl'] = (current_price - self.entry_price) / self.entry_price
            else:  # Short position
                trade_info['pnl'] = (self.entry_price - current_price) / self.entry_price
            
            # بررسی stop loss و take profit
            if (self.position > 0 and current_price <= self.stop_loss) or \
               (self.position < 0 and current_price >= self.stop_loss):
                trade_info['action'] = 'CLOSE_STOP_LOSS'
                self.position = 0
                
            elif (self.position > 0 and current_price >= self.take_profit) or \
                 (self.position < 0 and current_price <= self.take_profit):
                trade_info['action'] = 'CLOSE_TAKE_PROFIT'
                self.position = 0
        
        # بررسی سیگنال جدید
        if signal == 'BUY' and self.position <= 0:
            if self.position < 0:  # Close short position first
                trade_info['action'] = 'CLOSE_SHORT'
                self.position = 0
            
            # Open long position
            trade_info['action'] = 'BUY'
            self.position = 1
            self.entry_price = current_price
            trade_info['entry_price'] = current_price
            trade_info['position_size'] = self.calculate_position_size(capital)
            
            # محاسبه stop loss و take profit
            self.stop_loss, self.take_profit = self.calculate_stop_loss_take_profit(
                current_price, 'BUY'
            )
            trade_info['stop_loss'] = self.stop_loss
            trade_info['take_profit'] = self.take_profit
            
        elif signal == 'SELL' and self.position >= 0:
            if self.position > 0:  # Close long position first
                trade_info['action'] = 'CLOSE_LONG'
                self.position = 0
            
            # Open short position
            trade_info['action'] = 'SELL'
            self.position = -1
            self.entry_price = current_price
            trade_info['entry_price'] = current_price
            trade_info['position_size'] = self.calculate_position_size(capital)
            
            # محاسبه stop loss و take profit
            self.stop_loss, self.take_profit = self.calculate_stop_loss_take_profit(
                current_price, 'SELL'
            )
            trade_info['stop_loss'] = self.stop_loss
            trade_info['take_profit'] = self.take_profit
        
        trade_info['position'] = self.position
        return trade_info