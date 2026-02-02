"""
استراتژی‌های معاملاتی مبتنی بر ML
"""

import backtrader as bt
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from config.settings import settings

class MLStrategy(bt.Strategy):
    """
    استراتژی معاملاتی مبتنی بر مدل‌های یادگیری ماشین
    """
    
    params = (
        ('symbol', 'EURUSD'),
        ('position_size_pct', 0.02),
        ('stop_loss_pct', 0.02),
        ('take_profit_pct', 0.04),
        ('use_ensemble', True),
    )
    
    def __init__(self):
        # ذخیره مراجع به داده‌ها
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        
        # اندیکاتورهای تکنیکال
        self.sma20 = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=20
        )
        self.sma50 = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=50
        )
        self.rsi = bt.indicators.RSI(
            self.datas[0].close, period=14
        )
        self.macd = bt.indicators.MACD(
            self.datas[0].close
        )
        
        # بارگذاری مدل‌های ML
        self.models = self.load_models()
        
        # ذخیره سیگنال‌ها
        self.signals = []
        self.predictions = []
        
        # متغیرهای معاملاتی
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.trade_count = 0
        self.win_count = 0
        
        # توقف ضرر و حد سود
        self.stop_loss = None
        self.take_profit = None
        
    def load_models(self):
        """بارگذاری مدل‌های آموزش دیده"""
        try:
            models_dir = Path(settings.MODELS_DIR)
            
            if self.p.use_ensemble:
                # بارگذاری مدل ensemble
                ensemble_path = models_dir / "ensemble" / f"{self.p.symbol}_ensemble.pkl"
                if ensemble_path.exists():
                    return joblib.load(ensemble_path)
            
            # بارگذاری مدل Random Forest به عنوان fallback
            rf_path = models_dir / "ml" / f"{self.p.symbol}_random_forest.pkl"
            if rf_path.exists():
                return {'random_forest': joblib.load(rf_path)}
                
        except Exception as e:
            print(f"خطا در بارگذاری مدل‌ها: {e}")
        
        return None
    
    def calculate_features(self):
        """محاسبه ویژگی‌ها از داده‌های کنونی"""
        features = {}
        
        # قیمت‌ها
        features['close'] = self.dataclose[0]
        features['open'] = self.dataopen[0]
        features['high'] = self.datahigh[0]
        features['low'] = self.datalow[0]
        
        # بازده‌ها
        if len(self.dataclose) > 1:
            features['returns'] = (self.dataclose[0] - self.dataclose[-1]) / self.dataclose[-1]
            features['log_returns'] = np.log(self.dataclose[0] / self.dataclose[-1])
        
        # اندیکاتورها
        features['sma20'] = self.sma20[0]
        features['sma50'] = self.sma50[0]
        features['rsi'] = self.rsi[0]
        features['macd'] = self.macd.macd[0]
        features['macd_signal'] = self.macd.signal[0]
        
        # سایر ویژگی‌ها
        features['high_low_pct'] = (self.datahigh[0] - self.datalow[0]) / self.dataclose[0] * 100
        features['close_open_pct'] = (self.dataclose[0] - self.dataopen[0]) / self.dataopen[0] * 100
        
        # ویژگی‌های زمانی
        features['hour'] = self.data.datetime.time().hour
        features['day_of_week'] = self.data.datetime.date().weekday()
        
        return pd.DataFrame([features])
    
    def generate_signal(self):
        """تولید سیگنال از مدل‌های ML"""
        if self.models is None:
            return 0.5  # سیگنال خنثی
        
        try:
            # محاسبه ویژگی‌ها
            features_df = self.calculate_features()
            
            if self.p.use_ensemble and 'meta_ensemble' in self.models:
                # استفاده از ensemble
                meta_model = self.models['meta_ensemble']
                prediction = meta_model.predict_proba(features_df.values)[0, 1]
            elif 'random_forest' in self.models:
                # استفاده از Random Forest
                rf_model = self.models['random_forest']
                prediction = rf_model.predict_proba(features_df.values)[0, 1]
            else:
                prediction = 0.5
            
            return prediction
            
        except Exception as e:
            print(f"خطا در تولید سیگنال: {e}")
            return 0.5
    
    def calculate_position_size(self):
        """محاسبه اندازه پوزیشن با Kelly Criterion"""
        account_value = self.broker.getvalue()
        
        # Kelly Criterion اصلاح‌شده
        if self.trade_count > 0:
            win_rate = self.win_count / self.trade_count
            avg_win = 0.04  # فرضی
            avg_loss = 0.02  # فرضی
            
            if avg_loss > 0:
                kelly_f = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
                kelly_f = max(0, min(kelly_f, 0.5))  # محدود کردن
            else:
                kelly_f = 0.1
        else:
            kelly_f = 0.1
        
        position_value = account_value * kelly_f * self.p.position_size_pct
        size = position_value / self.dataclose[0]
        
        return size
    
    def next(self):
        # اگر سفارش معلقی داریم، منتظر می‌مانیم
        if self.order:
            return
        
        # تولید سیگنال
        signal_prob = self.generate_signal()
        
        # ذخیره سیگنال
        self.signals.append(signal_prob)
        self.predictions.append(signal_prob)
        
        # اگر در پوزیشن نیستیم
        if not self.position:
            # سیگنال خرید قوی
            if signal_prob > 0.7:
                self.buy_signal()
            # سیگنال فروش قوی (Short)
            elif signal_prob < 0.3:
                self.sell_signal()
        
        # مدیریت پوزیشن باز
        else:
            self.manage_position(signal_prob)
    
    def buy_signal(self):
        """اجرای سفارش خرید"""
        size = self.calculate_position_size()
        
        if size > 0:
            # خرید با حد ضرر و حد سود
            self.order = self.buy(size=size)
            
            # تنظیم حد ضرر
            stop_price = self.dataclose[0] * (1 - self.p.stop_loss_pct)
            self.stop_loss = self.sell(
                exectype=bt.Order.Stop, 
                price=stop_price, 
                size=size,
                parent=self.order
            )
            
            # تنظیم حد سود
            take_profit_price = self.dataclose[0] * (1 + self.p.take_profit_pct)
            self.take_profit = self.sell(
                exectype=bt.Order.Limit, 
                price=take_profit_price, 
                size=size,
                parent=self.order
            )
            
            self.log(f'BUY EXECUTED, Price: {self.dataclose[0]:.5f}, Size: {size:.2f}')
    
    def sell_signal(self):
        """اجرای سفارش فروش (Short)"""
        size = self.calculate_position_size()
        
        if size > 0:
            # فروش استقراضی
            self.order = self.sell(size=size)
            
            # تنظیم حد ضرر برای short
            stop_price = self.dataclose[0] * (1 + self.p.stop_loss_pct)
            self.stop_loss = self.buy(
                exectype=bt.Order.Stop, 
                price=stop_price, 
                size=size,
                parent=self.order
            )
            
            # تنظیم حد سود برای short
            take_profit_price = self.dataclose[0] * (1 - self.p.take_profit_pct)
            self.take_profit = self.buy(
                exectype=bt.Order.Limit, 
                price=take_profit_price, 
                size=size,
                parent=self.order
            )
            
            self.log(f'SELL EXECUTED, Price: {self.dataclose[0]:.5f}, Size: {size:.2f}')
    
    def manage_position(self, signal_prob):
        """مدیریت پوزیشن باز"""
        if self.position:
            current_price = self.dataclose[0]
            entry_price = self.position.price
            
            # محاسبه سود/زیان فعلی
            if self.position.size > 0:  # Long position
                pnl_pct = (current_price - entry_price) / entry_price
                
                # اگر سیگنال تغییر کرده، بررسی خروج
                if signal_prob < 0.4 and pnl_pct > 0:
                    self.close()
                    self.log(f'CLOSE LONG, Signal changed, PnL: {pnl_pct:.2%}')
                
                # Trailing stop
                if pnl_pct > 0.02:  # اگر 2% سود داریم
                    new_stop = entry_price * 1.01  # حد ضرر را به 1% سود می‌بریم
                    if new_stop > self.stop_loss.price:
                        self.cancel(self.stop_loss)
                        self.stop_loss = self.sell(
                            exectype=bt.Order.Stop, 
                            price=new_stop, 
                            size=abs(self.position.size)
                        )
                        
            else:  # Short position
                pnl_pct = (entry_price - current_price) / entry_price
                
                # اگر سیگنال تغییر کرده، بررسی خروج
                if signal_prob > 0.6 and pnl_pct > 0:
                    self.close()
                    self.log(f'CLOSE SHORT, Signal changed, PnL: {pnl_pct:.2%}')
    
    def notify_order(self, order):
        """مدیریت اطلاعیه‌های سفارش"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price:.5f}, '
                    f'Cost: {order.executed.value:.2f}, '
                    f'Comm: {order.executed.comm:.2f}'
                )
            else:
                self.log(
                    f'SELL EXECUTED, Price: {order.executed.price:.5f}, '
                    f'Cost: {order.executed.value:.2f}, '
                    f'Comm: {order.executed.comm:.2f}'
                )
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        self.order = None
    
    def notify_trade(self, trade):
        """مدیریت اطلاعیه‌های معامله"""
        if not trade.isclosed:
            return
        
        self.trade_count += 1
        if trade.pnl > 0:
            self.win_count += 1
        
        self.log(
            f'TRADE CLOSED, '
            f'PnL: {trade.pnl:.2f}, '
            f'Net PnL: {trade.pnlcomm:.2f}, '
            f'Win Rate: {self.win_count/self.trade_count:.2%}'
        )
    
    def log(self, txt, dt=None):
        """لاگ کردن"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
    
    def stop(self):
        """پایان بکتست"""
        self.log(f'پایان استراتژی. تعداد معاملات: {self.trade_count}')
        self.log(f'نرخ برد: {self.win_count/max(1, self.trade_count):.2%}')


class HybridStrategy(MLStrategy):
    """
    استراتژی ترکیبی ML + قوانین تکنیکال
    """
    
    params = (
        ('use_technical_rules', True),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
        ('macd_threshold', 0),
    )
    
    def generate_signal(self):
        """تولید سیگنال ترکیبی ML + تکنیکال"""
        # سیگنال ML
        ml_signal = super().generate_signal()
        
        if not self.p.use_technical_rules:
            return ml_signal
        
        # قوانین تکنیکال
        technical_signal = 0.5  # خنثی
        
        # قانون RSI
        if self.rsi[0] > self.p.rsi_overbought:
            technical_signal = 0.3  # فروش
        elif self.rsi[0] < self.p.rsi_oversold:
            technical_signal = 0.7  # خرید
        
        # قانون MACD
        if self.macd.macd[0] > self.macd.signal[0] + self.p.macd_threshold:
            technical_signal = max(technical_signal, 0.6)  # تمایل به خرید
        elif self.macd.macd[0] < self.macd.signal[0] - self.p.macd_threshold:
            technical_signal = min(technical_signal, 0.4)  # تمایل به فروش
        
        # قانون Moving Average
        if self.dataclose[0] > self.sma20[0] > self.sma50[0]:
            technical_signal = max(technical_signal, 0.65)  # روند صعودی
        elif self.dataclose[0] < self.sma20[0] < self.sma50[0]:
            technical_signal = min(technical_signal, 0.35)  # روند نزولی
        
        # ترکیب سیگنال‌ها (وزن بیشتر برای ML)
        combined_signal = (ml_signal * 0.7) + (technical_signal * 0.3)
        
        return combined_signal