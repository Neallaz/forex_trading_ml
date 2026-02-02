"""
مدیریت اندازه پوزیشن و ریسک
"""

import numpy as np
import pandas as pd
from scipy import stats

class PositionSizer:
    """
    کلاس مدیریت اندازه پوزیشن
    """
    
    def __init__(self, initial_capital=10000, max_risk_per_trade=0.02):
        """
        Args:
            initial_capital: سرمایه اولیه
            max_risk_per_trade: حداکثر ریسک در هر معامله (به عنوان درصدی از سرمایه)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.position_history = []
        
    def kelly_criterion(self, win_rate, avg_win, avg_loss):
        """
        محاسبه اندازه پوزیشن با Kelly Criterion
        
        Args:
            win_rate: نرخ برد
            avg_win: میانگین سود (به صورت نسبت)
            avg_loss: میانگین ضرر (به صورت نسبت)
        
        Returns:
            درصد سرمایه برای اختصاص به معامله
        """
        if avg_loss == 0:
            return 0.1  # مقدار پیش‌فرض
        
        kelly_f = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        
        # محدود کردن به بازه معقول
        kelly_f = max(0, min(kelly_f, 0.25))  # حداکثر 25% سرمایه
        
        return kelly_f
    
    def optimal_f(self, returns_series):
        """
        محاسبه Optimal f بر اساس تئوری Ralph Vince
        
        Args:
            returns_series: سری بازده‌های معاملات گذشته
        
        Returns:
            مقدار Optimal f
        """
        if len(returns_series) < 10:
            return 0.1
        
        try:
            # تبدیل به numpy array
            returns = np.array(returns_series)
            
            # محاسبه TWR برای مقادیر مختلف f
            best_f = 0
            best_twr = -np.inf
            
            for f in np.arange(0.01, 0.5, 0.01):
                twr = np.prod(1 + returns * f)
                
                if twr > best_twr:
                    best_twr = twr
                    best_f = f
            
            return best_f
            
        except:
            return 0.1
    
    def fixed_fractional(self, account_value, risk_per_trade=None):
        """
        روش Fixed Fractional
        
        Args:
            account_value: ارزش حساب فعلی
            risk_per_trade: ریسک در هر معامله (اگر None باشد از max_risk_per_trade استفاده می‌شود)
        
        Returns:
            حداکثر سرمایه قابل اختصاص
        """
        if risk_per_trade is None:
            risk_per_trade = self.max_risk_per_trade
        
        risk_amount = account_value * risk_per_trade
        return risk_amount
    
    def fixed_ratio(self, account_value, delta=5000):
        """
        روش Fixed Ratio (Ryan Jones)
        
        Args:
            account_value: ارزش حساب فعلی
            delta: پارامتر دلتا
        
        Returns:
            تعداد واحدهای معاملاتی
        """
        n = (account_value - self.initial_capital) / delta
        n = max(1, n)  # حداقل یک واحد
        
        return n
    
    def volatility_adjusted(self, current_price, atr, atr_period=14):
        """
        تنظیم اندازه پوزیشن بر اساس نوسان
        
        Args:
            current_price: قیمت فعلی
            atr: Average True Range
            atr_period: دوره ATR
        
        Returns:
            اندازه پوزیشن بر اساس واحدهای استاندارد شده
        """
        # محاسبه واحدهای استاندارد
        if atr > 0:
            risk_amount = self.current_capital * self.max_risk_per_trade
            unit_size = risk_amount / (atr * 2)  # ریسک 2 برابر ATR
            
            # تبدیل به تعداد سهام/واحد
            position_size = unit_size / current_price
        else:
            position_size = self.fixed_fractional(self.current_capital) / current_price
        
        return position_size
    
    def calculate_position_size(self, method='kelly', **kwargs):
        """
        محاسبه اندازه پوزیشن با روش انتخابی
        
        Args:
            method: روش محاسبه ('kelly', 'fixed_fractional', 'volatility_adjusted', 'optimal_f')
            **kwargs: پارامترهای مورد نیاز برای هر روش
        
        Returns:
            اندازه پوزیشن
        """
        if method == 'kelly':
            win_rate = kwargs.get('win_rate', 0.5)
            avg_win = kwargs.get('avg_win', 0.04)
            avg_loss = kwargs.get('avg_loss', 0.02)
            
            kelly_f = self.kelly_criterion(win_rate, avg_win, avg_loss)
            position_value = self.current_capital * kelly_f * self.max_risk_per_trade
            
            return position_value
            
        elif method == 'fixed_fractional':
            risk_per_trade = kwargs.get('risk_per_trade', self.max_risk_per_trade)
            return self.fixed_fractional(self.current_capital, risk_per_trade)
            
        elif method == 'volatility_adjusted':
            current_price = kwargs.get('current_price')
            atr = kwargs.get('atr', 0)
            
            if current_price is None:
                raise ValueError("current_price required for volatility_adjusted method")
            
            position_size = self.volatility_adjusted(current_price, atr)
            return position_size
            
        elif method == 'optimal_f':
            returns_series = kwargs.get('returns_series', [])
            
            if len(returns_series) == 0:
                return self.fixed_fractional(self.current_capital)
            
            optimal_f = self.optimal_f(returns_series)
            position_value = self.current_capital * optimal_f * self.max_risk_per_trade
            
            return position_value
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def update_capital(self, new_capital):
        """به‌روزرسانی سرمایه فعلی"""
        self.current_capital = new_capital
    
    def add_trade(self, entry_price, exit_price, position_size, trade_type='long'):
        """اضافه کردن معامله به تاریخچه"""
        if trade_type == 'long':
            pnl = (exit_price - entry_price) * position_size
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # short
            pnl = (entry_price - exit_price) * position_size
            pnl_pct = (entry_price - exit_price) / entry_price
        
        trade = {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'trade_type': trade_type
        }
        
        self.position_history.append(trade)
        
        # به‌روزرسانی سرمایه
        self.current_capital += pnl
    
    def get_performance_stats(self):
        """محاسبه آمار عملکرد"""
        if not self.position_history:
            return {}
        
        trades = pd.DataFrame(self.position_history)
        
        stats = {
            'total_trades': len(trades),
            'winning_trades': len(trades[trades['pnl'] > 0]),
            'losing_trades': len(trades[trades['pnl'] < 0]),
            'total_pnl': trades['pnl'].sum(),
            'avg_pnl': trades['pnl'].mean(),
            'max_win': trades['pnl'].max(),
            'max_loss': trades['pnl'].min(),
            'win_rate': len(trades[trades['pnl'] > 0]) / len(trades),
            'profit_factor': abs(trades[trades['pnl'] > 0]['pnl'].sum() / 
                                trades[trades['pnl'] < 0]['pnl'].sum()),
            'sharpe_ratio': self.calculate_sharpe(trades['pnl_pct']),
            'max_drawdown': self.calculate_max_drawdown(trades['pnl_pct']),
            'current_capital': self.current_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital
        }
        
        return stats
    
    def calculate_sharpe(self, returns, risk_free_rate=0.02):
        """محاسبه Sharpe Ratio"""
        if len(returns) < 2:
            return 0
        
        excess_returns = returns - (risk_free_rate / 252)  # فرض 252 روز معاملاتی
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        return sharpe
    
    def calculate_max_drawdown(self, returns):
        """محاسبه Maximum Drawdown"""
        if len(returns) == 0:
            return 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    def get_position_sizing_recommendation(self, current_price, volatility=None):
        """
        دریافت توصیه برای اندازه پوزیشن
        
        Args:
            current_price: قیمت فعلی
            volatility: نوسان (اختیاری)
        
        Returns:
            دیکشنری حاوی توصیه‌های مختلف
        """
        recommendations = {}
        
        # Fixed Fractional
        ff_amount = self.fixed_fractional(self.current_capital)
        recommendations['fixed_fractional'] = {
            'amount': ff_amount,
            'units': ff_amount / current_price if current_price > 0 else 0,
            'risk_percent': self.max_risk_per_trade * 100
        }
        
        # اگر تاریخچه معاملات وجود دارد
        if self.position_history:
            trades = pd.DataFrame(self.position_history)
            win_rate = len(trades[trades['pnl'] > 0]) / len(trades)
            avg_win = trades[trades['pnl'] > 0]['pnl_pct'].mean() if len(trades[trades['pnl'] > 0]) > 0 else 0.04
            avg_loss = abs(trades[trades['pnl'] < 0]['pnl_pct'].mean()) if len(trades[trades['pnl'] < 0]) > 0 else 0.02
            
            # Kelly Criterion
            kelly_f = self.kelly_criterion(win_rate, avg_win, avg_loss)
            kelly_amount = self.current_capital * kelly_f * self.max_risk_per_trade
            
            recommendations['kelly'] = {
                'amount': kelly_amount,
                'units': kelly_amount / current_price if current_price > 0 else 0,
                'kelly_f': kelly_f,
                'win_rate': win_rate
            }
            
            # Optimal f
            optimal_f = self.optimal_f(trades['pnl_pct'])
            optimal_amount = self.current_capital * optimal_f * self.max_risk_per_trade
            
            recommendations['optimal_f'] = {
                'amount': optimal_amount,
                'units': optimal_amount / current_price if current_price > 0 else 0,
                'optimal_f': optimal_f
            }
        
        # اگر volatility داده شده باشد
        if volatility is not None and volatility > 0:
            vol_amount = self.volatility_adjusted(current_price, volatility)
            recommendations['volatility_adjusted'] = {
                'amount': vol_amount * current_price,
                'units': vol_amount,
                'volatility': volatility
            }
        
        return recommendations