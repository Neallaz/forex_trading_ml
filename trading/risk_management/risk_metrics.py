"""
محاسبه معیارهای ریسک مالی
"""

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta

class RiskMetrics:
    """
    کلاس محاسبه معیارهای ریسک
    """
    
    def __init__(self, returns_series=None):
        """
        Args:
            returns_series: سری زمانی بازده‌ها
        """
        self.returns = returns_series if returns_series is not None else pd.Series(dtype=float)
    
    def set_returns(self, returns_series):
        """تنظیم سری بازده‌ها"""
        self.returns = returns_series
    
    def calculate_sharpe_ratio(self, risk_free_rate=0.02, annualization_factor=252):
        """
        محاسبه Sharpe Ratio
        
        Args:
            risk_free_rate: نرخ بدون ریسک سالانه
            annualization_factor: فاکتور سالانه‌سازی (252 برای روزهای معاملاتی)
        
        Returns:
            Sharpe Ratio
        """
        if len(self.returns) < 2:
            return 0
        
        # بازده اضافی
        excess_returns = self.returns - (risk_free_rate / annualization_factor)
        
        # محاسبه Sharpe
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(annualization_factor)
        
        return sharpe
    
    def calculate_sortino_ratio(self, risk_free_rate=0.02, annualization_factor=252):
        """
        محاسبه Sortino Ratio
        
        Args:
            risk_free_rate: نرخ بدون ریسک سالانه
            annualization_factor: فاکتور سالانه‌سازی
        
        Returns:
            Sortino Ratio
        """
        if len(self.returns) < 2:
            return 0
        
        # بازده اضافی
        excess_returns = self.returns - (risk_free_rate / annualization_factor)
        
        # فقط بازده‌های منفی (Downside Deviation)
        downside_returns = self.returns[self.returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_std = np.std(downside_returns)
        
        # محاسبه Sortino
        sortino = np.mean(excess_returns) / downside_std * np.sqrt(annualization_factor)
        
        return sortino
    
    def calculate_max_drawdown(self, prices=None):
        """
        محاسبه Maximum Drawdown
        
        Args:
            prices: سری قیمت‌ها (اگر None باشد، از بازده‌ها محاسبه می‌شود)
        
        Returns:
            حداکثر drawdown و جزئیات
        """
        if prices is not None:
            cumulative = prices
        elif len(self.returns) > 0:
            cumulative = (1 + self.returns).cumprod()
        else:
            return {'max_drawdown': 0, 'start': None, 'end': None, 'recovery': None}
        
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        # پیدا کردن نقطه شروع drawdown
        peak_before = cumulative[:max_dd_idx].idxmax()
        
        # پیدا کردن نقطه بازیابی
        if max_dd_idx < len(cumulative) - 1:
            recovery_idx = cumulative[max_dd_idx:][cumulative[max_dd_idx:] >= cumulative[peak_before]]
            recovery = recovery_idx.index[0] if len(recovery_idx) > 0 else None
        else:
            recovery = None
        
        result = {
            'max_drawdown': max_dd,
            'start': peak_before,
            'end': max_dd_idx,
            'recovery': recovery,
            'duration': (max_dd_idx - peak_before).days if isinstance(peak_before, datetime) else None
        }
        
        return result
    
    def calculate_var(self, confidence_level=0.95, method='historical'):
        """
        محاسبه Value at Risk (VaR)
        
        Args:
            confidence_level: سطح اطمینان
            method: روش محاسبه ('historical', 'parametric', 'monte_carlo')
        
        Returns:
            VaR
        """
        if len(self.returns) == 0:
            return 0
        
        if method == 'historical':
            # روش تاریخی
            var = np.percentile(self.returns, (1 - confidence_level) * 100)
            
        elif method == 'parametric':
            # روش پارامتریک (فرض توزیع نرمال)
            mean = np.mean(self.returns)
            std = np.std(self.returns)
            var = stats.norm.ppf(1 - confidence_level, mean, std)
            
        elif method == 'monte_carlo':
            # روش مونت کارلو
            n_simulations = 10000
            mean = np.mean(self.returns)
            std = np.std(self.returns)
            
            simulated_returns = np.random.normal(mean, std, n_simulations)
            var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return var
    
    def calculate_cvar(self, confidence_level=0.95):
        """
        محاسبه Conditional VaR (Expected Shortfall)
        
        Args:
            confidence_level: سطح اطمینان
        
        Returns:
            CVaR
        """
        if len(self.returns) == 0:
            return 0
        
        var = self.calculate_var(confidence_level, method='historical')
        cvar = self.returns[self.returns <= var].mean()
        
        return cvar
    
    def calculate_calmar_ratio(self, prices=None, risk_free_rate=0.02):
        """
        محاسبه Calmar Ratio
        
        Args:
            prices: سری قیمت‌ها
            risk_free_rate: نرخ بدون ریسک
        
        Returns:
            Calmar Ratio
        """
        if len(self.returns) < 2:
            return 0
        
        # بازده سالانه
        annual_return = self.calculate_annual_return()
        
        # Maximum Drawdown
        max_dd_info = self.calculate_max_drawdown(prices)
        max_dd = abs(max_dd_info['max_drawdown'])
        
        if max_dd == 0:
            return np.inf
        
        # Calmar Ratio
        calmar = (annual_return - risk_free_rate) / max_dd
        
        return calmar
    
    def calculate_annual_return(self):
        """محاسبه بازده سالانه"""
        if len(self.returns) < 2:
            return 0
        
        total_return = (1 + self.returns).prod() - 1
        
        # فرض 252 روز معاملاتی در سال
        n_days = len(self.returns)
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        
        return annual_return
    
    def calculate_volatility(self, annualized=True):
        """محاسبه نوسان"""
        if len(self.returns) < 2:
            return 0
        
        volatility = np.std(self.returns)
        
        if annualized:
            volatility *= np.sqrt(252)
        
        return volatility
    
    def calculate_beta(self, market_returns):
        """
        محاسبه Beta نسبت به بازار
        
        Args:
            market_returns: بازده‌های بازار
        
        Returns:
            Beta
        """
        if len(self.returns) < 2 or len(market_returns) < 2:
            return 0
        
        # همسو کردن داده‌ها
        aligned_returns = pd.concat([self.returns, market_returns], axis=1, join='inner')
        
        if len(aligned_returns) < 2:
            return 0
        
        cov_matrix = np.cov(aligned_returns.values.T)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        
        return beta
    
    def calculate_alpha(self, market_returns, risk_free_rate=0.02):
        """
        محاسبه Alpha
        
        Args:
            market_returns: بازده‌های بازار
            risk_free_rate: نرخ بدون ریسک
        
        Returns:
            Alpha
        """
        if len(self.returns) < 2:
            return 0
        
        beta = self.calculate_beta(market_returns)
        portfolio_return = self.calculate_annual_return()
        market_return = (1 + market_returns).prod() - 1
        
        # Annualize market return
        n_days = len(market_returns)
        market_return_annual = (1 + market_return) ** (252 / n_days) - 1
        
        alpha = portfolio_return - (risk_free_rate + beta * (market_return_annual - risk_free_rate))
        
        return alpha
    
    def calculate_treynor_ratio(self, market_returns, risk_free_rate=0.02):
        """
        محاسبه Treynor Ratio
        
        Args:
            market_returns: بازده‌های بازار
            risk_free_rate: نرخ بدون ریسک
        
        Returns:
            Treynor Ratio
        """
        if len(self.returns) < 2:
            return 0
        
        beta = self.calculate_beta(market_returns)
        
        if beta == 0:
            return np.inf
        
        portfolio_return = self.calculate_annual_return()
        treynor = (portfolio_return - risk_free_rate) / beta
        
        return treynor
    
    def calculate_information_ratio(self, benchmark_returns):
        """
        محاسبه Information Ratio
        
        Args:
            benchmark_returns: بازده‌های معیار
        
        Returns:
            Information Ratio
        """
        if len(self.returns) < 2 or len(benchmark_returns) < 2:
            return 0
        
        # همسو کردن داده‌ها
        aligned_data = pd.concat([self.returns, benchmark_returns], axis=1, join='inner')
        
        if len(aligned_data) < 2:
            return 0
        
        active_returns = aligned_data.iloc[:, 0] - aligned_data.iloc[:, 1]
        
        if len(active_returns) < 2:
            return 0
        
        information_ratio = np.mean(active_returns) / np.std(active_returns) * np.sqrt(252)
        
        return information_ratio
    
    def calculate_ulcer_index(self, prices=None):
        """
        محاسبه Ulcer Index
        
        Args:
            prices: سری قیمت‌ها
        
        Returns:
            Ulcer Index
        """
        if prices is not None:
            cumulative = prices
        elif len(self.returns) > 0:
            cumulative = (1 + self.returns).cumprod()
        else:
            return 0
        
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        ulcer_index = np.sqrt(np.mean(drawdown ** 2))
        
        return ulcer_index
    
    def get_all_metrics(self, prices=None, market_returns=None, benchmark_returns=None):
        """
        محاسبه تمام معیارهای ریسک
        
        Returns:
            دیکشنری حاوی تمام معیارها
        """
        metrics = {}
        
        # بازده‌ها
        metrics['annual_return'] = self.calculate_annual_return()
        metrics['volatility'] = self.calculate_volatility(annualized=True)
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio()
        metrics['sortino_ratio'] = self.calculate_sortino_ratio()
        
        # Drawdown
        dd_info = self.calculate_max_drawdown(prices)
        metrics['max_drawdown'] = dd_info['max_drawdown']
        metrics['drawdown_duration'] = dd_info['duration']
        
        # سایر معیارها
        metrics['calmar_ratio'] = self.calculate_calmar_ratio(prices)
        metrics['var_95'] = self.calculate_var(confidence_level=0.95)
        metrics['cvar_95'] = self.calculate_cvar(confidence_level=0.95)
        metrics['ulcer_index'] = self.calculate_ulcer_index(prices)
        
        # معیارهای نسبی
        if market_returns is not None:
            metrics['beta'] = self.calculate_beta(market_returns)
            metrics['alpha'] = self.calculate_alpha(market_returns)
            metrics['treynor_ratio'] = self.calculate_treynor_ratio(market_returns)
        
        if benchmark_returns is not None:
            metrics['information_ratio'] = self.calculate_information_ratio(benchmark_returns)
        
        # آمار توصیفی
        if len(self.returns) > 0:
            metrics['skewness'] = stats.skew(self.returns)
            metrics['kurtosis'] = stats.kurtosis(self.returns)
            metrics['jarque_bera'] = stats.jarque_bera(self.returns)[1]  # p-value
        
        return metrics
    
    def generate_risk_report(self, prices=None, market_returns=None, benchmark_returns=None):
        """
        تولید گزارش ریسک کامل
        
        Returns:
            گزارش ریسک به صورت متن
        """
        metrics = self.get_all_metrics(prices, market_returns, benchmark_returns)
        
        report = "=" * 60 + "\n"
        report += "گزارش تحلیل ریسک\n"
        report += "=" * 60 + "\n\n"
        
        report += "1. معیارهای بازده و ریسک:\n"
        report += "-" * 40 + "\n"
        report += f"بازده سالانه: {metrics['annual_return']:.2%}\n"
        report += f"نوسان سالانه: {metrics['volatility']:.2%}\n"
        report += f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        report += f"Sortino Ratio: {metrics['sortino_ratio']:.2f}\n"
        report += f"Calmar Ratio: {metrics['calmar_ratio']:.2f}\n\n"
        
        report += "2. معیارهای Drawdown:\n"
        report += "-" * 40 + "\n"
        report += f"حداکثر Drawdown: {metrics['max_drawdown']:.2%}\n"
        report += f"مدت Drawdown: {metrics['drawdown_duration']} روز\n"
        report += f"Ulcer Index: {metrics['ulcer_index']:.4f}\n\n"
        
        report += "3. معیارهای ریسک مطلق:\n"
        report += "-" * 40 + "\n"
        report += f"VaR (95%): {metrics['var_95']:.2%}\n"
        report += f"CVaR (95%): {metrics['cvar_95']:.2%}\n\n"
        
        if 'beta' in metrics:
            report += "4. معیارهای ریسک نسبی:\n"
            report += "-" * 40 + "\n"
            report += f"Beta: {metrics['beta']:.2f}\n"
            report += f"Alpha: {metrics['alpha']:.2%}\n"
            report += f"Treynor Ratio: {metrics['treynor_ratio']:.2f}\n"
        
        if 'information_ratio' in metrics:
            report += f"Information Ratio: {metrics['information_ratio']:.2f}\n\n"
        
        report += "5. آمار توصیفی بازده‌ها:\n"
        report += "-" * 40 + "\n"
        if 'skewness' in metrics:
            report += f"چولگی: {metrics['skewness']:.2f}\n"
            report += f"کشیدگی: {metrics['kurtosis']:.2f}\n"
        
        if 'jarque_bera' in metrics:
            normality = "توزیع نرمال" if metrics['jarque_bera'] > 0.05 else "توزیع غیرنرمال"
            report += f"آزمون نرمال بودن (Jarque-Bera): {normality}\n"
        
        report += "\n" + "=" * 60 + "\n"
        
        return report