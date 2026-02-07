# utils/metrics.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
import warnings
warnings.filterwarnings('ignore')

class TradingMetrics:
    """کلاس محاسبه معیارهای معاملاتی"""
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """محاسبه Sharpe Ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # نرخ بدون ریسک روزانه
        if excess_returns.std() == 0:
            return 0.0
        
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """محاسبه Sortino Ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """محاسبه Maximum Drawdown"""
        if len(equity_curve) == 0:
            return 0.0, None, None
        
        # محاسبه cumulative maximum
        cumulative_max = equity_curve.expanding().max()
        drawdown = (equity_curve - cumulative_max) / cumulative_max
        
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        # پیدا کردن peak قبل از drawdown
        peak_idx = equity_curve[:max_dd_idx].idxmax()
        
        return max_dd, peak_idx, max_dd_idx
    
    @staticmethod
    def calmar_ratio(returns: pd.Series, equity_curve: pd.Series) -> float:
        """محاسبه Calmar Ratio"""
        if len(returns) == 0:
            return 0.0
        
        annual_return = returns.mean() * 252
        max_dd, _, _ = TradingMetrics.max_drawdown(equity_curve)
        
        if max_dd == 0:
            return 0.0
        
        return annual_return / abs(max_dd)
    
    @staticmethod
    def win_rate(trades: pd.DataFrame) -> float:
        """محاسبه Win Rate"""
        if len(trades) == 0:
            return 0.0
        
        winning_trades = trades[trades['pnl'] > 0]
        return len(winning_trades) / len(trades)
    
    @staticmethod
    def profit_factor(trades: pd.DataFrame) -> float:
        """محاسبه Profit Factor"""
        if len(trades) == 0:
            return 0.0
        
        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        
        if gross_loss == 0:
            return float('inf')
        
        return gross_profit / gross_loss
    
    @staticmethod
    def average_win_loss(trades: pd.DataFrame) -> Tuple[float, float]:
        """محاسبه میانگین سود و زیان"""
        if len(trades) == 0:
            return 0.0, 0.0
        
        avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if len(trades[trades['pnl'] > 0]) > 0 else 0.0
        avg_loss = trades[trades['pnl'] < 0]['pnl'].mean() if len(trades[trades['pnl'] < 0]) > 0 else 0.0
        
        return avg_win, abs(avg_loss)
    
    @staticmethod
    def expectancy(trades: pd.DataFrame) -> float:
        """محاسبه Expectancy"""
        if len(trades) == 0:
            return 0.0
        
        win_rate = TradingMetrics.win_rate(trades)
        avg_win, avg_loss = TradingMetrics.average_win_loss(trades)
        
        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    @staticmethod
    def value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """محاسبه Value at Risk (VaR)"""
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def conditional_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """محاسبه Conditional Value at Risk (CVaR)"""
        if len(returns) == 0:
            return 0.0
        
        var = TradingMetrics.value_at_risk(returns, confidence_level)
        cvar = returns[returns <= var].mean()
        
        return cvar
    
    @staticmethod
    def model_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict:
        """محاسبه معیارهای مدل"""
        metrics = {}
        
        # معیارهای کلاسیک
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) \
            if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # معیارهای معاملاتی
        metrics['true_signals'] = tp + tn
        metrics['false_signals'] = fp + fn
        metrics['signal_accuracy'] = metrics['true_signals'] / (metrics['true_signals'] + metrics['false_signals']) \
            if (metrics['true_signals'] + metrics['false_signals']) > 0 else 0
        
        return metrics