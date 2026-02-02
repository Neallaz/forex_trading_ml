"""
محاسبه معیارهای عملکرد مالی و آماری
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FinancialMetrics:
    """
    کلاس محاسبه معیارهای مالی
    """
    
    @staticmethod
    def calculate_returns(prices: pd.Series, method: str = 'log') -> pd.Series:
        """
        محاسبه بازده‌ها
        
        Args:
            prices: سری قیمت‌ها
            method: روش محاسبه ('log' یا 'simple')
        
        Returns:
            سری بازده‌ها
        """
        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:  # simple
            returns = prices.pct_change()
        
        return returns.dropna()
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, 
                              annualization_factor: int = 252) -> float:
        """
        محاسبه Sharpe Ratio
        
        Args:
            returns: بازده‌های روزانه
            risk_free_rate: نرخ بدون ریسک سالانه
            annualization_factor: فاکتور سالانه‌سازی
        
        Returns:
            Sharpe Ratio
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / annualization_factor)
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        sharpe *= np.sqrt(annualization_factor)
        
        return sharpe
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                               annualization_factor: int = 252) -> float:
        """
        محاسبه Sortino Ratio
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / annualization_factor)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_std = np.std(downside_returns)
        sortino = np.mean(excess_returns) / downside_std
        sortino *= np.sqrt(annualization_factor)
        
        return sortino
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> Dict:
        """
        محاسبه Maximum Drawdown
        """
        cumulative = prices if prices.iloc[0] != 0 else prices / prices.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        # پیدا کردن peak قبل از drawdown
        peak_before = cumulative[:max_dd_idx].idxmax()
        peak_value = cumulative.loc[peak_before]
        trough_value = cumulative.loc[max_dd_idx]
        
        return {
            'max_drawdown': max_dd,
            'peak_date': peak_before,
            'trough_date': max_dd_idx,
            'peak_value': peak_value,
            'trough_value': trough_value,
            'recovery_date': None,
            'drawdown_duration': (max_dd_idx - peak_before).days if hasattr(max_dd_idx, 'date') else None
        }
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, prices: pd.Series, 
                              risk_free_rate: float = 0.02) -> float:
        """
        محاسبه Calmar Ratio
        """
        if len(returns) < 2:
            return 0.0
        
        annual_return = FinancialMetrics.calculate_annual_return(returns)
        max_dd_info = FinancialMetrics.calculate_max_drawdown(prices)
        max_dd = abs(max_dd_info['max_drawdown'])
        
        if max_dd == 0:
            return np.inf
        
        calmar = (annual_return - risk_free_rate) / max_dd
        return calmar
    
    @staticmethod
    def calculate_annual_return(returns: pd.Series, days_per_year: int = 252) -> float:
        """
        محاسبه بازده سالانه
        """
        if len(returns) == 0:
            return 0.0
        
        total_return = np.prod(1 + returns) - 1
        n_days = len(returns)
        annual_return = (1 + total_return) ** (days_per_year / n_days) - 1
        
        return annual_return
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, annualized: bool = True, 
                            days_per_year: int = 252) -> float:
        """
        محاسبه نوسان
        """
        if len(returns) < 2:
            return 0.0
        
        volatility = np.std(returns)
        
        if annualized:
            volatility *= np.sqrt(days_per_year)
        
        return volatility
    
    @staticmethod
    def calculate_beta(portfolio_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        محاسبه Beta
        """
        if len(portfolio_returns) < 2 or len(market_returns) < 2:
            return 0.0
        
        aligned_returns = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
        
        if len(aligned_returns) < 2:
            return 0.0
        
        cov_matrix = np.cov(aligned_returns.values.T)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        
        return beta
    
    @staticmethod
    def calculate_alpha(portfolio_returns: pd.Series, market_returns: pd.Series,
                       risk_free_rate: float = 0.02) -> float:
        """
        محاسبه Alpha
        """
        if len(portfolio_returns) < 2:
            return 0.0
        
        beta = FinancialMetrics.calculate_beta(portfolio_returns, market_returns)
        portfolio_annual_return = FinancialMetrics.calculate_annual_return(portfolio_returns)
        market_annual_return = FinancialMetrics.calculate_annual_return(market_returns)
        
        alpha = portfolio_annual_return - (risk_free_rate + beta * (market_annual_return - risk_free_rate))
        
        return alpha
    
    @staticmethod
    def calculate_treynor_ratio(portfolio_returns: pd.Series, market_returns: pd.Series,
                               risk_free_rate: float = 0.02) -> float:
        """
        محاسبه Treynor Ratio
        """
        if len(portfolio_returns) < 2:
            return 0.0
        
        beta = FinancialMetrics.calculate_beta(portfolio_returns, market_returns)
        
        if beta == 0:
            return np.inf
        
        portfolio_annual_return = FinancialMetrics.calculate_annual_return(portfolio_returns)
        treynor = (portfolio_annual_return - risk_free_rate) / beta
        
        return treynor
    
    @staticmethod
    def calculate_information_ratio(portfolio_returns: pd.Series, 
                                   benchmark_returns: pd.Series) -> float:
        """
        محاسبه Information Ratio
        """
        if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
            return 0.0
        
        aligned_returns = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_returns) < 2:
            return 0.0
        
        active_returns = aligned_returns.iloc[:, 0] - aligned_returns.iloc[:, 1]
        tracking_error = np.std(active_returns)
        
        if tracking_error == 0:
            return np.inf
        
        information_ratio = np.mean(active_returns) / tracking_error * np.sqrt(252)
        
        return information_ratio
    
    @staticmethod
    def calculate_ulcer_index(prices: pd.Series) -> float:
        """
        محاسبه Ulcer Index
        """
        if len(prices) < 2:
            return 0.0
        
        cumulative = prices / prices.iloc[0] if prices.iloc[0] != 0 else prices
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        ulcer_index = np.sqrt(np.mean(drawdown ** 2))
        
        return ulcer_index
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95, 
                     method: str = 'historical') -> float:
        """
        محاسبه Value at Risk
        """
        if len(returns) == 0:
            return 0.0
        
        if method == 'historical':
            var = np.percentile(returns, (1 - confidence_level) * 100)
        elif method == 'parametric':
            mean = np.mean(returns)
            std = np.std(returns)
            var = stats.norm.ppf(1 - confidence_level, mean, std)
        elif method == 'monte_carlo':
            n_simulations = 10000
            mean = np.mean(returns)
            std = np.std(returns)
            simulated_returns = np.random.normal(mean, std, n_simulations)
            var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return var
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        محاسبه Conditional VaR (Expected Shortfall)
        """
        if len(returns) == 0:
            return 0.0
        
        var = FinancialMetrics.calculate_var(returns, confidence_level, 'historical')
        cvar = returns[returns <= var].mean()
        
        return cvar
    
    @staticmethod
    def calculate_win_rate(trades: pd.DataFrame) -> float:
        """
        محاسبه Win Rate
        """
        if len(trades) == 0:
            return 0.0
        
        winning_trades = trades[trades['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades)
        
        return win_rate
    
    @staticmethod
    def calculate_profit_factor(trades: pd.DataFrame) -> float:
        """
        محاسبه Profit Factor
        """
        if len(trades) == 0:
            return 0.0
        
        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        
        if gross_loss == 0:
            return np.inf
        
        profit_factor = gross_profit / gross_loss
        
        return profit_factor
    
    @staticmethod
    def calculate_average_win(trades: pd.DataFrame) -> float:
        """
        محاسبه میانگین سود
        """
        winning_trades = trades[trades['pnl'] > 0]
        
        if len(winning_trades) == 0:
            return 0.0
        
        avg_win = winning_trades['pnl'].mean()
        
        return avg_win
    
    @staticmethod
    def calculate_average_loss(trades: pd.DataFrame) -> float:
        """
        محاسبه میانگین ضرر
        """
        losing_trades = trades[trades['pnl'] < 0]
        
        if len(losing_trades) == 0:
            return 0.0
        
        avg_loss = losing_trades['pnl'].mean()
        
        return avg_loss
    
    @staticmethod
    def calculate_expectancy(trades: pd.DataFrame) -> float:
        """
        محاسبه Expectancy
        """
        win_rate = FinancialMetrics.calculate_win_rate(trades)
        avg_win = FinancialMetrics.calculate_average_win(trades)
        avg_loss = FinancialMetrics.calculate_average_loss(trades)
        
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
        return expectancy
    
    @staticmethod
    def calculate_risk_reward_ratio(trades: pd.DataFrame) -> float:
        """
        محاسبه نسبت Risk/Reward
        """
        avg_win = FinancialMetrics.calculate_average_win(trades)
        avg_loss = FinancialMetrics.calculate_average_loss(trades)
        
        if avg_loss == 0:
            return np.inf
        
        risk_reward = abs(avg_win / avg_loss)
        
        return risk_reward
    
    @staticmethod
    def calculate_sqn(trades: pd.DataFrame) -> float:
        """
        محاسبه System Quality Number (SQN)
        """
        if len(trades) < 10:
            return 0.0
        
        expectancy = FinancialMetrics.calculate_expectancy(trades)
        std_pnl = trades['pnl'].std()
        
        if std_pnl == 0:
            return 0.0
        
        sqn = (expectancy / std_pnl) * np.sqrt(len(trades))
        
        return sqn
    
    @staticmethod
    def calculate_kelly_criterion(win_rate: float, avg_win_pct: float, 
                                 avg_loss_pct: float) -> float:
        """
        محاسبه Kelly Criterion
        """
        if avg_loss_pct == 0:
            return 0.0
        
        kelly = win_rate - ((1 - win_rate) / (avg_win_pct / avg_loss_pct))
        kelly = max(0, min(kelly, 1))  # محدود کردن به بازه [0, 1]
        
        return kelly
    
    @staticmethod
    def calculate_mar_ratio(annual_return: float, max_drawdown: float) -> float:
        """
        محاسبه MAR Ratio (Return over Maximum Drawdown)
        """
        if max_drawdown == 0:
            return np.inf
        
        mar = annual_return / abs(max_drawdown)
        
        return mar
    
    @staticmethod
    def calculate_omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
        """
        محاسبه Omega Ratio
        """
        if len(returns) == 0:
            return 0.0
        
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        
        if losses == 0:
            return np.inf
        
        omega = gains / losses
        
        return omega
    
    @staticmethod
    def calculate_gain_to_pain_ratio(returns: pd.Series) -> float:
        """
        محاسبه Gain to Pain Ratio
        """
        if len(returns) == 0:
            return 0.0
        
        total_gain = returns[returns > 0].sum()
        total_pain = abs(returns[returns < 0].sum())
        
        if total_pain == 0:
            return np.inf
        
        gpr = total_gain / total_pain
        
        return gpr
    
    @staticmethod
    def calculate_tail_ratio(returns: pd.Series, percentile: float = 0.95) -> float:
        """
        محاسبه Tail Ratio
        """
        if len(returns) < 10:
            return 0.0
        
        positive_tail = np.percentile(returns, percentile * 100)
        negative_tail = abs(np.percentile(returns, (1 - percentile) * 100))
        
        if negative_tail == 0:
            return np.inf
        
        tail_ratio = positive_tail / negative_tail
        
        return tail_ratio
    
    @staticmethod
    def calculate_common_sense_ratio(returns: pd.Series) -> float:
        """
        محاسبه Common Sense Ratio
        """
        if len(returns) < 10:
            return 0.0
        
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return np.inf
        
        avg_positive = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_negative = abs(negative_returns.mean())
        
        if avg_negative == 0:
            return np.inf
        
        csr = avg_positive / avg_negative
        
        return csr
    
    @staticmethod
    def calculate_all_metrics(returns: pd.Series, prices: pd.Series = None,
                            trades: pd.DataFrame = None, 
                            benchmark_returns: pd.Series = None) -> Dict:
        """
        محاسبه تمام معیارهای عملکرد
        """
        metrics = {}
        
        # معیارهای بازده
        metrics['total_return'] = np.prod(1 + returns) - 1
        metrics['annual_return'] = FinancialMetrics.calculate_annual_return(returns)
        metrics['cagr'] = metrics['annual_return']
        
        # معیارهای ریسک
        metrics['volatility'] = FinancialMetrics.calculate_volatility(returns)
        metrics['sharpe_ratio'] = FinancialMetrics.calculate_sharpe_ratio(returns)
        metrics['sortino_ratio'] = FinancialMetrics.calculate_sortino_ratio(returns)
        metrics['var_95'] = FinancialMetrics.calculate_var(returns, 0.95)
        metrics['cvar_95'] = FinancialMetrics.calculate_cvar(returns, 0.95)
        
        # معیارهای Drawdown
        if prices is not None:
            dd_info = FinancialMetrics.calculate_max_drawdown(prices)
            metrics['max_drawdown'] = dd_info['max_drawdown']
            metrics['calmar_ratio'] = FinancialMetrics.calculate_calmar_ratio(returns, prices)
            metrics['ulcer_index'] = FinancialMetrics.calculate_ulcer_index(prices)
            metrics['mar_ratio'] = FinancialMetrics.calculate_mar_ratio(
                metrics['annual_return'], metrics['max_drawdown']
            )
        
        # معیارهای معاملاتی
        if trades is not None:
            metrics['win_rate'] = FinancialMetrics.calculate_win_rate(trades)
            metrics['profit_factor'] = FinancialMetrics.calculate_profit_factor(trades)
            metrics['expectancy'] = FinancialMetrics.calculate_expectancy(trades)
            metrics['sqn'] = FinancialMetrics.calculate_sqn(trades)
            metrics['avg_win'] = FinancialMetrics.calculate_average_win(trades)
            metrics['avg_loss'] = FinancialMetrics.calculate_average_loss(trades)
            metrics['risk_reward_ratio'] = FinancialMetrics.calculate_risk_reward_ratio(trades)
        
        # معیارهای نسبی
        if benchmark_returns is not None:
            metrics['beta'] = FinancialMetrics.calculate_beta(returns, benchmark_returns)
            metrics['alpha'] = FinancialMetrics.calculate_alpha(returns, benchmark_returns)
            metrics['treynor_ratio'] = FinancialMetrics.calculate_treynor_ratio(returns, benchmark_returns)
            metrics['information_ratio'] = FinancialMetrics.calculate_information_ratio(returns, benchmark_returns)
        
        # سایر معیارها
        metrics['omega_ratio'] = FinancialMetrics.calculate_omega_ratio(returns)
        metrics['gain_to_pain_ratio'] = FinancialMetrics.calculate_gain_to_pain_ratio(returns)
        metrics['tail_ratio'] = FinancialMetrics.calculate_tail_ratio(returns)
        metrics['common_sense_ratio'] = FinancialMetrics.calculate_common_sense_ratio(returns)
        
        # آمار توصیفی
        metrics['skewness'] = stats.skew(returns)
        metrics['kurtosis'] = stats.kurtosis(returns)
        metrics['jarque_bera_pvalue'] = stats.jarque_bera(returns)[1]
        
        return metrics
    
    @staticmethod
    def generate_performance_report(returns: pd.Series, prices: pd.Series = None,
                                  trades: pd.DataFrame = None,
                                  benchmark_returns: pd.Series = None) -> str:
        """
        تولید گزارش عملکرد
        """
        metrics = FinancialMetrics.calculate_all_metrics(
            returns, prices, trades, benchmark_returns
        )
        
        report = "=" * 70 + "\n"
        report += "گزارش تحلیل عملکرد مالی\n"
        report += "=" * 70 + "\n\n"
        
        report += "1. معیارهای بازده:\n"
        report += "-" * 40 + "\n"
        report += f"بازده کل: {metrics['total_return']:+.2%}\n"
        report += f"بازده سالانه (CAGR): {metrics['annual_return']:+.2%}\n\n"
        
        report += "2. معیارهای ریسک-بازده:\n"
        report += "-" * 40 + "\n"
        report += f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        report += f"Sortino Ratio: {metrics['sortino_ratio']:.2f}\n"
        report += f"نوسان سالانه: {metrics['volatility']:.2%}\n"
        report += f"VaR (95%): {metrics['var_95']:.2%}\n"
        report += f"CVaR (95%): {metrics['cvar_95']:.2%}\n\n"
        
        if 'max_drawdown' in metrics:
            report += "3. معیارهای Drawdown:\n"
            report += "-" * 40 + "\n"
            report += f"حداکثر Drawdown: {metrics['max_drawdown']:.2%}\n"
            report += f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}\n"
            report += f"MAR Ratio: {metrics.get('mar_ratio', 0):.2f}\n"
            report += f"Ulcer Index: {metrics.get('ulcer_index', 0):.4f}\n\n"
        
        if trades is not None:
            report += "4. معیارهای معاملاتی:\n"
            report += "-" * 40 + "\n"
            report += f"نرخ برد: {metrics.get('win_rate', 0):.2%}\n"
            report += f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
            report += f"Expectancy: {metrics.get('expectancy', 0):.2f}\n"
            report += f"SQN: {metrics.get('sqn', 0):.2f}\n"
            report += f"نسبت سود/ضرر: {metrics.get('risk_reward_ratio', 0):.2f}\n"
            report += f"میانگین سود: {metrics.get('avg_win', 0):.2f}\n"
            report += f"میانگین ضرر: {metrics.get('avg_loss', 0):.2f}\n\n"
        
        if benchmark_returns is not None:
            report += "5. معیارهای نسبی:\n"
            report += "-" * 40 + "\n"
            report += f"Beta: {metrics.get('beta', 0):.2f}\n"
            report += f"Alpha: {metrics.get('alpha', 0):.2%}\n"
            report += f"Treynor Ratio: {metrics.get('treynor_ratio', 0):.2f}\n"
            report += f"Information Ratio: {metrics.get('information_ratio', 0):.2f}\n\n"
        
        report += "6. سایر معیارها:\n"
        report += "-" * 40 + "\n"
        report += f"Omega Ratio: {metrics.get('omega_ratio', 0):.2f}\n"
        report += f"Gain to Pain Ratio: {metrics.get('gain_to_pain_ratio', 0):.2f}\n"
        report += f"Tail Ratio: {metrics.get('tail_ratio', 0):.2f}\n"
        report += f"Common Sense Ratio: {metrics.get('common_sense_ratio', 0):.2f}\n\n"
        
        report += "7. آمار توصیفی:\n"
        report += "-" * 40 + "\n"
        report += f"چولگی: {metrics.get('skewness', 0):.2f}\n"
        report += f"کشیدگی: {metrics.get('kurtosis', 0):.2f}\n"
        report += f"نرمال بودن (p-value): {metrics.get('jarque_bera_pvalue', 0):.4f}\n\n"
        
        # ارزیابی عملکرد
        report += "8. ارزیابی عملکرد:\n"
        report += "-" * 40 + "\n"
        
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe > 1.5:
            report += "✅ Sharpe Ratio عالی\n"
        elif sharpe > 1.0:
            report += "🟡 Sharpe Ratio خوب\n"
        elif sharpe > 0.5:
            report += "🟠 Sharpe Ratio متوسط\n"
        else:
            report += "🔴 Sharpe Ratio ضعیف\n"
        
        if 'max_drawdown' in metrics:
            max_dd = abs(metrics['max_drawdown'])
            if max_dd < 0.1:
                report += "✅ Drawdown بسیار کم\n"
            elif max_dd < 0.2:
                report += "🟡 Drawdown قابل قبول\n"
            elif max_dd < 0.3:
                report += "🟠 Drawdown بالا\n"
            else:
                report += "🔴 Drawdown بسیار بالا\n"
        
        if 'profit_factor' in metrics:
            pf = metrics['profit_factor']
            if pf > 2.0:
                report += "✅ Profit Factor عالی\n"
            elif pf > 1.5:
                report += "🟡 Profit Factor خوب\n"
            elif pf > 1.0:
                report += "🟠 Profit Factor قابل قبول\n"
            else:
                report += "🔴 Profit Factor ضعیف\n"
        
        report += "\n" + "=" * 70 + "\n"
        
        return report

class ModelMetrics:
    """
    کلاس محاسبه معیارهای مدل‌های ML/DL
    """
    
    @staticmethod
    def calculate_classification_metrics(y_true, y_pred, y_pred_proba=None):
        """
        محاسبه معیارهای classification
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, confusion_matrix,
            classification_report
        )
        
        metrics = {}
        
        # معیارهای پایه
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # ROC-AUC اگر احتمالات موجود باشد
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = 0.0
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # معیارهای مالی
        try:
            # محاسبه precision/recall برای کلاس مثبت (خرید)
            tn, fp, fn, tp = cm.ravel()
            
            metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            # Matthews Correlation Coefficient
            mcc_numerator = (tp * tn) - (fp * fn)
            mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            metrics['mcc'] = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0
            
        except:
            pass
        
        return metrics
    
    @staticmethod
    def calculate_regression_metrics(y_true, y_pred):
        """
        محاسبه معیارهای regression
        """
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error,
            mean_absolute_percentage_error, r2_score
        )
        
        metrics = {}
        
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # معیارهای مالی خاص
        metrics['directional_accuracy'] = np.mean(
            np.sign(y_true[1:]) == np.sign(y_pred[1:])
        ) if len(y_true) > 1 else 0
        
        return metrics
    
    @staticmethod
    def calculate_feature_importance(model, feature_names):
        """
        محاسبه اهمیت ویژگی‌ها
        """
        importance_dict = {}
        
        if hasattr(model, 'feature_importances_'):
            # مدل‌های tree-based
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            for idx in indices:
                if idx < len(feature_names):
                    importance_dict[feature_names[idx]] = importances[idx]
        
        elif hasattr(model, 'coef_'):
            # مدل‌های linear
            if len(model.coef_.shape) == 1:
                importances = np.abs(model.coef_)
            else:
                importances = np.mean(np.abs(model.coef_), axis=0)
            
            indices = np.argsort(importances)[::-1]
            
            for idx in indices:
                if idx < len(feature_names):
                    importance_dict[feature_names[idx]] = importances[idx]
        
        return importance_dict
    
    @staticmethod
    def calculate_model_stability(predictions_list, method='std'):
        """
        محاسبه پایداری مدل
        """
        if len(predictions_list) == 0:
            return 0.0
        
        predictions_array = np.array(predictions_list)
        
        if method == 'std':
            stability = 1 / (1 + np.std(predictions_array, axis=0).mean())
        elif method == 'correlation':
            corr_matrix = np.corrcoef(predictions_array.T)
            stability = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return stability
    
    @staticmethod
    def calculate_forecast_metrics(actual_returns, predicted_returns, predicted_directions):
        """
        محاسبه معیارهای پیش‌بینی مالی
        """
        metrics = {}
        
        # دقت جهت
        metrics['direction_accuracy'] = np.mean(
            np.sign(actual_returns) == np.sign(predicted_returns)
        )
        
        # دقت باینری
        if predicted_directions is not None:
            actual_directions = (actual_returns > 0).astype(int)
            metrics['binary_accuracy'] = np.mean(
                actual_directions == predicted_directions
            )
        
        # بازده مبتنی بر سیگنال
        long_returns = actual_returns[predicted_returns > 0]
        short_returns = actual_returns[predicted_returns < 0]
        
        if len(long_returns) > 0:
            metrics['long_returns_mean'] = np.mean(long_returns)
            metrics['long_returns_std'] = np.std(long_returns)
            metrics['long_sharpe'] = metrics['long_returns_mean'] / metrics['long_returns_std'] * np.sqrt(252)
        
        if len(short_returns) > 0:
            metrics['short_returns_mean'] = np.mean(short_returns)
            metrics['short_returns_std'] = np.std(short_returns)
            metrics['short_sharpe'] = metrics['short_returns_mean'] / metrics['short_returns_std'] * np.sqrt(252)
        
        # Hit Rate
        positive_hits = np.sum((predicted_returns > 0) & (actual_returns > 0))
        negative_hits = np.sum((predicted_returns < 0) & (actual_returns < 0))
        total_predictions = len(actual_returns)
        
        metrics['hit_rate'] = (positive_hits + negative_hits) / total_predictions
        
        return metrics

class StatisticalTests:
    """
    کلاس انجام تست‌های آماری
    """
    
    @staticmethod
    def stationarity_test(series, test_type='adf'):
        """
        تست stationarity
        """
        from statsmodels.tsa.stattools import adfuller, kpss
        
        if test_type == 'adf':
            # Augmented Dickey-Fuller test
            result = adfuller(series.dropna())
            test_statistic = result[0]
            p_value = result[1]
            is_stationary = p_value < 0.05
            
            return {
                'test': 'ADF',
                'statistic': test_statistic,
                'p_value': p_value,
                'is_stationary': is_stationary,
                'critical_values': result[4]
            }
        
        elif test_type == 'kpss':
            # KPSS test
            result = kpss(series.dropna(), regression='c')
            test_statistic = result[0]
            p_value = result[1]
            is_stationary = p_value > 0.05
            
            return {
                'test': 'KPSS',
                'statistic': test_statistic,
                'p_value': p_value,
                'is_stationary': is_stationary,
                'critical_values': result[3]
            }
        
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    @staticmethod
    def normality_test(series, test_type='jarque_bera'):
        """
        تست نرمال بودن
        """
        if test_type == 'jarque_bera':
            statistic, p_value = stats.jarque_bera(series.dropna())
            is_normal = p_value > 0.05
            
            return {
                'test': 'Jarque-Bera',
                'statistic': statistic,
                'p_value': p_value,
                'is_normal': is_normal
            }
        
        elif test_type == 'shapiro':
            statistic, p_value = stats.shapiro(series.dropna())
            is_normal = p_value > 0.05
            
            return {
                'test': 'Shapiro-Wilk',
                'statistic': statistic,
                'p_value': p_value,
                'is_normal': is_normal
            }
        
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    @staticmethod
    def autocorrelation_test(series, lags=20):
        """
        تست autocorrelation
        """
        from statsmodels.tsa.stattools import acf, pacf
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
        acf_values = acf(series.dropna(), nlags=lags)
        pacf_values = pacf(series.dropna(), nlags=lags)
        
        # Ljung-Box test
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(series.dropna(), lags=[lags])
        
        return {
            'acf': acf_values,
            'pacf': pacf_values,
            'ljung_box': {
                'statistic': lb_test['lb_stat'].iloc[0],
                'p_value': lb_test['lb_pvalue'].iloc[0]
            }
        }
    
    @staticmethod
    def correlation_test(series1, series2, test_type='pearson'):
        """
        تست correlation
        """
        aligned_data = pd.concat([series1, series2], axis=1).dropna()
        
        if len(aligned_data) < 3:
            return None
        
        if test_type == 'pearson':
            corr, p_value = stats.pearsonr(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
        elif test_type == 'spearman':
            corr, p_value = stats.spearmanr(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
        elif test_type == 'kendall':
            corr, p_value = stats.kendalltau(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        return {
            'test': test_type,
            'correlation': corr,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def granger_causality_test(series1, series2, max_lag=10):
        """
        تست Granger Causality
        """
        from statsmodels.tsa.stattools import grangercausalitytests
        
        aligned_data = pd.concat([series1, series2], axis=1).dropna()
        
        if len(aligned_data) < max_lag * 2:
            return None
        
        try:
            test_result = grangercausalitytests(aligned_data, maxlag=max_lag, verbose=False)
            
            results = {}
            for lag in range(1, max_lag + 1):
                f_test = test_result[lag][0]['ssr_ftest']
                results[lag] = {
                    'f_statistic': f_test[0],
                    'p_value': f_test[1],
                    'significant': f_test[1] < 0.05
                }
            
            return results
        
        except Exception as e:
            print(f"Error in Granger test: {e}")
            return None
    
    @staticmethod
    def cointegration_test(series1, series2, test_type='eg'):
        """
        تست cointegration
        """
        from statsmodels.tsa.stattools import coint
        
        if test_type == 'eg':
            # Engle-Granger test
            statistic, p_value, critical_values = coint(series1, series2)
            
            return {
                'test': 'Engle-Granger',
                'statistic': statistic,
                'p_value': p_value,
                'critical_values': critical_values,
                'is_cointegrated': p_value < 0.05
            }
        
        else:
            raise ValueError(f"Unknown test type: {test_type}")

def calculate_portfolio_metrics(portfolio_returns, weights=None):
    """
    محاسبه معیارهای پرتفولیو
    """
    metrics = {}
    
    if weights is not None:
        # محاسبه بازده و ریسک پرتفولیو
        portfolio_return = np.sum(portfolio_returns.mean() * weights)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(portfolio_returns.cov(), weights)))
        
        metrics['portfolio_return'] = portfolio_return
        metrics['portfolio_volatility'] = portfolio_vol
        metrics['portfolio_sharpe'] = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
    
    # معیارهای تنوع
    if len(portfolio_returns.columns) > 1:
        corr_matrix = portfolio_returns.corr()
        avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        metrics['avg_correlation'] = avg_correlation
        metrics['diversification_ratio'] = 1 - avg_correlation
    
    return metrics

# توابع کمکی
def format_percentage(value):
    """فرمت کردن درصد"""
    return f"{value:.2%}"

def format_currency(value):
    """فرمت کردن پول"""
    return f"${value:,.2f}"

def format_metric(value, metric_type='float'):
    """فرمت کردن معیار"""
    if metric_type == 'percentage':
        return format_percentage(value)
    elif metric_type == 'currency':
        return format_currency(value)
    elif metric_type == 'ratio':
        return f"{value:.2f}"
    else:
        return f"{value:.4f}"

if __name__ == "__main__":
    print("ماژول معیارهای مالی و آماری")