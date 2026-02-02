"""
Backtesting سیستم تریدینگ با Backtrader
"""

import backtrader as bt
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings
from trading.strategies.ml_strategy import MLStrategy, HybridStrategy
from trading.risk_management.risk_metrics import RiskMetrics

class ForexBacktester:
    """کلاس اصلی Backtesting برای فارکس"""
    
    def __init__(self):
        self.data_dir = Path(settings.PROCESSED_DATA_DIR)
        self.models_dir = Path(settings.MODELS_DIR)
        self.results_dir = Path("trading/backtesting/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_data_for_backtrader(self, symbol):
        """آماده‌سازی داده‌ها برای Backtrader"""
        data_path = self.data_dir / f"{symbol}_processed.csv"
        
        if not data_path.exists():
            print(f"فایل داده برای {symbol} یافت نشد")
            return None
        
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # انتخاب ستون‌های مورد نیاز
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        df = df[required_cols]
        
        # اطمینان از عدم وجود NaN
        df = df.dropna()
        
        print(f"داده‌های {symbol} بارگذاری شد: {len(df)} رکورد")
        return df
    
    def run_backtest(self, symbol, strategy_class=MLStrategy, strategy_params=None, 
                    plot_results=True, save_results=True):
        """
        اجرای بکتست برای یک جفت ارز
        
        Args:
            symbol: نماد جفت ارز
            strategy_class: کلاس استراتژی
            strategy_params: پارامترهای استراتژی
            plot_results: نمایش نمودارها
            save_results: ذخیره نتایج
        
        Returns:
            نتایج بکتست
        """
        print(f"\n{'='*60}")
        print(f"شروع Backtesting برای {symbol}")
        print(f"{'='*60}")
        
        # آماده‌سازی داده‌ها
        df = self.prepare_data_for_backtrader(symbol)
        if df is None:
            return None
        
        # ایجاد cerebro engine
        cerebro = bt.Cerebro()
        
        # اضافه کردن داده‌ها
        data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1
        )
        cerebro.adddata(data)
        
        # تنظیم پارامترهای استراتژی
        if strategy_params is None:
            strategy_params = {
                'symbol': symbol,
                'position_size_pct': settings.POSITION_SIZE_PCT,
                'stop_loss_pct': settings.STOP_LOSS_PCT,
                'take_profit_pct': settings.TAKE_PROFIT_PCT,
                'use_ensemble': True,
            }
        
        # اضافه کردن استراتژی
        cerebro.addstrategy(strategy_class, **strategy_params)
        
        # تنظیمات کارگزار
        cerebro.broker.setcash(settings.INITIAL_CAPITAL)
        cerebro.broker.setcommission(commission=settings.COMMISSION)
        
        # اضافه کردن آنالایزرها
        self.add_analyzers(cerebro)
        
        # اجرای بکتست
        print(f'\nسرمایه اولیه: ${cerebro.broker.getvalue():,.2f}')
        
        results = cerebro.run()
        strat = results[0]
        
        print(f'سرمایه نهایی: ${cerebro.broker.getvalue():,.2f}')
        
        # جمع‌آوری نتایج
        performance_metrics = self.collect_results(strat, symbol)
        
        # ذخیره نتایج
        if save_results:
            self.save_results(performance_metrics, symbol, strat)
        
        # نمایش نمودارها
        if plot_results:
            cerebro.plot(style='candlestick', volume=False)
        
        return performance_metrics
    
    def add_analyzers(self, cerebro):
        """اضافه کردن آنالایزرهای Backtrader"""
        # Sharpe Ratio
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe',
                           riskfreerate=0.02, annualize=True)
        
        # Returns
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # DrawDown
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        
        # Trade Analyzer
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # Time Return
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        
        # VWR (Volume Weighted Return)
        cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')
        
        # SQN (System Quality Number)
        cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        
        # PeriodStats
        cerebro.addanalyzer(bt.analyzers.PeriodStats, _name='periodstats')
        
        # LogReturnsRolling
        cerebro.addanalyzer(bt.analyzers.LogReturnsRolling, _name='logreturns')
        
        # Transactions
        cerebro.addanalyzer(bt.analyzers.Transactions, _name='transactions')
        
        # PyFolio (برای تحلیل پیشرفته)
        try:
            cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
        except:
            pass
    
    def collect_results(self, strategy, symbol):
        """جمع‌آوری نتایج از آنالایزرها"""
        metrics = {
            'symbol': symbol,
            'initial_capital': settings.INITIAL_CAPITAL,
            'final_capital': strategy.broker.getvalue(),
            'total_return': 0,
            'annual_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
        }
        
        # Sharpe Ratio
        sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
        if 'sharperatio' in sharpe_analysis:
            metrics['sharpe_ratio'] = sharpe_analysis['sharperatio']
        
        # Returns
        returns_analysis = strategy.analyzers.returns.get_analysis()
        if 'rtot' in returns_analysis:
            metrics['total_return'] = returns_analysis['rtot']
        if 'rnorm100' in returns_analysis:
            metrics['annual_return'] = returns_analysis['rnorm100'] / 100
        
        # DrawDown
        drawdown_analysis = strategy.analyzers.drawdown.get_analysis()
        if 'max' in drawdown_analysis:
            metrics['max_drawdown'] = drawdown_analysis['max']['drawdown']
            metrics['max_drawdown_money'] = drawdown_analysis['max']['moneydown']
            metrics['max_drawdown_length'] = drawdown_analysis['max']['len']
        
        # Trade Analysis
        trades_analysis = strategy.analyzers.trades.get_analysis()
        
        if 'total' in trades_analysis and 'total' in trades_analysis['total']:
            metrics['total_trades'] = trades_analysis['total']['total']
            
            if 'won' in trades_analysis and 'total' in trades_analysis['won']:
                metrics['winning_trades'] = trades_analysis['won']['total']
                metrics['win_rate'] = metrics['winning_trades'] / max(1, metrics['total_trades'])
            
            if 'lost' in trades_analysis and 'total' in trades_analysis['lost']:
                metrics['losing_trades'] = trades_analysis['lost']['total']
            
            # Profit Factor
            if 'won' in trades_analysis and 'pnl' in trades_analysis['won'] and 'total' in trades_analysis['won']['pnl']:
                total_won = trades_analysis['won']['pnl']['total']
                total_lost = abs(trades_analysis['lost']['pnl']['total']) if 'lost' in trades_analysis else 0
                
                if total_lost > 0:
                    metrics['profit_factor'] = total_won / total_lost
        
        # Time Return برای محاسبه Sortino و سایر معیارها
        try:
            timereturn = strategy.analyzers.timereturn.get_analysis()
            returns_series = pd.Series(timereturn)
            
            # محاسبه معیارهای ریسک اضافی
            risk_metrics = RiskMetrics(returns_series)
            
            metrics['sortino_ratio'] = risk_metrics.calculate_sortino_ratio()
            metrics['calmar_ratio'] = risk_metrics.calculate_calmar_ratio()
            metrics['var_95'] = risk_metrics.calculate_var(confidence_level=0.95)
            metrics['volatility'] = risk_metrics.calculate_volatility(annualized=True)
            
        except:
            pass
        
        # SQN
        try:
            sqn_analysis = strategy.analyzers.sqn.get_analysis()
            if 'sqn' in sqn_analysis:
                metrics['sqn'] = sqn_analysis['sqn']
        except:
            metrics['sqn'] = 0
        
        # محاسبه بازده کل
        metrics['total_return_pct'] = (metrics['final_capital'] - metrics['initial_capital']) / metrics['initial_capital'] * 100
        
        return metrics
    
    def save_results(self, metrics, symbol, strategy):
        """ذخیره نتایج بکتست"""
        # ذخیره در CSV
        results_df = pd.DataFrame([metrics])
        results_path = self.results_dir / f"{symbol}_backtest_results.csv"
        results_df.to_csv(results_path)
        
        # ذخیره در Pickle
        pickle_path = self.results_dir / f"{symbol}_backtest_results.pkl"
        results_df.to_pickle(pickle_path)
        
        # تولید گزارش متنی
        report = self.generate_report(metrics, symbol)
        report_path = self.results_dir / f"{symbol}_backtest_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nنتایج بکتست برای {symbol}:")
        print(report)
        print(f"\nنتایج در {results_path} ذخیره شد")
        
        return results_df
    
    def generate_report(self, metrics, symbol):
        """تولید گزارش متنی از نتایج"""
        report = "=" * 70 + "\n"
        report += f"گزارش Backtesting - {symbol}\n"
        report += "=" * 70 + "\n\n"
        
        report += "خلاصه عملکرد:\n"
        report += "-" * 40 + "\n"
        report += f"سرمایه اولیه: ${metrics['initial_capital']:,.2f}\n"
        report += f"سرمایه نهایی: ${metrics['final_capital']:,.2f}\n"
        report += f"بازده کل: {metrics['total_return_pct']:.2f}%\n"
        report += f"بازده سالانه: {metrics['annual_return']:.2%}\n\n"
        
        report += "معیارهای ریسک-بازده:\n"
        report += "-" * 40 + "\n"
        report += f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        report += f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}\n"
        report += f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}\n"
        report += f"نوسان سالانه: {metrics.get('volatility', 0):.2%}\n"
        report += f"VaR (95%): {metrics.get('var_95', 0):.2%}\n\n"
        
        report += "معیارهای Drawdown:\n"
        report += "-" * 40 + "\n"
        report += f"حداکثر Drawdown: {metrics['max_drawdown']:.2%}\n"
        report += f"حداکثر Drawdown (ارزش): ${metrics.get('max_drawdown_money', 0):,.2f}\n"
        report += f"مدت Drawdown: {metrics.get('max_drawdown_length', 0)} دوره\n\n"
        
        report += "آمار معاملات:\n"
        report += "-" * 40 + "\n"
        report += f"تعداد کل معاملات: {metrics['total_trades']}\n"
        report += f"معاملات برنده: {metrics['winning_trades']}\n"
        report += f"معاملات بازنده: {metrics['losing_trades']}\n"
        report += f"نرخ برد: {metrics['win_rate']:.2%}\n"
        report += f"ضریب سوددهی: {metrics['profit_factor']:.2f}\n"
        report += f"SQN: {metrics.get('sqn', 0):.2f}\n\n"
        
        # ارزیابی عملکرد
        report += "ارزیابی عملکرد:\n"
        report += "-" * 40 + "\n"
        
        if metrics['sharpe_ratio'] > 1.5:
            report += "✅ Sharpe Ratio عالی (بیشتر از 1.5)\n"
        elif metrics['sharpe_ratio'] > 1.0:
            report += "🟡 Sharpe Ratio خوب (بین 1.0 و 1.5)\n"
        else:
            report += "🔴 Sharpe Ratio نیاز به بهبود (کمتر از 1.0)\n"
        
        if metrics['max_drawdown'] < 0.1:
            report += "✅ Drawdown بسیار کم (کمتر از 10%)\n"
        elif metrics['max_drawdown'] < 0.2:
            report += "🟡 Drawdown قابل قبول (بین 10% و 20%)\n"
        else:
            report += "🔴 Drawdown بالا (بیشتر از 20%)\n"
        
        if metrics['win_rate'] > 0.55:
            report += "✅ نرخ برد عالی (بیشتر از 55%)\n"
        elif metrics['win_rate'] > 0.45:
            report += "🟡 نرخ برد قابل قبول (بین 45% و 55%)\n"
        else:
            report += "🔴 نرخ برد پایین (کمتر از 45%)\n"
        
        if metrics['profit_factor'] > 1.5:
            report += "✅ Profit Factor عالی (بیشتر از 1.5)\n"
        elif metrics['profit_factor'] > 1.0:
            report += "🟡 Profit Factor قابل قبول (بین 1.0 و 1.5)\n"
        else:
            report += "🔴 Profit Factor ضعیف (کمتر از 1.0)\n"
        
        report += "\n" + "=" * 70 + "\n"
        
        return report
    
    def run_comparative_backtest(self, symbols=None, strategies=None):
        """
        اجرای بکتست مقایسه‌ای برای چندین جفت ارز و استراتژی
        
        Args:
            symbols: لیست نمادها
            strategies: دیکشنری استراتژی‌ها
        
        Returns:
            نتایج مقایسه‌ای
        """
        if symbols is None:
            symbols = settings.FOREX_PAIRS[:3]
        
        if strategies is None:
            strategies = {
                'ML Strategy': MLStrategy,
                'Hybrid Strategy': HybridStrategy,
            }
        
        all_results = {}
        
        for symbol in symbols:
            print(f"\n{'='*60}")
            print(f"Backtesting مقایسه‌ای برای {symbol}")
            print(f"{'='*60}")
            
            symbol_results = {}
            
            for strategy_name, strategy_class in strategies.items():
                print(f"\nاستراتژی: {strategy_name}")
                
                try:
                    results = self.run_backtest(
                        symbol=symbol,
                        strategy_class=strategy_class,
                        strategy_params={'symbol': symbol},
                        plot_results=False,
                        save_results=False
                    )
                    
                    if results:
                        symbol_results[strategy_name] = results
                        print(f"   بازده: {results['total_return_pct']:.2f}%")
                        print(f"   Sharpe: {results['sharpe_ratio']:.2f}")
                        print(f"   Max DD: {results['max_drawdown']:.2%}")
                    
                except Exception as e:
                    print(f"   خطا: {e}")
            
            all_results[symbol] = symbol_results
        
        # ایجاد جدول مقایسه
        comparison_df = self.create_comparison_table(all_results)
        
        # ذخیره نتایج مقایسه
        comparison_path = self.results_dir / "comparative_results.csv"
        comparison_df.to_csv(comparison_path)
        
        print(f"\n{'='*60}")
        print("نتایج مقایسه‌ای:")
        print('='*60)
        print(comparison_df.to_string())
        
        return all_results
    
    def create_comparison_table(self, all_results):
        """ایجاد جدول مقایسه نتایج"""
        comparison_data = []
        
        for symbol, strategies in all_results.items():
            for strategy_name, results in strategies.items():
                row = {
                    'Symbol': symbol,
                    'Strategy': strategy_name,
                    'Total Return %': results['total_return_pct'],
                    'Annual Return %': results['annual_return'] * 100,
                    'Sharpe Ratio': results['sharpe_ratio'],
                    'Max Drawdown %': results['max_drawdown'] * 100,
                    'Win Rate %': results['win_rate'] * 100,
                    'Profit Factor': results['profit_factor'],
                    'Total Trades': results['total_trades'],
                    'Final Capital': results['final_capital'],
                }
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # مرتب‌سازی بر اساس بازده کل
        comparison_df = comparison_df.sort_values('Total Return %', ascending=False)
        
        return comparison_df
    
    def run_walk_forward_analysis(self, symbol, strategy_class=MLStrategy, 
                                 train_size=0.7, n_splits=5):
        """
        اجرای تحلیل Walk-Forward
        
        Args:
            symbol: نماد جفت ارز
            strategy_class: کلاس استراتژی
            train_size: سایز داده آموزش
            n_splits: تعداد تقسیم‌ها
        
        Returns:
            نتایج Walk-Forward
        """
        print(f"\n{'='*60}")
        print(f"تحلیل Walk-Forward برای {symbol}")
        print(f"{'='*60}")
        
        # بارگذاری داده‌ها
        df = self.prepare_data_for_backtrader(symbol)
        if df is None:
            return None
        
        total_samples = len(df)
        train_samples = int(total_samples * train_size)
        test_samples = total_samples - train_samples
        
        print(f"کل نمونه‌ها: {total_samples}")
        print(f"نمونه‌های آموزش: {train_samples}")
        print(f"نمونه‌های تست: {test_samples}")
        
        wf_results = []
        
        for i in range(n_splits):
            # محاسبه بازه‌های آموزش و تست
            train_start = i * test_samples
            train_end = train_start + train_samples
            test_start = train_end
            test_end = min(test_start + test_samples, total_samples)
            
            # تقسیم داده‌ها
            train_data = df.iloc[train_start:train_end]
            test_data = df.iloc[test_start:test_end]
            
            print(f"\nSplit {i+1}/{n_splits}:")
            print(f"  آموزش: {train_start} تا {train_end}")
            print(f"  تست: {test_start} تا {test_end}")
            
            # آموزش مدل روی داده‌های آموزش (در واقعیت باید مدل را دوباره آموزش دهیم)
            # برای سادگی، از مدل از پیش آموزش دیده استفاده می‌کنیم
            
            # اجرای بکتست روی داده‌های تست
            try:
                # ایجاد cerebro جدید
                cerebro = bt.Cerebro()
                
                # اضافه کردن داده‌های تست
                data = bt.feeds.PandasData(
                    dataname=test_data,
                    datetime=None,
                    open='open',
                    high='high',
                    low='low',
                    close='close',
                    volume='volume',
                    openinterest=-1
                )
                cerebro.adddata(data)
                
                # اضافه کردن استراتژی
                cerebro.addstrategy(strategy_class, symbol=symbol)
                
                # تنظیمات کارگزار
                cerebro.broker.setcash(settings.INITIAL_CAPITAL)
                cerebro.broker.setcommission(commission=settings.COMMISSION)
                
                # اضافه کردن آنالایزرها
                self.add_analyzers(cerebro)
                
                # اجرای بکتست
                results = cerebro.run()
                strat = results[0]
                
                # جمع‌آوری نتایج
                metrics = self.collect_results(strat, symbol)
                metrics['split'] = i + 1
                metrics['train_period'] = f"{train_data.index[0].date()} تا {train_data.index[-1].date()}"
                metrics['test_period'] = f"{test_data.index[0].date()} تا {test_data.index[-1].date()}"
                
                wf_results.append(metrics)
                
                print(f"  بازده تست: {metrics['total_return_pct']:.2f}%")
                print(f"  Sharpe تست: {metrics['sharpe_ratio']:.2f}")
                
            except Exception as e:
                print(f"  خطا در Split {i+1}: {e}")
        
        # جمع‌آوری و تحلیل نتایج Walk-Forward
        if wf_results:
            wf_df = pd.DataFrame(wf_results)
            
            # محاسبه میانگین‌ها
            avg_metrics = {
                'avg_return': wf_df['total_return_pct'].mean(),
                'std_return': wf_df['total_return_pct'].std(),
                'avg_sharpe': wf_df['sharpe_ratio'].mean(),
                'std_sharpe': wf_df['sharpe_ratio'].std(),
                'avg_max_dd': wf_df['max_drawdown'].mean(),
                'avg_win_rate': wf_df['win_rate'].mean(),
                'consistency': len(wf_df[wf_df['total_return_pct'] > 0]) / len(wf_df),
            }
            
            print(f"\nنتایج Walk-Forward Analysis:")
            print(f"  میانگین بازده: {avg_metrics['avg_return']:.2f}%")
            print(f"  انحراف معیار بازده: {avg_metrics['std_return']:.2f}%")
            print(f"  میانگین Sharpe: {avg_metrics['avg_sharpe']:.2f}")
            print(f"  انحراف معیار Sharpe: {avg_metrics['std_sharpe']:.2f}")
            print(f"  نرخ ثبات: {avg_metrics['consistency']:.2%}")
            
            # ذخیره نتایج
            wf_path = self.results_dir / f"{symbol}_walk_forward_results.csv"
            wf_df.to_csv(wf_path)
            
            avg_path = self.results_dir / f"{symbol}_walk_forward_avg.csv"
            pd.DataFrame([avg_metrics]).to_csv(avg_path)
            
            return wf_df, avg_metrics
        
        return None

if __name__ == "__main__":
    # نمونه استفاده از بکتستر
    backtester = ForexBacktester()
    
    # اجرای بکتست برای یک جفت ارز
    results = backtester.run_backtest(
        symbol="EURUSD",
        strategy_class=MLStrategy,
        plot_results=True,
        save_results=True
    )
    
    # اجرای تحلیل مقایسه‌ای
    # comparative_results = backtester.run_comparative_backtest()
    
    # اجرای تحلیل Walk-Forward
    # wf_results = backtester.run_walk_forward_analysis("EURUSD")