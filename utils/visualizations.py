"""
توابع ویژوال‌سازی داده‌های مالی
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# تنظیمات matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
sns.set_palette("husl")

class FinancialVisualizations:
    """
    کلاس ویژوال‌سازی داده‌های مالی
    """
    
    @staticmethod
    def plot_price_chart(df: pd.DataFrame, symbol: str = '', 
                        indicators: List[str] = None,
                        volume: bool = True) -> go.Figure:
        """
        رسم نمودار قیمت با اندیکاتورها
        
        Args:
            df: DataFrame حاوی داده‌ها
            symbol: نام نماد
            indicators: لیست اندیکاتورها برای نمایش
            volume: نمایش حجم
        
        Returns:
            نمودار Plotly
        """
        # ایجاد subplot
        if volume and 'volume' in df.columns:
            rows = 2
            row_heights = [0.7, 0.3]
            subplot_titles = (f'{symbol} Price', 'Volume')
        else:
            rows = 1
            row_heights = [1.0]
            subplot_titles = (f'{symbol} Price',)
        
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=subplot_titles,
            row_heights=row_heights
        )
        
        # نمودار کندل‌استیک
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # اضافه کردن اندیکاتورها
        if indicators:
            for indicator in indicators:
                if indicator in df.columns:
                    if indicator.startswith('SMA_') or indicator.startswith('EMA_'):
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df[indicator],
                                name=indicator,
                                line=dict(width=1)
                            ),
                            row=1, col=1
                        )
        
        # نمودار حجم
        if volume and 'volume' in df.columns:
            # رنگ‌آمیزی حجم بر اساس قیمت
            colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' 
                     for i in range(len(df))]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name='Volume',
                    marker_color=colors,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=f'{symbol} Price Chart',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            height=600,
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def plot_technical_indicators(df: pd.DataFrame, indicators_config: Dict = None) -> go.Figure:
        """
        رسم اندیکاتورهای تکنیکال
        """
        if indicators_config is None:
            indicators_config = {
                'RSI': {'row': 1, 'range': [0, 100], 'lines': [30, 70]},
                'MACD': {'row': 2, 'components': ['MACD', 'MACD_Signal']},
                'Stoch_K': {'row': 3, 'range': [0, 100], 'lines': [20, 80]},
                'BB_Upper': {'row': 4, 'components': ['BB_Upper', 'BB_Middle', 'BB_Lower']}
            }
        
        rows = len(indicators_config)
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=list(indicators_config.keys())
        )
        
        for i, (indicator_name, config) in enumerate(indicators_config.items(), 1):
            row = config.get('row', i)
            
            if 'components' in config:
                # اندیکاتور با چند component
                for component in config['components']:
                    if component in df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df[component],
                                name=component,
                                line=dict(width=1)
                            ),
                            row=row, col=1
                        )
            else:
                # اندیکاتور تک component
                if indicator_name in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[indicator_name],
                            name=indicator_name,
                            line=dict(width=1)
                        ),
                        row=row, col=1
                    )
                    
                    # خطوط اضافی
                    if 'lines' in config:
                        for line_value in config['lines']:
                            fig.add_hline(
                                y=line_value,
                                line_dash="dash",
                                line_color="gray",
                                row=row, col=1
                            )
            
            # تنظیم range
            if 'range' in config:
                fig.update_yaxes(range=config['range'], row=row, col=1)
        
        fig.update_layout(
            title='Technical Indicators',
            height=300 * rows,
            showlegend=True,
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def plot_returns_distribution(returns: pd.Series, bins: int = 50) -> go.Figure:
        """
        رسم توزیع بازده‌ها
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Distribution of Returns', 'QQ Plot'),
            vertical_spacing=0.1
        )
        
        # هیستوگرام
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=bins,
                name='Returns',
                histnorm='probability',
                marker_color='steelblue',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # اضافه کردن خط نرمال
        x_norm = np.linspace(returns.min(), returns.max(), 100)
        y_norm = stats.norm.pdf(x_norm, returns.mean(), returns.std())
        
        fig.add_trace(
            go.Scatter(
                x=x_norm,
                y=y_norm,
                name='Normal Distribution',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # QQ Plot
        from scipy import stats
        qq = stats.probplot(returns.dropna(), dist="norm")
        x_theoretical = qq[0][0]
        y_sample = qq[0][1]
        
        fig.add_trace(
            go.Scatter(
                x=x_theoretical,
                y=y_sample,
                mode='markers',
                name='QQ Plot',
                marker=dict(color='blue', size=6)
            ),
            row=2, col=1
        )
        
        # خط 45 درجه
        min_val = min(x_theoretical.min(), y_sample.min())
        max_val = max(x_theoretical.max(), y_sample.max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='45° Line',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Returns Distribution Analysis',
            height=600,
            showlegend=True,
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame, method: str = 'pearson') -> go.Figure:
        """
        رسم ماتریس همبستگی
        """
        # محاسبه همبستگی
        corr_matrix = df.corr(method=method)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f'Correlation Matrix ({method.title()})',
            height=600,
            width=800,
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def plot_feature_importance(importance_dict: Dict, top_n: int = 20) -> go.Figure:
        """
        رسم اهمیت ویژگی‌ها
        """
        # مرتب‌سازی و انتخاب top_n
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_items[:top_n]
        
        features = [item[0] for item in top_features]
        importances = [item[1] for item in top_features]
        
        fig = go.Figure(data=go.Bar(
            x=importances,
            y=features,
            orientation='h',
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importances',
            xaxis_title='Importance',
            yaxis_title='Features',
            height=max(400, len(features) * 20),
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def plot_equity_curve(equity_series: pd.Series, 
                         drawdown_series: pd.Series = None) -> go.Figure:
        """
        رسم منحنی سرمایه و drawdown
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Equity Curve', 'Drawdown'),
            vertical_spacing=0.05
        )
        
        # منحنی سرمایه
        fig.add_trace(
            go.Scatter(
                x=equity_series.index,
                y=equity_series.values,
                mode='lines',
                name='Equity',
                line=dict(color='green', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.1)'
            ),
            row=1, col=1
        )
        
        # Drawdown
        if drawdown_series is not None:
            fig.add_trace(
                go.Scatter(
                    x=drawdown_series.index,
                    y=drawdown_series.values,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='red', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 0, 0.1)'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title='Equity Curve Analysis',
            height=600,
            showlegend=True,
            template='plotly_dark'
        )
        
        fig.update_yaxes(title_text='Account Value', row=1, col=1)
        fig.update_yaxes(title_text='Drawdown (%)', row=2, col=1)
        
        return fig
    
    @staticmethod
    def plot_trades_on_chart(df: pd.DataFrame, trades: pd.DataFrame) -> go.Figure:
        """
        رسم معاملات روی نمودار قیمت
        """
        fig = go.Figure()
        
        # نمودار کندل‌استیک
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ))
        
        # معاملات خرید
        buy_trades = trades[trades['type'] == 'BUY']
        if not buy_trades.empty:
            fig.add_trace(go.Scatter(
                x=buy_trades['entry_time'],
                y=buy_trades['entry_price'],
                mode='markers',
                name='Buy',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='green',
                    line=dict(width=2, color='darkgreen')
                )
            ))
        
        # معاملات فروش
        sell_trades = trades[trades['type'] == 'SELL']
        if not sell_trades.empty:
            fig.add_trace(go.Scatter(
                x=sell_trades['entry_time'],
                y=sell_trades['entry_price'],
                mode='markers',
                name='Sell',
                marker=dict(
                    symbol='triangle-down',
                    size=15,
                    color='red',
                    line=dict(width=2, color='darkred')
                )
            ))
        
        # خطوط حد ضرر و حد سود
        for _, trade in trades.iterrows():
            if 'stop_loss' in trade and pd.notna(trade['stop_loss']):
                fig.add_trace(go.Scatter(
                    x=[trade['entry_time'], trade['exit_time']],
                    y=[trade['stop_loss'], trade['stop_loss']],
                    mode='lines',
                    line=dict(color='red', width=1, dash='dash'),
                    showlegend=False
                ))
            
            if 'take_profit' in trade and pd.notna(trade['take_profit']):
                fig.add_trace(go.Scatter(
                    x=[trade['entry_time'], trade['exit_time']],
                    y=[trade['take_profit'], trade['take_profit']],
                    mode='lines',
                    line=dict(color='green', width=1, dash='dash'),
                    showlegend=False
                ))
        
        fig.update_layout(
            title='Trades on Price Chart',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            height=600,
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def plot_returns_scatter(actual_returns: pd.Series, 
                            predicted_returns: pd.Series) -> go.Figure:
        """
        رسم scatter plot بازده‌های واقعی در مقابل پیش‌بینی شده
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=predicted_returns,
            y=actual_returns,
            mode='markers',
            name='Returns',
            marker=dict(
                size=8,
                color=actual_returns,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Actual Return")
            )
        ))
        
        # خط 45 درجه
        min_val = min(predicted_returns.min(), actual_returns.min())
        max_val = max(predicted_returns.max(), actual_returns.max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        # خط رگرسیون
        from sklearn.linear_model import LinearRegression
        X = predicted_returns.values.reshape(-1, 1)
        y = actual_returns.values
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        
        fig.add_trace(go.Scatter(
            x=predicted_returns,
            y=y_pred,
            mode='lines',
            name='Regression Line',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Actual vs Predicted Returns',
            xaxis_title='Predicted Returns',
            yaxis_title='Actual Returns',
            height=500,
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, classes: List[str] = None) -> go.Figure:
        """
        رسم confusion matrix
        """
        if classes is None:
            classes = ['Down', 'Up']
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16}
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=400,
            width=500,
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, 
                      roc_auc: float) -> go.Figure:
        """
        رسم ROC curve
        """
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC (AUC = {roc_auc:.3f})',
            line=dict(color='darkorange', width=2)
        ))
        
        # خط تصادفی
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='navy', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            height=500,
            width=500,
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def plot_precision_recall_curve(precision: np.ndarray, 
                                   recall: np.ndarray, 
                                   average_precision: float) -> go.Figure:
        """
        رسم Precision-Recall curve
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'AP = {average_precision:.3f}',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=500,
            width=500,
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def plot_learning_curve(train_scores: np.ndarray, 
                           val_scores: np.ndarray, 
                           train_sizes: np.ndarray) -> go.Figure:
        """
        رسم learning curve
        """
        fig = go.Figure()
        
        # Train scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=np.mean(train_scores, axis=1),
            mode='lines+markers',
            name='Training Score',
            line=dict(color='blue', width=2),
            error_y=dict(
                type='data',
                array=np.std(train_scores, axis=1),
                visible=True
            )
        ))
        
        # Validation scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=np.mean(val_scores, axis=1),
            mode='lines+markers',
            name='Validation Score',
            line=dict(color='green', width=2),
            error_y=dict(
                type='data',
                array=np.std(val_scores, axis=1),
                visible=True
            )
        ))
        
        fig.update_layout(
            title='Learning Curve',
            xaxis_title='Training Size',
            yaxis_title='Score',
            height=500,
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def plot_time_series_decomposition(series: pd.Series, 
                                      model: str = 'additive',
                                      period: int = None) -> go.Figure:
        """
        رسم تجزیه سری زمانی
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        if period is None:
            # حدس period بر اساس داده
            if len(series) > 365:
                period = 365  # روزانه
            elif len(series) > 52:
                period = 52   # هفتگی
            elif len(series) > 12:
                period = 12   # ماهانه
            else:
                period = 7    # پیش‌فرض
        
        decomposition = seasonal_decompose(series.dropna(), model=model, period=period)
        
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.05
        )
        
        # Original
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode='lines',
                name='Original',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # Trend
        fig.add_trace(
            go.Scatter(
                x=decomposition.trend.index,
                y=decomposition.trend.values,
                mode='lines',
                name='Trend',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        # Seasonal
        fig.add_trace(
            go.Scatter(
                x=decomposition.seasonal.index,
                y=decomposition.seasonal.values,
                mode='lines',
                name='Seasonal',
                line=dict(color='green', width=1)
            ),
            row=3, col=1
        )
        
        # Residual
        fig.add_trace(
            go.Scatter(
                x=decomposition.resid.index,
                y=decomposition.resid.values,
                mode='lines',
                name='Residual',
                line=dict(color='orange', width=1)
            ),
            row=4, col=1
        )
        
        fig.update_layout(
            title='Time Series Decomposition',
            height=800,
            showlegend=False,
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def plot_risk_reward_scatter(trades: pd.DataFrame) -> go.Figure:
        """
        رسم scatter plot ریسک در مقابل بازده
        """
        fig = go.Figure()
        
        # معاملات برنده
        winning_trades = trades[trades['pnl'] > 0]
        if not winning_trades.empty:
            fig.add_trace(go.Scatter(
                x=winning_trades['risk'],
                y=winning_trades['reward'],
                mode='markers',
                name='Winning Trades',
                marker=dict(
                    symbol='circle',
                    size=10,
                    color='green',
                    opacity=0.7
                )
            ))
        
        # معاملات بازنده
        losing_trades = trades[trades['pnl'] < 0]
        if not losing_trades.empty:
            fig.add_trace(go.Scatter(
                x=losing_trades['risk'],
                y=losing_trades['reward'],
                mode='markers',
                name='Losing Trades',
                marker=dict(
                    symbol='circle',
                    size=10,
                    color='red',
                    opacity=0.7
                )
            ))
        
        # خط 1:1 (ریسک = بازده)
        max_risk = trades['risk'].max() if 'risk' in trades.columns else 1
        fig.add_trace(go.Scatter(
            x=[0, max_risk],
            y=[0, max_risk],
            mode='lines',
            name='Risk = Reward',
            line=dict(color='gray', dash='dash')
        ))
        
        fig.update_layout(
            title='Risk vs Reward Scatter Plot',
            xaxis_title='Risk',
            yaxis_title='Reward',
            height=500,
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def plot_monthly_returns_heatmap(returns: pd.Series) -> go.Figure:
        """
        رسم heatmap بازده‌های ماهانه
        """
        # تبدیل به DataFrame ماهانه
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # ایجاد DataFrame برای heatmap
        monthly_returns_df = monthly_returns.to_frame('return')
        monthly_returns_df['year'] = monthly_returns_df.index.year
        monthly_returns_df['month'] = monthly_returns_df.index.month_name()
        
        # pivot table
        heatmap_data = monthly_returns_df.pivot_table(
            values='return',
            index='year',
            columns='month',
            aggfunc='mean'
        )
        
        # ترتیب ماه‌ها
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        heatmap_data = heatmap_data[month_order]
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values * 100,  # تبدیل به درصد
            x=heatmap_data.columns,
            y=heatmap_data.index.astype(str),
            colorscale='RdYlGn',
            text=np.round(heatmap_data.values * 100, 2),
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Monthly Returns Heatmap (%)',
            height=400,
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def plot_daily_returns_heatmap(returns: pd.Series) -> go.Figure:
        """
        رسم heatmap بازده‌های روزانه
        """
        # تبدیل به DataFrame روزانه
        daily_returns = returns.to_frame('return')
        daily_returns['year'] = daily_returns.index.year
        daily_returns['month'] = daily_returns.index.month
        daily_returns['day'] = daily_returns.index.day
        
        # ایجاد pivot table
        pivot_data = daily_returns.pivot_table(
            values='return',
            index='day',
            columns=['year', 'month'],
            aggfunc='mean'
        )
        
        # میانگین ماهانه
        monthly_avg = pivot_data.mean(axis=0).unstack()
        
        fig = go.Figure(data=go.Heatmap(
            z=monthly_avg.values * 100,
            x=[f"{y}-{m:02d}" for y, m in monthly_avg.columns],
            y=monthly_avg.index,
            colorscale='RdYlGn',
            text=np.round(monthly_avg.values * 100, 2),
            texttemplate='%{text:.1f}%',
            textfont={"size": 8}
        ))
        
        fig.update_layout(
            title='Average Daily Returns by Month (%)',
            height=500,
            template='plotly_dark'
        )
        
        return fig

class MatplotlibVisualizations:
    """
    ویژوال‌سازی با matplotlib (برای گزارش‌های چاپی)
    """
    
    @staticmethod
    def create_comprehensive_chart(df: pd.DataFrame, symbol: str = '') -> plt.Figure:
        """
        ایجاد نمودار جامع با matplotlib
        """
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), 
                                gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        
        # 1. نمودار قیمت
        ax1 = axes[0]
        ax1.plot(df.index, df['close'], label='Close Price', color='blue', linewidth=1)
        
        if 'SMA_20' in df.columns:
            ax1.plot(df.index, df['SMA_20'], label='SMA 20', color='orange', linewidth=1, alpha=0.7)
        if 'SMA_50' in df.columns:
            ax1.plot(df.index, df['SMA_50'], label='SMA 50', color='red', linewidth=1, alpha=0.7)
        
        ax1.set_title(f'{symbol} - Price Chart')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. RSI
        ax2 = axes[1]
        if 'RSI' in df.columns:
            ax2.plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=1)
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax2.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
        
        ax2.set_title('RSI')
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # 3. MACD
        ax3 = axes[2]
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            ax3.plot(df.index, df['MACD'], label='MACD', color='blue', linewidth=1)
            ax3.plot(df.index, df['MACD_Signal'], label='Signal', color='red', linewidth=1)
            ax3.bar(df.index, df.get('MACD_Hist', 0), label='Histogram', 
                   color=np.where(df.get('MACD_Hist', 0) > 0, 'g', 'r'), alpha=0.3)
        
        ax3.set_title('MACD')
        ax3.set_ylabel('MACD')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Volume
        ax4 = axes[3]
        if 'volume' in df.columns:
            colors = ['r' if df['close'].iloc[i] < df['open'].iloc[i] else 'g' 
                     for i in range(len(df))]
            ax4.bar(df.index, df['volume'], color=colors, alpha=0.5)
        
        ax4.set_title('Volume')
        ax4.set_ylabel('Volume')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_correlation_heatmap_matplotlib(df: pd.DataFrame, 
                                          method: str = 'pearson') -> plt.Figure:
        """
        رسم heatmap همبستگی با matplotlib
        """
        corr_matrix = df.corr(method=method)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # رسم heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # اضافه کردن متن
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        # تنظیمات محور
        ax.set_xticks(range(len(corr_matrix)))
        ax.set_yticks(range(len(corr_matrix)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)
        
        # رنگ‌بار
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom")
        
        ax.set_title(f'Correlation Matrix ({method.title()})')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_distribution_matplotlib(series: pd.Series, 
                                    title: str = 'Distribution') -> plt.Figure:
        """
        رسم توزیع با matplotlib
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # هیستوگرام
        ax1.hist(series.dropna(), bins=50, density=True, alpha=0.6, color='blue')
        
        # اضافه کردن KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(series.dropna())
        x_range = np.linspace(series.min(), series.max(), 100)
        ax1.plot(x_range, kde(x_range), 'r-', linewidth=2)
        
        ax1.set_title(f'{title} - Histogram')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(series.dropna(), vert=True, patch_artist=True)
        ax2.set_title(f'{title} - Box Plot')
        ax2.set_ylabel('Value')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# توابع کمکی
def save_plot(fig, filename: str, format: str = 'png', dpi: int = 300):
    """
    ذخیره نمودار
    """
    if hasattr(fig, 'write_image'):  # Plotly figure
        fig.write_image(filename, format=format, scale=2)
    else:  # Matplotlib figure
        fig.savefig(filename, format=format, dpi=dpi, bbox_inches='tight')
    
    print(f"Plot saved as {filename}")

def create_dashboard_figures(df: pd.DataFrame, trades: pd.DataFrame = None, 
                           metrics: Dict = None) -> Dict[str, go.Figure]:
    """
    ایجاد تمام نمودارهای لازم برای داشبورد
    """
    figures = {}
    
    # 1. نمودار قیمت اصلی
    figures['price_chart'] = FinancialVisualizations.plot_price_chart(
        df.tail(500),  # آخرین 500 نقطه
        indicators=['SMA_20', 'SMA_50'],
        volume=True
    )
    
    # 2. اندیکاتورهای تکنیکال
    figures['technical_indicators'] = FinancialVisualizations.plot_technical_indicators(
        df.tail(500)
    )
    
    # 3. توزیع بازده‌ها
    if 'returns' in df.columns:
        figures['returns_distribution'] = FinancialVisualizations.plot_returns_distribution(
            df['returns'].dropna()
        )
    
    # 4. ماتریس همبستگی
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 2:
        figures['correlation_matrix'] = FinancialVisualizations.plot_correlation_matrix(
            df[numeric_cols].tail(200)
        )
    
    # 5. معاملات روی نمودار
    if trades is not None and len(trades) > 0:
        figures['trades_chart'] = FinancialVisualizations.plot_trades_on_chart(
            df.tail(500), trades.tail(50)
        )
    
    # 6. Heatmap ماهانه
    if 'returns' in df.columns and len(df) > 365:
        figures['monthly_heatmap'] = FinancialVisualizations.plot_monthly_returns_heatmap(
            df['returns']
        )
    
    return figures

if __name__ == "__main__":
    print("ماژول ویژوال‌سازی داده‌های مالی")