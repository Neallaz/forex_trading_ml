# utils/visualizations.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class TradingVisualizations:
    """کلاس ایجاد ویژوال‌های معاملاتی"""
    
    @staticmethod
    def plot_equity_curve(equity_curve: pd.Series, 
                         benchmark: Optional[pd.Series] = None,
                         title: str = "Equity Curve") -> go.Figure:
        """رسم نمودار equity curve"""
        fig = go.Figure()
        
        # اضافه کردن equity curve
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Strategy',
            line=dict(color='green', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)'
        ))
        
        # اضافه کردن benchmark (اختیاری)
        if benchmark is not None:
            fig.add_trace(go.Scatter(
                x=benchmark.index,
                y=benchmark.values,
                mode='lines',
                name='Benchmark',
                line=dict(color='blue', width=1, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Equity ($)',
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    @staticmethod
    def plot_drawdown(drawdown: pd.Series, title: str = "Drawdown") -> go.Figure:
        """رسم نمودار drawdown"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=1),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.3)'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template='plotly_dark',
            yaxis=dict(tickformat='.1%')
        )
        
        return fig
    
    @staticmethod
    def plot_returns_distribution(returns: pd.Series, 
                                 title: str = "Returns Distribution") -> go.Figure:
        """رسم توزیع بازده‌ها"""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns',
            marker_color='skyblue',
            opacity=0.7
        ))
        
        # اضافه کردن خط میانگین
        mean_return = returns.mean()
        fig.add_vline(
            x=mean_return,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_return:.4f}"
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Returns',
            yaxis_title='Frequency',
            template='plotly_dark',
            bargap=0.05
        )
        
        return fig
    
    @staticmethod
    def plot_monthly_returns(monthly_returns: pd.Series,
                            title: str = "Monthly Returns Heatmap") -> go.Figure:
        """رسم heatmap بازده‌های ماهانه"""
        # آماده‌سازی داده برای heatmap
        monthly_returns_df = monthly_returns.unstack()
        
        fig = go.Figure(data=go.Heatmap(
            z=monthly_returns_df.values,
            x=monthly_returns_df.columns,
            y=monthly_returns_df.index,
            colorscale='RdYlGn',
            colorbar=dict(title='Returns'),
            text=monthly_returns_df.applymap(lambda x: f'{x:.2%}'),
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Month',
            yaxis_title='Year',
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def plot_trade_analysis(trades: pd.DataFrame,
                           title: str = "Trade Analysis") -> go.Figure:
        """تحلیل معاملات"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cumulative P&L', 'Trade Duration',
                           'P&L Distribution', 'Win/Loss by Time of Day'),
            specs=[[{'type': 'scatter'}, {'type': 'histogram'}],
                   [{'type': 'histogram'}, {'type': 'bar'}]]
        )
        
        # 1. Cumulative P&L
        if 'entry_time' in trades.columns and 'pnl' in trades.columns:
            trades_sorted = trades.sort_values('entry_time')
            cum_pnl = trades_sorted['pnl'].cumsum()
            
            fig.add_trace(
                go.Scatter(x=trades_sorted['entry_time'], y=cum_pnl,
                          mode='lines+markers', name='Cumulative P&L'),
                row=1, col=1
            )
        
        # 2. Trade Duration
        if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
            duration = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / 3600  # به ساعت
            
            fig.add_trace(
                go.Histogram(x=duration, nbinsx=20, name='Trade Duration'),
                row=1, col=2
            )
        
        # 3. P&L Distribution
        if 'pnl' in trades.columns:
            fig.add_trace(
                go.Histogram(x=trades['pnl'], nbinsx=30, name='P&L Distribution'),
                row=2, col=1
            )
        
        # 4. Win/Loss by Time of Day
        if 'entry_time' in trades.columns and 'pnl' in trades.columns:
            trades['hour'] = trades['entry_time'].dt.hour
            trades['result'] = trades['pnl'].apply(lambda x: 'Win' if x > 0 else 'Loss')
            
            hourly_stats = trades.groupby(['hour', 'result']).size().unstack(fill_value=0)
            
            fig.add_trace(
                go.Bar(x=hourly_stats.index, y=hourly_stats['Win'], name='Wins'),
                row=2, col=2
            )
            fig.add_trace(
                go.Bar(x=hourly_stats.index, y=hourly_stats['Loss'], name='Losses'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text=title, template='plotly_dark')
        return fig
    
    @staticmethod
    def plot_model_performance_comparison(metrics_dict: Dict[str, Dict],
                                         title: str = "Model Performance Comparison") -> go.Figure:
        """مقایسه عملکرد مدل‌ها"""
        models = list(metrics_dict.keys())
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [metrics_dict[model].get(metric, 0) for model in models]
            fig.add_trace(go.Bar(
                name=metric,
                x=models,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Models',
            yaxis_title='Score',
            barmode='group',
            template='plotly_dark',
            yaxis=dict(range=[0, 1])
        )
        
        return fig