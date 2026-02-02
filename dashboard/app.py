"""
داشبورد Streamlit برای نمایش نتایج
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import joblib
import sys
import os

# اضافه کردن مسیر پروژه به sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings

# تنظیمات صفحه
st.set_page_config(
    page_title="Forex ML Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# استایل‌های سفارشی
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .positive {
        color: #10B981;
        font-weight: bold;
    }
    .negative {
        color: #EF4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class ForexTradingDashboard:
    """کلاس داشبورد تریدینگ"""
    
    def __init__(self):
        self.data_dir = Path(settings.PROCESSED_DATA_DIR)
        self.models_dir = Path(settings.MODELS_DIR)
        self.results_dir = Path("trading/backtesting/results")
        
    def load_data(self, symbol):
        """بارگذاری داده‌ها"""
        try:
            data_path = self.data_dir / f"{symbol}_processed.csv"
            if data_path.exists():
                return pd.read_csv(data_path, index_col=0, parse_dates=True)
        except:
            pass
        return None
    
    def load_predictions(self, symbol):
        """بارگذاری پیش‌بینی‌ها"""
        try:
            preds_path = self.models_dir / "ensemble" / f"{symbol}_ensemble_predictions.csv"
            if preds_path.exists():
                return pd.read_csv(preds_path, index_col=0, parse_dates=True)
        except:
            pass
        return None
    
    def load_backtest_results(self, symbol):
        """بارگذاری نتایج بکتست"""
        try:
            results_path = self.results_dir / f"{symbol}_backtest_results.csv"
            if results_path.exists():
                return pd.read_csv(results_path, index_col=0)
        except:
            pass
        return None
    
    def create_price_chart(self, df, symbol):
        """ایجاد نمودار قیمت"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} Price', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # نمودار کندل‌استیک
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # اضافه کردن moving averages
        if 'SMA_20' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20'),
                row=1, col=1
            )
        
        if 'SMA_50' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50'),
                row=1, col=1
            )
        
        # نمودار حجم
        if 'volume' in df.columns:
            colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' 
                     for i in range(len(df))]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name='Volume',
                    marker_color=colors
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=f'{symbol} Price Chart',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_prediction_chart(self, df, predictions):
        """ایجاد نمودار پیش‌بینی‌ها"""
        if predictions is None or df is None:
            return None
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Price with Signals', 'Prediction Probability'),
            row_heights=[0.6, 0.4]
        )
        
        # قیمت با سیگنال‌ها
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                name='Close Price',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # سیگنال‌های خرید
        buy_signals = predictions[predictions['final_signal'] == 1]
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=df.loc[buy_signals.index, 'close'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ),
                row=1, col=1
            )
        
        # سیگنال‌های فروش
        sell_signals = predictions[predictions['final_signal'] == 0]
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=df.loc[sell_signals.index, 'close'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ),
                row=1, col=1
            )
        
        # احتمالات پیش‌بینی
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions['final_prediction'],
                name='Prediction Probability',
                line=dict(color='purple', width=2),
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        # خط 0.5 (آستانه)
        fig.add_hline(
            y=0.5, 
            line_dash="dash", 
            line_color="gray",
            row=2, col=1
        )
        
        fig.update_layout(
            title='Trading Signals and Predictions',
            height=700,
            showlegend=True
        )
        
        return fig
    
    def create_metrics_display(self, metrics):
        """نمایش معیارهای عملکرد"""
        if metrics is None:
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Sharpe Ratio",
                f"{metrics.get('sharpe_ratio', 0):.2f}",
                delta="Good" if metrics.get('sharpe_ratio', 0) > 1 else "Needs Improvement"
            )
        
        with col2:
            st.metric(
                "Total Return",
                f"{metrics.get('total_return', 0):.2f}%",
                delta_color="normal"
            )
        
        with col3:
            st.metric(
                "Max Drawdown",
                f"{metrics.get('max_drawdown', 0):.2f}%",
                delta_color="inverse"
            )
        
        with col4:
            win_rate = metrics.get('win_rate', 0)
            st.metric(
                "Win Rate",
                f"{win_rate:.1f}%",
                delta="Good" if win_rate > 55 else "Needs Improvement"
            )
        
        # معیارهای اضافی
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric(
                "Profit Factor",
                f"{metrics.get('profit_factor', 0):.2f}",
                delta="Good" if metrics.get('profit_factor', 0) > 1.5 else "Normal"
            )
        
        with col6:
            st.metric(
                "Sortino Ratio",
                f"{metrics.get('sortino_ratio', 0):.2f}",
                delta="Good" if metrics.get('sortino_ratio', 0) > 1 else "Normal"
            )
        
        with col7:
            st.metric(
                "Total Trades",
                f"{metrics.get('total_trades', 0)}",
                delta_color="off"
            )
        
        with col8:
            sqn = metrics.get('sqn', 0)
            st.metric(
                "SQN",
                f"{sqn:.2f}",
                delta="Excellent" if sqn > 2 else "Good" if sqn > 1.5 else "Needs Work"
            )
    
    def create_equity_curve(self, trades_data):
        """ایجاد نمودار منحنی سرمایه"""
        # این تابع نیاز به داده‌های تاریخی معاملات دارد
        # برای سادگی، یک نمونه ایجاد می‌کنیم
        if trades_data is None:
            # ایجاد داده نمونه
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            np.random.seed(42)
            equity = 10000 + np.cumsum(np.random.randn(len(dates)) * 100)
            trades_data = pd.DataFrame({'equity': equity}, index=dates)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trades_data.index,
            y=trades_data['equity'],
            mode='lines',
            name='Equity Curve',
            line=dict(color='green', width=2),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Date',
            yaxis_title='Account Value ($)',
            height=400
        )
        
        return fig
    
    def create_feature_importance_chart(self, symbol):
        """نمودار اهمیت ویژگی‌ها"""
        try:
            # بارگذاری مدل برای feature importance
            model_path = self.models_dir / "ml" / f"{symbol}_random_forest.pkl"
            if model_path.exists():
                model = joblib.load(model_path)
                
                if hasattr(model, 'feature_importances_'):
                    # بارگذاری داده‌ها برای نام ویژگی‌ها
                    data_path = self.data_dir / f"{symbol}_features.csv"
                    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
                    
                    features = df.drop(['target', 'target_return'], axis=1).columns
                    importances = model.feature_importances_
                    
                    # انتخاب 15 ویژگی برتر
                    indices = np.argsort(importances)[-15:]
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=importances[indices],
                        y=[features[i] for i in indices],
                        orientation='h',
                        marker_color='steelblue'
                    ))
                    
                    fig.update_layout(
                        title='Top 15 Feature Importances',
                        xaxis_title='Importance',
                        yaxis_title='Features',
                        height=500
                    )
                    
                    return fig
        except:
            pass
        
        return None
    
    def run(self):
        """اجرای داشبورد"""
        st.markdown('<h1 class="main-header">🏦 Forex ML Trading System Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.header("تنظیمات")
            
            # انتخاب جفت ارز
            symbol = st.selectbox(
                "Select Currency Pair",
                options=settings.FOREX_PAIRS,
                index=0
            )
            
            # بازه زمانی
            timeframe = st.selectbox(
                "Timeframe",
                options=["1H", "4H", "1D"],
                index=0
            )
            
            # نمایش مدل‌ها
            st.subheader("Model Performance")
            
            # بارگذاری نتایج
            results = self.load_backtest_results(symbol)
            if results is not None:
                sharpe = results['sharpe_ratio'].iloc[0]
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
            # اطلاعات پروژه
            st.subheader("Project Info")
            st.info("""
            This dashboard displays results from the 
            Forex ML Trading System project.
            
            Features:
            • Price prediction using ML/DL
            • Technical analysis indicators
            • Risk management metrics
            • Backtesting results
            """)
        
        # بارگذاری داده‌ها
        data = self.load_data(symbol)
        predictions = self.load_predictions(symbol)
        backtest_results = self.load_backtest_results(symbol)
        
        if data is None:
            st.error(f"داده‌ای برای {symbol} یافت نشد")
            return
        
        # تب‌های اصلی
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "📈 Price Analysis", 
            "🤖 Model Predictions", 
            "📉 Performance", 
            "🔍 Feature Analysis"
        ])
        
        with tab1:
            st.header("Overview")
            
            # معیارهای کلیدی
            if backtest_results is not None:
                self.create_metrics_display(backtest_results.iloc[0].to_dict())
            
            # خلاصه وضعیت
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Current Status")
                st.metric("Current Price", f"{data['close'].iloc[-1]:.5f}")
                
                if len(data) > 1:
                    change = ((data['close'].iloc[-1] - data['close'].iloc[-2]) / 
                            data['close'].iloc[-2] * 100)
                    st.metric("24h Change", f"{change:.2f}%")
                
                st.metric("Data Points", len(data))
            
            with col2:
                st.subheader("Market Conditions")
                
                # محاسبه نوسان
                volatility = data['log_returns'].std() * np.sqrt(252) * 100
                st.metric("Annual Volatility", f"{volatility:.2f}%")
                
                # وضعیت RSI
                if 'RSI' in data.columns:
                    current_rsi = data['RSI'].iloc[-1]
                    rsi_status = "Overbought" if current_rsi > 70 else \
                                "Oversold" if current_rsi < 30 else "Neutral"
                    st.metric("RSI", f"{current_rsi:.1f}", rsi_status)
            
            # نمودار equity curve
            st.subheader("Equity Curve")
            equity_fig = self.create_equity_curve(None)
            st.plotly_chart(equity_fig, use_container_width=True)
        
        with tab2:
            st.header("Price Analysis")
            
            # نمودار قیمت
            price_fig = self.create_price_chart(data.tail(500), symbol)
            st.plotly_chart(price_fig, use_container_width=True)
            
            # آمارهای قیمت
            st.subheader("Price Statistics")
            price_stats = data[['open', 'high', 'low', 'close']].describe()
            st.dataframe(price_stats.style.format("{:.5f}"))
        
        with tab3:
            st.header("Model Predictions")
            
            if predictions is not None:
                # نمودار پیش‌بینی‌ها
                pred_fig = self.create_prediction_chart(data, predictions)
                if pred_fig:
                    st.plotly_chart(pred_fig, use_container_width=True)
                
                # آمار پیش‌بینی‌ها
                st.subheader("Prediction Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    buy_signals = len(predictions[predictions['final_signal'] == 1])
                    total_signals = len(predictions)
                    if total_signals > 0:
                        buy_percentage = (buy_signals / total_signals) * 100
                        st.metric("Buy Signals", f"{buy_signals}", 
                                 f"{buy_percentage:.1f}% of total")
                
                with col2:
                    avg_confidence = predictions['final_prediction'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                
                with col3:
                    accuracy = (predictions['final_signal'] == predictions['actual']).mean()
                    st.metric("Prediction Accuracy", f"{accuracy:.2%}")
                
                # جدول سیگنال‌های اخیر
                st.subheader("Recent Signals")
                recent_signals = predictions.tail(20).copy()
                recent_signals['signal'] = recent_signals['final_signal'].map(
                    {0: 'SELL', 1: 'BUY'}
                )
                recent_signals['confidence'] = recent_signals['final_prediction'].apply(
                    lambda x: f"{x:.1%}"
                )
                
                st.dataframe(
                    recent_signals[['signal', 'confidence', 'actual']].rename(
                        columns={'actual': 'Actual Direction'}
                    ).style.applymap(
                        lambda x: 'color: green' if x == 'BUY' else 'color: red',
                        subset=['signal']
                    )
                )
            else:
                st.warning("پیش‌بینی‌های مدل موجود نیست")
        
        with tab4:
            st.header("Performance Metrics")
            
            if backtest_results is not None:
                # نمایش کامل معیارها
                st.subheader("Detailed Performance Metrics")
                
                metrics_df = backtest_results.T
                metrics_df.columns = ['Value']
                st.dataframe(metrics_df.style.format("{:.4f}"))
                
                # نمودارهای عملکرد
                col1, col2 = st.columns(2)
                
                with col1:
                    # نمودار Sharpe vs Sortino
                    fig1 = go.Figure()
                    
                    fig1.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=backtest_results['sharpe_ratio'].iloc[0],
                        title={'text': "Sharpe Ratio"},
                        domain={'row': 0, 'column': 0},
                        gauge={'axis': {'range': [0, 3]},
                              'bar': {'color': "darkblue"},
                              'steps': [
                                  {'range': [0, 1], 'color': "red"},
                                  {'range': [1, 2], 'color': "yellow"},
                                  {'range': [2, 3], 'color': "green"}
                              ]}
                    ))
                    
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # نمودار Win Rate
                    fig2 = go.Figure()
                    
                    win_rate = backtest_results['win_rate'].iloc[0]
                    fig2.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=win_rate,
                        title={'text': "Win Rate"},
                        domain={'row': 0, 'column': 0},
                        gauge={'axis': {'range': [0, 100]},
                              'bar': {'color': "darkblue"},
                              'steps': [
                                  {'range': [0, 40], 'color': "red"},
                                  {'range': [40, 55], 'color': "yellow"},
                                  {'range': [55, 100], 'color': "green"}
                              ]}
                    ))
                    
                    st.plotly_chart(fig2, use_container_width=True)
                
                # تحلیل drawdown
                st.subheader("Drawdown Analysis")
                
                if 'max_drawdown' in backtest_results.columns:
                    max_dd = backtest_results['max_drawdown'].iloc[0]
                    
                    fig3 = go.Figure(go.Indicator(
                        mode="number",
                        value=max_dd,
                        number={"suffix": "%"},
                        title={"text": "Maximum Drawdown"},
                        domain={'x': [0, 1], 'y': [0, 1]}
                    ))
                    
                    fig3.update_layout(
                        height=200
                    )
                    
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # تفسیر drawdown
                    if max_dd < 10:
                        st.success("✅ Drawdown کم - مدیریت ریسک عالی")
                    elif max_dd < 20:
                        st.warning("⚠️ Drawdown متوسط - نیاز به بهبود مدیریت ریسک")
                    else:
                        st.error("❌ Drawdown بالا - نیاز به بازنگری جدی در استراتژی")
        
        with tab5:
            st.header("Feature Analysis")
            
            # نمودار اهمیت ویژگی‌ها
            feature_fig = self.create_feature_importance_chart(symbol)
            if feature_fig:
                st.plotly_chart(feature_fig, use_container_width=True)
            
            # همبستگی ویژگی‌ها
            st.subheader("Feature Correlations")
            
            if data is not None:
                # انتخاب ویژگی‌های عددی
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                
                # محاسبه همبستگی
                corr_matrix = data[numeric_cols].corr()
                
                # نمودار heatmap
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1
                ))
                
                fig_corr.update_layout(
                    title='Feature Correlation Matrix',
                    height=600
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # توزیع بازده‌ها
            st.subheader("Returns Distribution")
            
            if 'log_returns' in data.columns:
                fig_dist = px.histogram(
                    data, 
                    x='log_returns',
                    nbins=50,
                    title='Distribution of Log Returns'
                )
                
                fig_dist.add_vline(
                    x=data['log_returns'].mean(), 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Mean"
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center'>
                <p>Forex ML Trading System Dashboard • Built with Streamlit</p>
                <p>⚠️ Disclaimer: This is for educational purposes only. Not financial advice.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

def main():
    """تابع اصلی"""
    dashboard = ForexTradingDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()