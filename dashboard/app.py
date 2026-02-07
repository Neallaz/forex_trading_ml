# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import path_config
import warnings
warnings.filterwarnings('ignore')

# تنظیمات صفحه
st.set_page_config(
    page_title="Forex ML Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS سفارشی
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .positive {
        color: #00C853;
        font-weight: bold;
    }
    .negative {
        color: #FF5252;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class TradingDashboard:
    def __init__(self):
        self.data_dir = Path(path_config.PROCESSED_DATA_DIR)
        self.models_dir = Path(path_config.ML_MODELS_DIR)
        self.results_dir = Path(path_config.RESULTS_DIR)
        
    def load_data(self, symbol: str = 'EURUSD'):
        """بارگذاری داده‌ها"""
        filepath = self.data_dir / f"{symbol}_features.csv"
        if filepath.exists():
            return pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        return pd.DataFrame()
    
    def load_metrics(self, symbol: str = 'EURUSD'):
        """بارگذاری معیارهای مدل"""
        filepath = self.models_dir / f"{symbol}_ml_metrics.csv"
        if filepath.exists():
            return pd.read_csv(filepath)
        return pd.DataFrame()
    
    def create_price_chart(self, df: pd.DataFrame, symbol: str):
        """ایجاد نمودار قیمت"""
        fig = go.Figure()
        
        # کندل‌استیک
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ))
        
        # Moving Averages
        if 'sma_20' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['sma_20'],
                name='SMA 20',
                line=dict(color='orange', width=1)
            ))
        
        if 'sma_50' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['sma_50'],
                name='SMA 50',
                line=dict(color='red', width=1)
            ))
        
        # تنظیمات layout
        fig.update_layout(
            title=f'{symbol} Price Chart',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark',
            height=500,
            showlegend=True
        )
        
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig
    
    def create_technical_indicators(self, df: pd.DataFrame):
        """ایجاد نمودارهای اندیکاتورهای تکنیکال"""
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('RSI', 'MACD', 'Bollinger Bands', 'Volume'),
            vertical_spacing=0.1,
            row_heights=[0.25, 0.25, 0.25, 0.25]
        )
        
        # RSI
        if 'rsi_14' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['rsi_14'], name='RSI'),
                row=1, col=1
            )
            # اضافه کردن خطوط overbought/oversold
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        
        # MACD
        if 'macd_12_26' in df.columns and 'macd_signal_9' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['macd_12_26'], name='MACD'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['macd_signal_9'], name='Signal'),
                row=2, col=1
            )
        
        # Bollinger Bands
        if 'bb_upper_20' in df.columns and 'bb_lower_20' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['bb_upper_20'], name='Upper BB',
                          line=dict(color='gray', dash='dash')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['bb_lower_20'], name='Lower BB',
                          line=dict(color='gray', dash='dash')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['close'], name='Price'),
                row=3, col=1
            )
        
        # Volume
        if 'volume' in df.columns:
            colors = ['green' if row['close'] >= row['open'] else 'red' 
                     for _, row in df.iterrows()]
            fig.add_trace(
                go.Bar(x=df.index, y=df['volume'], name='Volume',
                      marker_color=colors),
                row=4, col=1
            )
        
        fig.update_layout(height=800, showlegend=True, template='plotly_dark')
        return fig
    
    def create_model_performance(self, metrics_df: pd.DataFrame):
        """نمایش عملکرد مدل‌ها"""
        if metrics_df.empty:
            return None
        
        fig = go.Figure()
        
        metrics = ['accuracy', 'f1_score', 'roc_auc']
        
        for metric in metrics:
            fig.add_trace(go.Bar(
                x=metrics_df['model'],
                y=metrics_df[metric],
                name=metric.replace('_', ' ').title(),
                text=metrics_df[metric].round(3),
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            barmode='group',
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    def run(self):
        """اجرای داشبورد"""
        # عنوان اصلی
        st.markdown('<h1 class="main-header">📈 Forex ML Trading Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.header("⚙️ تنظیمات")
            
            # انتخاب جفت ارز
            symbol = st.selectbox(
                "جفت ارز",
                ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'],
                index=0
            )
            
            # انتخاب تایم‌فریم
            timeframe = st.selectbox(
                "تایم‌فریم",
                ['1h', '4h', '1d', '1w'],
                index=0
            )
            
            # تاریخ شروع و پایان
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("از تاریخ", value=pd.to_datetime('2023-01-01'))
            with col2:
                end_date = st.date_input("تا تاریخ", value=pd.to_datetime('2023-12-31'))
            
            # نمایش معیارهای کلیدی
            st.header("📊 معیارهای کلیدی")
            
            # این بخش را با داده‌های واقعی پر کنید
            st.metric("Sharpe Ratio", "1.45", "0.12")
            st.metric("Win Rate", "56.3%", "2.1%")
            st.metric("Max Drawdown", "-12.4%", "-1.2%")
            st.metric("Total Return", "24.7%", "3.2%")
            
            # دکمه‌های عملیاتی
            st.header("🚀 عملیات")
            if st.button("🔄 بروزرسانی داده‌ها"):
                st.info("در حال بروزرسانی داده‌ها...")
                # کد بروزرسانی داده‌ها
            
            if st.button("🤖 آموزش مدل‌ها"):
                st.info("در حال آموزش مدل‌ها...")
                # کد آموزش مدل‌ها
        
        # بخش اصلی داشبورد
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 قیمت و نمودارها", 
            "🤖 عملکرد مدل‌ها", 
            "📈 معاملات", 
            "📋 گزارش‌ها"
        ])
        
        with tab1:
            # بارگذاری داده‌ها
            df = self.load_data(symbol)
            
            if not df.empty:
                # فیلتر بر اساس تاریخ
                df_filtered = df.loc[start_date:end_date]
                
                # نمودار قیمت
                st.plotly_chart(
                    self.create_price_chart(df_filtered, symbol),
                    use_container_width=True
                )
                
                # نمودارهای تکنیکال
                st.plotly_chart(
                    self.create_technical_indicators(df_filtered),
                    use_container_width=True
                )
            else:
                st.warning("داده‌ای برای نمایش وجود ندارد.")
        
        with tab2:
            # بارگذاری معیارهای مدل
            metrics_df = self.load_metrics(symbol)
            
            if not metrics_df.empty:
                # نمودار عملکرد مدل‌ها
                st.plotly_chart(
                    self.create_model_performance(metrics_df),
                    use_container_width=True
                )
                
                # جدول معیارها
                st.subheader("📋 جدول معیارهای مدل‌ها")
                st.dataframe(
                    metrics_df.style.format({
                        'accuracy': '{:.3f}',
                        'f1_score': '{:.3f}',
                        'roc_auc': '{:.3f}',
                        'precision': '{:.3f}',
                        'recall': '{:.3f}'
                    }).background_gradient(cmap='Blues', subset=['accuracy', 'f1_score', 'roc_auc']),
                    use_container_width=True
                )
            else:
                st.info("لطفاً ابتدا مدل‌ها را آموزش دهید.")
        
        with tab3:
            st.subheader("📊 وضعیت معاملات اخیر")
            
            # نمونه داده معاملات
            trades_data = {
                'تاریخ': ['2023-12-01 10:00', '2023-12-01 14:30', '2023-12-02 09:15'],
                'نماد': ['EURUSD', 'GBPUSD', 'EURUSD'],
                'نوع': ['خرید', 'فروش', 'خرید'],
                'حجم': [0.1, 0.15, 0.2],
                'قیمت ورود': [1.0985, 1.2650, 1.0960],
                'قیمت خروج': [1.1020, 1.2620, 'در حال معامله'],
                'سود/زیان': ['+35 پیپ', '+30 پیپ', '+15 پیپ'],
                'وضعیت': ['بسته شده', 'بسته شده', 'باز']
            }
            
            trades_df = pd.DataFrame(trades_data)
            st.dataframe(trades_df, use_container_width=True)
            
            # نمودار equity curve
            st.subheader("📈 منحنی سرمایه")
            
            # داده نمونه
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            equity = 10000 + np.cumsum(np.random.randn(len(dates)) * 100)
            
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=dates, y=equity,
                mode='lines',
                name='Equity',
                line=dict(color='green', width=2)
            ))
            
            fig_eq.update_layout(
                title='Equity Curve',
                xaxis_title='Date',
                yaxis_title='Equity ($)',
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig_eq, use_container_width=True)
        
        with tab4:
            st.subheader("📊 گزارش عملکرد کلی")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("تعداد معاملات", "125")
            with col2:
                st.metric("Win Rate", "56.8%")
            with col3:
                st.metric("Profit Factor", "1.65")
            with col4:
                st.metric("Sharpe Ratio", "1.45")
            
            # گزارش‌های تفصیلی
            st.subheader("📋 گزارش‌های تفصیلی")
            
            report_type = st.selectbox(
                "نوع گزارش",
                ['معاملات سودده', 'معاملات زیانده', 'همه معاملات', 'تحلیل ریسک']
            )
            
            # دکمه دانلود گزارش
            if st.button("📥 دانلود گزارش Excel"):
                # کد تولید و دانلود گزارش
                st.success("گزارش با موفقیت دانلود شد!")

def main():
    dashboard = TradingDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()