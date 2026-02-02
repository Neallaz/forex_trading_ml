"""
توابع کمکی برای محاسبه اندیکاتورهای تکنیکال
"""

import numpy as np
import pandas as pd
import talib
from typing import Optional, Dict, List, Union

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    محاسبه اندیکاتور RSI
    
    Args:
        prices: سری قیمت‌ها
        period: دوره زمانی
    
    Returns:
        سری RSI
    """
    if len(prices) < period:
        return pd.Series([np.nan] * len(prices), index=prices.index)
    
    return pd.Series(
        talib.RSI(prices.values, timeperiod=period),
        index=prices.index
    )

def calculate_macd(prices: pd.Series, fast_period: int = 12, 
                  slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
    """
    محاسبه اندیکاتور MACD
    
    Returns:
        دیکشنری شامل MACD, Signal, Histogram
    """
    if len(prices) < slow_period + signal_period:
        empty_series = pd.Series([np.nan] * len(prices), index=prices.index)
        return {'macd': empty_series, 'signal': empty_series, 'histogram': empty_series}
    
    macd, signal, hist = talib.MACD(
        prices.values,
        fastperiod=fast_period,
        slowperiod=slow_period,
        signalperiod=signal_period
    )
    
    return {
        'macd': pd.Series(macd, index=prices.index),
        'signal': pd.Series(signal, index=prices.index),
        'histogram': pd.Series(hist, index=prices.index)
    }

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, 
                             std_dev: float = 2) -> Dict[str, pd.Series]:
    """
    محاسبه باندهای بولینگر
    """
    if len(prices) < period:
        empty_series = pd.Series([np.nan] * len(prices), index=prices.index)
        return {'upper': empty_series, 'middle': empty_series, 'lower': empty_series}
    
    upper, middle, lower = talib.BBANDS(
        prices.values,
        timeperiod=period,
        nbdevup=std_dev,
        nbdevdn=std_dev
    )
    
    return {
        'upper': pd.Series(upper, index=prices.index),
        'middle': pd.Series(middle, index=prices.index),
        'lower': pd.Series(lower, index=prices.index)
    }

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                 period: int = 14) -> pd.Series:
    """محاسبه Average True Range"""
    if len(high) < period:
        return pd.Series([np.nan] * len(high), index=high.index)
    
    atr = talib.ATR(high.values, low.values, close.values, timeperiod=period)
    return pd.Series(atr, index=high.index)

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                        k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """محاسبه Stochastic Oscillator"""
    if len(high) < k_period + d_period:
        empty_series = pd.Series([np.nan] * len(high), index=high.index)
        return {'k': empty_series, 'd': empty_series}
    
    slowk, slowd = talib.STOCH(
        high.values, low.values, close.values,
        fastk_period=k_period,
        slowk_period=d_period,
        slowd_period=d_period
    )
    
    return {
        'k': pd.Series(slowk, index=high.index),
        'd': pd.Series(slowd, index=high.index)
    }

def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """محاسبه Exponential Moving Average"""
    if len(prices) < period:
        return pd.Series([np.nan] * len(prices), index=prices.index)
    
    ema = talib.EMA(prices.values, timeperiod=period)
    return pd.Series(ema, index=prices.index)

def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """محاسبه Simple Moving Average"""
    if len(prices) < period:
        return pd.Series([np.nan] * len(prices), index=prices.index)
    
    sma = talib.SMA(prices.values, timeperiod=period)
    return pd.Series(sma, index=prices.index)

def calculate_volume_indicators(volume: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
    """محاسبه اندیکاتورهای حجم"""
    results = {}
    
    # On-Balance Volume
    if len(volume) > 0:
        obv = talib.OBV(close.values, volume.values)
        results['obv'] = pd.Series(obv, index=close.index)
    
    # Money Flow Index (MFI)
    # نیاز به high, low دارد - در اینجا ساده شده
    if len(volume) > 14:
        # استفاده از close به جای high/low برای سادگی
        typical_price = close
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(close > close.shift(1), 0)
        negative_flow = money_flow.where(close < close.shift(1), 0)
        
        positive_sum = positive_flow.rolling(14).sum()
        negative_sum = negative_flow.rolling(14).sum()
        
        money_ratio = positive_sum / negative_sum.replace(0, np.nan)
        mfi = 100 - (100 / (1 + money_ratio))
        
        results['mfi'] = mfi
    
    return results

def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    محاسبه تمام اندیکاتورهای تکنیکال برای یک DataFrame
    
    Args:
        df: DataFrame حاوی ستون‌های open, high, low, close, volume
    
    Returns:
        DataFrame با اندیکاتورهای اضافه شده
    """
    result_df = df.copy()
    
    # Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        result_df[f'SMA_{period}'] = calculate_sma(df['close'], period)
        result_df[f'EMA_{period}'] = calculate_ema(df['close'], period)
    
    # RSI
    result_df['RSI'] = calculate_rsi(df['close'], 14)
    
    # MACD
    macd = calculate_macd(df['close'])
    result_df['MACD'] = macd['macd']
    result_df['MACD_Signal'] = macd['signal']
    result_df['MACD_Hist'] = macd['histogram']
    
    # Bollinger Bands
    bb = calculate_bollinger_bands(df['close'])
    result_df['BB_Upper'] = bb['upper']
    result_df['BB_Middle'] = bb['middle']
    result_df['BB_Lower'] = bb['lower']
    
    # ATR
    result_df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])
    
    # Stochastic
    stoch = calculate_stochastic(df['high'], df['low'], df['close'])
    result_df['Stoch_K'] = stoch['k']
    result_df['Stoch_D'] = stoch['d']
    
    # Volume Indicators
    vol_indicators = calculate_volume_indicators(df['volume'], df['close'])
    for name, series in vol_indicators.items():
        result_df[name] = series
    
    # Custom Indicators
    result_df = add_custom_indicators(result_df)
    
    return result_df

def add_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """اضافه کردن اندیکاتورهای سفارشی"""
    result_df = df.copy()
    
    # Price Rate of Change
    result_df['ROC'] = df['close'].pct_change(periods=12) * 100
    
    # Commodity Channel Index (CCI)
    if len(df) >= 20:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(20).mean()
        mean_dev = abs(typical_price - sma_tp).rolling(20).mean()
        result_df['CCI'] = (typical_price - sma_tp) / (0.015 * mean_dev)
    
    # Average Directional Index (ADX)
    if len(df) >= 14:
        adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        result_df['ADX'] = pd.Series(adx, index=df.index)
    
    # Parabolic SAR
    sar = talib.SAR(df['high'].values, df['low'].values, acceleration=0.02, maximum=0.2)
    result_df['SAR'] = pd.Series(sar, index=df.index)
    
    # Williams %R
    willr = talib.WILLR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
    result_df['Williams_R'] = pd.Series(willr, index=df.index)
    
    # Ichimoku Cloud
    if len(df) >= 52:
        # Tenkan-sen (Conversion Line)
        period9_high = df['high'].rolling(9).max()
        period9_low = df['low'].rolling(9).min()
        result_df['Ichimoku_Tenkan'] = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line)
        period26_high = df['high'].rolling(26).max()
        period26_low = df['low'].rolling(26).min()
        result_df['Ichimoku_Kijun'] = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A)
        result_df['Ichimoku_Senkou_A'] = ((result_df['Ichimoku_Tenkan'] + result_df['Ichimoku_Kijun']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        period52_high = df['high'].rolling(52).max()
        period52_low = df['low'].rolling(52).min()
        result_df['Ichimoku_Senkou_B'] = ((period52_high + period52_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        result_df['Ichimoku_Chikou'] = df['close'].shift(-26)
    
    # قیمت نسبت به اندیکاتورها
    if 'SMA_20' in result_df.columns:
        result_df['Price_vs_SMA20'] = (df['close'] - result_df['SMA_20']) / result_df['SMA_20'] * 100
    
    if 'BB_Middle' in result_df.columns:
        result_df['BB_Position'] = (df['close'] - result_df['BB_Middle']) / (result_df['BB_Upper'] - result_df['BB_Middle']) * 100
    
    # Volatility Measures
    result_df['Daily_Return'] = df['close'].pct_change()
    result_df['Volatility_10'] = result_df['Daily_Return'].rolling(10).std() * np.sqrt(252)
    result_df['Volatility_30'] = result_df['Daily_Return'].rolling(30).std() * np.sqrt(252)
    
    return result_df

def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    تشخیص الگوهای کندل‌استیک
    
    Returns:
        DataFrame با ستون‌های الگوهای کندل‌استیک
    """
    result_df = df.copy()
    
    open_price = df['open'].values
    high_price = df['high'].values
    low_price = df['low'].values
    close_price = df['close'].values
    
    # الگوهای بازگشتی صعودی
    result_df['Hammer'] = talib.CDLHAMMER(open_price, high_price, low_price, close_price)
    result_df['Inverted_Hammer'] = talib.CDLINVERTEDHAMMER(open_price, high_price, low_price, close_price)
    result_df['Morning_Star'] = talib.CDLMORNINGSTAR(open_price, high_price, low_price, close_price)
    result_df['Piercing_Line'] = talib.CDLPIERCING(open_price, high_price, low_price, close_price)
    
    # الگوهای بازگشتی نزولی
    result_df['Shooting_Star'] = talib.CDLSHOOTINGSTAR(open_price, high_price, low_price, close_price)
    result_df['Hanging_Man'] = talib.CDLHANGINGMAN(open_price, high_price, low_price, close_price)
    result_df['Evening_Star'] = talib.CDLEVENINGSTAR(open_price, high_price, low_price, close_price)
    result_df['Dark_Cloud_Cover'] = talib.CDLDARKCLOUDCOVER(open_price, high_price, low_price, close_price)
    
    # الگوهای ادامه‌دهنده
    result_df['Doji'] = talib.CDLDOJI(open_price, high_price, low_price, close_price)
    result_df['Spinning_Top'] = talib.CDLSPINNINGTOP(open_price, high_price, low_price, close_price)
    result_df['Marubozu'] = talib.CDLMARUBOZU(open_price, high_price, low_price, close_price)
    
    # الگوهای ترکیبی
    result_df['Engulfing'] = talib.CDLENGULFING(open_price, high_price, low_price, close_price)
    result_df['Harami'] = talib.CDLHARAMI(open_price, high_price, low_price, close_price)
    result_df['Three_White_Soldiers'] = talib.CDL3WHITESOLDIERS(open_price, high_price, low_price, close_price)
    result_df['Three_Black_Crows'] = talib.CDL3BLACKCROWS(open_price, high_price, low_price, close_price)
    
    # تبدیل به مقادیر باینری (1 برای وجود الگو)
    pattern_columns = [col for col in result_df.columns if col not in df.columns]
    for col in pattern_columns:
        result_df[col] = (result_df[col] != 0).astype(int)
    
    # تعداد کل الگوهای شناسایی شده
    result_df['Total_Patterns'] = result_df[pattern_columns].sum(axis=1)
    result_df['Bullish_Patterns'] = result_df[['Hammer', 'Inverted_Hammer', 'Morning_Star', 
                                               'Piercing_Line', 'Three_White_Soldiers']].sum(axis=1)
    result_df['Bearish_Patterns'] = result_df[['Shooting_Star', 'Hanging_Man', 'Evening_Star',
                                               'Dark_Cloud_Cover', 'Three_Black_Crows']].sum(axis=1)
    
    return result_df

def generate_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    تولید سیگنال‌های معاملاتی بر اساس اندیکاتورها
    
    Returns:
        DataFrame با سیگنال‌های معاملاتی
    """
    result_df = df.copy()
    
    # سیگنال RSI
    if 'RSI' in result_df.columns:
        result_df['RSI_Signal'] = 0
        result_df.loc[result_df['RSI'] < 30, 'RSI_Signal'] = 1  # خرید
        result_df.loc[result_df['RSI'] > 70, 'RSI_Signal'] = -1  # فروش
    
    # سیگنال MACD
    if all(col in result_df.columns for col in ['MACD', 'MACD_Signal']):
        result_df['MACD_Signal'] = 0
        macd_cross_above = (result_df['MACD'] > result_df['MACD_Signal']) & \
                          (result_df['MACD'].shift(1) <= result_df['MACD_Signal'].shift(1))
        macd_cross_below = (result_df['MACD'] < result_df['MACD_Signal']) & \
                          (result_df['MACD'].shift(1) >= result_df['MACD_Signal'].shift(1))
        
        result_df.loc[macd_cross_above, 'MACD_Signal'] = 1
        result_df.loc[macd_cross_below, 'MACD_Signal'] = -1
    
    # سیگنال Bollinger Bands
    if all(col in result_df.columns for col in ['BB_Upper', 'BB_Lower']):
        result_df['BB_Signal'] = 0
        result_df.loc[result_df['close'] <= result_df['BB_Lower'], 'BB_Signal'] = 1  # خرید
        result_df.loc[result_df['close'] >= result_df['BB_Upper'], 'BB_Signal'] = -1  # فروش
    
    # سیگنال Moving Average
    if all(col in result_df.columns for col in ['SMA_20', 'SMA_50']):
        result_df['MA_Signal'] = 0
        ma_cross_above = (result_df['SMA_20'] > result_df['SMA_50']) & \
                        (result_df['SMA_20'].shift(1) <= result_df['SMA_50'].shift(1))
        ma_cross_below = (result_df['SMA_20'] < result_df['SMA_50']) & \
                        (result_df['SMA_20'].shift(1) >= result_df['SMA_50'].shift(1))
        
        result_df.loc[ma_cross_above, 'MA_Signal'] = 1
        result_df.loc[ma_cross_below, 'MA_Signal'] = -1
    
    # سیگنال Stochastic
    if all(col in result_df.columns for col in ['Stoch_K', 'Stoch_D']):
        result_df['Stoch_Signal'] = 0
        result_df.loc[(result_df['Stoch_K'] < 20) & (result_df['Stoch_D'] < 20), 'Stoch_Signal'] = 1
        result_df.loc[(result_df['Stoch_K'] > 80) & (result_df['Stoch_D'] > 80), 'Stoch_Signal'] = -1
    
    # سیگنال ترکیبی (میانگین وزنی)
    signal_columns = [col for col in result_df.columns if col.endswith('_Signal')]
    if signal_columns:
        weights = {
            'RSI_Signal': 0.2,
            'MACD_Signal': 0.3,
            'BB_Signal': 0.2,
            'MA_Signal': 0.2,
            'Stoch_Signal': 0.1
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for signal_col in signal_columns:
            if signal_col in weights:
                weighted_sum += result_df[signal_col] * weights[signal_col]
                total_weight += weights[signal_col]
        
        if total_weight > 0:
            result_df['Combined_Signal'] = weighted_sum / total_weight
            result_df['Final_Signal'] = np.sign(result_df['Combined_Signal'])
    
    return result_df

def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    محاسبه سطح‌های حمایت و مقاومت
    
    Args:
        df: DataFrame قیمت
        window: پنجره برای شناسایی سقف و کف
    
    Returns:
        DataFrame با سطح‌های حمایت و مقاومت
    """
    result_df = df.copy()
    
    # شناسایی سقف‌های محلی
    result_df['Local_High'] = df['high'].rolling(window, center=True).max()
    result_df['Is_Resistance'] = (df['high'] == result_df['Local_High']).astype(int)
    
    # شناسایی کف‌های محلی
    result_df['Local_Low'] = df['low'].rolling(window, center=True).min()
    result_df['Is_Support'] = (df['low'] == result_df['Local_Low']).astype(int)
    
    # جمع‌آوری سطح‌های مقاومت
    resistance_levels = result_df[result_df['Is_Resistance'] == 1]['high'].unique()
    resistance_levels = sorted(resistance_levels, reverse=True)[:10]  # 10 مقاومت برتر
    
    # جمع‌آوری سطح‌های حمایت
    support_levels = result_df[result_df['Is_Support'] == 1]['low'].unique()
    support_levels = sorted(support_levels)[:10]  # 10 حمایت برتر
    
    # فاصله از نزدیک‌ترین سطح
    if len(resistance_levels) > 0:
        closest_resistance = min(resistance_levels, key=lambda x: abs(x - df['close'].iloc[-1]))
        result_df['Distance_to_Resistance'] = (closest_resistance - df['close']) / df['close'] * 100
    
    if len(support_levels) > 0:
        closest_support = min(support_levels, key=lambda x: abs(x - df['close'].iloc[-1]))
        result_df['Distance_to_Support'] = (df['close'] - closest_support) / df['close'] * 100
    
    return result_df, support_levels, resistance_levels

# توابع کمکی
def normalize_series(series: pd.Series) -> pd.Series:
    """نرمال‌سازی سری به بازه [0, 1]"""
    if series.std() == 0:
        return pd.Series([0.5] * len(series), index=series.index)
    
    normalized = (series - series.min()) / (series.max() - series.min())
    return normalized

def calculate_correlation_matrix(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """محاسبه ماتریس همبستگی"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    corr_matrix = df[columns].corr()
    return corr_matrix

def remove_multicollinearity(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """
    حذف ویژگی‌های با همبستگی بالا
    
    Args:
        df: DataFrame ویژگی‌ها
        threshold: آستانه همبستگی برای حذف
    
    Returns:
        DataFrame با ویژگی‌های کاهش یافته
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    return df.drop(columns=to_drop)

def calculate_technical_score(df: pd.DataFrame) -> pd.Series:
    """
    محاسبه امتیاز تکنیکال ترکیبی
    
    Returns:
        سری امتیاز تکنیکال (0-100)
    """
    score = pd.Series(50, index=df.index)  # امتیاز خنثی اولیه
    
    # RSI Score
    if 'RSI' in df.columns:
        rsi_score = np.where(df['RSI'] < 30, 80,
                           np.where(df['RSI'] > 70, 20, 50))
        score = score * 0.3 + rsi_score * 0.7
    
    # MACD Score
    if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
        macd_score = np.where(df['MACD'] > df['MACD_Signal'], 70, 30)
        score = score * 0.6 + macd_score * 0.4
    
    # Price vs MA Score
    if 'SMA_50' in df.columns:
        ma_score = np.where(df['close'] > df['SMA_50'], 65, 35)
        score = score * 0.7 + ma_score * 0.3
    
    # Volume Score (ساده شده)
    if 'volume' in df.columns:
        vol_avg = df['volume'].rolling(20).mean()
        vol_score = np.where(df['volume'] > vol_avg * 1.5, 60, 40)
        score = score * 0.8 + vol_score * 0.2
    
    return score

if __name__ == "__main__":
    # نمونه استفاده
    print("ماژول اندیکاتورهای تکنیکال")