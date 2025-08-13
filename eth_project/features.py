"""
features.py
This module provides functionality for generating numerical features from raw
candlestick data.  The primary entry point is ``generate_features`` which
accepts a pandas DataFrame containing at least the columns ``open``, ``high``,
``low``, ``close`` and ``volume``.  The function returns a new DataFrame with
additional feature columns suitable for training a machine learning model.

Feature engineering is a critical step in any systematic trading strategy.
Here we derive a modest set of indicators and ratios inspired by common
technical analysis (moving averages, exponential moving averages, MACD,
relative strength index, rolling volatility and volume measures).  These
features are intentionally simple and avoid future leakage: everything is
computed using only historical data up to and including the current candle.
"""

from typing import Optional
import pandas as pd
import numpy as np


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute a simple Relative Strength Index (RSI) over ``series``.

    The RSI is calculated by taking the ratio of average gains to average
    losses over a rolling window.  This implementation uses an exponential
    moving average (EMA) for efficiency and to mirror typical TA library
    behaviour.  If there is insufficient data for the given period the
    resulting RSI value will be NaN.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Use exponential moving average for average gain/loss
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a comprehensive suite of features from raw candlestick data.

    The intent of this function is to capture short‑, medium‑ and long‑term
    dynamics of price and volume using a diverse set of technical
    indicators.  All computations use only historical data up to and
    including the current candle to avoid look‑ahead bias.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing at least ``open``, ``high``, ``low``, ``close``
        and ``volume`` columns.  The ``open_time`` column, if present,
        will be preserved.  All numeric columns will be converted to floats.

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing the original columns plus engineered
        features.  No rows are dropped; NaN values may be present for the
        initial periods where rolling windows cannot be computed.
    """
    df = df.copy()

    # Ensure numeric columns are floats to avoid integer division
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # Basic price and volume derived features
    df['ret_1'] = df['close'].pct_change().fillna(0)
    df['ret_open_close'] = (df['close'] - df['open']) / df['open']
    df['range_rel'] = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
    df['vol_change'] = df['volume'].pct_change().fillna(0)

    # Rolling statistics over multiple horizons to capture momentum and volatility
    periods = [3, 5, 10, 15, 30, 60, 120, 240, 360]
    for window in periods:
        # Simple moving average of close
        sma = df['close'].rolling(window=window, min_periods=1).mean()
        df[f'sma_{window}'] = sma
        # EMA of close
        ema = df['close'].ewm(span=window, adjust=False).mean()
        df[f'ema_{window}'] = ema
        # Relative position of close to SMA
        df[f'close_sma_ratio_{window}'] = (df['close'] - sma) / sma.replace(0, np.nan)
        # Rolling return over the window (percentage change from window steps ago)
        df[f'ret_{window}'] = df['close'].pct_change(periods=window).fillna(0)
        # Rolling volatility: standard deviation of returns over the window
        ret_series = df['close'].pct_change().rolling(window=window, min_periods=2)
        vol = ret_series.std().replace(0, np.nan)
        df[f'volatility_{window}'] = vol
        # Rolling max and min ranges normalised by current close
        df[f'range_max_{window}'] = (df['high'].rolling(window=window, min_periods=1).max() - df['close']) / df['close'].replace(0, np.nan)
        df[f'range_min_{window}'] = (df['close'] - df['low'].rolling(window=window, min_periods=1).min()) / df['close'].replace(0, np.nan)
        # Rolling average volume and volume ratio
        vol_ma = df['volume'].rolling(window=window, min_periods=1).mean()
        df[f'vol_ma_{window}'] = vol_ma
        df[f'vol_ratio_{window}'] = (df['volume'] - vol_ma) / vol_ma.replace(0, np.nan)

    # MACD (12, 26, 9) – classic momentum indicator
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_hist'] = macd - signal

    # RSI (Relative Strength Index) capturing overbought/oversold conditions
    df['rsi_14'] = compute_rsi(df['close'], period=14)

    # Position of the closing price within the bar: 0 at low, 1 at high
    df['rel_close_in_bar'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)

    # Slope of medium‑term moving average as momentum proxy (difference between current and past SMA)
    for window, lag in [(30, 10), (60, 20), (120, 30)]:
        sma_col = f'sma_{window}'
        lagged = df[sma_col].shift(lag)
        df[f'sma_slope_{window}_{lag}'] = df[sma_col] - lagged

    # Replace any infinities resulting from division by zero with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df