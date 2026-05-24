#!/usr/bin/env python3
"""技术指标计算模块"""
import pandas as pd
import numpy as np


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算全套技术指标"""
    df = df.copy()
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)

    # MA
    for p in [5, 10, 20, 60]:
        df[f"ma{p}"] = df["close"].rolling(p).mean()

    # RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean().values
    avg_loss = pd.Series(loss).rolling(14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger
    ma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["boll_upper"] = ma20 + 2 * std20
    df["boll_lower"] = ma20 - 2 * std20
    df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / ma20

    # Volume 指标
    df["volume_ma5"] = df["volume"].rolling(5).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma5"]

    # ATR
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    df["atr"] = pd.Series(tr).rolling(14).mean().values

    # 价格位置
    df["price_position"] = ((close - df["boll_lower"]) / (df["boll_upper"] - df["boll_lower"])).clip(0, 1)

    return df


def compute_trend_intensity(df: pd.DataFrame, window: int = 20) -> float:
    """计算趋势强度 (0-1)"""
    close = df["close"].values[-window:]
    if len(close) < window:
        return 0.0
    returns = np.diff(close) / close[:-1]
    # 趋势强度 = 均值绝对值 / 标准差
    if np.std(returns) == 0:
        return 0.0
    intensity = min(1.0, abs(np.mean(returns)) / (np.std(returns) / np.sqrt(window)))
    return round(float(intensity), 4)


def compute_volatility(df: pd.DataFrame, window: int = 20) -> float:
    """计算波动率"""
    returns = df["close"].pct_change().dropna().values[-window:]
    if len(returns) < 2:
        return 0.0
    return round(float(np.std(returns) * np.sqrt(252)), 4)
