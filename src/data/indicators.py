#!/usr/bin/env python3
"""技术指标计算模块"""
import pandas as pd
import numpy as np


def _rolling_mean(s: pd.Series, window: int) -> pd.Series:
    """兼容 pandas 3.x 的 rolling mean（绕过 CoW 内部缓存问题）"""
    arr = s.to_numpy(dtype=float, na_value=float("nan"))
    result = pd.Series(np.full(len(arr), float("nan")), index=s.index, dtype=float)
    for i in range(window - 1, len(arr)):
        result.iloc[i] = np.nanmean(arr[i - window + 1 : i + 1])
    return result


def _rolling_std(s: pd.Series, window: int) -> pd.Series:
    arr = s.to_numpy(dtype=float, na_value=float("nan"))
    result = pd.Series(np.full(len(arr), float("nan")), index=s.index, dtype=float)
    for i in range(window - 1, len(arr)):
        result.iloc[i] = np.nanstd(arr[i - window + 1 : i + 1])
    return result


def _ewm_mean(s: pd.Series, span: int) -> pd.Series:
    arr = s.to_numpy(dtype=float, na_value=float("nan"))
    alpha = 2 / (span + 1)
    result = np.full(len(arr), float("nan"))
    result[0] = arr[0]
    for i in range(1, len(arr)):
        if np.isnan(arr[i]):
            result[i] = result[i - 1]
        else:
            result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return pd.Series(result, index=s.index, dtype=float)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算全套技术指标"""
    close = df["close"].to_numpy(dtype=float, na_value=float("nan"))
    high = df["high"].to_numpy(dtype=float, na_value=float("nan"))
    low = df["low"].to_numpy(dtype=float, na_value=float("nan"))
    volume = df["volume"].to_numpy(dtype=float, na_value=float("nan"))
    s_close = pd.Series(close, dtype=float)
    s_volume = pd.Series(volume, dtype=float)

    # MA
    for p in [5, 10, 20, 60]:
        df[f"ma{p}"] = _rolling_mean(s_close, p)

    # RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    n = len(gain)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14 - 1, n):
        window = gain[max(0, i - 13) : i + 1]
        avg_gain[i] = np.nanmean(window)
        avg_loss[i] = np.nanmean(loss[max(0, i - 13) : i + 1])
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = _ewm_mean(s_close, 12)
    ema26 = _ewm_mean(s_close, 26)
    df["macd"] = ema12 - ema26
    df["macd_signal"] = _ewm_mean(pd.Series(df["macd"].to_numpy(dtype=float), dtype=float), 9)
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger
    ma20 = _rolling_mean(s_close, 20)
    std20 = _rolling_std(s_close, 20)
    df["boll_upper"] = ma20 + 2 * std20
    df["boll_lower"] = ma20 - 2 * std20
    df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / ma20.replace(0, float("nan"))

    # Volume 指标
    df["volume_ma5"] = _rolling_mean(s_volume, 5)
    df["volume_ratio"] = s_volume / df["volume_ma5"].replace(0, float("nan"))

    # ATR
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    df["atr"] = _rolling_mean(pd.Series(tr, dtype=float), 14)

    # 价格位置
    bolu = df["boll_upper"].to_numpy(dtype=float)
    boll = df["boll_lower"].to_numpy(dtype=float)
    width = bolu - boll
    df["price_position"] = np.where(width > 0, np.clip((close - boll) / width, 0, 1), 0.5)

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
