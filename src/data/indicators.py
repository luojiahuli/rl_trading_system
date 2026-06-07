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


def compute_indicators(df: pd.DataFrame, add_weekly: bool = True) -> pd.DataFrame:
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

    if add_weekly and "date" in df.columns:
        df = add_weekly_indicators(df)

    return df


def add_weekly_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """在日线 DataFrame 上计算周线级别指标。

    将日线 OHLCV 按 ISO 周重采样为周线，计算周线指标，
    然后合并回日线（同周内的每日共享相同的周线值）。
    前向填充确保本周指标对本周每日可见。
    """
    if len(df) < 15:
        df["week_ma5"] = float("nan")
        df["week_ma10"] = float("nan")
        df["week_rsi_14"] = float("nan")
        df["week_macd_hist"] = float("nan")
        df["week_trend"] = 0
        return df

    # 1. 构建 ISO 周键
    iso = df["date"].dt.isocalendar()
    df["_week_key"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)

    # 2. 聚合周线 OHLCV
    weekly = df.groupby("_week_key").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()
    weekly = weekly.sort_values("_week_key").reset_index(drop=True)

    w_close = weekly["close"].to_numpy(dtype=float)

    # 3. 周线指标
    s_w_close = pd.Series(w_close, dtype=float)
    weekly["w_ma5"] = _rolling_mean(s_w_close, min(5, len(weekly)))
    weekly["w_ma10"] = _rolling_mean(s_w_close, min(10, len(weekly)))

    # 周线 RSI(14)
    delta = np.diff(w_close, prepend=w_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    n_w = len(gain)
    avg_gain = np.full(n_w, np.nan)
    avg_loss = np.full(n_w, np.nan)
    for i in range(min(14, n_w) - 1, n_w):
        window = gain[max(0, i - 13) : i + 1]
        avg_gain[i] = np.nanmean(window)
        avg_loss[i] = np.nanmean(loss[max(0, i - 13) : i + 1])
    rs_w = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    weekly["w_rsi_14"] = 100 - (100 / (1 + rs_w))

    # 周线 MACD
    ema12 = _ewm_mean(s_w_close, 12)
    ema26 = _ewm_mean(s_w_close, 26)
    weekly["w_macd"] = ema12 - ema26
    weekly["w_macd_signal"] = _ewm_mean(pd.Series(weekly["w_macd"].to_numpy(dtype=float), dtype=float), 9)
    weekly["w_macd_hist"] = weekly["w_macd"] - weekly["w_macd_signal"]

    # 周线趋势: 1=看涨, -1=看跌, 0=中性 (1% 阈值)
    w_trend = np.zeros(n_w, dtype=int)
    has_ma5 = ~np.isnan(weekly["w_ma5"].to_numpy(dtype=float))
    has_ma10 = ~np.isnan(weekly["w_ma10"].to_numpy(dtype=float))
    for i in range(n_w):
        if has_ma5[i] and has_ma10[i]:
            ratio = weekly["w_ma5"].iloc[i] / weekly["w_ma10"].iloc[i]
            if ratio > 1.01:
                w_trend[i] = 1
            elif ratio < 0.99:
                w_trend[i] = -1
            else:
                w_trend[i] = 0
    weekly["w_trend"] = w_trend

    # 4. 合并回日线
    df = df.merge(
        weekly[["_week_key", "w_ma5", "w_ma10", "w_rsi_14", "w_macd_hist", "w_trend"]],
        on="_week_key", how="left"
    )

    # 5. 前向填充周线值（周初的日线需要上周数据）
    for col in ["w_ma5", "w_ma10", "w_rsi_14", "w_macd_hist", "w_trend"]:
        df[col] = df[col].ffill().fillna(0 if col == "w_trend" else float("nan"))

    # 6. 重命名添加 week_ 前缀
    rename_map = {
        "w_ma5": "week_ma5",
        "w_ma10": "week_ma10",
        "w_rsi_14": "week_rsi_14",
        "w_macd_hist": "week_macd_hist",
        "w_trend": "week_trend",
    }
    df = df.rename(columns=rename_map)
    df.drop(columns=["_week_key"], inplace=True)

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
