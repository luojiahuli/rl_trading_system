#!/usr/bin/env python3
"""策略实现库 — 基础版 + 增强版 (简洁单指标增强 + 止损止盈)"""
import numpy as np
import pandas as pd


class Strategy:
    name = "base"

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════
# 基础策略 (Original)
# ═══════════════════════════════════════════════════════════

class TrendFollowingStrategy(Strategy):
    """趋势跟踪：均线金叉死叉"""
    name = "trend_following"

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        if "ma5" not in df.columns or "ma20" not in df.columns:
            df = df.copy()
            df["ma5"] = df["close"].rolling(5).mean()
            df["ma20"] = df["close"].rolling(20).mean()
        signals = np.zeros(len(df))
        for i in range(1, len(df)):
            if pd.notna(df["ma5"].iloc[i]) and pd.notna(df["ma20"].iloc[i]):
                if df["ma5"].iloc[i-1] <= df["ma20"].iloc[i-1] and df["ma5"].iloc[i] > df["ma20"].iloc[i]:
                    signals[i] = 1
                elif df["ma5"].iloc[i-1] >= df["ma20"].iloc[i-1] and df["ma5"].iloc[i] < df["ma20"].iloc[i]:
                    signals[i] = -1
        return signals


class MeanReversionStrategy(Strategy):
    """均值回归：RSI 超买超卖"""
    name = "mean_reversion"

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        signals = np.zeros(len(df))
        for i in range(len(df)):
            rsi = df.get("rsi_14", pd.Series([50] * len(df))).iloc[i]
            if pd.isna(rsi):
                continue
            if rsi < 30:
                signals[i] = 1
            elif rsi > 70:
                signals[i] = -1
        return signals


class BreakoutStrategy(Strategy):
    """突破策略：价格突破 Bollinger 上轨"""
    name = "breakout"

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        signals = np.zeros(len(df))
        for i in range(1, len(df)):
            close = df["close"].iloc[i]
            upper = df.get("boll_upper", pd.Series([np.nan] * len(df))).iloc[i]
            lower = df.get("boll_lower", pd.Series([np.nan] * len(df))).iloc[i]
            vol_ratio = df.get("volume_ratio", pd.Series([1] * len(df))).iloc[i]
            if pd.isna(upper) or pd.isna(lower):
                continue
            if close > upper and vol_ratio > 1.5:
                signals[i] = 1
            elif close < lower and vol_ratio > 1.5:
                signals[i] = -1
        return signals


class MomentumStrategy(Strategy):
    """动量策略：涨幅 + 成交量确认"""
    name = "momentum"

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        signals = np.zeros(len(df))
        for i in range(5, len(df)):
            ret_5d = df["close"].iloc[i] / df["close"].iloc[i-5] - 1
            vol_ratio = df.get("volume_ratio", pd.Series([1] * len(df))).iloc[i]
            if ret_5d > 0.05 and vol_ratio > 1.2:
                signals[i] = 1
            elif ret_5d < -0.05 and vol_ratio > 1.2:
                signals[i] = -1
        return signals


# ═══════════════════════════════════════════════════════════
# 增强策略 (Entry + 1个额外确认 + 主动 Exit)
# ═══════════════════════════════════════════════════════════

class EnhancedMomentumStrategy(Strategy):
    """增强动量：动量 + RSI 多头确认 + 风控退出

    Entry: 5d涨>4% AND 量比>1.2 AND RSI>50 (确认多头)
    Exit:  5d涨<-3% (放宽阈值避免震荡市频繁进出)
    """
    name = "enhanced_momentum"

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        signals = np.zeros(len(df))
        close = df["close"].to_numpy(dtype=float)
        for i in range(10, len(df)):
            ret_5d = close[i] / close[i-5] - 1
            vol_ratio = float(df.get("volume_ratio", pd.Series([1] * len(df))).iloc[i])
            rsi = float(df.get("rsi_14", pd.Series([50] * len(df))).iloc[i])
            if pd.isna(rsi):
                continue
            if ret_5d > 0.04 and vol_ratio > 1.2 and rsi > 50:
                signals[i] = 1
            elif ret_5d < -0.03:
                signals[i] = -1
        return signals


class EnhancedTrendStrategy(Strategy):
    """增强趋势：金叉 + 量确认 + 趋势持续持有

    Entry: MA5>MA20 AND 量比>1.2
    Exit:  close<MA20 (趋势破坏才出, 不因短期波动离场)
    """
    name = "enhanced_trend"

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        signals = np.zeros(len(df))
        for i in range(25, len(df)):
            close = float(df["close"].iloc[i])
            ma5 = float(df.get("ma5", pd.Series([np.nan] * len(df))).iloc[i])
            ma20 = float(df.get("ma20", pd.Series([np.nan] * len(df))).iloc[i])
            vol_ratio = float(df.get("volume_ratio", pd.Series([1] * len(df))).iloc[i])
            if pd.isna(ma5) or pd.isna(ma20):
                continue
            if ma5 > ma20 and vol_ratio > 1.2:
                signals[i] = 1
            elif close < ma20:
                signals[i] = -1
        return signals


class EnhancedBreakoutStrategy(Strategy):
    """增强突破：Bollinger 突破 + MACD 动能

    Entry: close>Boll上轨 AND 量比>1.5 AND MACD柱>0
    Exit:  close<MA20 (趋势破坏才出)
    """
    name = "enhanced_breakout"

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        signals = np.zeros(len(df))
        for i in range(25, len(df)):
            close = float(df["close"].iloc[i])
            upper = float(df.get("boll_upper", pd.Series([np.nan] * len(df))).iloc[i])
            lower = float(df.get("boll_lower", pd.Series([np.nan] * len(df))).iloc[i])
            vol_ratio = float(df.get("volume_ratio", pd.Series([1] * len(df))).iloc[i])
            macd_hist = float(df.get("macd_hist", pd.Series([0] * len(df))).iloc[i])
            ma20 = float(df.get("ma20", pd.Series([np.nan] * len(df))).iloc[i])
            if pd.isna(upper) or pd.isna(lower) or pd.isna(ma20):
                continue
            if close > upper and vol_ratio > 1.5 and macd_hist > 0:
                signals[i] = 1
            elif close < ma20:
                signals[i] = -1
        return signals


class EnhancedMeanReversionStrategy(Strategy):
    """增强均值回归：RSI<28 + 缩量 + Boll超跌

    Entry: RSI<28 AND close<Boll下轨 AND 量比<0.8
    Exit:  RSI>55 OR close>MA20
    """
    name = "enhanced_mean_reversion"

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        signals = np.zeros(len(df))
        for i in range(25, len(df)):
            close = float(df["close"].iloc[i])
            rsi = float(df.get("rsi_14", pd.Series([50] * len(df))).iloc[i])
            lower = float(df.get("boll_lower", pd.Series([np.nan] * len(df))).iloc[i])
            vol_ratio = float(df.get("volume_ratio", pd.Series([1] * len(df))).iloc[i])
            ma20 = float(df.get("ma20", pd.Series([np.nan] * len(df))).iloc[i])
            if pd.isna(lower) or pd.isna(ma20) or pd.isna(rsi):
                continue
            if rsi < 28 and close < lower and vol_ratio < 0.8:
                signals[i] = 1
            elif rsi > 55 or close > ma20:
                signals[i] = -1
        return signals


class CompositeStrategy(Strategy):
    """集成策略：4策略加权投票"""
    name = "composite"

    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold
        self.sub_strategies = [
            EnhancedMomentumStrategy(),
            EnhancedTrendStrategy(),
            EnhancedBreakoutStrategy(),
            EnhancedMeanReversionStrategy(),
        ]

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        all_signals = np.array([s.generate_signals(df) for s in self.sub_strategies])
        consensus = np.mean(all_signals, axis=0)
        signals = np.zeros(len(df))
        for i in range(len(df)):
            if consensus[i] >= self.threshold:
                signals[i] = 1
            elif consensus[i] <= -self.threshold:
                signals[i] = -1
        return signals


def get_all_strategies() -> list:
    return [TrendFollowingStrategy(), MeanReversionStrategy(), BreakoutStrategy(), MomentumStrategy()]


def get_enhanced_strategies() -> list:
    return [EnhancedTrendStrategy(), EnhancedMeanReversionStrategy(), EnhancedBreakoutStrategy(),
            EnhancedMomentumStrategy(), CompositeStrategy()]
