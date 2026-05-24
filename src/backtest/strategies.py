#!/usr/bin/env python3
"""策略实现库"""
import numpy as np
import pandas as pd


class Strategy:
    name = "base"

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        """返回信号数组: 1=买入, -1=卖出, 0=持有"""
        raise NotImplementedError


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
                    signals[i] = 1   # 金叉买入
                elif df["ma5"].iloc[i-1] >= df["ma20"].iloc[i-1] and df["ma5"].iloc[i] < df["ma20"].iloc[i]:
                    signals[i] = -1  # 死叉卖出
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
                signals[i] = 1   # 超卖买入
            elif rsi > 70:
                signals[i] = -1  # 超买卖出
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
                signals[i] = 1   # 放量突破买入
            elif close < lower and vol_ratio > 1.5:
                signals[i] = -1  # 放量跌破卖出
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


def get_all_strategies() -> list:
    """获取所有策略实例"""
    return [
        TrendFollowingStrategy(),
        MeanReversionStrategy(),
        BreakoutStrategy(),
        MomentumStrategy(),
    ]
