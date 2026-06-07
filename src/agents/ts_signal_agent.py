#!/usr/bin/env python3
"""时间序列信号检测 Agent - 第一层信号"""
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from ..agents.base import AgentContext, BaseAgent


class TimeSeriesSignalAgent(BaseAgent):
    name = "ts_signal"
    description = "时间序列模式检测 - CUSUM + 峰值谷值 + 趋势强度"

    def execute(self, context: AgentContext) -> AgentContext:
        signals = []

        for code, df in context.market_data.items():
            close = df["close"].values.astype(float)
            volume = df["volume"].values.astype(float)
            if len(close) < 30:
                continue

            # 获取该股票的最新周线趋势
            week_trend = 0
            if "week_trend" in df.columns:
                wt_val = df["week_trend"].iloc[-1]
                week_trend = int(wt_val) if not pd.isna(wt_val) else 0

            # 1. CUSUM 趋势变化检测
            cusum_signals = self._detect_cusum(close)
            for cs in cusum_signals:
                signals.append({
                    "stock": code,
                    "type": cs["type"],
                    "confidence": cs["confidence"],
                    "index": cs["index"],
                    "method": "cusum",
                    "date": str(df.iloc[cs["index"]]["date"]) if cs["index"] < len(df) else "",
                    "week_trend": week_trend,
                })

            # 2. 峰值/谷值检测（局部极值）
            peak_valley = self._detect_peaks_valleys(close)
            for pv in peak_valley:
                signals.append({
                    "stock": code,
                    "type": pv["type"],
                    "confidence": pv["confidence"],
                    "index": pv["index"],
                    "method": "extremum",
                    "date": str(df.iloc[pv["index"]]["date"]) if pv["index"] < len(df) else "",
                    "week_trend": week_trend,
                })

            # 3. 突破检测（价格突破 Bollinger 带）
            if "boll_upper" in df.columns and "boll_lower" in df.columns:
                boll_signals = self._detect_breakout(df)
                signals.extend(boll_signals)

        # 去重排序
        signals.sort(key=lambda x: x["confidence"], reverse=True)
        context.ts_signals = signals[:50]
        context.warnings.append(f"检测到 {len(signals)} 个时间窗口信号")
        return context

    def _detect_cusum(self, prices, threshold=0.02, drift=0.005):
        """CUSUM 变化点检测"""
        signals = []
        cum_sum = 0
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            cum_sum += ret - drift
            if abs(cum_sum) > threshold:
                signals.append({
                    "type": "up_trend_start" if cum_sum > 0 else "down_trend_start",
                    "confidence": min(1.0, abs(cum_sum)),
                    "index": i,
                })
                cum_sum = 0
        return signals[-5:]  # 取最近 5 个

    def _detect_peaks_valleys(self, prices, order=5):
        """检测局部峰值和谷值"""
        signals = []
        peaks = argrelextrema(prices, np.greater, order=order)[0]
        valleys = argrelextrema(prices, np.less, order=order)[0]

        for p in peaks[-5:]:
            confidence = min(1.0, abs(prices[p] - prices[max(0, p-order)]) / prices[max(0, p-order)])
            signals.append({"type": "peak", "confidence": round(confidence, 3), "index": int(p)})

        for v in valleys[-5:]:
            confidence = min(1.0, abs(prices[max(0, v-order)] - prices[v]) / prices[max(0, v-order)])
            signals.append({"type": "valley", "confidence": round(confidence, 3), "index": int(v)})
        return signals

    def _detect_breakout(self, df):
        """Bollinger 带突破检测"""
        signals = []
        close = df["close"].values
        upper = df["boll_upper"].values
        lower = df["boll_lower"].values
        week_trend_break = 0
        if "week_trend" in df.columns:
            wt_val = df["week_trend"].iloc[-1]
            week_trend_break = int(wt_val) if not pd.isna(wt_val) else 0

        for i in range(max(1, len(close)-5), len(close)):
            if close[i] > upper[i]:
                strength = (close[i] - upper[i]) / upper[i]
                signals.append({
                    "stock": df.get("stock_code", ""), "type": "upper_breakout",
                    "confidence": min(1.0, strength * 10),
                    "index": int(i), "method": "bollinger",
                    "date": str(df.iloc[i]["date"]) if "date" in df.columns else "",
                    "week_trend": week_trend_break,
                })
            elif close[i] < lower[i]:
                strength = (lower[i] - close[i]) / lower[i]
                signals.append({
                    "stock": df.get("stock_code", ""), "type": "lower_breakout",
                    "confidence": min(1.0, strength * 10),
                    "index": int(i), "method": "bollinger",
                    "date": str(df.iloc[i]["date"]) if "date" in df.columns else "",
                    "week_trend": week_trend_break,
                })
        return signals[-5:]
