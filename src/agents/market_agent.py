#!/usr/bin/env python3
"""市场整体研判 Agent — 基于指数趋势/成交量/板块轮动/个股宽度的四维加权研判"""
import numpy as np
import pandas as pd

from ..agents.base import AgentContext, BaseAgent
from ..data.fetcher import fetch_index_daily

# ── 滚动均值（pandas 3.x CoW 兼容） ──────────────────────

def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(arr), np.nan)
    if len(arr) < window:
        return result
    for i in range(window - 1, len(arr)):
        result[i] = np.nanmean(arr[i - window + 1 : i + 1])
    return result


# ── Agent ────────────────────────────────────────────────

class MarketJudgementAgent(BaseAgent):
    name = "market_judgement"
    description = "基于指数趋势/成交量/板块轮动/个股宽度进行市场研判"

    def execute(self, context: AgentContext) -> AgentContext:
        # 1. 获取上证指数数据
        df = self._try_fetch_index(context)
        if df is None or len(df) < 30:
            context.warnings.append("市场研判: 上证指数数据不足")
            return context

        close = df["close"].to_numpy(dtype=float)
        volume = df["volume"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)

        # 2. 计算各项指标
        idx_trend = self._analyze_trend(close)
        vol_analysis = self._analyze_volume(volume)
        sector_br = self._analyze_sector_breadth(context.hot_sectors)
        stock_br = self._analyze_stock_breadth(context.market_data)

        # 3. 加权综合评分
        trend_score = idx_trend["trend_score"]    # [-1, 1]
        vol_score = vol_analysis["vol_score"]     # [-1, 1]
        sector_score = sector_br["sector_score"]   # [-1, 1]
        stock_score = stock_br["stock_score"]      # [-1, 1]

        composite = (
            trend_score * 0.40 +
            vol_score * 0.20 +
            sector_score * 0.20 +
            stock_score * 0.20
        )
        composite = max(-1.0, min(1.0, composite))

        # 4. 市场阶段判定
        if composite > 0.3:
            market_phase = "牛市"
        elif composite < -0.3:
            market_phase = "熊市"
        else:
            market_phase = "震荡"

        # 5. 置信度
        if abs(composite) > 0.6:
            confidence = "high"
        elif abs(composite) > 0.2:
            confidence = "medium"
        else:
            confidence = "low"

        # 6. 趋势方向（基于近期涨跌幅）
        ret_5d = idx_trend.get("ret_5d", 0)
        ret_20d = idx_trend.get("ret_20d", 0)
        if ret_5d > 2.0 and ret_20d > 0:
            trend_direction = "up"
        elif ret_5d < -2.0 and ret_20d < 0:
            trend_direction = "down"
        else:
            trend_direction = "sideways"

        # 7. 政策预期
        rsi = idx_trend.get("rsi_14", 50)
        if market_phase == "熊市":
            policy_outlook = "宽松"
        elif market_phase == "牛市" and rsi > 70 and confidence == "high":
            policy_outlook = "收紧"
        else:
            policy_outlook = "中性"

        # 8. 走势预判
        next_trend = self._predict_next(market_phase, rsi, confidence)

        # 9. 汇总摘要
        close_price = int(close[-1]) if len(close) > 0 else 0
        ma_alignment = idx_trend.get("ma_alignment", "mixed")
        pct_above = stock_br.get("pct_above_ma50", 0)
        sector_breadth = sector_br.get("breadth", "unknown")
        summary = (
            f"上证指数{close_price}点，MA排列{ma_alignment}。"
            f"板块活跃度{sector_breadth}，{pct_above}%个股站上MA50。"
            f"综合判断市场处于{market_phase}，{trend_direction}趋势，"
            f"政策预期{policy_outlook}，置信度{confidence}。"
        )

        context.market_judgement = {
            "market_phase": market_phase,
            "trend_direction": trend_direction,
            "policy_outlook": policy_outlook,
            "confidence": confidence,
            "next_trend": next_trend,
            "summary": summary,
            "details": {
                "index_trend": {
                    "price_vs_ma200_pct": round(idx_trend.get("price_vs_ma200_pct", 0), 2),
                    "ma_alignment": ma_alignment,
                    "slope_20d": round(idx_trend.get("slope_20d", 0), 4),
                    "ret_5d": round(ret_5d, 2),
                    "ret_20d": round(ret_20d, 2),
                    "rsi_14": round(rsi, 1),
                    "trend_score": round(trend_score, 3),
                },
                "volume_analysis": {
                    "volume_ratio": round(vol_analysis.get("volume_ratio", 1), 2),
                    "vol_trend": vol_analysis.get("vol_trend", "flat"),
                    "vol_score": round(vol_score, 3),
                },
                "sector_breadth": {
                    "total_sectors": sector_br.get("total_sectors", 0),
                    "high_heat_count": sector_br.get("high_heat_count", 0),
                    "breadth": sector_breadth,
                    "sector_score": round(sector_score, 3),
                },
                "stock_breadth": {
                    "total_stocks": stock_br.get("total_stocks", 0),
                    "above_ma50": stock_br.get("above_ma50", 0),
                    "pct_above_ma50": round(pct_above, 1),
                    "breadth": stock_br.get("breadth", "unknown"),
                    "stock_score": round(stock_score, 3),
                },
                "composite_score": round(composite, 3),
            },
        }

        return context

    # ── 私有方法 ──────────────────────────────────────────

    def _try_fetch_index(self, ctx: AgentContext) -> pd.DataFrame | None:
        """尝试获取上证指数数据"""
        try:
            df = fetch_index_daily("000001", start_date="2023-06-01")
            if df is not None and len(df) >= 30:
                return df
        except Exception:
            pass
        return None

    def _analyze_trend(self, close: np.ndarray) -> dict:
        """维度1: 指数趋势分析"""
        n = len(close)
        # MA200 / MA120 后备
        if n >= 200:
            ma_long = _rolling_mean(close, 200)
            price_vs_ma200 = (close[-1] / ma_long[-1] - 1) * 100
        elif n >= 120:
            ma_long = _rolling_mean(close, 120)
            price_vs_ma200 = (close[-1] / ma_long[-1] - 1) * 100
        else:
            price_vs_ma200 = 0

        # 均线排列: MA5, MA20, MA60
        if n >= 60:
            ma5 = _rolling_mean(close, 5)
            ma20 = _rolling_mean(close, 20)
            ma60 = _rolling_mean(close, 60)
            aligned = (ma5[-1] > ma20[-1] > ma60[-1])
            bearish = (ma5[-1] < ma20[-1] < ma60[-1])
            ma_alignment = "bullish" if aligned else ("bearish" if bearish else "mixed")
        else:
            ma_alignment = "mixed"

        # 20日线性斜率（简单差分均值）
        if n >= 20:
            slope_20d = (close[-1] - close[-20]) / close[-20] * 100
        else:
            slope_20d = 0

        # 短期涨跌幅
        ret_5d = (close[-1] / close[-5] - 1) * 100 if n >= 5 else 0
        ret_20d = (close[-1] / close[-20] - 1) * 100 if n >= 20 else 0

        # 趋势评分 [-6, +6] → [-1, +1]
        score = 0
        if price_vs_ma200 > 5:
            score += 2
        elif price_vs_ma200 > 0:
            score += 1
        elif price_vs_ma200 < -5:
            score -= 2
        elif price_vs_ma200 < 0:
            score -= 1

        if ma_alignment == "bullish":
            score += 2
        elif ma_alignment == "bearish":
            score -= 2

        if slope_20d > 10:
            score += 2
        elif slope_20d > 0:
            score += 1
        elif slope_20d < -10:
            score -= 2
        elif slope_20d < 0:
            score -= 1

        trend_score = max(-1.0, min(1.0, score / 6))

        # RSI(14)
        rsi_14 = 50
        if n >= 15:
            delta = np.diff(close)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_g = np.array([np.nanmean(gain[i - 13 : i + 1]) for i in range(13, len(gain))])
            avg_l = np.array([np.nanmean(loss[i - 13 : i + 1]) for i in range(13, len(loss))])
            if len(avg_l) > 0 and avg_l[-1] != 0:
                rsi_14 = 100 - 100 / (1 + avg_g[-1] / avg_l[-1])

        return {
            "price_vs_ma200_pct": price_vs_ma200,
            "ma_alignment": ma_alignment,
            "slope_20d": slope_20d,
            "ret_5d": ret_5d,
            "ret_20d": ret_20d,
            "rsi_14": rsi_14,
            "trend_score": trend_score,
        }

    def _analyze_volume(self, volume: np.ndarray) -> dict:
        """维度2: 成交量分析"""
        n = len(volume)
        if n < 10:
            return {"volume_ratio": 1, "vol_trend": "flat", "vol_score": 0}

        vol_ma5 = _rolling_mean(volume, 5)
        vol_ma10 = _rolling_mean(volume, 10)

        volume_ratio = volume[-1] / vol_ma5[-1] if vol_ma5[-1] > 0 else 1
        vol_trend5 = np.nanmean(volume[-5:]) if n >= 5 else 0
        vol_trend10 = np.nanmean(volume[-10:]) if n >= 10 else 0

        if vol_trend5 > vol_trend10 * 1.1:
            vol_trend = "increasing"
        elif vol_trend5 < vol_trend10 * 0.9:
            vol_trend = "decreasing"
        else:
            vol_trend = "flat"

        score = 0
        if volume_ratio > 1.5:
            score += 2
        elif volume_ratio > 1.2:
            score += 1
        elif volume_ratio < 0.5:
            score -= 2
        elif volume_ratio < 0.8:
            score -= 1

        if vol_trend == "increasing":
            score += 1
        elif vol_trend == "decreasing":
            score -= 1

        vol_score = max(-1.0, min(1.0, score / 3))
        return {"volume_ratio": volume_ratio, "vol_trend": vol_trend, "vol_score": vol_score}

    def _analyze_sector_breadth(self, hot_sectors: list) -> dict:
        """维度3: 板块轮动宽度"""
        if not hot_sectors:
            return {"total_sectors": 0, "high_heat_count": 0, "breadth": "unknown", "sector_score": 0}

        total = len(hot_sectors)
        high_heat = sum(1 for s in hot_sectors if s.get("heat_score", 0) >= 70)
        medium_heat = sum(1 for s in hot_sectors if 60 <= s.get("heat_score", 0) < 70)

        if high_heat >= 5:
            breadth = "broad"
            score = 2
        elif high_heat >= 3 or total >= 5:
            breadth = "moderate"
            score = 1
        elif total < 3:
            breadth = "narrow"
            score = -1
        else:
            breadth = "moderate"
            score = 1

        sector_score = max(-1.0, min(1.0, score / 2))
        return {
            "total_sectors": total,
            "high_heat_count": high_heat,
            "breadth": breadth,
            "sector_score": sector_score,
        }

    def _analyze_stock_breadth(self, market_data: dict) -> dict:
        """维度4: 个股宽度（% 股票站上 MA50）"""
        if not market_data:
            return {"total_stocks": 0, "above_ma50": 0, "pct_above_ma50": 0, "breadth": "unknown", "stock_score": 0}

        above = 0
        total = 0
        for code, df in market_data.items():
            if df is None or len(df) < 50:
                continue
            close = df["close"].to_numpy(dtype=float)
            if len(close) < 50:
                continue
            total += 1
            ma50 = np.nanmean(close[-50:])
            if close[-1] > ma50:
                above += 1

        if total == 0:
            return {"total_stocks": 0, "above_ma50": 0, "pct_above_ma50": 0, "breadth": "unknown", "stock_score": 0}

        pct = above / total * 100
        if pct >= 60:
            breadth = "broad"
            score = 2
        elif pct >= 40:
            breadth = "moderate"
            score = 1
        elif pct >= 20:
            breadth = "narrow"
            score = -1
        else:
            breadth = "very_narrow"
            score = -2

        stock_score = max(-1.0, min(1.0, score / 2))
        return {
            "total_stocks": total,
            "above_ma50": above,
            "pct_above_ma50": pct,
            "breadth": breadth,
            "stock_score": stock_score,
        }

    @staticmethod
    def _predict_next(market_phase: str, rsi: float, confidence: str) -> str:
        """走势预判"""
        if market_phase == "牛市":
            if confidence == "high" and rsi < 75:
                return "趋势延续"
            elif rsi > 75:
                return "可能的反转"
            return "继续震荡上行"
        elif market_phase == "熊市":
            if confidence == "high" and rsi > 25:
                return "趋势延续"
            elif rsi < 25:
                return "可能出现反弹"
            return "继续探底"
        else:  # 震荡
            if rsi > 70:
                return "可能面临压力"
            elif rsi < 30:
                return "可能出现反弹"
            return "继续震荡"
