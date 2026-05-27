#!/usr/bin/env python3
"""RL 交易决策 Agent - 第二层信号"""
import os
import numpy as np
import pandas as pd
from ..agents.base import AgentContext, BaseAgent
from config import RL_BUY_POSITION_PCT, RL_ADD_POSITION_PCT, RL_STOP_LOSS


class RLTradingAgent(BaseAgent):
    name = "rl_trading"
    description = "强化学习交易决策 - 基于时间窗口判断买卖点"

    def execute(self, context: AgentContext) -> AgentContext:
        signals = []

        # 如果没有训练好的 RL 模型，使用启发式策略
        for code in context.stock_pool:
            ind = context.indicators.get(code, {})
            if not ind:
                continue

            close = ind.get("close", 0)
            rsi = ind.get("rsi_14", 50)
            price_pos = ind.get("price_position", 0.5)
            volume_ratio = ind.get("volume_ratio", 1)
            pct_chg = ind.get("pct_chg", 0)

            # 检测时间窗口信号中是否有此股票
            ts_hits = [s for s in context.ts_signals if s.get("stock") == code]

            # 判断买入
            buy_score = 0
            if rsi < 35 and price_pos < 0.3:
                buy_score += 2  # 超卖区
            if volume_ratio > 1.5 and pct_chg > 0:
                buy_score += 2  # 放量上涨
            if any(s["type"] in ("valley", "up_trend_start") for s in ts_hits):
                buy_score += 2  # 时间窗口支持
            if any(s["type"] == "lower_breakout" for s in ts_hits):
                buy_score += 1

            # 判断卖出
            sell_score = 0
            if rsi > 70 and price_pos > 0.8:
                sell_score += 2  # 超买区
            if volume_ratio > 2 and pct_chg < 0:
                sell_score += 2  # 放量下跌
            if any(s["type"] in ("peak", "down_trend_start") for s in ts_hits):
                sell_score += 2

            # 最终决策
            if buy_score >= 2 and buy_score > sell_score:
                action = "buy"
                confidence = min(1.0, buy_score / 5)
            elif sell_score >= 2 and sell_score > buy_score:
                action = "sell"
                confidence = min(1.0, sell_score / 5)
            else:
                action = "hold"
                confidence = 0

            if action != "hold":
                signals.append({
                    "stock": code,
                    "action": action,
                    "confidence": round(confidence, 3),
                    "price": close,
                    "position_pct": RL_BUY_POSITION_PCT if action == "buy" else 1.0,
                    "reason": f"RSI={rsi}, 价格位置={price_pos:.2f}, 量比={volume_ratio:.2f}",
                })

        signals.sort(key=lambda x: x["confidence"], reverse=True)
        context.rl_signals = signals
        context.warnings.append(f"生成 {len(signals)} 个交易信号")
        return context
