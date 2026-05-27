#!/usr/bin/env python3
"""每日交易流水 Agent — 生成操作记录（从 RL 信号生成模拟操作流水）"""
import math
from ..agents.base import AgentContext, BaseAgent


class TradeJournalAgent(BaseAgent):
    name = "trade_journal"
    description = "从 RL 信号生成每日操作记录（买入/卖出/持有）"

    def execute(self, context: AgentContext) -> AgentContext:
        trades = []
        signals = context.rl_signals

        if not signals:
            context.warnings.append("无 RL 信号，跳过交易流水")
            return context

        initial_cash = 100000.0  # 模拟初始资金 10 万
        cash = initial_cash
        position = None  # 当前持仓 {stock, qty, avg_price}

        for sig in signals:
            stock = sig.get("stock", "")
            action = sig.get("action", "hold")
            price = sig.get("price", 0)
            confidence = sig.get("confidence", 0)
            reason = sig.get("reason", "")

            if action == "buy" and cash >= price * 100:
                # 模拟买入（每只股票最多投入 25% 仓位）
                invest = min(cash * 0.25, cash)
                qty = math.floor(invest / price / 100) * 100
                if qty >= 100:
                    cost = qty * price
                    cash -= cost
                    position = {"stock": stock, "qty": qty, "avg_price": price}
                    trades.append({
                        "stock": stock,
                        "action": "buy",
                        "price": price,
                        "quantity": qty,
                        "pnl": 0,
                        "confidence": confidence,
                        "reason": reason,
                    })

            elif action == "sell" and position and position["stock"] == stock:
                # 卖出持仓
                proceeds = position["qty"] * price
                pnl = proceeds - position["qty"] * position["avg_price"]
                cash += proceeds
                trades.append({
                    "stock": stock,
                    "action": "sell",
                    "price": price,
                    "quantity": position["qty"],
                    "pnl": round(pnl, 2),
                    "confidence": confidence,
                    "reason": reason,
                })
                position = None

        # 剩余持仓也算一条 hold 记录
        if position:
            trades.append({
                "stock": position["stock"],
                "action": "hold",
                "price": position["avg_price"],
                "quantity": position["qty"],
                "pnl": 0,
                "confidence": 0,
                "reason": "持仓中",
            })

        context.trades = trades
        context.warnings.append(f"生成 {len(trades)} 条操作记录")

        # 持久化到数据库
        if context.db and trades:
            context.db.save_trade_journal(context.date, trades)

        return context