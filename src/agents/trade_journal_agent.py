#!/usr/bin/env python3
"""每日交易流水 Agent — 生成操作记录（从 RL 信号生成模拟操作流水）"""
import math
from ..agents.base import AgentContext, BaseAgent
from config import INITIAL_CASH


class TradeJournalAgent(BaseAgent):
    name = "trade_journal"
    description = "从 RL 信号生成每日操作记录（买入/卖出/持有）"

    def execute(self, context: AgentContext) -> AgentContext:
        trades = []
        signals = context.rl_signals

        if not signals:
            context.warnings.append("无 RL 信号，跳过交易流水")
            return context

        cash = float(INITIAL_CASH)
        position = None  # 当前持仓 {stock, qty, avg_price, strategy}

        for sig in signals:
            stock = sig.get("stock", "")
            action = sig.get("action", "hold")
            price = sig.get("price", 0)
            confidence = sig.get("confidence", 0)
            reason = sig.get("reason", "")
            strategy = sig.get("strategy", "")

            if action == "buy" and cash >= price * 100:
                invest = min(cash * 0.25, cash)
                qty = math.floor(invest / price / 100) * 100
                if qty >= 100:
                    cost = qty * price
                    cash -= cost
                    position = {
                        "stock": stock, "qty": qty,
                        "avg_price": price, "strategy": strategy,
                    }
                    trades.append({
                        "stock": stock, "action": "buy",
                        "price": price, "quantity": qty, "pnl": 0,
                        "confidence": confidence, "reason": reason,
                        "strategy": strategy, "cash_after": round(cash, 2),
                    })

            elif action == "sell" and position and position["stock"] == stock:
                proceeds = position["qty"] * price
                pnl = proceeds - position["qty"] * position["avg_price"]
                cash += proceeds
                trades.append({
                    "stock": stock, "action": "sell",
                    "price": price, "quantity": position["qty"],
                    "pnl": round(pnl, 2),
                    "confidence": confidence, "reason": reason,
                    "strategy": position.get("strategy", ""),
                    "cash_after": round(cash, 2),
                })
                position = None

        if position:
            trades.append({
                "stock": position["stock"], "action": "hold",
                "price": position["avg_price"], "quantity": position["qty"],
                "pnl": 0, "confidence": 0, "reason": "持仓中",
                "strategy": position.get("strategy", ""),
                "cash_after": round(cash, 2),
            })

        context.trades = trades
        context.warnings.append(
            f"生成 {len(trades)} 条操作记录, 剩余现金 ¥{cash:,.2f}"
        )

        # 计算账户总资产
        total_assets = cash
        if position:
            total_assets += position["qty"] * position["avg_price"]

        context.portfolio = {
            "initial_cash": INITIAL_CASH,
            "cash": round(cash, 2),
            "total_assets": round(total_assets, 2),
            "total_return": round((total_assets / INITIAL_CASH - 1) * 100, 2),
            "position_count": len(trades),
        }

        if context.db and trades:
            context.db.save_trade_journal(context.date, trades)

        return context
