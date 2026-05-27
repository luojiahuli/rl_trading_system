#!/usr/bin/env python3
"""持仓明细分析 Agent — 从交易流水和市场数据计算持仓盈亏、权重、策略归属"""
import math
from .base import AgentContext, BaseAgent
from config import INITIAL_CASH


class PositionAnalysisAgent(BaseAgent):
    name = "position_analysis"
    description = "从交易流水和市场数据计算持仓明细（盈亏/权重/策略/账户总额）"

    def execute(self, context: AgentContext) -> AgentContext:
        trades = context.trades
        market_data = context.market_data
        indicators = context.indicators
        signals = context.rl_signals
        risk_config = context.risk_metrics
        portfolio = context.portfolio or {}

        if not trades:
            context.warnings.append("无交易流水，持仓分析跳过")
            context.position_analysis = self._empty_result()
            return context

        stop_loss_pct = -0.08
        if risk_config and "stop_loss" in risk_config:
            stop_loss_pct = risk_config.get("stop_loss", -0.08)

        # 构建信号查找表
        signal_map = {}
        for s in signals:
            signal_map[s.get("stock", "")] = {
                "strategy": s.get("strategy", ""),
                "confidence": s.get("confidence", 0),
                "reason": s.get("reason", ""),
            }

        positions = []
        cash = portfolio.get("cash", 0)

        for t in trades:
            stock = t.get("stock", "")
            action = t.get("action", "hold")
            quantity = t.get("quantity", 0)
            entry_price = t.get("price", 0)
            pnl_from_trade = t.get("pnl", 0)
            trade_strategy = t.get("strategy", "")

            # 从信号查找策略
            sig_info = signal_map.get(stock, {})
            strategy = trade_strategy or sig_info.get("strategy", "")
            reason = t.get("reason", "") or sig_info.get("reason", "")
            confidence = t.get("confidence", 0) or sig_info.get("confidence", 0)

            if action == "sell":
                positions.append({
                    "stock": stock,
                    "action": "卖出",
                    "strategy": strategy,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "current_price": 0,
                    "market_value": 0,
                    "pnl": round(pnl_from_trade, 2),
                    "pnl_pct": 0,
                    "cost_basis": 0,
                    "weight": 0,
                    "confidence": round(confidence * 100),
                    "rsi": None,
                    "stop_loss": None,
                    "signal_reason": reason,
                    "status": "已平仓",
                })
                continue

            stock_data = market_data.get(stock) if market_data else None
            current_price = 0
            if stock_data is not None and not stock_data.empty:
                current_price = float(stock_data["close"].iloc[-1])

            stock_indicators = indicators.get(stock, {}) if indicators else {}
            rsi_val = stock_indicators.get("rsi_14", None)
            if rsi_val is not None:
                try:
                    rsi_val = round(float(rsi_val), 1)
                except (ValueError, TypeError):
                    rsi_val = None

            market_value = round(quantity * current_price, 2)
            cost_basis = round(quantity * entry_price, 2)
            pnl = round(market_value - cost_basis, 2)
            pnl_pct = round((current_price / entry_price - 1) * 100, 2) if entry_price else 0
            stop_loss_price = round(entry_price * (1 + stop_loss_pct), 2) if entry_price else None

            positions.append({
                "stock": stock,
                "action": "持有" if action == "hold" else ("买入" if action == "buy" else action),
                "strategy": strategy or "—",
                "quantity": quantity,
                "entry_price": entry_price,
                "current_price": current_price,
                "market_value": market_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "cost_basis": cost_basis,
                "weight": 0,
                "confidence": round(confidence * 100),
                "rsi": rsi_val,
                "stop_loss": stop_loss_price,
                "signal_reason": reason,
                "status": "持仓中",
            })

        # 计算权重
        total_market_value = sum(
            p.get("market_value", 0) for p in positions if p["status"] == "持仓中"
        )
        for p in positions:
            if total_market_value > 0 and p["status"] == "持仓中":
                p["weight"] = round(p["market_value"] / total_market_value * 100, 1)

        positions.sort(key=lambda x: x.get("market_value", 0), reverse=True)

        # 汇总
        total_pnl = sum(p.get("pnl", 0) for p in positions)
        total_cost = sum(
            p.get("cost_basis", 0) for p in positions if p["status"] == "持仓中"
        )
        active_count = sum(1 for p in positions if p["status"] == "持仓中")
        total_pnl_pct = round((total_pnl / total_cost) * 100, 2) if total_cost > 0 else 0

        # 策略分布
        strategy_allocation = {}
        for p in positions:
            if p["status"] == "持仓中" and p.get("strategy"):
                s = p["strategy"]
                if s not in strategy_allocation:
                    strategy_allocation[s] = {
                        "count": 0, "market_value": 0, "weight": 0,
                    }
                strategy_allocation[s]["count"] += 1
                strategy_allocation[s]["market_value"] += p["market_value"]

        for s in strategy_allocation:
            if total_market_value > 0:
                strategy_allocation[s]["weight"] = round(
                    strategy_allocation[s]["market_value"] / total_market_value * 100, 1
                )
            strategy_allocation[s]["market_value"] = round(
                strategy_allocation[s]["market_value"], 2
            )

        # 账户总览（基于实际持仓重新计算）
        total_assets = total_market_value + cash
        total_return_pct = round((total_assets / INITIAL_CASH - 1) * 100, 4)
        total_return_raw = round(total_assets - INITIAL_CASH, 2)
        account_summary = {
            "initial_cash": INITIAL_CASH,
            "cash": round(cash, 2),
            "stock_value": round(total_market_value, 2),
            "total_assets": round(total_assets, 2),
            "total_return": round(total_return_pct, 2),
            "total_return_raw": total_return_raw,
            "active_positions": active_count,
            "total_positions": len(positions),
            "strategy_allocation": strategy_allocation,
        }

        context.position_analysis = {
            "positions": positions,
            "summary": account_summary,
        }

        context.warnings.append(
            f"持仓分析完成: {active_count} 只持仓中, "
            f"总资产 ¥{total_assets:,.2f}, "
            f"收益率 {account_summary['total_return']:+.2f}%"
        )
        return context

    @staticmethod
    def _empty_result() -> dict:
        return {
            "positions": [],
            "summary": {
                "initial_cash": INITIAL_CASH,
                "cash": 0,
                "stock_value": 0,
                "total_assets": 0,
                "total_return": 0,
                "total_return_raw": 0,
                "active_positions": 0,
                "total_positions": 0,
                "strategy_allocation": {},
            },
        }
