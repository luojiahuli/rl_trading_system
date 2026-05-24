#!/usr/bin/env python3
"""风险管理 Agent"""
import numpy as np
from ..agents.base import AgentContext, BaseAgent
from ..risk.manager import RiskManager
from config import RISK_MAX_DRAWDOWN, RISK_KELLY_FRACTION


class RiskManagementAgent(BaseAgent):
    name = "risk_management"
    description = "回撤预警 + 仓位管理 + VaR 计算"

    def execute(self, context: AgentContext) -> AgentContext:
        manager = RiskManager(
            max_drawdown=RISK_MAX_DRAWDOWN,
            kelly_fraction=RISK_KELLY_FRACTION,
        )

        # 模拟持仓和净值（回测结果中取）
        equity_values = []
        for result in context.backtest_results:
            curve = result.get("equity_curve", [])
            equity_values.extend(curve)

        peak_equity = max(equity_values) if equity_values else 100000
        current_equity = equity_values[-1] if equity_values else 100000

        # 回撤检查
        dd_result = manager.check_drawdown(current_equity, peak_equity)

        # VaR 计算
        returns = []
        for result in context.backtest_results:
            curve = result.get("equity_curve", [])
            if len(curve) > 1:
                r = np.diff(curve) / curve[:-1]
                returns.extend(r)
        var_95 = manager.compute_var(np.array(returns)) if returns else 0

        # 信号仓位建议
        position_advice = []
        for signal in context.rl_signals:
            win_rate = 0.55  # 默认值，可从历史数据中计算
            avg_win = 0.05
            avg_loss = 0.03
            kelly = manager.compute_kelly_position(win_rate, avg_win, avg_loss)
            position_advice.append({
                "stock": signal.get("stock"),
                "action": signal.get("action"),
                "kelly_pct": round(kelly, 3),
                "suggested_pct": round(min(kelly, signal.get("position_pct", 0.2)), 3),
            })

        context.risk_metrics = {
            "drawdown": dd_result,
            "var_95": var_95,
            "peak_equity": round(peak_equity, 2),
            "current_equity": round(current_equity, 2),
            "position_advice": position_advice,
        }
        context.portfolio = {
            "cash": 100000 - sum(a.get("suggested_pct", 0) * 100000
                                 for a in position_advice if a["action"] in ("buy",)),
            "positions": [a for a in position_advice if a["action"] in ("buy", "hold")],
        }
        return context
