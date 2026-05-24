#!/usr/bin/env python3
"""风险管理模块"""
import numpy as np


class RiskManager:
    """风控管理器"""

    def __init__(self, max_drawdown=-0.15, kelly_fraction=0.25):
        self.max_drawdown = max_drawdown
        self.kelly_fraction = kelly_fraction

    def compute_kelly_position(self, win_rate: float, avg_win: float,
                                avg_loss: float) -> float:
        """Kelly 公式计算最优仓位比例"""
        if avg_loss == 0:
            return self.kelly_fraction
        b = avg_win / avg_loss  # 盈亏比
        p = win_rate           # 胜率
        q = 1 - p              # 败率
        kelly = (b * p - q) / b
        return max(0, min(1, kelly * self.kelly_fraction))

    def check_drawdown(self, current_equity: float, peak_equity: float) -> dict:
        """检查回撤状态"""
        if peak_equity == 0:
            return {"level": "normal", "dd_pct": 0}

        dd = (current_equity - peak_equity) / peak_equity
        if dd < self.max_drawdown:
            return {
                "level": "critical",
                "dd_pct": round(dd, 4),
                "action": "stop_trading",
                "message": f"回撤 {dd:.1%} 超过阈值 {self.max_drawdown:.1%}，暂停交易",
            }
        elif dd < self.max_drawdown * 0.7:
            return {
                "level": "warning",
                "dd_pct": round(dd, 4),
                "action": "reduce_position",
                "message": f"回撤 {dd:.1%}，建议减仓",
            }
        return {
            "level": "normal",
            "dd_pct": round(dd, 4),
            "action": "normal",
            "message": f"回撤 {dd:.1%}，正常交易",
        }

    def compute_var(self, returns: np.ndarray, confidence=0.95) -> float:
        """计算 VaR"""
        if len(returns) == 0:
            return 0
        return round(float(np.percentile(returns, (1 - confidence) * 100)), 4)

    def position_sizing(self, capital: float, risk_per_trade: float,
                        stop_loss_pct: float, price: float) -> int:
        """基于风险计算仓位大小"""
        risk_amount = capital * risk_per_trade
        shares = int(risk_amount / (price * abs(stop_loss_pct)))
        return max(0, shares)
