#!/usr/bin/env python3
"""回测引擎"""
import numpy as np
import pandas as pd


class BacktestEngine:
    """多策略回测引擎"""

    def __init__(self, initial_cash=100000):
        self.initial_cash = initial_cash

    def run(self, df: pd.DataFrame, signals: np.ndarray,
            strategy_name: str = "strategy") -> dict:
        """执行单策略回测"""
        if len(df) != len(signals):
            raise ValueError("数据长度与信号长度不匹配")

        cash = self.initial_cash
        position = 0      # 持仓股数
        trades = []
        equity_curve = []

        for i in range(len(df)):
            price = df["close"].iloc[i]
            date = df["date"].iloc[i] if "date" in df.columns else i

            if signals[i] == 1 and cash > price:
                # 买入
                buy_amount = cash * 0.95  # 95% 资金
                shares = int(buy_amount / price)
                cost = shares * price
                cash -= cost
                position += shares
                trades.append({"date": date, "type": "buy", "price": price,
                               "shares": shares, "cash_after": cash})

            elif signals[i] == -1 and position > 0:
                # 卖出
                revenue = position * price
                cash += revenue
                trades.append({"date": date, "type": "sell", "price": price,
                               "shares": position, "cash_after": cash})
                position = 0

            # 记录净值
            equity = cash + position * price
            equity_curve.append(equity)

        # 最终清仓
        if position > 0:
            final_price = df["close"].iloc[-1]
            cash += position * final_price
            position = 0

        # 计算指标
        equity_series = pd.Series(equity_curve)
        total_return = equity_series.iloc[-1] / self.initial_cash - 1
        daily_returns = equity_series.pct_change().dropna()

        if len(daily_returns) > 0:
            sharpe = np.sqrt(252) * daily_returns.mean() / (daily_returns.std() + 1e-8)
            max_dd = self._max_drawdown(equity_series.values)
        else:
            sharpe = 0
            max_dd = 0

        return {
            "strategy": strategy_name,
            "total_return": round(total_return, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "num_trades": len([t for t in trades if t["type"] == "buy"]),
            "final_value": round(cash, 2),
            "trades": trades,
            "equity_curve": equity_curve,
        }

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        """计算最大回撤"""
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / (peak + 1e-8)
        return float(np.min(dd))
