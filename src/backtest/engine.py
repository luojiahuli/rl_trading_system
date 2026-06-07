#!/usr/bin/env python3
"""回测引擎"""
import numpy as np
import pandas as pd


class PortfolioBacktestEngine:
    """组合回测引擎：¥1M 总资金，在多只股票间动态分配

    每策略总资金 ¥1M，最多同时持有 max_positions 只股票。
    有信号才开仓，无信号资金闲置。通过集中资金到最佳信号提升收益。
    """

    def __init__(self, initial_cash: float = 1_000_000,
                 max_positions: int = 5,
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.15):
        self.initial_cash = initial_cash
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def run(self, market_data: dict[str, pd.DataFrame],
            strategy, strategy_name: str = "portfolio") -> dict:
        """运行组合回测

        Args:
            market_data: {stock_code: DataFrame with indicators}
            strategy: 策略实例
            strategy_name: 策略名称

        Returns:
            dict with total_return, sharpe_ratio, max_drawdown, etc.
        """
        # 1. 对所有股票生成信号
        signals = {}
        max_len = 0
        for code, df in market_data.items():
            sig = strategy.generate_signals(df)
            signals[code] = sig
            max_len = max(max_len, len(sig))

        if max_len == 0:
            return {"strategy": strategy_name, "total_return": 0,
                    "sharpe_ratio": 0, "max_drawdown": 0, "num_trades": 0,
                    "num_stop_loss": 0, "num_take_profit": 0,
                    "final_value": self.initial_cash,
                    "trades": [], "equity_curve": [self.initial_cash]}

        # 获取统一日期序列（用第一个有数据的股票）
        date_codes = [c for c, df in market_data.items() if len(df) == max_len]
        date_series = None
        if date_codes:
            date_series = market_data[date_codes[0]]["date"]

        cash = self.initial_cash
        # positions: {code: {"shares": int, "entry_price": float, "entry_day": int}}
        positions = {}
        trades = []
        equity_curve = []

        # 最大单仓分配金额（防止止损→加仓→大亏的恶性循环）
        max_alloc_per_pos = self.initial_cash * 0.95 / self.max_positions

        for day in range(max_len):
            # ── 本日被卖出/止损的股票，禁止同日再次买入 ──
            sold_today = set()

            # ── 检查现有持仓的止损/止盈 ──
            closed_positions = []
            for code in list(positions.keys()):
                df = market_data.get(code)
                if df is None or day >= len(df):
                    continue
                price = float(df["close"].iloc[day])
                pos = positions[code]
                pnl_pct = price / pos["entry_price"] - 1
                date = df["date"].iloc[day] if "date" in df.columns else day

                if self.stop_loss_pct > 0 and pnl_pct <= -self.stop_loss_pct:
                    revenue = pos["shares"] * price
                    cash += revenue
                    trades.append({"date": str(date), "stock": code, "type": "stop_loss",
                                   "price": price, "shares": pos["shares"],
                                   "pnl_pct": round(pnl_pct, 4), "cash_after": cash})
                    closed_positions.append(code)
                    sold_today.add(code)
                elif self.take_profit_pct > 0 and pnl_pct >= self.take_profit_pct:
                    revenue = pos["shares"] * price
                    cash += revenue
                    trades.append({"date": str(date), "stock": code, "type": "take_profit",
                                   "price": price, "shares": pos["shares"],
                                   "pnl_pct": round(pnl_pct, 4), "cash_after": cash})
                    closed_positions.append(code)
                    sold_today.add(code)

            for code in closed_positions:
                del positions[code]

            # ── 检查卖出信号 ──
            for code in list(positions.keys()):
                df = market_data.get(code)
                if df is None or day >= len(df):
                    continue
                if signals[code][day] == -1:
                    price = float(df["close"].iloc[day])
                    pos = positions[code]
                    revenue = pos["shares"] * price
                    cash += revenue
                    date = df["date"].iloc[day] if "date" in df.columns else day
                    pnl = price / pos["entry_price"] - 1 if pos["entry_price"] > 0 else 0
                    trades.append({"date": str(date), "stock": code, "type": "sell",
                                   "price": price, "shares": pos["shares"],
                                   "pnl_pct": round(pnl, 4), "cash_after": cash})
                    del positions[code]
                    sold_today.add(code)

            # ── 检查买入信号 ──
            # 有信号就买，按可用资金等量分配，但不超过最大单仓金额
            buy_codes = []
            for code in market_data:
                if code in positions or code in sold_today:
                    continue
                if day >= len(market_data[code]):
                    continue
                if signals[code][day] == 1:
                    buy_codes.append(code)

            slots = self.max_positions - len(positions)
            if slots > 0 and buy_codes:
                to_buy = buy_codes[:slots]
                alloc_per = min(cash * 0.95 / len(to_buy), max_alloc_per_pos)
                for code in to_buy:
                    price = float(market_data[code]["close"].iloc[day])
                    if alloc_per <= price * 100:
                        continue
                    shares = int(alloc_per / price)
                    if shares <= 0:
                        continue
                    cost = shares * price
                    cash -= cost
                    positions[code] = {
                        "shares": shares,
                        "entry_price": price,
                        "entry_day": day,
                    }
                    df = market_data[code]
                    date = df["date"].iloc[day] if "date" in df.columns else day
                    trades.append({"date": str(date), "stock": code, "type": "buy",
                                   "price": price, "shares": shares, "cash_after": cash})

            # ── 计算当日净值 ──
            portfolio_value = cash
            for code, pos in positions.items():
                df = market_data.get(code)
                if df is not None and day < len(df):
                    price = float(df["close"].iloc[day])
                    portfolio_value += pos["shares"] * price
            equity_curve.append(portfolio_value)

        # 最终清仓
        for code in list(positions.keys()):
            df = market_data.get(code)
            if df is not None:
                price = float(df["close"].iloc[-1])
                cash += positions[code]["shares"] * price
            del positions[code]

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
            "num_stop_loss": len([t for t in trades if t["type"] == "stop_loss"]),
            "num_take_profit": len([t for t in trades if t["type"] == "take_profit"]),
            "final_value": round(cash, 2),
            "trades": trades,
            "equity_curve": equity_curve,
        }

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / (peak + 1e-8)
        return float(np.min(dd))


class BacktestEngine:
    """多策略回测引擎，支持止损/止盈"""

    def __init__(self, initial_cash=100000, stop_loss_pct: float = 0.0,
                 take_profit_pct: float = 0.0):
        """
        Args:
            initial_cash: 初始资金
            stop_loss_pct: 止损比例 (如 0.05 = -5% 止损)，0 表示不启用
            take_profit_pct: 止盈比例 (如 0.15 = +15% 止盈)，0 表示不启用
        """
        self.initial_cash = initial_cash
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def run(self, df: pd.DataFrame, signals: np.ndarray,
            strategy_name: str = "strategy") -> dict:
        """执行单策略回测，支持盘中止损/止盈检查"""
        if len(df) != len(signals):
            raise ValueError("数据长度与信号长度不匹配")

        cash = self.initial_cash
        position = 0          # 持仓股数
        entry_price = 0.0     # 持仓均价（用于止损/止盈）
        trades = []
        equity_curve = []

        for i in range(len(df)):
            price = df["close"].iloc[i]
            date = df["date"].iloc[i] if "date" in df.columns else i

            # ── 止损/止盈检查 ──
            if position > 0 and entry_price > 0:
                pnl_pct = price / entry_price - 1
                if self.stop_loss_pct > 0 and pnl_pct <= -self.stop_loss_pct:
                    revenue = position * price
                    cash += revenue
                    trades.append({"date": date, "type": "stop_loss", "price": price,
                                   "shares": position, "pnl_pct": round(pnl_pct, 4),
                                   "cash_after": cash})
                    position = 0
                    entry_price = 0.0
                elif self.take_profit_pct > 0 and pnl_pct >= self.take_profit_pct:
                    revenue = position * price
                    cash += revenue
                    trades.append({"date": date, "type": "take_profit", "price": price,
                                   "shares": position, "pnl_pct": round(pnl_pct, 4),
                                   "cash_after": cash})
                    position = 0
                    entry_price = 0.0

            # ── 策略信号执行 ──
            if signals[i] == 1 and cash > price:
                buy_amount = cash * 0.95  # 95% 资金
                shares = int(buy_amount / price)
                cost = shares * price
                cash -= cost
                position += shares
                entry_price = price  # 记录入场价
                trades.append({"date": date, "type": "buy", "price": price,
                               "shares": shares, "cash_after": cash})

            elif signals[i] == -1 and position > 0:
                revenue = position * price
                cash += revenue
                trades.append({"date": date, "type": "sell", "price": price,
                               "shares": position, "pnl_pct": round(price / entry_price - 1, 4)
                               if entry_price > 0 else 0,
                               "cash_after": cash})
                position = 0
                entry_price = 0.0

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
            "num_stop_loss": len([t for t in trades if t["type"] == "stop_loss"]),
            "num_take_profit": len([t for t in trades if t["type"] == "take_profit"]),
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
