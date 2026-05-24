#!/usr/bin/env python3
"""多策略回测 + 市场状态匹配 Agent"""
import pandas as pd
import numpy as np
from ..agents.base import AgentContext, BaseAgent
from ..backtest.strategies import get_all_strategies
from ..backtest.engine import BacktestEngine
from ..backtest.regime import MarketRegimeClassifier


class MultiStrategyAgent(BaseAgent):
    name = "multi_strategy"
    description = "多策略回测 + 市场状态匹配"

    def execute(self, context: AgentContext) -> AgentContext:
        engine = BacktestEngine(initial_cash=100000)
        all_results = []

        # 评估每个股票上的策略表现
        for code, df in context.market_data.items():
            if len(df) < 30:
                continue

            # 市场状态分类
            classifier = MarketRegimeClassifier()
            try:
                classifier.fit(df)
                regime = classifier.predict(df)
                context.regime = classifier.get_regime_name(regime)
            except Exception:
                context.regime = "未知"

            # 每个策略回测
            for strategy in get_all_strategies():
                try:
                    signals = strategy.generate_signals(df)
                    result = engine.run(df, signals, strategy.name)
                    result["stock"] = code
                    all_results.append(result)
                except Exception:
                    continue

        # 计算市场状态匹配度
        if all_results:
            df_results = pd.DataFrame(all_results)
            strategy_perf = df_results.groupby("strategy").agg({
                "total_return": "mean",
                "sharpe_ratio": "mean",
                "max_drawdown": "mean",
            }).to_dict("index")

            # 找最佳策略
            best_strategy = max(strategy_perf.items(),
                                key=lambda x: x[1]["sharpe_ratio"])[0] if strategy_perf else ""
            best_return = max(strategy_perf.items(),
                              key=lambda x: x[1]["total_return"])[0] if strategy_perf else ""

            context.strategy_results = {
                "regime": context.regime,
                "strategy_performance": strategy_perf,
                "best_sharpe_strategy": best_strategy,
                "best_return_strategy": best_return,
                "total_backtests": len(all_results),
            }
            context.backtest_results = all_results

        context.warnings.append(f"完成 {len(all_results)} 次回测")
        return context
