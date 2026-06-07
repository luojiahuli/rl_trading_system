#!/usr/bin/env python3
"""多策略回测 + 市场状态匹配 Agent"""
import pandas as pd
import numpy as np
from ..agents.base import AgentContext, BaseAgent
from ..backtest.strategies import get_all_strategies, get_enhanced_strategies, get_mtf_strategies, get_enhanced_mtf_strategies
from ..backtest.engine import BacktestEngine, PortfolioBacktestEngine
from ..backtest.regime import MarketRegimeClassifier


from config import INITIAL_CASH


class MultiStrategyAgent(BaseAgent):
    name = "multi_strategy"
    description = "多策略回测 + 市场状态匹配"

    def execute(self, context: AgentContext) -> AgentContext:
        engine = BacktestEngine(initial_cash=INITIAL_CASH)
        portfolio_engine = PortfolioBacktestEngine(initial_cash=INITIAL_CASH)
        all_results = []
        portfolio_results = []

        # 评估每个股票上的策略表现（用于策略对比）
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

        # 组合级回测（多股票分散，用于展示收益曲线）
        portfolio_strategies = get_enhanced_strategies() + get_enhanced_mtf_strategies()
        all_strategy_names = set()
        for strategy in portfolio_strategies:
            if strategy.name in all_strategy_names:
                continue
            all_strategy_names.add(strategy.name)
            try:
                result = portfolio_engine.run(
                    context.market_data, strategy, strategy.name
                )
                portfolio_results.append(result)
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
            # 用组合级回测结果替换单股结果（用于收益曲线展示，按收益率排序）
            portfolio_results.sort(key=lambda x: x.get("total_return", 0), reverse=True)
            context.backtest_results = portfolio_results

        context.warnings.append(f"完成 {len(all_results)} 次单股回测 + {len(portfolio_results)} 次组合回测")
        return context
