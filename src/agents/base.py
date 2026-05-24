#!/usr/bin/env python3
"""
Agent 基类 + 共享上下文 + 编排器
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class AgentContext:
    """智能体共享上下文"""

    # === 输入 ===
    date: str = ""                     # 当前日期 YYYY-MM-DD

    # === 新闻 & 板块 ===
    news_data: list = field(default_factory=list)
    hot_sectors: list = field(default_factory=list)  # [{sector, heat_score, summary, stocks}]

    # === 市场数据 ===
    market_data: dict = field(default_factory=dict)   # {stock_code: DataFrame OHLCV}
    stock_pool: list = field(default_factory=list)    # 当前关注的股票列表
    indicators: dict = field(default_factory=dict)    # {stock_code: {indicator: value}}

    # === 信号 ===
    ts_signals: list = field(default_factory=list)    # 时间序列信号
    rl_signals: list = field(default_factory=list)    # RL 交易信号

    # === 策略 & 回测 ===
    regime: str = ""                   # 当前市场状态
    strategy_results: dict = field(default_factory=dict)
    backtest_results: list = field(default_factory=list)

    # === 风险管理 ===
    portfolio: dict = field(default_factory=dict)     # 持仓信息
    risk_metrics: dict = field(default_factory=dict)  # 风控指标

    # === 输出 ===
    report_text: str = ""
    report_html: str = ""
    viz_path: str = ""

    # === 状态 ===
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class BaseAgent(ABC):
    """Agent 基类"""
    name: str = "base"
    description: str = ""

    @abstractmethod
    def execute(self, context: AgentContext) -> AgentContext:
        ...

    def __repr__(self) -> str:
        return f"<Agent {self.name}>"


class OrchestratorAgent(BaseAgent):
    """编排器：按序执行 Agent"""
    name = "orchestrator"
    description = "编排所有 Agent 按依赖顺序执行"

    def __init__(self, agents: list):
        self.agents = agents

    def execute(self, context: AgentContext) -> AgentContext:
        for agent in self.agents:
            if context.errors:
                context.warnings.append(f"跳过 {agent.name}（上游失败）")
                continue
            try:
                context = agent.execute(context)
            except Exception as e:
                context.errors.append(f"{agent.name} 失败: {e}")
        return context
