"""
Agent 基类 + 共享上下文 + 编排器

支持消息队列通信和 SQLite 持久化:
  - AgentContext 挂载 message_bus 和 db 供所有 Agent 使用
  - Orchestrator 创建 bus → 注入 agent → 每个 agent publish 结果到 topic
  - StorageAgent 订阅所有 topic 落库
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

    # === 市场研判 ===
    market_judgement: dict = field(default_factory=dict)  # 市场整体研判结果

    # === 输出 ===
    report_text: str = ""
    report_html: str = ""
    viz_path: str = ""

    # === 状态 ===
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    # === 基础设施（由 Orchestrator 注入） ===
    bus: Any = None                    # MessageBus 实例
    db: Any = None                     # DatabaseManager 实例


class BaseAgent(ABC):
    """Agent 基类"""
    name: str = "base"
    description: str = ""

    @abstractmethod
    def execute(self, context: AgentContext) -> AgentContext:
        ...

    # ── 消息队列辅助 ──────────────────────────────────────

    def publish(self, context: AgentContext, topic: str, message: Any = None):
        """向消息总线发布本 Agent 的结果"""
        if context.bus is not None:
            context.bus.publish(topic, message if message is not None else self._snapshot(context))

    def consume(self, context: AgentContext, topic: str, timeout: float = None) -> Any:
        """从消息总线消费指定主题的消息"""
        if context.bus is not None:
            return context.bus.consume(topic, timeout=timeout)
        return None

    @staticmethod
    def _snapshot(context: AgentContext) -> dict:
        """从 context 中提取通用快照供 bus 传递"""
        return {
            "hot_sectors": context.hot_sectors,
            "ts_signals": context.ts_signals,
            "rl_signals": context.rl_signals,
            "backtest_results": context.backtest_results,
            "regime": context.regime,
            "strategy_results": context.strategy_results,
            "risk_metrics": context.risk_metrics,
            "stock_pool": context.stock_pool,
            "market_judgement": context.market_judgement,
        }

    def __repr__(self) -> str:
        return f"<Agent {self.name}>"


class OrchestratorAgent(BaseAgent):
    """编排器：按序执行 Agent，通过消息总线传递数据"""
    name = "orchestrator"
    description = "编排所有 Agent 按依赖顺序执行"

    def __init__(self, agents: list, message_bus=None, database=None):
        self.agents = agents
        self.message_bus = message_bus
        self.database = database

    def execute(self, context: AgentContext) -> AgentContext:
        # 注入基础设施
        context.bus = self.message_bus
        context.db = self.database

        import time
        for agent in self.agents:
            if context.errors:
                context.warnings.append(f"跳过 {agent.name}（上游失败）")
                continue

            aid = agent.name
            t0 = time.time()
            try:
                # 执行 agent
                context = agent.execute(context)

                # Agent 通过 bus publish 自己的结果
                if context.bus is not None:
                    topic_map = {
                        "hot_sector_mining": "sectors",
                        "data_fetch": "market_data",
                        "ts_signal": "ts_signals",
                        "rl_trading": "rl_signals",
                        "multi_strategy": "backtest",
                        "risk_management": "risk",
                        "market_judgement": "market",
                        "report_generator": "report",
                        "visualization": "viz",
                        "feishu_push": "feishu",
                        "storage": "storage",
                    }
                    topic = topic_map.get(aid)
                    if topic:
                        agent.publish(context, topic)

                # 记录执行日志
                elapsed = int((time.time() - t0) * 1000)
                if context.db is not None:
                    context.db.save_agent_log(
                        agent_name=aid,
                        date=context.date,
                        status="ok" if not context.errors else "error",
                        execution_time_ms=elapsed,
                    )

                print(f"  ✓ {aid} ({elapsed}ms)")

            except Exception as e:
                elapsed = int((time.time() - t0) * 1000)
                context.errors.append(f"{aid} 失败: {e}")
                print(f"  ✗ {aid} 失败: {e} ({elapsed}ms)")
                if context.db is not None:
                    context.db.save_agent_log(
                        agent_name=aid, date=context.date,
                        status="error", execution_time_ms=elapsed, error=str(e),
                    )

        return context
