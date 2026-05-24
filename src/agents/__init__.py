"""多智能体框架 + 所有 Specialist Agent"""
from .base import AgentContext, BaseAgent, OrchestratorAgent
from .hot_sector_agent import HotSectorMiningAgent
from .data_agent import DataFetchAgent
from .ts_signal_agent import TimeSeriesSignalAgent
from .rl_agent import RLTradingAgent
from .strategy_agent import MultiStrategyAgent
from .risk_agent import RiskManagementAgent
from .qa_agent import QAAgent
from .viz_agent import VisualizationAgent
from .feishu_agent import FeishuPushAgent
from .report_agent import ReportGeneratorAgent
from .storage_agent import StorageAgent


def build_daily_pipeline() -> list:
    """构建每日交易管线（含存储 Agent）"""
    return [
        HotSectorMiningAgent(),
        DataFetchAgent(),
        TimeSeriesSignalAgent(),
        RLTradingAgent(),
        MultiStrategyAgent(),
        RiskManagementAgent(),
        ReportGeneratorAgent(),
        VisualizationAgent(),
        FeishuPushAgent(),
        StorageAgent(),              # ← 持久化所有结果到 SQLite
    ]


__all__ = [
    "AgentContext", "BaseAgent", "OrchestratorAgent",
    "HotSectorMiningAgent", "DataFetchAgent", "TimeSeriesSignalAgent",
    "RLTradingAgent", "MultiStrategyAgent", "RiskManagementAgent",
    "QAAgent", "VisualizationAgent", "FeishuPushAgent",
    "ReportGeneratorAgent", "StorageAgent", "build_daily_pipeline",
]
