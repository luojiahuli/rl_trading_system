"""多智能体框架 + 所有 Specialist Agent

Pipeline (merged original + TradingAgents LLM debate):
  HotSectorMining → DataFetch → TimeSeriesSignal
  → DebatePipeline (Bull/Bear debate → ResearchManager → RiskDebate → PortfolioManager)
  → RLTrading (original heuristic)
  → MultiStrategy → RiskManagement → MarketJudgement
  → ReportGenerator → Visualization → FeishuPush
  → TradeJournal → PositionAnalysis → Storage → Reflection
"""
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
from .market_agent import MarketJudgementAgent
from .trade_journal_agent import TradeJournalAgent
from .position_agent import PositionAnalysisAgent

# === TradingAgents 集成 ===
from .debate_coordinator import DebatePipelineCoordinator
from .memory_agent import ReflectionAgent
from ..llm.client import LLMClient


def build_daily_pipeline(enable_llm_debate: bool = True) -> list:
    """构建每日交易管线（含 TradingAgents LLM debate 集成）

    Args:
        enable_llm_debate: 是否启用 LLM 辩论管线（默认为 True）。
                           False 时回退到纯原始管线。
    """
    pipeline = [
        HotSectorMiningAgent(),
        DataFetchAgent(),
        TimeSeriesSignalAgent(),
    ]

    if enable_llm_debate:
        # TradingAgents 辩论管线（并行于原始 RL 信号）
        pipeline.append(DebatePipelineCoordinator())

    pipeline.extend([
        RLTradingAgent(),                     # 原始启发式 RL 信号（与 LLM 信号共存）
        MultiStrategyAgent(),
        RiskManagementAgent(),
        MarketJudgementAgent(),
        ReportGeneratorAgent(),
        VisualizationAgent(),
        FeishuPushAgent(),
        TradeJournalAgent(),
        PositionAnalysisAgent(),
        StorageAgent(),                       # 持久化所有结果（含 debate 结果）
    ])

    if enable_llm_debate:
        pipeline.append(ReflectionAgent())    # 记忆 + 事后反思（需在 storage 之后）

    return pipeline


def build_debate_only_pipeline() -> list:
    """仅 TradingAgents 辩论管线（不含原始 RL 信号，用于对比测试）"""
    return [
        HotSectorMiningAgent(),
        DataFetchAgent(),
        TimeSeriesSignalAgent(),
        DebatePipelineCoordinator(),
        MultiStrategyAgent(),
        RiskManagementAgent(),
        MarketJudgementAgent(),
        ReportGeneratorAgent(),
        VisualizationAgent(),
        FeishuPushAgent(),
        StorageAgent(),
        ReflectionAgent(),
    ]


__all__ = [
    "AgentContext", "BaseAgent", "OrchestratorAgent",
    "HotSectorMiningAgent", "DataFetchAgent", "TimeSeriesSignalAgent",
    "RLTradingAgent", "MultiStrategyAgent", "RiskManagementAgent",
    "QAAgent", "VisualizationAgent", "FeishuPushAgent",
    "ReportGeneratorAgent", "StorageAgent", "MarketJudgementAgent",
    "TradeJournalAgent", "PositionAnalysisAgent",
    # TradingAgents integration
    "DebatePipelineCoordinator", "ReflectionAgent", "LLMClient",
    "build_daily_pipeline", "build_debate_only_pipeline",
]
