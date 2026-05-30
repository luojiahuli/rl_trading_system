"""Debate Pipeline Coordinator - runs the full TradingAgents-style debate flow.

Encapsulates the entire LLM-driven debate pipeline:
  1. For each stock with strong TS signals, run Bull ↔ Bear debate
  2. Research Manager synthesises debate into ResearchPlan
  3. Run Aggressive ↔ Conservative ↔ Neutral risk debate
  4. Portfolio Manager produces final PortfolioDecision
  5. Appends LLM-generated signals to rl_signals alongside existing heuristic signals

This runs as a SINGLE agent in the sequential pipeline, keeping the orchestration
clean while internally managing the multi-agent debate flow.

Inspired by TradingAgents' StateGraph with debate cycles.
"""
import json
from ..agents.base import AgentContext, BaseAgent
from ..agents.debate_state import InvestDebateState, RiskDebateState
from ..agents.debate_agent import BullResearcherAgent, BearResearcherAgent
from ..agents.research_manager_agent import ResearchManagerAgent
from ..agents.risk_debate_agent import AggressiveRiskAgent, ConservativeRiskAgent, NeutralRiskAgent
from ..agents.portfolio_manager_agent import PortfolioManagerAgent
from ..llm.client import LLMClient
from ..agents.memory_agent import TradingMemoryLog


class DebatePipelineCoordinator(BaseAgent):
    """Coordinates the full LLM debate pipeline for all stocks with strong signals."""

    name = "debate_pipeline"
    description = "TradingAgents-style LLM debate pipeline for A-share stocks"

    def __init__(self, llm_client: LLMClient = None, max_debate_rounds: int = 2,
                 max_risk_rounds: int = 1, min_signal_confidence: float = 0.3):
        self.llm = llm_client or LLMClient.from_config()
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_rounds = max_risk_rounds
        self.min_signal_confidence = min_signal_confidence
        # Sub-agents (lazy init per debate round)
        self._bull = None
        self._bear = None
        self._research_mgr = None
        self._risk_agg = None
        self._risk_con = None
        self._risk_neu = None
        self._portfolio_mgr = None
        self._memory = TradingMemoryLog()

    def _init_agents(self):
        if self._bull is None:
            self._bull = BullResearcherAgent(self.llm)
            self._bear = BearResearcherAgent(self.llm)
            self._research_mgr = ResearchManagerAgent(self.llm)
            self._risk_agg = AggressiveRiskAgent(self.llm)
            self._risk_con = ConservativeRiskAgent(self.llm)
            self._risk_neu = NeutralRiskAgent(self.llm)
            self._portfolio_mgr = PortfolioManagerAgent(self.llm)

    def execute(self, context: AgentContext) -> AgentContext:
        self._init_agents()

        # Get stocks with TS signals (strongest candidates for debate)
        debate_candidates = self._get_debate_candidates(context)
        if not debate_candidates:
            context.warnings.append("[DebatePipeline] No strong signal stocks for debate")
            return context

        # Inject memory context into the debate
        for stock in debate_candidates:
            past_ctx = self._memory.get_past_context(stock)
            if past_ctx:
                context.past_context = past_ctx
            break  # Just inject once for context

        # Run debate for each candidate stock
        context.warnings.append(
            f"[DebatePipeline] Running debate pipeline for {len(debate_candidates)} stocks"
        )

        for stock in debate_candidates:
            self._debate_one_stock(context, stock)

        context.warnings.append(
            f"[DebatePipeline] LLM debate generated "
            f"{sum(1 for s in context.rl_signals if s.get('strategy')=='llm_debate')} signals "
            f"(total signals: {len(context.rl_signals)})"
        )
        return context

    def _debate_one_stock(self, context: AgentContext, stock: str):
        """Run the full debate pipeline for a single stock."""
        # Store current debate stock
        context.current_debate_stock = stock

        # Phase 1: Bull/Bear debate
        context.invest_debate = InvestDebateState(max_rounds=self.max_debate_rounds)
        debate = context.invest_debate

        # Run alternating debate rounds
        while not debate.is_concluded:
            if debate.current_speaker == "bull":
                context = self._bull.execute(context)
            else:
                context = self._bear.execute(context)

        # Phase 2: Research Manager synthesises
        context = self._research_mgr.execute(context)

        # Phase 3: Risk debate (three perspectives)
        context.risk_debate = RiskDebateState(max_rounds=self.max_risk_rounds)
        context.risk_assessments = {}
        risk_debate = context.risk_debate

        while not risk_debate.is_concluded:
            speaker = risk_debate.current_speaker
            if speaker == "aggressive":
                context = self._risk_agg.execute(context)
            elif speaker == "conservative":
                context = self._risk_con.execute(context)
            elif speaker == "neutral":
                context = self._risk_neu.execute(context)

        # Phase 4: Portfolio Manager final decision
        context = self._portfolio_mgr.execute(context)

        # Cleanup
        context.current_debate_stock = ""
        context.invest_debate = None
        context.risk_debate = None

    def _get_debate_candidates(self, context: AgentContext) -> list[str]:
        """Select stocks with strongest TS signals for LLM debate."""
        signal_counts = {}
        for s in context.ts_signals:
            stock = s.get("stock", "")
            conf = s.get("confidence", 0)
            if conf >= self.min_signal_confidence:
                signal_counts[stock] = signal_counts.get(stock, 0) + 1

        # Sort by signal count, take top stocks
        sorted_stocks = sorted(signal_counts.items(), key=lambda x: -x[1])
        # Limit to stocks in our pool with data
        valid = [s for s, _ in sorted_stocks[:10] if s in context.indicators]
        return valid[:5]  # Max 5 stocks per run (LLM cost control)
