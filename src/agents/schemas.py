"""Pydantic structured output schemas for LLM-driven trading decisions.

Inspired by TradingAgents: structured output forces consistent decision format
from LLMs, making downstream parsing reliable and enabling the debate pipeline
to pass clean, machine-readable decisions between agents.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PortfolioRating(str, Enum):
    """5-tier rating for investment recommendations."""
    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction."""
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Synthesises bull/bear debate into a clear recommendation with rationale.
    """
    recommendation: PortfolioRating = Field(
        description="Buy/Overweight/Hold/Underweight/Sell. Commit to one side."
    )
    rationale: str = Field(
        description="Summary of key bull/bear arguments that led to this recommendation."
    )
    strategic_actions: str = Field(
        description="Concrete steps: position sizing, entry timing, risk controls."
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Confidence score 0-1 based on signal strength and debate quality.",
    )


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader agent."""
    action: TraderAction = Field(
        description="Buy/Hold/Sell. The concrete transaction direction."
    )
    reasoning: str = Field(
        description="2-4 sentence case for this action, anchored in signals and research plan."
    )
    entry_price: Optional[float] = Field(default=None, description="Entry price target.")
    stop_loss: Optional[float] = Field(default=None, description="Stop-loss price.")
    position_sizing: Optional[str] = Field(default=None, description="e.g. '5% of portfolio'.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence 0-1.")


class PortfolioDecision(BaseModel):
    """Final position decision after risk debate."""
    rating: PortfolioRating = Field(
        description="Final rating after risk consideration."
    )
    executive_summary: str = Field(
        description="Concise action plan: entry, sizing, risk levels, horizon."
    )
    investment_thesis: str = Field(
        description="Detailed reasoning anchored in debate evidence."
    )
    price_target: Optional[float] = Field(default=None, description="Target price.")
    time_horizon: Optional[str] = Field(default=None, description="e.g. '1-3 months'.")
    position_pct: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Final position size as fraction of portfolio.",
    )


class RiskAssessment(BaseModel):
    """Risk assessment from a single risk debate perspective."""
    perspective: str = Field(description="Aggressive/Conservative/Neutral")
    max_position_pct: float = Field(ge=0.0, le=1.0, description="Max position size allowed.")
    stop_loss_pct: float = Field(ge=0.0, le=1.0, description="Stop loss threshold.")
    verdict: str = Field(description="Risk verdict and key concerns.")
    score: float = Field(default=0.0, ge=-1.0, le=1.0, description="Risk score -1 to 1.")
