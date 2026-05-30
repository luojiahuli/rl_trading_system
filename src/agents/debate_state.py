"""Debate state management for Bull/Bear and Risk debates.

Inspired by TradingAgents' InvestDebateState and RiskDebateState patterns.
Tracks conversation rounds, history, and current speaker for alternating debates.
"""


class InvestDebateState:
    """Bull/Bear researcher debate state.

    Tracks alternating debate rounds between Bull and Bear researchers.
    After max_rounds * 2 exchanges, the debate concludes and the Research Manager
    synthesises a decision.
    """

    def __init__(self, max_rounds: int = 2):
        self.max_rounds = max_rounds
        self.reset()

    def reset(self):
        self.bull_history: list[str] = []
        self.bear_history: list[str] = []
        self.current_speaker: str = "bull"  # bull or bear
        self.count: int = 0
        self.is_concluded: bool = False

    def add_bull_argument(self, argument: str):
        self.bull_history.append(argument)
        self.count += 1
        self.current_speaker = "bear"
        self._check_concluded()

    def add_bear_argument(self, argument: str):
        self.bear_history.append(argument)
        self.count += 1
        self.current_speaker = "bull"
        self._check_concluded()

    def _check_concluded(self):
        if self.count >= 2 * self.max_rounds:
            self.is_concluded = True

    def get_context(self) -> dict:
        return {
            "bull_arguments": self.bull_history,
            "bear_arguments": self.bear_history,
            "round": self.count // 2 + 1,
            "total_rounds": self.max_rounds,
            "current_speaker": self.current_speaker,
            "is_concluded": self.is_concluded,
        }


class RiskDebateState:
    """Three-perspective risk debate state.

    Cycles through Aggressive → Conservative → Neutral → Aggressive.
    After max_rounds * 3 exchanges, the Portfolio Manager synthesises the
    final decision.
    """

    # Rotation order
    SPEAKER_ORDER = ["aggressive", "conservative", "neutral"]

    def __init__(self, max_rounds: int = 1):
        self.max_rounds = max_rounds
        self.reset()

    def reset(self):
        self.aggressive_history: list[str] = []
        self.conservative_history: list[str] = []
        self.neutral_history: list[str] = []
        self.speaker_index: int = 0
        self.count: int = 0
        self.is_concluded: bool = False

    @property
    def current_speaker(self) -> str:
        return self.SPEAKER_ORDER[self.speaker_index]

    def add_argument(self, perspective: str, argument: str):
        if perspective == "aggressive":
            self.aggressive_history.append(argument)
        elif perspective == "conservative":
            self.conservative_history.append(argument)
        elif perspective == "neutral":
            self.neutral_history.append(argument)
        self.count += 1
        self.speaker_index = (self.speaker_index + 1) % 3
        self._check_concluded()

    def _check_concluded(self):
        if self.count >= 3 * self.max_rounds:
            self.is_concluded = True

    def get_context(self) -> dict:
        return {
            "aggressive_arguments": self.aggressive_history,
            "conservative_arguments": self.conservative_history,
            "neutral_arguments": self.neutral_history,
            "round": self.count // 3 + 1,
            "total_rounds": self.max_rounds,
            "current_speaker": self.current_speaker,
            "is_concluded": self.is_concluded,
        }
