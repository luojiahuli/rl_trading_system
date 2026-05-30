"""
系统配置
"""
import os

# ====== 飞书推送 ======
FEISHU_WEBHOOK_URL = os.getenv(
    "FEISHU_WEBHOOK_URL",
    "https://open.feishu.cn/open-apis/bot/v2/hook/your_webhook_id",
)
FEISHU_SECRET = os.getenv("FEISHU_SECRET", "")

# ====== 数据配置 ======
START_DATE = "2024-01-01"
END_DATE = None  # None = 至今

# ====== 数据源优先级 ======
# "akshare" = AKShare 主 + BaoStock 备（需 VPN 访问国内源）
# "baostock" = 仅 BaoStock（外网直连，无需 VPN）
# "auto" = 自动检测代理，有代理用 akshare，否则 baostock
DATA_SOURCE = os.getenv("DATA_SOURCE", "auto")

# ====== 代理配置 ======
# 留空则自动检测 HTTP_PROXY / HTTPS_PROXY / ALL_PROXY 环境变量
PROXY_URL = os.getenv("PROXY_URL", "")

# ====== 板块配置 ======
EXCLUDED_SECTORS = [
    "银行", "保险", "证券", "信托", "金融",
    "券商", "多元金融", "房地产",
]

# ====== 资金配置 ======
INITIAL_CASH = 1000000             # 初始资金 100 万

# ====== RL 训练配置 ======
RL_TOTAL_TIMESTEPS = 200000
RL_LEARNING_RATE = 3e-4
RL_BUY_POSITION_PCT = 0.2       # 单次买入占总资金比例
RL_ADD_POSITION_PCT = 0.1       # 补仓比例
RL_MAX_POSITIONS = 5            # 最大持仓数
RL_STOP_LOSS = -0.08            # 硬止损 -8%
RL_STOP_LOSS_WARN = -0.05       # 软预警 -5%

# ====== 风控配置 ======
RISK_MAX_DRAWDOWN = -0.15       # 最大回撤阈值
RISK_KELLY_FRACTION = 0.25      # Kelly 系数（保守）

# ====== LLM 配置 (TradingAgents 集成) ======
# --- 两层 LLM 策略 ---
# quick_thinking_llm:  快速/便宜的模型 (DeepSeek)，用于分析师、研究员、交易员、风险辩论
# deep_thinking_llm:   更强大的模型 (MiniMax)，用于管理者决策（Research Manager, Portfolio Manager）
#
# 支持三种 provider: "ollama" (本地), "deepseek", "minimax", 或 "openai" (通用 OpenAI-compatible)
#
# 默认从 Hermes (~/.hermes/.env) 读取 API Key

def _load_hermes_env(key: str) -> str:
    """从 Hermes .env 读取配置"""
    import os
    val = os.getenv(key, "")
    if val:
        return val
    try:
        for line in open(os.path.expanduser("~/.hermes/.env")):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                if k == key:
                    return v
    except (FileNotFoundError, IOError):
        pass
    # 尝试从 config.yaml 读取 minimax base_url
    if key == "MINIMAX_BASE_URL":
        try:
            import yaml
            cfg = yaml.safe_load(open(os.path.expanduser("~/.hermes/config.yaml")))
            return cfg.get("model", {}).get("base_url", "https://api.minimax.chat/v1")
        except Exception:
            return "https://api.minimax.chat/v1"
    if key == "MINIMAX_MODEL":
        try:
            import yaml
            cfg = yaml.safe_load(open(os.path.expanduser("~/.hermes/config.yaml")))
            return cfg.get("model", {}).get("default", "MiniMax-M2.7").split("/")[-1]
        except Exception:
            return "MiniMax-M2.7"
    return ""

# Quick thinking LLM → DeepSeek (便宜、快速)
QUICK_LLM_PROVIDER = os.getenv("QUICK_LLM_PROVIDER", "deepseek")
QUICK_LLM_MODEL = os.getenv("QUICK_LLM_MODEL", "deepseek-chat")
QUICK_LLM_API_KEY = os.getenv("QUICK_LLM_API_KEY", _load_hermes_env("DEEPSEEK_API_KEY"))
QUICK_LLM_BASE_URL = os.getenv("QUICK_LLM_BASE_URL", "https://api.deepseek.com/v1")

# Deep thinking LLM → MiniMax M2.7 (更强大)
DEEP_LLM_PROVIDER = os.getenv("DEEP_LLM_PROVIDER", "minimax")
DEEP_LLM_MODEL = os.getenv("DEEP_LLM_MODEL", _load_hermes_env("MINIMAX_MODEL"))
DEEP_LLM_API_KEY = os.getenv("DEEP_LLM_API_KEY", _load_hermes_env("MINIMAX_API_KEY"))
DEEP_LLM_BASE_URL = os.getenv("DEEP_LLM_BASE_URL", _load_hermes_env("MINIMAX_BASE_URL"))

# LLM 辩论配置
LLM_DEBATE_ENABLED = os.getenv("LLM_DEBATE_ENABLED", "true").lower() == "true"
LLM_DEBATE_MAX_ROUNDS = int(os.getenv("LLM_DEBATE_MAX_ROUNDS", "2"))  # Bull/Bear 辩论轮数
LLM_RISK_MAX_ROUNDS = int(os.getenv("LLM_RISK_MAX_ROUNDS", "1"))      # 风险辩论轮数
LLM_MIN_SIGNAL_CONFIDENCE = float(os.getenv("LLM_MIN_SIGNAL_CONFIDENCE", "0.3"))

# Ollama (本地 LLM, 仅 ollama provider 时使用)
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

# ====== 输出目录 ======
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# ====== 数据库 ======
DB_PATH = os.path.join(OUTPUT_DIR, "trading.db")
