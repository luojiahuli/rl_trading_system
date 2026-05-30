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
# quick_thinking_llm: 快速/便宜的模型，用于分析师、研究员、交易员、风险辩论
# deep_thinking_llm:  更强大的模型，用于管理者决策（Research Manager, Portfolio Manager）
#
# 支持两种 provider: "ollama" (本地) 或 "openai" (OpenAI-compatible API)

# Quick thinking LLM (analysts, researchers, trader, risk debaters)
QUICK_LLM_PROVIDER = os.getenv("QUICK_LLM_PROVIDER", "ollama")
QUICK_LLM_MODEL = os.getenv("QUICK_LLM_MODEL", "qwen2.5:1.5b")
QUICK_LLM_API_KEY = os.getenv("QUICK_LLM_API_KEY", "")
QUICK_LLM_BASE_URL = os.getenv("QUICK_LLM_BASE_URL", "http://localhost:11434")

# Deep thinking LLM (managers)
DEEP_LLM_PROVIDER = os.getenv("DEEP_LLM_PROVIDER", "ollama")
DEEP_LLM_MODEL = os.getenv("DEEP_LLM_MODEL", "qwen2.5:1.5b")
DEEP_LLM_API_KEY = os.getenv("DEEP_LLM_API_KEY", "")
DEEP_LLM_BASE_URL = os.getenv("DEEP_LLM_BASE_URL", "http://localhost:11434")

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
