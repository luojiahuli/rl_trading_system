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

# ====== 板块配置 ======
EXCLUDED_SECTORS = [
    "银行", "保险", "证券", "信托", "金融",
    "券商", "多元金融", "房地产",
]

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

# ====== 本地模型 ======
LLM_MODEL_PATH = os.getenv(
    "LLM_MODEL_PATH",
    "",  # 设为 "" 时使用 Ollama，填入路径则用 llama-cpp-python
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

# ====== 输出目录 ======
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
