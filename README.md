# 智能量化交易系统

基于 **强化学习 + 多智能体架构** 的 A 股每日动态机会点挖掘系统。从新闻热点出发，融合时间序列信号与 RL 决策，通过多策略回测匹配最优市场状态，最终推送可视化结果到飞书。

系统内置 **SQLite 数据库后台** 持久化所有计算结果，并通过 **消息总线 (MessageBus)** 实现智能体间解耦通信。

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           OrchestratorAgent                                 │
│  每日触发: 新闻→板块→数据→信号→RL→回测→风控→报告→可视化→飞书→存储          │
└────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┘
     │    │    │    │    │    │    │    │    │    │    │    │    │    │    │
     ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
 ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
 │Hot │ │Data│ │TS │ │RL │ │Str-│ │Risk│ │Rep-│ │Viz │ │Fei-│ │Sto-│
 │Sect│ │Fetc│ │Sig-│ │Tra-│ │ate-│ │Mgmt│ │ort │ │Age-│ │shu │ │rage│
 │or  │ │h   │ │nal │ │ding│ │gy  │ │    │ │Gen │ │nt  │ │Push│ │Age-│
 │Min-│ │Age-│ │Age-│ │Age-│ │Age-│ │Age-│ │Age-│ │    │ │Age-│ │nt  │
 │ing │ │nt  │ │nt  │ │nt  │ │nt  │ │nt  │ │nt  │ │    │ │nt  │ │    │
 └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘
   │      │      │      │      │      │      │      │      │      │
   └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
                                    │
                          ┌─────────▼──────────┐
                          │   MessageBus       │
                          │  (消息队列中间件)    │
                          └─────────┬──────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  StorageAgent      │
                          │  (SQLite 持久化)    │
                          └─────────┬──────────┘
                                    │
                          ┌─────────▼──────────┐
                          │   trading.db       │
                          │  6 表 · 索引       │
                          └────────────────────┘
```

### 数据流

```
新闻/政策 ──→ 热点板块 ──→ 成分股 ──→ 日线数据 ──→ 技术指标
                                                    │
              ┌────────────────────────────────────┘
              ▼
     时间序列信号 (CUSUM / 峰值谷值 / Bollinger)
              │
              ▼
       RL 交易决策 (启发式评分)
              │
              ▼
     多策略回测 ──→ 市场状态分类 ──→ 风控检查 ──→ 报告/可视化 ──→ 飞书推送
                                                                   │
                                                           ┌───────▼───────┐
                                                           │  MessageBus   │
                                                           │  → 各 topic   │
                                                           └───────┬───────┘
                                                                   ▼
                                                           ┌───────────────┐
                                                           │  StorageAgent │
                                                           │  → SQLite 入库│
                                                           └───────────────┘
```

---

## 多智能体结构

系统由 10 个 Specialist Agent 组成，通过 OrchestratorAgent 编排，Agent 间通过 **MessageBus** 消息队列通信：

| Agent | 职责 | 核心算法 |
|-------|------|---------|
| **HotSectorMiningAgent** | 从央视新闻联播、概念板块热度挖掘热门板块 | jieba 分词 + TF-IDF 关键词→板块映射 |
| **DataFetchAgent** | 获取 A 股日线数据，计算技术指标 | AKShare + BaoStock 双源备份, 多窗口指标计算 |
| **TimeSeriesSignalAgent** | 检测趋势变化、突破、反转时间窗口 | CUSUM 过滤, argrelextrema 峰值检测, Bollinger 突破 |
| **RLTradingAgent** | 在第一层信号基础上做出买卖决策 | 启发式多因子评分 (RSI+价格位置+成交量+TS信号) |
| **MultiStrategyAgent** | 多策略回测 + 市场状态匹配 | 4 种策略并行回测, KMeans 聚类 |
| **RiskManagementAgent** | 回撤预警 + 仓位管理 | Kelly Criterion, VaR(95%), 动态止损 |
| **ReportGeneratorAgent** | 汇总各 Agent 结果生成综合报告 | 模板引擎 |
| **VisualizationAgent** | PyECharts 生成 HTML 可视化报告 | 5 种图表类型 |
| **FeishuPushAgent** | 推送分析报告到飞书 | 飞书 webhook 卡片消息 |
| **StorageAgent** | 持久化各 Agent 结果到 SQLite | 消息订阅 + 自动打标 |

### 编排方式

```python
pipeline = [
    HotSectorMiningAgent(),  # 1. 挖掘热门板块
    DataFetchAgent(),         # 2. 获取数据+计算指标
    TimeSeriesSignalAgent(),  # 3. 时间序列第一层信号
    RLTradingAgent(),         # 4. RL 第二层决策
    MultiStrategyAgent(),     # 5. 多策略回测
    RiskManagementAgent(),    # 6. 风控检查
    ReportGeneratorAgent(),   # 7. 生成报告
    VisualizationAgent(),     # 8. 可视化
    FeishuPushAgent(),        # 9. 飞书推送
    StorageAgent(),           # 10. SQLite 持久化 ← 新增
]
```

每个 Agent 共享 `AgentContext`，基础设施（`MessageBus` / `DatabaseManager`）由 Orchestrator 自动注入。

---

## 存储层 (SQLite)

所有 Agent 的计算结果最终由 **StorageAgent** 持久化到 SQLite 数据库，支持历史回溯和分析。

### 数据库表结构

| 表名 | 用途 | 关键字段 |
|------|------|---------|
| `agent_logs` | Agent 执行日志 | agent_name, date, status, execution_time_ms, error |
| `hot_sectors` | 热门板块快照 | date, sector, heat_score, source, stocks_json |
| `trading_signals` | 交易信号（含 TS 窗口关联） | date, stock, action, confidence, reason |
| `backtest_results` | 回测结果 | date, strategy_name, total_return, sharpe_ratio, max_drawdown |
| `model_labels` | **模型有效标签**（核心） | date, model_name, label_type, label_value, confidence, is_effective |
| `market_cache` | 市场数据缓存 | stock_code, date, data_type, data_json, expires_at |

### 模型标签沉淀

`model_labels` 表是系统的核心设计之一 — 每次管线运行后，StorageAgent 自动：

- 为交易信号打标（买入/卖出信号置信度评估）
- 为回测策略打标（Sharpe > 0.5 标记为有效）
- 为市场状态打标（KMeans 分类结果）
- 为风控指标打标（VaR, 回撤等级）
- 通过 `is_effective` 字段区分高质量 vs 低质量标签

## 消息队列 (MessageBus)

智能体之间通过 **进程内消息总线** 解耦通信，基于 Python `queue.Queue` 实现：

```
Agent A ──publish("sectors")──→ MessageBus ──consume──→ Agent B
Agent B ──publish("signals")──→ MessageBus ──consume──→ Agent C
                                        │
                                        └── StorageAgent 订阅所有 topic
```

- **Topic 机制**: 每个 Agent 完成后 publish 到特定 topic（如 `sectors`、`signals`、`backtest`）
- **非阻塞消费**: `consume(topic, timeout)` 支持超时，防止无限等待
- **解耦**: Agent 不直接依赖下一个 Agent 的接口，只依赖消息格式
- **可观测**: `message_count(topic)` 可查看积压情况

---

## 核心算法

### 1. 热门板块挖掘

- **数据源**: 新浪财经 / 财联社新闻 → 东方财富板块热度排行 → AKShare 概念板块 + BaoStock 日线数据
- **外网访问**: AKShare 需 VPN 访问国内数据源；BaoStock 全球直连（无需 VPN），系统自动降级
- **排除板块**: 银行、保险、证券、信托、金融、券商、多元金融、房地产
- **映射方式**: 20 个预设热点板块关键词表，jieba 分词后匹配

### 2. 时间序列信号 (第一层)

三种检测方法并行：

- **CUSUM**: 累计和检测，threshold=0.02, drift=0.005 — 捕捉趋势变化点
- **峰值/谷值**: `scipy.signal.argrelextrema(order=5)` — 检测局部极值
- **Bollinger 突破**: 价格突破布林带上轨/下轨 — 捕捉突破信号

### 3. RL 交易决策 (第二层)

当前实现为**启发式多因子评分系统**，未来规划接入 PPO：

- **买入评分**: RSI<35 + 价格位置<0.3 → +2 | 成交量比>1.5 + 涨幅>0 → +2 | TS 信号支持 → +2
- **卖出评分**: RSI>70 + 价格位置>0.8 → +2 | 成交量比>1.5 + 跌幅<0 → +2
- **决策规则**: buy_score >= 3 且 > sell_score → 买入；sell_score >= 3 且 > buy_score → 卖出

### 4. 多策略回测

四种策略并行回测：

| 策略 | 逻辑 | 参数 |
|------|------|------|
| 趋势跟踪 | MA5/20 金叉买入、死叉卖出 | 5/20 日均线 |
| 均值回归 | RSI<30 买入、>70 卖出 | RSI(14) |
| 突破策略 | 价格放量突破 Bollinger 上轨买入 | 布林带(20,2) + 量比>1.5 |
| 动量策略 | 5 日涨幅>5%+放量买入 | 5 日收益率 |

- **收益率**: 最终净值 / 初始资金 - 1
- **Sharpe 比率**: sqrt(252) * 日均收益 / 日收益标准差
- **最大回撤**: 净值峰值到谷底的最大跌幅

### 5. 市场状态分类

- **算法**: KMeans(n_clusters=3)
- **特征**: 5 日收益率、20 日波动率(年化)、成交量趋势、5 日 RSI 均值
- **输出**: 牛市 / 熊市 / 震荡市

### 6. 风险管理

- **Kelly Criterion**: `f* = (b*p - q) / b` 计算最优仓位
- **VaR(95%)**: 历史模拟法，95% 置信水平下日最大预期损失
- **回撤预警**: 硬止损 -8%，软预警 -5%
- **仓位控制**: 单笔不超过 Kelly 比例的 25%

---

## 项目结构

```
rl_trading_system/
├── main.py                           # 主入口（终端 / 问答模式）
├── config.py                         # 全局配置
├── requirements.txt
├── README.md
├── src/
│   ├── agents/
│   │   ├── __init__.py               # Agent 注册 + 管线定义
│   │   ├── base.py                   # AgentContext + BaseAgent + OrchestratorAgent
│   │   ├── hot_sector_agent.py       # 热门板块挖掘
│   │   ├── data_agent.py             # 数据获取
│   │   ├── ts_signal_agent.py        # 时间序列信号
│   │   ├── rl_agent.py               # RL 交易决策
│   │   ├── strategy_agent.py         # 多策略回测
│   │   ├── risk_agent.py             # 风险管理
│   │   ├── qa_agent.py               # 本地 LLM 问答 (Qwen2.5-1.5B)
│   │   ├── viz_agent.py              # 可视化
│   │   ├── feishu_agent.py           # 飞书推送
│   │   ├── report_agent.py           # 报告生成
│   │   └── storage_agent.py          # SQLite 持久化 ← 新增
│   ├── storage/                      # ← 新增存储层
│   │   ├── __init__.py
│   │   ├── database.py               # DatabaseManager (SQLite 6 表)
│   │   └── message_bus.py            # MessageBus (Queue 消息队列)
│   ├── data/
│   │   ├── fetcher.py                # AKShare 数据获取
│   │   ├── indicators.py             # 技术指标计算
│   │   └── sector_map.py             # 板块关键词映射
│   ├── backtest/
│   │   ├── engine.py                 # 回测引擎
│   │   ├── strategies.py             # 策略实现库
│   │   └── regime.py                 # 市场状态分类
│   ├── risk/
│   │   └── manager.py                # 风控模型
│   └── viz/
│       ├── charts.py                 # PyECharts 图表生成
│       └── templates/
└── output/
    ├── reports/                      # 可视化报告 HTML
    ├── models/                       # RL 模型
    └── logs/                         # 日志
```

---

## 快速开始

### 安装

```bash
git clone https://github.com/luojiahuli/rl_trading_system.git
cd rl_trading_system
pip install -r requirements.txt
```

### 配置

数据源自动选择（`config.py` 或环境变量）：

```bash
# 自动检测代理（有 VPN 用 AKShare，否则用 BaoStock）
export DATA_SOURCE=auto

# 手动指定代理地址
export PROXY_URL=http://127.0.0.1:7890
```

飞书推送配置：

```python
FEISHU_WEBHOOK_URL = "https://open.feishu.cn/open-apis/bot/v2/hook/your_webhook_id"
FEISHU_SECRET = "your_signature_secret"  # 可选
```

> **数据源说明**: 系统支持双数据源 — **AKShare**（数据最全，含板块/概念数据，需 VPN 访问国内源）和 **BaoStock**（外网直连，无需 VPN，仅历史日线）。默认自动模式，有代理时优先 AKShare，失败自动切换 BaoStock。

### 运行

```bash
# 终端模式 — 执行完整分析管线
python main.py

# 指定日期
python main.py --date 2026-05-24

# 问答模式 — 分析完成后可提问
python main.py --qa
```

### 运行回测 Demo

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from src.backtest.strategies import get_all_strategies
from src.backtest.engine import BacktestEngine
from src.backtest.regime import MarketRegimeClassifier
from src.data.indicators import compute_indicators
import numpy as np, pandas as pd

# 生成合成数据
np.random.seed(42)
n = 252
returns = np.random.randn(n) * 0.015 + 0.0005
returns[100:180] += 0.003
price = 100 * np.exp(np.cumsum(returns))
df = pd.DataFrame({'close': price, 'high': price*1.01, 'low': price*0.99,
                   'volume': np.abs(1e6+np.random.randn(n)*2e5), 'open': price*0.995},
                  index=pd.date_range('2025-06-01', periods=n, freq='B'))
df = compute_indicators(df)

for s in get_all_strategies():
    r = BacktestEngine().run(df, s.generate_signals(df), s.name)
    print(f'{s.name:20s} 收益: {r[\"total_return\"]*100:+.2f}%  Sharpe: {r[\"sharpe_ratio\"]:.3f}')

reg = MarketRegimeClassifier().fit(df)
print(f'市场状态: {reg.get_regime_name(reg.predict(df))}')
"
```

---

## 设计思路

### 为什么是两层信号？

```
时间序列 (第一层)  ──→  识别"什么时候"可能有交易机会
                           ↓
RL 决策 (第二层)    ──→  判断"是否"执行交易
```

- **第一层** 负责缩小搜索空间，过滤掉大部分无效时间窗口
- **第二层** 在精选窗口内做精确决策，降低 RL 探索难度

### 为什么是多策略？

单一策略无法适应所有市场状态。趋势跟踪在牛市中表现优异，但在震荡市中频繁亏损；均值回归在震荡市中表现稳健，在牛市中可能过早卖出。KMeans 聚类识别当前市场状态，选择最适配的策略。

### 为什么排除金融板块？

金融、券商等大盘股受宏观经济和政策影响大，走势相对独立，且与新闻联播/政策热点的关联性较弱。系统聚焦中小盘股，与新闻热点有更强的联动性。

---

## 依赖

| 包 | 用途 |
|------|---------|
| akshare | A 股数据源（主） |
| baostock | A 股数据源（备，外网直连） |
| pandas, numpy | 数据处理 |
| ta | 技术指标计算 |
| jieba, snownlp | 中文分词与新闻 NLP |
| scikit-learn | 市场状态 KMeans 聚类 |
| pyecharts | HTML 可视化报告 |
| requests | 飞书 webhook 推送 |
| stable-baselines3 | RL PPO 训练 (可选) |
| gymnasium | RL 交易环境 (可选) |

---

---

---

## 最新回测结果 (2026-05-27)

**数据**: 实盘 A 股日线（BaoStock）+ 合成数据后备

### 策略绩效

| 策略 | 收益率 | Sharpe | 最大回撤 |
|------|--------|--------|---------|
| momentum | **+155.60%** | **0.830** 🏆 | -30.42% |
| trend_following | +64.16% | 0.553 | -27.67% |
| breakout | +41.33% | 0.225 | -33.28% |
| mean_reversion | +20.84% | 0.461 | -27.70% |

### 市场研判

| 维度 | 判断 |
|------|------|
| 市场阶段 | 震荡 |
| 趋势方向 | sideways |
| 政策预期 | 中性 |
| 置信度 | medium |

**市场宽度**: 80.0% 股票站上MA50 | **指数**: 上证 4093 点 (MA200 +3.36%) | **RSI**: 40.6

### 交易信号

| 股票 | 操作 | 置信度 |
|------|------|--------|
| 002475 立讯精密 | **买入** | 80% |
| 300433 蓝思科技 | **卖出** | 80% |

### 风险分析

| 指标 | 值 |
|------|-----|
| 当前回撤 | **-41.98%** (critical) |
| 风控状态 | 暂停交易（回撤超阈值 -15.0%） |

### 最佳策略

- **最佳 Sharpe**: momentum (0.830)
- **最佳收益**: momentum (+155.60%)

### 可视化报告

[📊 查看完整报告](output/reports/daily_report_20260527.html)


## 路线图

- [x] 多智能体管线 + 热点板块挖掘
- [x] 时间序列信号检测 (CUSUM + 峰值 + Bollinger)
- [x] RL 启发式评分决策
- [x] 4 策略回测 + 市场状态分类
- [x] 风险管理 (Kelly + VaR + 回撤预警)
- [x] PyECharts 可视化报告
- [x] 飞书推送
- [ ] Gymnasium 交易环境 + PPO 在线训练
- [ ] Ollama Qwen2.5-1.5B 本地问答
- [ ] Gradio UI 面板
- [x] 实盘数据验证 (AKShare + BaoStock 双源备份)
- [ ] 多时间周期信号融合
