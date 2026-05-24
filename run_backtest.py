#!/usr/bin/env python3
"""完整回测管线：合成数据 → 策略回测 → 市场分类 → 风控 → 可视化报告"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.base import AgentContext
from src.backtest.strategies import get_all_strategies
from src.backtest.engine import BacktestEngine
from src.backtest.regime import MarketRegimeClassifier
from src.risk.manager import RiskManager
from src.data.indicators import compute_indicators
from src.viz.charts import create_report_html
from config import REPORT_DIR, OUTPUT_DIR

def generate_synthetic_data(n=252, seed=42):
    """生成合成 OHLCV 数据"""
    np.random.seed(seed)
    dates = pd.date_range('2025-06-01', periods=n, freq='B')

    returns = np.random.randn(n) * 0.015 + 0.0005
    returns[100:180] += 0.003   # 上升趋势
    returns[180:] -= 0.002      # 下跌趋势

    price = 100 * np.exp(np.cumsum(returns))
    high = price * (1 + np.abs(np.random.randn(n)) * 0.012)
    low = price * (1 - np.abs(np.random.randn(n)) * 0.012)
    volume = np.abs(1e6 + np.random.randn(n) * 2e5)

    df = pd.DataFrame({
        'date': dates, 'open': price * 0.995,
        'high': high, 'low': low, 'close': price, 'volume': volume,
    }).set_index('date')

    return compute_indicators(df)


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 1. 生成数据
    print("📊 生成合成交易数据...")
    df = generate_synthetic_data()

    # 2. 策略回测
    print("📈 执行多策略回测...")
    backtest_results = []
    strategy_perf = {}

    for strat in get_all_strategies():
        signals = strat.generate_signals(df)
        engine = BacktestEngine(initial_cash=100000)
        result = engine.run(df, signals, strategy_name=strat.name)
        backtest_results.append(result)
        strategy_perf[strat.name] = {
            "total_return": result["total_return"],
            "sharpe_ratio": result["sharpe_ratio"],
            "max_drawdown": result["max_drawdown"],
            "num_trades": result["num_trades"],
        }
        print(f"   {strat.name:20s} 收益: {result['total_return']*100:+7.2f}%  "
              f"Sharpe: {result['sharpe_ratio']:.3f}  "
              f"回撤: {result['max_drawdown']*100:.2f}%  "
              f"交易: {result['num_trades']}")

    # 3. 市场状态
    print("\n🏷️  市场状态分类...")
    regime = MarketRegimeClassifier(n_regimes=3)
    regime.fit(df)
    regime_names = {0: "震荡市", 1: "牛市", 2: "熊市"}

    labels = regime._labels
    for i, name in regime_names.items():
        cnt = sum(labels == i) if labels is not None else 0
        print(f"   {name}: {cnt}/{len(labels)} 天")

    current_regime = regime.predict(df)
    current_regime_name = regime_names.get(current_regime, "未知")
    print(f"   当前状态: {current_regime_name}")

    # 4. 风控
    print("\n⚠️  风险管理...")
    risk = RiskManager()
    peak = df['close'].max()
    current = df['close'].iloc[-1]
    dd_info = risk.check_drawdown(current, peak)
    var_val = risk.compute_var(df['close'].pct_change().dropna())
    print(f"   回撤: {dd_info['dd_pct']*100:.2f}% | 状态: {dd_info['level']}")
    print(f"   VaR(95%): {var_val*100:.2f}%")

    # 5. 构建 AgentContext
    print("\n📦 构建分析上下文...")
    date_str = datetime.now().strftime("%Y-%m-%d")
    context = AgentContext(date=date_str)

    # 热门板块（模拟）
    context.hot_sectors = [
        {"sector": "人工智能", "heat_score": 95, "summary": "政策利好+产业突破", "stocks": ["000001", "000002"]},
        {"sector": "半导体", "heat_score": 88, "summary": "国产替代加速", "stocks": ["000003", "000004"]},
        {"sector": "新能源", "heat_score": 82, "summary": "产业链景气度提升", "stocks": ["000005", "000006"]},
        {"sector": "机器人", "heat_score": 78, "summary": "人形机器人产业进展", "stocks": ["000007", "000008"]},
        {"sector": "低空经济", "heat_score": 72, "summary": "政策试点扩大", "stocks": ["000009", "000010"]},
    ]

    # RL 模拟信号
    context.rl_signals = [
        {"stock": "000001.XSHE", "action": "buy", "confidence": 0.85, "reason": "TS突破+RSI适中"},
        {"stock": "000002.XSHE", "action": "buy", "confidence": 0.72, "reason": "放量突破Bollinger上轨"},
        {"stock": "000003.XSHG", "action": "buy", "confidence": 0.68, "reason": "均线金叉+量比>1.5"},
        {"stock": "000004.XSHG", "action": "sell", "confidence": 0.76, "reason": "RSI>70超买"},
        {"stock": "000005.XSHE", "action": "hold", "confidence": 0.55, "reason": "震荡区间等待方向"},
    ]

    # 回测结果
    context.backtest_results = backtest_results

    # 策略表现
    best_sharpe_name = max(strategy_perf, key=lambda k: strategy_perf[k]["sharpe_ratio"])
    best_return_name = max(strategy_perf, key=lambda k: strategy_perf[k]["total_return"])
    context.strategy_results = {
        "strategy_performance": strategy_perf,
        "best_sharpe_strategy": best_sharpe_name,
        "best_return_strategy": best_return_name,
    }

    # 市场状态
    context.regime = current_regime_name

    # 风控指标
    equity_curve = backtest_results[0]["equity_curve"] if backtest_results else [100000]
    context.risk_metrics = {
        "drawdown": {"dd_pct": dd_info["dd_pct"], "level": dd_info["level"],
                      "message": dd_info["message"]},
        "var_95": var_val,
        "current_equity": equity_curve[-1],
        "peak_equity": max(equity_curve),
        "position_size": risk.position_sizing(100000, 0.02, 0.05, df["close"].iloc[-1]),
    }

    # 6. 生成可视化报告
    print("🎨 生成可视化报告...")
    date_num = date_str.replace("-", "")
    output_path = os.path.join(REPORT_DIR, f"daily_report_{date_num}.html")
    context.viz_path = create_report_html(context, output_path)
    print(f"   报告已生成: {output_path}")

    # 7. 输出摘要
    print(f"\n{'='*50}")
    print(f"📊 分析完成: {date_str}")
    print(f"  热门板块: {len(context.hot_sectors)} 个")
    print(f"  交易信号: {len(context.rl_signals)} 个")
    print(f"  回测次数: {len(context.backtest_results)} 次")
    print(f"  市场状态: {context.regime}")
    print(f"  可视化: {context.viz_path}")
    print(f"{'='*50}")

    # 8. 更新 README 回测结果
    update_readme_md(strategy_perf, current_regime_name, dd_info, var_val, best_sharpe_name, best_return_name)

    print("\n✅ 完成！可视化报告和 README 已更新。")


def update_readme_md(perf, regime, dd_info, var_val, best_sharpe, best_return):
    """更新 README 中的回测结果表格"""
    readme_path = "README.md"
    if not os.path.exists(readme_path):
        return

    # 生成策略表格
    table_lines = [
        "| 策略 | 收益率 | Sharpe | 最大回撤 | 交易次数 |",
        "|------|--------|--------|---------|---------|",
    ]
    for name, m in perf.items():
        table_lines.append(
            f"| {name} | {m['total_return']*100:+.2f}% | "
            f"{m['sharpe_ratio']:.3f} | "
            f"{m['max_drawdown']*100:.2f}% | "
            f"{m['num_trades']} |"
        )

    with open(readme_path, "r") as f:
        content = f.read()

    # 替换或追加结果章节
    results_block = f"""---

## 最新回测结果 ({datetime.now().strftime('%Y-%m-%d')})

**数据**: 252 个交易日合成数据（含上升/下跌/震荡三阶段）

### 策略绩效

{'    '.join(table_lines)}

### 市场状态

- 分类: {regime}
- VaR(95%): {var_val*100:.2f}%
- 当前回撤: {dd_info['dd_pct']*100:.2f}% ({dd_info['level']})

### 最佳策略

- **最佳 Sharpe**: {best_sharpe} ({perf[best_sharpe]['sharpe_ratio']:.3f})
- **最佳收益**: {best_return} ({perf[best_return]['total_return']*100:.2f}%)

### 可视化报告

![报告截图](output/reports/daily_report_{datetime.now().strftime('%Y%m%d')}.html)

"""

    # 替换已有结果区块或追加
    marker = "## 最新回测结果"
    if marker in content:
        before = content.split(marker)[0]
        after = content.split(marker)[1]
        if "## " in after:
            after = after[after.index("## "):]
        else:
            after = ""
        content = before + results_block + "\n" + after
    else:
        # 在路线图前插入
        roadmap_marker = "## 路线图"
        if roadmap_marker in content:
            content = content.replace(roadmap_marker, results_block + "\n" + roadmap_marker)
        else:
            content += "\n" + results_block

    with open(readme_path, "w") as f:
        f.write(content)


if __name__ == "__main__":
    main()
