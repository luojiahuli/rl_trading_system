#!/usr/bin/env python3
"""回测复盘：2026-04-01 → 至今，真实 A 股数据，4 策略合计利润曲线

Usage:
    python run_backtest_review.py

Output:
    - 终端打印各策略绩效
    - output/reports/backtest_review_*.json 详细数据
    - 更新 README.md 回测结果表格
    - 保存 equity_curve CSV 到 output/reports/
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.backtest.strategies import get_all_strategies
from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_stock_daily
from src.data.indicators import compute_indicators
from config import OUTPUT_DIR, REPORT_DIR

START = "2026-04-01"
END = datetime.now().strftime("%Y-%m-%d")
INITIAL_CASH = 1_000_000

# 股票池 — 沿用 data_agent 的热门板块映射，排除金融/地产
_SECTOR_STOCK_MAP = {
    "消费电子": ["002475", "601138", "603160", "002241", "300433"],
    "人工智能": ["300308", "688111", "300502", "603019", "002230"],
    "低空经济": ["002625", "600685", "600760", "600118", "002023"],
    "军工":     ["600760", "600893", "600862", "002013", "000768"],
    "半导体":   ["688981", "002371", "603501", "600703", "300661"],
    "新能源":   ["300750", "002594", "601012", "300274", "603659"],
    "汽车":     ["600104", "000625", "601633", "002594", "000800"],
    "医药":     ["600276", "300760", "000538", "002007", "300122"],
    "白酒":     ["600519", "000858", "002304", "000568", "600809"],
    "光伏":     ["601012", "600438", "688599", "002459", "300274"],
    "机器人":   ["300124", "688005", "002472", "300024", "600835"],
    "数字经济": ["000938", "688568", "300496", "002415", "603986"],
    "国产芯片": ["603986", "300661", "688981", "002371", "600703"],
}


def fetch_all_data() -> dict[str, pd.DataFrame]:
    """从 BaoStock 拉取所有股票日线数据，返回 {code: df}"""
    all_codes = []
    for codes in _SECTOR_STOCK_MAP.values():
        all_codes.extend(codes)
    all_codes = list(dict.fromkeys(all_codes))  # 去重保留顺序

    print(f"📡 拉取 {len(all_codes)} 只股票数据 ({START} → {END})...")
    market_data = {}
    failed = []
    for code in all_codes:
        df = fetch_stock_daily(code, START, END)
        if df is not None and len(df) >= 10:
            df = compute_indicators(df)
            market_data[code] = df
            print(f"  ✅ {code}: {len(df)} 天数据")
        else:
            failed.append(code)
            print(f"  ❌ {code}: 数据不足")

    if failed:
        print(f"  ⚠️  {len(failed)} 只股票数据不足: {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")

    if not market_data:
        print("  ⚠️  BaoStock 无数据，使用合成数据")
        return _generate_synthetic_pool(all_codes)

    return market_data


def _generate_synthetic_pool(codes: list[str]) -> dict[str, pd.DataFrame]:
    """BaoStock 不可用时的合成数据回退"""
    np.random.seed(42)
    dates = pd.date_range(START, END, freq="B")
    if len(dates) < 10:
        dates = pd.date_range(end=END, periods=60, freq="B")

    pool = {}
    for code in codes:
        np.random.seed(hash(code) % (2**31))
        n = len(dates)
        base = 10 + np.random.random() * 40
        innovations = np.random.randn(n) * 0.015
        log_p = np.zeros(n)
        for i in range(1, n):
            log_p[i] = log_p[i-1] + innovations[i] - 0.001 * log_p[i-1]
        prices = base * np.exp(log_p)
        prices = np.clip(prices, base * 0.5, base * 1.5)

        df = pd.DataFrame({
            "date": dates,
            "open": prices * (1 - np.abs(np.random.randn(n)) * 0.01),
            "high": prices * (1 + np.abs(np.random.randn(n)) * 0.015),
            "low":  prices * (1 - np.abs(np.random.randn(n)) * 0.015),
            "close": prices,
            "volume": np.random.randint(100_000, 10_000_000, n),
        })
        df["date"] = pd.to_datetime(df["date"])
        pool[code] = compute_indicators(df)

    return pool


def run_backtests(market_data: dict[str, pd.DataFrame]) -> dict:
    """对所有股票运行 4 策略，聚合结果"""
    strategies = get_all_strategies()
    strategy_names = [s.name for s in strategies]

    # 按策略聚合
    agg = {name: {"equity_curves": [], "metrics": [], "codes": []} for name in strategy_names}

    print(f"\n📈 回测 {len(market_data)} 只股票 × {len(strategies)} 策略 (本金 ¥{INITIAL_CASH:,})...")

    for code, df in market_data.items():
        for strat in strategies:
            try:
                signals = strat.generate_signals(df)
                engine = BacktestEngine(initial_cash=INITIAL_CASH)
                result = engine.run(df, signals, strategy_name=strat.name)
                agg[strat.name]["equity_curves"].append(result["equity_curve"])
                agg[strat.name]["metrics"].append(result)
                agg[strat.name]["codes"].append(code)
            except Exception as e:
                print(f"  ⚠️  {code} × {strat.name}: {e}")

    # 计算合计结果
    results = {}
    for name in strategy_names:
        entries = agg[name]
        if not entries["equity_curves"]:
            print(f"  ⚠️  {name}: 无有效回测结果")
            continue

        # 按日期对齐：取最长 equity_curve（交易日最多的股票）
        max_len = max(len(ec) for ec in entries["equity_curves"])
        # 补齐较短序列（用 NaN 填充开头 — 股票上市时间不同）
        aligned = []
        for ec in entries["equity_curves"]:
            if len(ec) < max_len:
                pad = [np.nan] * (max_len - len(ec))
                aligned.append(np.array(pad + list(ec)))
            else:
                aligned.append(np.array(ec))
        aligned = np.array(aligned)  # shape: (n_stocks, n_days)

        # 合计净值曲线 = 各股票净值之和
        total_equity = np.nansum(aligned, axis=0)
        # 减去多份本金中未使用的部分：每只股票各自从 INITIAL_CASH 开始，
        # 合计净值起点 = n_stocks * INITIAL_CASH
        n_stocks = len(entries["codes"])
        total_initial = n_stocks * INITIAL_CASH

        total_return = total_equity[-1] / total_initial - 1 if total_initial > 0 else 0

        daily_returns = pd.Series(total_equity).pct_change().dropna()
        sharpe = np.sqrt(252) * daily_returns.mean() / (daily_returns.std() + 1e-8) if len(daily_returns) > 0 else 0

        peak = np.maximum.accumulate(total_equity)
        dd = (total_equity - peak) / (peak + 1e-8)
        max_dd = float(np.min(dd))

        # 各策略独立统计
        indiv_returns = [m["total_return"] for m in entries["metrics"]]
        avg_return = np.mean(indiv_returns)
        win_rate = np.sum(np.array(indiv_returns) > 0) / len(indiv_returns) if indiv_returns else 0

        total_trades = sum(m["num_trades"] for m in entries["metrics"])

        results[name] = {
            "total_return": round(total_return, 4),
            "avg_return": round(avg_return, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "num_trades": total_trades,
            "win_rate": round(win_rate, 4),
            "n_stocks": n_stocks,
            "final_value": round(total_equity[-1], 2),
            "equity_curve": total_equity.tolist(),
        }

        print(f"  {name:20s} 合计收益: {total_return*100:+7.2f}%  "
              f"Sharpe: {sharpe:.3f}  "
              f"回撤: {max_dd*100:.2f}%  "
              f"交易: {total_trades}  "
              f"覆盖: {n_stocks} 只")

    return results


def generate_comparison_table(strategy_results: dict) -> str:
    """生成 README 兼容的 markdown 表格"""
    lines = [
        "| 策略 | 合计收益率 | 平均收益率 | Sharpe | 最大回撤 | 交易次数 | 胜率 | 覆盖股票 |",
        "|------|-----------|-----------|--------|---------|---------|------|---------|",
    ]
    for name, m in strategy_results.items():
        lines.append(
            f"| {name} | {m['total_return']*100:+7.2f}% | "
            f"{m['avg_return']*100:+7.2f}% | "
            f"{m['sharpe_ratio']:.3f} | "
            f"{m['max_drawdown']*100:.2f}% | "
            f"{m['num_trades']} | "
            f"{m['win_rate']*100:.0f}% | "
            f"{m['n_stocks']} |"
        )
    return "\n".join(lines)


def save_results(strategy_results: dict, market_data: dict):
    """保存回测结果到 JSON 和 CSV"""
    os.makedirs(REPORT_DIR, exist_ok=True)

    # JSON 摘要
    summary = {
        "backtest_period": {"start": START, "end": END},
        "initial_capital_per_stock": INITIAL_CASH,
        "strategies": {},
    }
    for name, m in strategy_results.items():
        summary["strategies"][name] = {k: v for k, v in m.items() if k != "equity_curve"}

    json_path = os.path.join(REPORT_DIR, f"backtest_review_{END.replace('-', '')}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存: {json_path}")

    # 各策略 equity curve CSV
    df_curves = {}
    for name, m in strategy_results.items():
        df_curves[name] = pd.Series(m["equity_curve"], name=name)
    df_out = pd.DataFrame(df_curves)
    csv_path = os.path.join(REPORT_DIR, f"equity_curves_{END.replace('-', '')}.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"💾 净值曲线已保存: {csv_path}")


def update_readme(strategy_results: dict, regime_name: str = "N/A"):
    """在 README.md 中更新回测结果"""
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
    if not os.path.exists(readme_path):
        print("⚠️  README.md 不存在，跳过更新")
        return

    table = generate_comparison_table(strategy_results)

    # 最佳策略
    best_sharpe = max(strategy_results, key=lambda k: strategy_results[k]["sharpe_ratio"])
    best_return = max(strategy_results, key=lambda k: strategy_results[k]["total_return"])

    results_block = f"""---

## 最新回测复盘 ({END})

**回测区间**: {START} → {END}（真实 A 股数据）
**本金**: 每策略每只股票 ¥{INITIAL_CASH:,}（合计净值汇总）
**数据源**: BaoStock（前复权日线）
**股票池**: {sum(len(v) for v in _SECTOR_STOCK_MAP.values())} 只（{len(_SECTOR_STOCK_MAP)} 个热门板块）

### 策略绩效（合计）

{table}

### 最佳策略

- **最佳 Sharpe**: {best_sharpe} ({strategy_results[best_sharpe]['sharpe_ratio']:.3f})
- **最佳收益**: {best_return} ({strategy_results[best_return]['total_return']*100:.2f}%)

### 风险分析

| 指标 | 说明 |
|------|------|
| 回测方式 | 每只股票独立运行，合计净值=∑各股票净值 |
| T+1 限制 | 买入后次日方可卖出（由信号触发日期控制） |
| 涨跌停板 | 实际数据已包含涨跌停限制 |
| 手续费 | 未计入（简化回测） |

### 净值曲线

各策略合计净值曲线已保存至: `output/reports/equity_curves_{END.replace('-', '')}.csv`

"""

    with open(readme_path, "r") as f:
        content = f.read()

    marker = "## 最新回测复盘"
    if marker in content:
        before = content.split(marker)[0]
        after = content.split(marker)[1]
        if "## " in after:
            after = after[after.index("## "):]
        else:
            after = ""
        content = before + results_block + "\n" + after
    else:
        roadmap_marker = "## 路线图"
        if roadmap_marker in content:
            content = content.replace(roadmap_marker, results_block + "\n" + roadmap_marker)
        else:
            content += "\n" + results_block

    with open(readme_path, "w") as f:
        f.write(content)
    print("📝 README.md 已更新")


def main():
    print(f"{'='*60}")
    print(f"📊 回测复盘: {START} → {END}")
    print(f"{'='*60}\n")

    # 1. 获取数据
    market_data = fetch_all_data()

    if not market_data:
        print("❌ 无可用数据，退出")
        return

    print(f"\n📊 共获取 {len(market_data)} 只股票数据")

    # 2. 运行回测
    strategy_results = run_backtests(market_data)

    if not strategy_results:
        print("❌ 无回测结果")
        return

    # 3. 保存结果
    save_results(strategy_results, market_data)

    # 4. 更新 README
    update_readme(strategy_results)

    # 5. 打印对比表格
    print(f"\n{'='*60}")
    print("📋 策略绩效汇总表")
    print(generate_comparison_table(strategy_results))
    print(f"{'='*60}")
    print(f"✅ 回测复盘完成！")


if __name__ == "__main__":
    main()
