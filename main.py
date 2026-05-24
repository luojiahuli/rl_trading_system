#!/usr/bin/env python3
"""
智能量化交易系统 - 主入口
每日动态机会点分析: 新闻→板块→信号→RL→回测→飞书
"""
import os
import sys
import json
from datetime import datetime

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents import AgentContext, OrchestratorAgent, build_daily_pipeline
from config import REPORT_DIR, LOG_DIR, OUTPUT_DIR


def run_daily_analysis(date_str: str = None) -> AgentContext:
    """执行每日全流程分析"""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"🚀 启动 {date_str} 量化交易分析...")

    context = AgentContext(date=date_str)
    pipeline = build_daily_pipeline()
    orchestrator = OrchestratorAgent(pipeline)

    print(f"📋 Agent 管线: {[a.name for a in pipeline]}")
    context = orchestrator.execute(context)

    # 输出摘要
    print(f"\n{'='*50}")
    print(f"📊 分析完成: {date_str}")
    print(f"  热门板块: {len(context.hot_sectors)} 个")
    print(f"  股票池: {len(context.stock_pool)} 只")
    print(f"  时间信号: {len(context.ts_signals)} 个")
    print(f"  交易信号: {len(context.rl_signals)} 个")
    print(f"  回测次数: {len(context.backtest_results)} 次")
    print(f"  市场状态: {context.regime}")
    print(f"  可视化: {context.viz_path}")

    if context.errors:
        print(f"  ❌ 错误: {len(context.errors)} 个")
        for e in context.errors:
            print(f"    - {e}")

    return context


def run_in_terminal(date_str: str = None):
    """终端模式运行"""
    context = run_daily_analysis(date_str)
    print(f"\n{'='*50}")
    print(context.report_text)
    print(f"\n{'='*50}")
    print("💡 提示: 运行 `python main.py --qa` 启动问答模式")
    return context


def run_qa_mode(date_str: str = None):
    """问答模式 - 先分析再对话"""
    context = run_daily_analysis(date_str)
    from src.agents.qa_agent import QAAgent

    qa = QAAgent()
    qa.context = context

    print("\n💬 问答模式 (输入 'exit' 退出)")
    print("示例问题: 今天哪些板块有机会？")
    while True:
        try:
            question = input("\n❓ ")
            if question.lower() in ("exit", "quit", "q"):
                break
            qa_context = context
            answer = qa._fallback_answer(qa_context)
            print(f"\n{answer}")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="智能量化交易系统")
    parser.add_argument("--date", "-d", help="分析日期 YYYY-MM-DD")
    parser.add_argument("--qa", action="store_true", help="启动问答模式")
    args = parser.parse_args()

    if args.qa:
        run_qa_mode(args.date)
    else:
        run_in_terminal(args.date)
