#!/usr/bin/env python3
"""
智能量化交易系统 - Gradio Web UI
多标签页：运行分析 / 持仓明细 / Q&A 问答
"""
import os
import sys
import json
import threading
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import pandas as pd

from src.agents import AgentContext, OrchestratorAgent, build_daily_pipeline
from src.agents.qa_agent import QAAgent
from src.storage import DatabaseManager, MessageBus
from config import REPORT_DIR, LOG_DIR, OUTPUT_DIR, DB_PATH

_qa_agent = QAAgent()


def run_analysis(date_str: str):
    """执行完整分析管线，返回各组件数据"""
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    bus = MessageBus()
    db = DatabaseManager(DB_PATH).connect()
    context = AgentContext(date=date_str)
    pipeline = build_daily_pipeline()
    orchestrator = OrchestratorAgent(pipeline, message_bus=bus, database=db)
    context = orchestrator.execute(context)
    db.close()

    # --- Tab 1: 报告文本 ---
    report_text = context.report_text or "无报告数据"

    # --- Tab 1: 可视化 HTML ---
    viz_html = ""
    if context.viz_path and os.path.exists(context.viz_path):
        with open(context.viz_path, "r", encoding="utf-8") as f:
            viz_html = f.read()

    # --- Tab 2: 持仓明细 DataFrame ---
    pos = context.position_analysis or {}
    positions = pos.get("positions", [])
    summary = pos.get("summary", {})
    if positions:
        pos_df = pd.DataFrame([{
            "股票": p.get("stock", ""),
            "策略": p.get("strategy", ""),
            "方向": p.get("action", ""),
            "数量": p.get("quantity", 0),
            "均价": p.get("entry_price", 0),
            "现价": p.get("current_price", 0),
            "市值": p.get("market_value", 0),
            "盈亏": p.get("pnl", 0),
            "盈亏%": p.get("pnl_pct", 0),
            "权重%": p.get("weight", 0),
            "RSI": p.get("rsi", "-"),
            "状态": p.get("status", ""),
        } for p in positions])
    else:
        pos_df = pd.DataFrame()

    # 账户总览
    s = summary
    account_overview = {
        "💰 初始资金": f"¥{s.get('initial_cash', 0):,.0f}",
        "💵 剩余现金": f"¥{s.get('cash', 0):,.2f}",
        "📊 持仓市值": f"¥{s.get('stock_value', 0):,.2f}",
        "🏦 总资产": f"¥{s.get('total_assets', 0):,.2f}",
        "📈 总收益率": f"{s.get('total_return', 0):+.2f}%",
        "📊 持仓数量": f"{s.get('active_positions', 0)} 只",
        "📋 策略分布": str(s.get('strategy_allocation', {})),
    }
    summary_json = account_overview

    # --- Tab 3: QA 缓存 ---
    _qa_agent.context = context

    return report_text, viz_html, pos_df, summary_json


def answer_question(question: str, history: list):
    """Q&A 回复"""
    if not question.strip():
        return "", history or []
    if history is None:
        history = []
    ctx = _qa_agent.context if hasattr(_qa_agent, "context") else None
    if not ctx:
        reply = "请先在「运行分析」标签页执行分析管线。"
    else:
        reply = _qa_agent._fallback_answer(ctx)
    history.append((question, reply))
    return "", history


# ── Gradio UI ──

with gr.Blocks(title="智能量化交易系统") as demo:
    gr.Markdown(
        "# 🚀 智能量化交易系统\n"
        "基于强化学习 + 多智能体架构的 A 股每日动态机会点挖掘系统"
    )

    # ── Tab 1: 运行分析 ──
    with gr.Tab("📊 运行分析"):
        with gr.Row():
            date_input = gr.Textbox(
                label="分析日期", value=datetime.now().strftime("%Y-%m-%d"),
                scale=3,
            )
            run_btn = gr.Button("▶ 运行全部分析", variant="primary", scale=1, min_width=160)
        status_text = gr.Markdown("💡 点击上方按钮开始分析管线")
        report_md = gr.Markdown(label="分析报告", visible=True)
        viz_html = gr.HTML(label="可视化图表")

    # ── Tab 2: 持仓明细 ──
    with gr.Tab("💰 持仓明细"):
        summary_json = gr.JSON(label="持仓摘要")
        pos_table = gr.DataFrame(
            label="持仓明细",
            wrap=True,
        )
        with gr.Row():
            refresh_pos_btn = gr.Button("🔄 刷新持仓", variant="secondary")
            pos_hint = gr.Markdown("💡 请先在「运行分析」标签页执行分析管线")

    # ── Tab 3: Q&A 问答 ──
    with gr.Tab("💬 智能问答"):
        chatbot = gr.Chatbot(label="对话", height=400)
        with gr.Row():
            question_input = gr.Textbox(
                label="输入问题", placeholder="例如：今天哪些板块有机会？",
                scale=4,
            )
            send_btn = gr.Button("发送", variant="primary", scale=1, min_width=100)
        clear_btn = gr.Button("清空对话", variant="secondary", size="sm")
        qa_hint = gr.Markdown("💡 请先在「运行分析」标签页执行分析管线后再提问")

    # ── 事件绑定 ──

    # 运行分析
    def on_run(date_str):
        yield "⏳ 正在执行分析管线（约 30-60 秒）...", "", "", pd.DataFrame(), {"status": "running"}
        try:
            report, viz, pos_df, summary = run_analysis(date_str)
            yield "✅ 分析完成", report, viz, pos_df, summary
        except Exception as e:
            yield f"❌ 分析失败: {e}", f"❌ 错误: {e}", "", pd.DataFrame(), {"error": str(e)}

    run_btn.click(
        fn=on_run,
        inputs=date_input,
        outputs=[status_text, report_md, viz_html, pos_table, summary_json],
    )

    # 刷新持仓（读取已缓存的分析结果）
    def refresh_positions():
        ctx = getattr(_qa_agent, "context", None)
        if not ctx or not ctx.position_analysis:
            return "⚠️ 请先在「运行分析」标签页执行分析管线", pd.DataFrame(), {"提示": "无数据"}
        pos = ctx.position_analysis
        positions = pos.get("positions", [])
        summary = pos.get("summary", {})
        if positions:
            df = pd.DataFrame([{
                "股票": p.get("stock", ""),
                "策略": p.get("strategy", ""),
                "方向": p.get("action", ""),
                "数量": p.get("quantity", 0),
                "均价": p.get("entry_price", 0),
                "现价": p.get("current_price", 0),
                "市值": p.get("market_value", 0),
                "盈亏": p.get("pnl", 0),
                "盈亏%": p.get("pnl_pct", 0),
                "权重%": p.get("weight", 0),
                "RSI": p.get("rsi", "-"),
                "状态": p.get("status", ""),
            } for p in positions])
        else:
            df = pd.DataFrame()
        s = summary
        overview = {
            "💰 初始资金": f"¥{s.get('initial_cash', 0):,.0f}",
            "💵 剩余现金": f"¥{s.get('cash', 0):,.2f}",
            "📊 持仓市值": f"¥{s.get('stock_value', 0):,.2f}",
            "🏦 总资产": f"¥{s.get('total_assets', 0):,.2f}",
            "📈 总收益率": f"{s.get('total_return', 0):+.2f}%",
            "📊 持仓数量": f"{s.get('active_positions', 0)} 只",
            "📋 策略分布": str(s.get('strategy_allocation', {})),
        }
        return f"✅ 持仓数据已刷新（{s.get('active_positions', 0)} 只持仓中）", df, overview

    refresh_pos_btn.click(
        fn=refresh_positions,
        outputs=[pos_hint, pos_table, summary_json],
    )

    # Q&A 发送
    send_btn.click(
        fn=answer_question,
        inputs=[question_input, chatbot],
        outputs=[question_input, chatbot],
    )
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, chatbot],
        outputs=[question_input, chatbot],
    )
    clear_btn.click(fn=lambda: [], outputs=[chatbot])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="量化交易系统 Web UI")
    parser.add_argument("--port", type=int, default=7860, help="端口号")
    parser.add_argument("--share", action="store_true", help="创建公网链接")
    args = parser.parse_args()
    demo.launch(server_port=args.port, share=args.share, theme=gr.themes.Soft(), css="footer{display:none !important}")
