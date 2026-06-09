#!/usr/bin/env python3
"""
智能量化交易系统 - Streamlit Web UI
多页面：运行分析 / 持仓明细 / Q&A 问答
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
from config import REPORT_DIR, LOG_DIR, OUTPUT_DIR, DB_PATH
from src.agents import AgentContext, OrchestratorAgent, build_daily_pipeline
from src.agents.qa_agent import QAAgent
from src.storage import DatabaseManager, MessageBus

st.set_page_config(
    page_title="智能量化交易系统",
    page_icon="🚀",
    layout="wide",
    menu_items={
        "about": "基于强化学习 + 多智能体架构的 A 股每日动态机会点挖掘系统"
    }
})

# ── 会话状态初始化 ──
if "context" not in st.session_state:
    st.session_state.context = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# ── 分析管线 ──
def run_analysis(date_str: str):
    """执行完整分析管线"""
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
    return context


# ── 侧边栏设置 ──
with st.sidebar:
    st.title("⚙️ 设置")
    date_input = st.text_input(
        "分析日期",
        value=datetime.now().strftime("%Y-%m-%d"),
        help="格式：YYYY-MM-DD"
    )
    st.divider()
    st.caption("🚀 智能量化交易系统")
    st.caption("基于 RL + 多智能体架构")


# ── 主页面 ──
tab1, tab2, tab3 = st.tabs(["📊 运行分析", "💰 持仓明细", "💬 智能问答"])

# ── Tab 1: 运行分析 ──
with tab1:
    col_header = st.columns([4, 1])
    with col_header[0]:
        st.title("🚀 智能量化交易系统")
        st.markdown("基于强化学习 + 多智能体架构的 A 股每日动态机会点挖掘系统")
    with col_header[1]:
        st.write("")
        run_btn = st.button("▶ 运行全部分析", type="primary", use_container_width=True)

    st.divider()

    if run_btn:
        with st.spinner("⏳ 正在执行分析管线（约 30-60 秒）..."):
            try:
                st.session_state.context = run_analysis(date_input)
                st.session_state.analysis_done = True
                st.success("✅ 分析完成", icon="✅")
            except Exception as e:
                st.error(f"❌ 分析失败: {e}", icon="❌")
                st.session_state.analysis_done = False

    # 显示分析结果
    if st.session_state.analysis_done and st.session_state.context:
        ctx = st.session_state.context

        col_report, col_viz = st.columns([1, 1])
        with col_report:
            st.subheader("📋 分析报告")
            report_text = ctx.report_text or "无报告数据"
            st.markdown(report_text)

        with col_viz:
            st.subheader("📈 可视化图表")
            if ctx.viz_path and os.path.exists(ctx.viz_path):
                with open(ctx.viz_path, "r", encoding="utf-8") as f:
                    viz_html = f.read()
                st.components.v1.html(viz_html, height=600, scrolling=True)
            else:
                st.info("无可视化数据")

        # 持仓预览
        pos = ctx.position_analysis or {}
        positions = pos.get("positions", [])
        if positions:
            st.subheader("💰 持仓预览（Top 5）")
            preview_df = pd.DataFrame([{
                "股票": p.get("stock", ""),
                "策略": p.get("strategy", ""),
                "方向": p.get("action", ""),
                "盈亏%": f"{p.get('pnl_pct', 0):+.2f}%",
                "状态": p.get("status", ""),
            } for p in positions[:5]])
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
    else:
        st.info("💡 点击上方「运行全部分析」按钮开始分析管线", icon="ℹ️")


# ── Tab 2: 持仓明细 ──
with tab2:
    st.title("💰 持仓明细")
    st.divider()

    if st.session_state.analysis_done and st.session_state.context:
        ctx = st.session_state.context
        pos = ctx.position_analysis or {}
        positions = pos.get("positions", [])
        summary = pos.get("summary", {})

        # 账户总览指标卡
        s = summary
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("💰 初始资金", f"¥{s.get('initial_cash', 0):,.0f}")
        m2.metric("💵 剩余现金", f"¥{s.get('cash', 0):,.2f}")
        m3.metric("📊 持仓市值", f"¥{s.get('stock_value', 0):,.2f}")
        m4.metric("🏦 总资产", f"¥{s.get('total_assets', 0):,.2f}")
        m5.metric("📈 总收益率", f"{s.get('total_return', 0):+.2f}%",
                  delta=f"{s.get('total_return', 0):+.2f}%")
        m6.metric("📊 持仓数量", f"{s.get('active_positions', 0)} 只")

        st.divider()

        if positions:
            st.subheader("📋 持仓明细")
            pos_df = pd.DataFrame([{
                "股票": p.get("stock", ""),
                "策略": p.get("strategy", ""),
                "方向": p.get("action", ""),
                "数量": p.get("quantity", 0),
                "均价": f"¥{p.get('entry_price', 0):.2f}",
                "现价": f"¥{p.get('current_price', 0):.2f}",
                "市值": f"¥{p.get('market_value', 0):,.2f}",
                "盈亏": f"¥{p.get('pnl', 0):,.2f}",
                "盈亏%": f"{p.get('pnl_pct', 0):+.2f}%",
                "权重%": f"{p.get('weight', 0):.1f}%",
                "RSI": p.get("rsi", "-"),
                "状态": p.get("status", ""),
            } for p in positions])
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
        else:
            st.info("无持仓数据")
    else:
        st.info("💡 请先在「运行分析」标签页执行分析管线后再查看持仓", icon="ℹ️")


# ── Tab 3: Q&A 问答 ──
with tab3:
    st.title("💬 智能问答")
    st.divider()

    # 显示聊天历史
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)

    # 输入
    if prompt := st.chat_input("输入问题，例如：今天哪些板块有机会？"):
        with st.chat_message("user"):
            st.markdown(prompt)

        ctx = st.session_state.context if st.session_state.context else None
        if not ctx:
            reply = "请先在「运行分析」标签页执行分析管线。"
        else:
            qa = QAAgent()
            qa.context = ctx
            reply = qa._fallback_answer(ctx) or "抱歉，暂时无法回答这个问题。"

        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.chat_history.append((prompt, reply))

    st.divider()
    if st.button("清空对话", icon="🗑️"):
        st.session_state.chat_history = []
        st.rerun()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="量化交易系统 Web UI")
    parser.add_argument("--port", type=int, default=8501, help="端口号")
    args = parser.parse_args()
    # 本地调试用 `streamlit run app_streamlit.py`
    # Cloudflare 部署用 streamlit-cloudflare