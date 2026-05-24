#!/usr/bin/env python3
"""飞书推送 Agent"""
import json
import time
import hmac
import hashlib
import base64
import requests
from ..agents.base import AgentContext, BaseAgent
from config import FEISHU_WEBHOOK_URL, FEISHU_SECRET


class FeishuPushAgent(BaseAgent):
    name = "feishu_push"
    description = "推送分析报告到飞书"

    def execute(self, context: AgentContext) -> AgentContext:
        if not FEISHU_WEBHOOK_URL or "your_webhook_id" in FEISHU_WEBHOOK_URL:
            context.warnings.append("飞书 webhook 未配置，跳过推送")
            return context

        try:
            card = self._build_card(context)
            payload = {
                "timestamp": str(int(time.time())),
                "msg_type": "interactive",
                "card": card,
            }
            if FEISHU_SECRET:
                sign_str = f"{payload['timestamp']}\n{FEISHU_SECRET}"
                payload["sign"] = base64.b64encode(
                    hmac.new(b"", sign_str.encode(), hashlib.sha256).digest()
                ).decode()

            resp = requests.post(FEISHU_WEBHOOK_URL, json=payload, timeout=10)
            if resp.status_code == 200:
                context.warnings.append("飞书推送成功")
            else:
                context.warnings.append(f"飞书推送失败: {resp.status_code}")
        except Exception as e:
            context.warnings.append(f"飞书推送异常: {e}")

        return context

    def _build_card(self, ctx: AgentContext) -> dict:
        """构建飞书卡片消息"""
        elements = [{"tag": "markdown", "content": f"**日期**: {ctx.date}"}]

        # 热门板块
        if ctx.hot_sectors:
            sector_text = "\n".join([
                f"🔥 **{s['sector']}** (热度: {s['heat_score']})"
                for s in ctx.hot_sectors[:5]
            ])
            elements.append({"tag": "markdown", "content": f"**热门板块:**\n{sector_text}"})

        # 交易信号
        if ctx.rl_signals:
            signal_text = "\n".join([
                f"{'🟢' if s['action']=='buy' else '🔴'} **{s['stock']}**: {s['action']} "
                f"(置信度: {s['confidence']:.0%})"
                for s in ctx.rl_signals[:5]
            ])
            elements.append({"tag": "markdown", "content": f"**交易信号:**\n{signal_text}"})

        # 策略和风控
        if ctx.risk_metrics:
            dd = ctx.risk_metrics.get("drawdown", {})
            elements.append({"tag": "markdown", "content":
                f"**风控:** 回撤 {dd.get('dd_pct', 0):.2%} | "
                f"状态: {dd.get('level', 'normal')}"})

        # 查看报告链接
        if ctx.viz_path:
            elements.append({
                "tag": "action",
                "actions": [{
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "📊 查看完整报告"},
                    "type": "primary",
                    "multi_url": {
                        "url": f"file://{ctx.viz_path}",
                        "pc_url": f"file://{ctx.viz_path}",
                    },
                }],
            })

        return {
            "header": {
                "title": {"tag": "plain_text", "content": f"🚀 量化交易日报 {ctx.date}"},
                "template": "blue",
            },
            "elements": elements,
        }
