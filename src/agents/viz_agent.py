#!/usr/bin/env python3
"""可视化 Agent"""
import os
from ..agents.base import AgentContext, BaseAgent
from ..viz.charts import create_report_html
from config import REPORT_DIR


class VisualizationAgent(BaseAgent):
    name = "visualization"
    description = "生成 PyECharts 可视化报告"

    def execute(self, context: AgentContext) -> AgentContext:
        os.makedirs(REPORT_DIR, exist_ok=True)
        date_str = context.date.replace("-", "")
        output_path = os.path.join(REPORT_DIR, f"daily_report_{date_str}.html")

        try:
            context.viz_path = create_report_html(context, output_path)
            # 修复: 将 CDN 的 ECharts 改为本地文件
            self._fix_echarts_local(context.viz_path)
            context.warnings.append(f"报告已生成: {output_path}")
        except Exception as e:
            context.warnings.append(f"可视化生成失败: {e}")

        return context

    def _fix_echarts_local(self, html_path: str):
        """将 HTML 中的 pyecharts CDN ECharts 替换为本地文件"""
        import shutil, os
        if not html_path or not os.path.exists(html_path):
            return
        reports_dir = os.path.dirname(html_path)
        local_js = os.path.join(reports_dir, "echarts.min.js")
        # 如果本地没有则复制
        if not os.path.exists(local_js):
            src = "/Users/mac13/workspace/rl_trading_system/output/echarts.min.js"
            if os.path.exists(src):
                shutil.copy(src, local_js)
        # 替换 HTML 中的 CDN URL
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                content = f.read()
            if "assets.pyecharts.org" in content or "cdnjs.cloudflare.com" in content:
                content = content.replace(
                    "https://assets.pyecharts.org/assets/v6/echarts.min.js",
                    "echarts.min.js"
                ).replace(
                    "https://cdnjs.cloudflare.com/ajax/libs/echarts/5.4.3/echarts.min.js",
                    "echarts.min.js"
                )
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(content)
        except Exception:
            pass
