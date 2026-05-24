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
            context.warnings.append(f"报告已生成: {output_path}")
        except Exception as e:
            context.warnings.append(f"可视化生成失败: {e}")

        return context
