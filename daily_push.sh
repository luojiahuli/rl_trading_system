#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

REPORT_LOG="output/logs/daily_run_$(date +%Y%m%d).log"
mkdir -p output/logs

echo "=== 🚀 量化交易日报 $(date +%Y-%m-%d) ===" > "$REPORT_LOG"

# 运行交易系统
python3 main.py >> "$REPORT_LOG" 2>&1

# 提取关键信息推送飞书
python3 -c "
import requests, json, time, os, sys
sys.path.insert(0, '.')
from config import START_DATE, END_DATE

APP_ID = os.environ.get('FEISHU_APP_ID', '')
APP_SECRET = os.environ.get('FEISHU_APP_SECRET', '')
CHAT_ID = os.environ.get('FEISHU_CHAT_ID', '')

# Get token
r = requests.post(
    'https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal',
    json={'app_id': APP_ID, 'app_secret': APP_SECRET},
    timeout=10
)
token = r.json().get('tenant_access_token', '')
if not token:
    print('Failed to get token')
    sys.exit(1)

# Try to read the report output
report = open('$REPORT_LOG').read()

# Parse key metrics
lines = report.strip().split('\n')
summary = {}
for line in lines:
    if '分析完成:' in line:
        parts = line.strip().split('|')
        for p in parts:
            p = p.strip()
            if ':' in p:
                k, v = p.split(':', 1)
                summary[k.strip()] = v.strip()

hot_sectors_text = '无'
signals_text = '无'
strategy_text = '无'
risk_text = '无'
market_state = summary.get('市场状态', '未知')

# Parse report sections
in_section = None
sections = {'板块': [], '信号': [], '策略': [], '风控': []}
for line in lines:
    line_s = line.strip()
    if line_s.startswith('| 板块'):
        in_section = '板块'
        continue
    elif line_s.startswith('| 股票'):
        in_section = '信号'
        continue
    elif line_s.startswith('| 策略'):
        in_section = '策略'
        continue
    elif line_s.startswith('| 收益率'):
        continue
    elif line_s.startswith('| 当前回撤'):
        risk_text = line_s
    elif in_section == '板块' and line_s.startswith('|') and '|' in line_s:
        cols = [c.strip() for c in line_s.split('|') if c.strip()]
        if len(cols) >= 2 and cols[0] not in ('板块', '---'):
            sections['板块'].append(f\"{cols[0]}(热度{cols[1] if len(cols)>1 else '?'})\")
    elif in_section == '信号' and line_s.startswith('|') and '|' in line_s:
        cols = [c.strip() for c in line_s.split('|') if c.strip()]
        if len(cols) >= 2 and cols[0] not in ('股票', '---'):
            sections['信号'].append(f\"{cols[0]}→{cols[1]}({cols[2] if len(cols)>2 else '?'})\")
    elif in_section == '策略' and line_s.startswith('|') and '|' in line_s:
        cols = [c.strip() for c in line_s.split('|') if c.strip()]
        if len(cols) >= 2 and cols[0] not in ('策略', '---'):
            sections['策略'].append(f\"{cols[0]}: {cols[1]}|S{cols[2] if len(cols)>2 else '?'}|D{cols[3] if len(cols)>3 else '?'}\")
    if not line_s.startswith('|'):
        in_section = None

if sections['板块']:
    hot_sectors_text = ' | '.join(sections['板块'][:5])
if sections['信号']:
    signals_text = '\\n'.join(sections['信号'][:3])
if sections['策略']:
    strategy_lines = sections['策略'][:4]
    strategy_text = '\\n'.join(strategy_lines)

stock_count = summary.get('股票池', '?')
signal_count = summary.get('交易信号', '?')
backtest_count = summary.get('回测次数', '?')

# Build rich text content
content = {
    'zh_cn': {
        'title': f'🚀 量化交易日报 {time.strftime(\"%Y-%m-%d\")}',
        'content': [
            [{'tag': 'text', 'text': f'📊 市场: {market_state} | 股票: {stock_count}只 | 信号: {signal_count}个 | 回测: {backtest_count}次'}],
            [{'tag': 'text', 'text': ''}],
            [{'tag': 'text', 'text': '🔥 热门板块:'}],
            [{'tag': 'text', 'text': f'  {hot_sectors_text}'}],
            [{'tag': 'text', 'text': ''}],
        ]
    }
}

# Parse market judgement section
mj_fields = {}
in_mj = False
for line in lines:
    line_s = line.strip()
    if line_s.startswith('##') and '市场研判' in line_s:
        in_mj = True
        continue
    if in_mj:
        if line_s.startswith('|') and '|' in line_s:
            cols = [c.strip() for c in line_s.split('|') if c.strip()]
            if len(cols) >= 2 and cols[0] in ('市场阶段', '趋势方向', '政策预期', '置信度'):
                mj_fields[cols[0]] = cols[1]
        elif not line_s.startswith('|') and not line_s.startswith('|---'):
            in_mj = False

if mj_fields:
    mj_line = ' | '.join([f'{k}:{v}' for k, v in mj_fields.items()])
    content['zh_cn']['content'].append([{'tag': 'text', 'text': ''}])
    content['zh_cn']['content'].append([{'tag': 'text', 'text': '📊 市场研判:'}])
    content['zh_cn']['content'].append([{'tag': 'text', 'text': f'  {mj_line}'}])

if sections['信号']:
    signals_block = [{'tag': 'text', 'text': '📈 交易信号:'}]
    for s in sections['信号'][:3]:
        signals_block.append({'tag': 'text', 'text': f'  {s}'})
    content['zh_cn']['content'].append(signals_block)
    content['zh_cn']['content'].append([{'tag': 'text', 'text': ''}])

if sections['策略']:
    strategy_block = [{'tag': 'text', 'text': '📊 策略绩效:'}]
    for s in sections['策略'][:4]:
        strategy_block.append({'tag': 'text', 'text': f'  {s}'})
    content['zh_cn']['content'].append(strategy_block)
    content['zh_cn']['content'].append([{'tag': 'text', 'text': ''}])

# Best strategy
best_sharpe = ''
best_return = ''
for line in lines:
    if '最佳 Sharpe' in line:
        best_sharpe = line.strip()
    if '最佳收益' in line:
        best_return = line.strip()
if best_sharpe or best_return:
    lines_block = [{'tag': 'text', 'text': '🏆 最佳:'}]
    if best_sharpe:
        lines_block.append({'tag': 'text', 'text': f'  {best_sharpe}'})
    if best_return:
        lines_block.append({'tag': 'text', 'text': f'  {best_return}'})
    content['zh_cn']['content'].append(lines_block)
    content['zh_cn']['content'].append([{'tag': 'text', 'text': ''}])

# Risk status
if risk_text:
    content['zh_cn']['content'].append([{'tag': 'text', 'text': f'⚠️ {risk_text}'}])
    content['zh_cn']['content'].append([{'tag': 'text', 'text': ''}])

# Footer
footer_text = '💡 运行 main.py --qa 进入问答模式'
content['zh_cn']['content'].append([{'tag': 'text', 'text': footer_text}])

payload = {
    'receive_id': CHAT_ID,
    'msg_type': 'post',
    'content': json.dumps(content, ensure_ascii=False)
}
headers = {
    'Authorization': f'Bearer {token}',
    'Content-Type': 'application/json; charset=utf-8'
}
r2 = requests.post(
    'https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id',
    headers=headers, json=payload, timeout=10
)
result = r2.json()
if result.get('code') == 0:
    print('Feishu push success')
else:
    print(f'Feishu push failed: {result}')
" >> "$REPORT_LOG" 2>&1

echo "=== ✅ 完成 $(date) ===" >> "$REPORT_LOG"
