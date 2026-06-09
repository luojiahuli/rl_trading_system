# Cloudflare Pages + Streamlit 部署指南

## 前置准备

1. 安装 streamlit-cloudflare：
   ```bash
   pip install streamlit>=1.40.0
   pip install streamlit-cloudflare>=0.1.0
   ```

2. 本地测试：
   ```bash
   cd rl_trading_system
   streamlit run app_streamlit.py --server.port 8501
   ```

## 部署到 Cloudflare Pages

### 方式 A：使用 streamlit-cloudflare CLI
```bash
# 安装 Cloudflare Wrangler
npm install -g wrangler

# 登录 Cloudflare
wrangler login

# 部署
cd rl_trading_system
streamlit-cloudflare deploy --project-name rl-trading-system
```

### 方式 B：使用 GitHub Actions 自动化部署

1. 在 Cloudflare Pages 创建项目，连接 GitHub 仓库
2. 设置构建设置：
   - **构建命令**: `pip install -r requirements.txt && streamlit-cloudflare build`
   - **构建输出目录**: `.streamlit-cloudflare`
   - **环境变量**: 添加 `PYTHON_VERSION = 3.11`
3. 设置环境变量（Secrets）：
   - `DEEPSEEK_API_KEY`
   - `MINIMAX_API_KEY`
   - `FEISHU_WEBHOOK_URL`

## 目录结构

```
rl_trading_system/
├── app_streamlit.py      # Streamlit 主应用 ← 上传这个
├── config.py             # 配置文件
├── requirements.txt     # Python 依赖
├── src/                  # 业务逻辑模块
├── output/              # 运行时输出（不在 Git 中）
└── README.md
```

## 注意事项

- `output/` 目录包含运行时数据库和日志，已在 .gitignore 中忽略
- 首次部署需要配置 LLM API Key 环境变量
- 分析管线运行需要 30-60 秒，确保页面 timeout 设置足够