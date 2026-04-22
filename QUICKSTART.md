# Quick Start Guide - PDF RAG Project

## 🚀 快速开始指南

### 第一步：配置 Azure 服务

1. 复制配置文件模板：
```powershell
cd backend
copy config.example.json config.json
```

2. 编辑 `backend/config.json`，填入你的 Azure 凭证：
   - Azure OpenAI API密钥和端点
   - Azure AI Search 服务名称和API密钥
   - Azure Document Intelligence 端点和密钥

### 第二步：安装后端依赖

```powershell
cd backend
python -m venv pdf_rag_venv
.\pdf_rag_venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 第三步：安装前端依赖

```powershell
cd frontend
npm install
```

### 第四步：启动应用

**方式A：使用启动脚本（推荐）**
```powershell
.\start.ps1
```

**方式B：手动启动**

终端1（后端）:
```powershell
cd backend
.\pdf_rag_venv\Scripts\Activate.ps1  # 激活虚拟环境
python app.py
```

终端2（前端）:
```powershell
cd frontend
npm run dev
```

### 第五步：访问应用

- 前端界面：http://localhost:5173
- 后端API：http://localhost:8000
- API文档：http://localhost:8000/docs

## 📝 使用流程

1. **创建索引**
   - 输入索引名称（如 `pdf-my-docs`）
   - 点击 "Create Index"

2. **上传PDF**
   - 选择索引
   - 选择PDF文件
   - 点击 "Upload & Index"
   - 等待处理完成

3. **提问**
   - 输入问题
   - 调整 Top-K 值（推荐3-10）
   - 可选：启用 Agent Mode 获得更智能的检索
   - 点击 "Ask"
   - 查看检索到的上下文和生成的答案

## 🤖 Agent Mode 说明

启用 Agent Mode 后，AI 代理会：
- 评估查询质量
- 重写模糊查询以获得更好的检索结果
- 自适应地扩大搜索范围（如果证据不足）
- 评估证据充分性
- 生成基于推理的答案

实时追踪显示代理的思考过程！

## ⚠️ 常见问题

### 后端无法启动
- 检查Python版本（需要3.9+）
- 确认所有依赖已安装
- 验证 `config.json` 中的 Azure 凭证

### 前端无法启动
- 检查Node.js版本（需要18+）
- 删除 `node_modules` 并重新安装
- 确认端口5173可用

### 上传失败
- 确保PDF有效且未加密
- 检查 Azure Document Intelligence 配额
- 验证索引名称符合规则（小写、字母数字、-、_）

## 📚 技术栈

**后端**
- FastAPI - 异步Web框架
- Azure OpenAI - GPT-4 问答、text-embedding-ada-002 嵌入
- Azure AI Search - 混合搜索（向量+关键词）
- Azure Document Intelligence - PDF文本提取

**前端**
- React 19 - UI框架
- Vite - 快速构建工具
- React Bootstrap - UI组件
- EventSource - 实时代理追踪流

## 🔧 项目结构

```
PDF_RAG_Project/
├── backend/                    # 后端（FastAPI）
│   ├── app.py                 # 主应用
│   ├── config.json            # 配置文件（Azure凭证）
│   ├── requirements.txt       # Python依赖
│   ├── routers/
│   │   └── pdf_rag_routes.py # PDF RAG API路由
│   └── AI_AGENT/
│       └── pdf_agent_engine.py # 智能代理逻辑
├── frontend/                   # 前端（React + Vite）
│   ├── src/
│   │   ├── App.jsx           # 主应用组件
│   │   ├── RAGPage.jsx       # PDF RAG界面
│   │   └── lib/              # API工具
│   ├── package.json
│   └── vite.config.js
├── start.ps1                   # 启动脚本
└── README.md                   # 完整文档
```

## 📞 获取帮助

详细文档请查看 `README.md`

祝使用愉快！🎉
