# JobFit Agent

智能求职匹配助手，基于大语言模型帮助用户分析简历与岗位的匹配度，并提供针对性的学习建议。

## 功能特性

- **简历分析**：提取简历中的关键技能和工作经历
- **JD解析**：解析岗位描述，了解岗位核心需求
- **匹配分析**：对比简历与JD，给出匹配分数和改进建议
- **学习路径**：根据缺失技能推荐学习资源
- **报告生成**：输出结构化的匹配分析报告

## 技术栈

- **后端**：Flask + LangChain/LangGraph
- **前端**：HTML + CSS + JavaScript
- **数据库**：SQLite（对话检查点持久化）
- **模型**：支持 OpenAI 兼容接口的LLM

## 项目结构

```
JobFit_Agent/
├── app.py              # Flask 后端服务
├── agent.py            # LangGraph Agent 核心逻辑
├── tool.py             # Agent 工具集
├── main.py             # Agent 交互入口
├── frontend/
│   ├── index.html      # 登录页
│   ├── register.html   # 注册页
│   └── dashboard.html  # 主界面
├── database/           # 用户数据
├── uploads/            # 上传文件
└── resources/          # 资源文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install flask langchain langgraph python-dotenv
```

### 2. 配置环境变量

创建 `.env` 文件：

```
API_KEY=your_api_key
BASE_URL=your_base_url
TAVIT_API_KEY=your_tavit_api_key
```

### 3. 启动服务

```bash
python app.py
```

访问 `http://localhost:5000`

## 使用流程

1. 注册/登录账号
2. 上传简历图片和岗位JD图片
3. 点击分析，查看匹配结果
4. 根据建议学习缺失技能

## Agent 工作流程

```
用户上传简历 + JD
       ↓
提取文本 (OCR)
       ↓
匹配分析 (加权评分)
       ↓
反思判断 (是否需要搜索)
       ↓
搜索补充资料 (可选)
       ↓
生成学习建议 + 报告
```
