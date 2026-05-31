# JobFit Agent v5.0

智能求职匹配助手，基于大语言模型帮助用户分析简历与岗位的匹配度，并提供针对性的学习建议。

## ✨ 功能特性

- **智能匹配分析**：对比简历与JD，使用加权算法计算匹配分数
- **技能识别**：自动识别必需技能和加分技能，精准定位缺失项
- **OCR文本提取**：支持从图片中提取简历和JD文字内容
- **反思机制**：自动评估匹配结果合理性，决定是否需要补充搜索
- **学习建议**：根据缺失技能提供针对性学习路径和资源推荐
- **报告生成**：输出结构化的匹配分析报告，支持PDF和Markdown下载
- **报告管理**：支持报告列表查看、详情查看和删除功能

## 🛠️ 技术栈

| 层次 | 技术 | 说明 |
|------|------|------|
| 后端框架 | Flask | 轻量级 Web 框架，支持蓝图架构 |
| Agent 框架 | LangChain + LangGraph | 构建智能 Agent 工作流 |
| 模型接口 | OpenAI 兼容 API | 支持多种 LLM 模型 |
| 搜索工具 | Tavily | 互联网搜索能力 |
| 持久化 | SQLite | 对话检查点存储 |
| 前端 | HTML5 + CSS3 + JavaScript | 响应式界面设计 |

## 📁 项目结构

```
JobFit_Agent/
├── app_v5.py                 # Flask 后端服务入口 (v5.0)
├── agent/                    # Agent 核心模块
│   ├── jobfit_agent.py       # Agent 工作流定义
│   ├── __init__.py
│   └── tools/                # 工具集
│       ├── match_tool.py     # 匹配分析工具
│       ├── ocr_tool.py       # OCR文本提取工具
│       ├── search_tool.py    # 互联网搜索工具
│       ├── learning_tool.py  # 学习建议工具
│       ├── reflection_tool.py# 反思评估工具
│       └── report_tool.py    # 报告生成工具
├── controllers/              # REST API 控制器
│   ├── auth_controller.py    # 用户认证接口
│   └── jobfit_controller.py  # 匹配分析接口
├── services/                 # 业务服务层
│   ├── auth_service.py       # 认证服务
│   └── jobfit_service.py     # 匹配分析服务
├── config/                   # 配置管理
│   └── settings.py           # 统一配置中心
├── exceptions/               # 自定义异常
│   └── jobfit_exceptions.py  # 异常类定义
├── utils/                    # 工具函数
│   └── crypto_utils.py       # 加密工具
├── frontend/                 # 前端静态文件
│   ├── index.html            # 登录页
│   ├── register.html         # 注册页
│   ├── dashboard.html        # 主界面
│   └── report.html           # 报告详情页
├── database/                 # 用户数据存储
├── uploads/                  # 上传文件存储
├── reports/                  # 生成的报告文件
├── resources/                # 资源文件
└── .env                      # 环境变量配置
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install flask langchain langgraph langchain-openai langchain-core \
            python-dotenv pydantic
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
# API 配置
API_KEY=your_api_key_here
BASE_URL=https://api.example.com/v1

# 搜索 API（可选）
TAVILY_API_KEY=your_tavily_api_key

# 模型配置
MODEL_NAME=qwen3.5-plus
MODEL_TEMPERATURE=0.0
```

### 3. 启动服务

```bash
python app_v5.py
```

访问 `http://localhost:5000` 即可使用。

## 📖 使用流程

```
┌─────────────────────────────────────────────────────┐
│  1. 注册/登录账号                                   │
│         ↓                                          │
│  2. 上传简历图片和岗位JD图片                        │
│         ↓                                          │
│  3. 点击「开始匹配」按钮                            │
│         ↓                                          │
│  4. 查看匹配分数和技能分析结果                      │
│         ↓                                          │
│  5. 点击「生成报告」查看详细报告                    │
│         ↓                                          │
│  6. 下载 PDF 或 Markdown 格式报告                  │
└─────────────────────────────────────────────────────┘
```

## 🔄 Agent 工作流程

```
用户输入（图片或文字）
       ↓
┌──────────────────┐
│  OCR文本提取     │  extract_text_from_image()
└────────┬─────────┘
         ↓
┌──────────────────┐
│  匹配分析        │  match_resume_to_jd()
│  (加权评分算法)   │
└────────┬─────────┘
         ↓
┌──────────────────┐
│  反思评估        │  reflect_on_match()
│  (判断是否搜索)   │
└────────┬─────────┘
         ↓
    需要搜索?
     /     \
    是      否
     ↓       ↓
┌─────────┐ ┌──────────────────┐
│ 搜索    │ │ 生成学习建议     │  suggest_learning()
│ 学习资源 │ └────────┬─────────┘
└────┬────┘          │
     ↓               ↓
     └──────┬────────┘
            ↓
┌──────────────────┐
│ 生成报告         │  generate_report()
│ (Markdown格式)   │
└──────────────────┘
```

## 📊 匹配评分算法

| 技能类型 | 权重 | 评分规则 |
|----------|------|----------|
| 必需技能 | 70% | 每缺失一项扣除 (70 / 总必需项数) |
| 加分技能 | 30% | 每匹配一项增加 (30 / 总加分项数) |

## 🔧 API 接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/v1/auth/register` | POST | 用户注册 |
| `/api/v1/auth/login` | POST | 用户登录 |
| `/api/v1/agent/chat` | POST | Agent 对话（流式响应） |
| `/api/v1/upload/<file_type>` | POST | 文件上传 |
| `/api/v1/report/generate` | POST | 生成报告 |
| `/api/v1/report/list` | GET | 获取报告列表 |
| `/api/v1/report/content` | GET | 获取报告内容 |
| `/api/v1/report/delete` | DELETE | 删除报告 |

## 📝 更新日志

### v5.0 (2026-05-31)
- 修复第一阶段 OCR 红框无输出问题（多线程 + SSE keepalive 机制）
- 丰富 Agent 最终输出文本，完整展示技能匹配详情与学习建议
- 重构报告生成系统，实现多维度可视化分层（表格 / 进度条 / 雷达图 / 时间线）
- 优化报告页面渲染逻辑，增强 Markdown CSS 样式
- 删除冗余代码（main.py、debug_tuple.py），版本号升级至 v5.0

### v3.0 (2026-05-15)
- 重构为蓝图架构，代码结构更清晰
- 新增报告查看页面，支持 PDF 和 Markdown 下载
- 新增报告删除功能
- 修复报告格式渲染异常问题
- 优化提示词系统，确保中文输出一致性
- 清理调试代码，提升代码质量

### v2.0
- 引入 LangGraph 构建 Agent 工作流
- 新增反思机制和搜索能力
- 支持流式响应
- 配置中心化管理

### v1.0
- 基础匹配分析功能
- 用户认证系统
- 报告生成功能

## 📄 许可证

MIT License