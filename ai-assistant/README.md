# 📚 个人知识库 AI 助手 v2.0

> 基于 RAG（检索增强生成）的个人知识问答系统  
> 从"能查到"升级到"能对话"——带记忆、能判断、会选工具

## 架构 v2.0

```
                        ┌──────────────────────┐
                        │   Chat 前端（纯 HTML）│
                        │  chat/index.html      │
                        └──────────┬───────────┘
                                   │ POST /ask
                                   ▼
┌────── 用户 ──────┐        ┌─────────────┐
│  query.py (CLI)  │◄──────►│  api.py     │
│  agent.py (ReAct)│        │  (FastAPI)  │
└──────────────────┘        └──────┬──────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
           ┌────────────┐  ┌──────────┐  ┌──────────┐
           │should_search│  │对话记忆   │  │ ReAct    │
           │(Agent判断)  │  │chat_history│  │多工具调用 │
           └──────┬─────┘  └──────────┘  └────┬─────┘
                  ▼                           ▼
        ┌──────────────────┐         ┌──────────────┐
        │  混合检索         │         │ search_kb    │
        │  关键词+语义      │         │ calc / time  │
        └──────┬───────────┘         └──────────────┘
               ▼
        ┌──────────────┐
        │  ChromaDB    │
        │  向量数据库   │
        └──────┬───────┘
               ▲
        ┌──────┴───────┐
        │  ingest.py   │
        │  PDF/Word/   │
        │  HTML/Code/  │
        │  TXT         │
        └──────────────┘
```

## 快速开始

```bash
# 1. 环境
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. 配置 .env
# LLM_API_KEY=你的key
# LLM_BASE_URL=https://api.deepseek.com/v1
# LLM_MODEL=deepseek-v4-flash

# 3. 文档入库（支持 PDF / Word / HTML / 代码 / 纯文本）
python ingest.py

# 4. 启动（四种方式任选）
python query.py                 # 命令行问答（带对话记忆）
python agent.py                 # ReAct Agent 模式（多工具调用）
uvicorn api:app --port 8000     # REST API
# 浏览器打开 chat/index.html    # 聊天界面（配合 API）
```

## v2.0 新增功能

| 功能 | 说明 | 文件 |
|------|------|------|
| 🗣️ **多轮对话记忆** | AI 记住上下文，支持"上一个呢""再解释一下" | `query.py`, `api.py` |
| 🎨 **Chat 前端** | 纯 HTML/CSS/JS，暗色主题，来源折叠 | `chat/index.html` |
| 🧠 **Agent 判断** | AI 自己决定查不查知识库，闲聊不浪费检索 | `query.py` `should_search()` |
| 🔧 **ReAct 多工具** | 手写 Agent 循环，搜知识库+算数学+看时间 | `agent.py` |
| 📄 **多格式导入** | PDF / Word / HTML / 代码 / 纯文本 自动识别 | `ingest.py` (v2) |
| 📊 **质量评估** | 忠实度 / 相关性 / 精确度 三项指标 | `evaluate.py` |

### 核心亮点：Agent 判断

```
以前（无脑检索）：
  🙋 你好 → 🔍 检索 → 找不到 → "不知道"              ← 不该查
  🙋 今天天气 → 🔍 检索 → 找不到 → "不知道"            ← 不该查

现在（Agent 判断）：
  🙋 你好 → 🤖 "你好！有什么可以帮你的？"              ← 不查，直接聊
  🙋 define → 🔍 检索知识库 → ✅ "下定义，定范围"        ← 查
  🙋 3.14*5² → 🧮 calculate → ✅ "3.14×25=78.5"        ← 用工具算
```

## 项目结构

```
ai-assistant/
├── .env                  ← API 密钥（不提交 Git）
├── .gitignore
├── requirements.txt
├── README.md
│
├── ingest.py             ← 文档入库 v2（PDF/Word/HTML/代码/文本）
├── query.py              ← 命令行问答（带对话记忆 + Agent 判断）
├── agent.py              ← ReAct Agent（多工具调用，手写循环）
├── api.py                ← FastAPI REST API（多会话支持）
├── evaluate.py           ← RAG 质量评估（20 组测试 + 三项指标）
│
├── chat/
│   └── index.html        ← 纯前端聊天页面（零依赖）
│
├── data/                 ← 你的文档（PDF/Word/HTML/代码/TXT）
├── storage/              ← ChromaDB 向量数据（不提交 Git）
└── .venv/                ← 虚拟环境（不提交 Git）
```

## 技术栈

| 组件 | 技术 | 原因 |
|------|------|------|
| RAG 框架 | LlamaIndex | 专注 RAG 场景，上手快 |
| Embedding | BAAI/bge-small-zh-v1.5 | 中文效果好，512维，免费 |
| 向量库 | ChromaDB | 本地运行，零配置 |
| LLM | DeepSeek | 便宜，OpenAI 兼容接口，128K 上下文 |
| PDF 解析 | PyMuPDF (fitz) | pypdf 对中文支持差 |
| Word 解析 | python-docx | 支持段落和表格提取 |
| HTML 解析 | BeautifulSoup + lxml | 自动移除 script/style |
| 前端 | 纯 HTML/CSS/JS | 零依赖，开箱即用 |
| API | FastAPI + uvicorn | 高性能异步，CORS 支持 |

## 遇到的问题 & 解决方案

### Week 1

#### 1. HuggingFace 模型下载超时
- **原因**：国内网络访问 huggingface.co 不通
- **解决**：`set HF_ENDPOINT=https://hf-mirror.com`

#### 2. PDF 中文乱码
- **原因**：pypdf 对部分中文 PDF 支持不好
- **解决**：改用 PyMuPDF（fitz），C 语言实现，中文支持好

#### 3. 词汇表无法通过语义检索找到
- **原因**：语义搜索对字典/词汇表天然失效
- **解决**：混合检索——英文关键词全库直搜 + 语义检索兜底

#### 4. 关键词匹配漏检
- **原因**：只在语义 top-20 里过滤，目标词排在第 25 位就看不到
- **解决**：ChromaDB 全库 `get()` + Python 文本匹配，100% 召回

### Week 2

#### 5. 纯中文闲聊触发无效检索
- **原因**：所有问题都走检索，包括"你好""天气"
- **解决**：`should_search()` 两步判断——有英文关键词直接查，纯中文问 LLM 判断

#### 6. 追问丢失上下文
- **原因**：每次请求独立，AI 不知道上一句
- **解决**：`chat_history` 列表存储最近 5 轮对话，携带到 prompt 中

#### 7. Agent 框架太重
- **原因**：LlamaIndex ReActAgent 依赖多，出问题难排查
- **解决**：手写 ReAct 循环——解析 LLM 输出中的 `ACTION: tool, ARGS: x`，自动循环调用，不到 100 行

#### 8. 代码文件 chunk 策略
- **原因**：代码用 512 字分块会截断函数/类定义
- **解决**：代码文件自动用 chunk_size=256，保留语言元数据

## 路线图

| 版本 | 内容 |
|------|------|
| v1.0 ✅ | 完整 RAG 链路：PDF 解析 + 混合检索 + Web + API |
| v2.0 ✅ | 多轮对话记忆 + Chat 前端 + Agent 判断 + 多工具 + 多格式导入 + RAGAS 评估 |
| v3.0 🔜 | 联网搜索、知识库管理（增删改）、多文档对话 |

## 评估结果

运行 `python evaluate.py` 查看最新评分。

指标说明：
| 指标 | 含义 |
|------|------|
| Faithfulness | AI 回答是否基于检索到的资料 |
| Answer Relevancy | 回答是否切题 |
| Context Precision | 检索到的片段是否相关 |
