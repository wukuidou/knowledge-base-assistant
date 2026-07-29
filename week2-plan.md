# 个人知识库 AI 助手 — 第二周实施计划

> **前置条件**：第一周已完成，跑通了完整 RAG 链路（PDF 解析 + 混合检索 + Web + API）
> **本周目标**：给 RAG 装上记忆和判断力——让 AI 能多轮对话、自主决定何时查知识库

---

## 项目概览

第一周我们搭好了一个**能用的** RAG 系统。但它有两个明显的局限：

| 局限 | 表现 |
|------|------|
| **没有记忆** | 每轮对话都是独立的，AI 不知道你上一句说了什么 |
| **没有判断力** | 所有问题都会去查知识库，包括"你好""今天天气怎么样" |

第二周的核心目标：**让系统从"能查到"变成"能对话"**。

---

## 技术选型

| 新增组件 | 选择 | 原因 |
|----------|------|------|
| 对话记忆 | LlamaIndex ChatMemory / 自建 | 存储对话历史，多轮追问 |
| Agent | LlamaIndex ReAct Agent | 让 AI 自己决定用不用知识库 |
| Chat 前端 | 纯 HTML/CSS/JS（单文件） | 零依赖，开箱即用，不装任何框架 |
| 文档扩展 | python-docx, beautifulsoup4 | 支持 Word 和网页 |

---

## 每日计划

### Day 1：多轮对话记忆

**目标**：让 AI 记住上下文，支持追问

```
📄 本日会修改的文件：
   D:\learn\chat\ai-assistant\query.py  ← 加入对话记忆
```

#### 为什么需要记忆

```
现在（无记忆）：
  🙋 realm是什么？   → 🤖 realm不在知识库中
  🙋 那上一个呢？    → 🤖 什么上一个？我不知道...（已失忆）

改后（有记忆）：
  🙋 vanish是什么意思？ → 🤖 vanish是动词，突然消失...
  🙋 上一个呢？         → 🤖 上一个单词是vanish...（有记忆，能理解指代）
```

#### 实现方式

不改 ChromaDB，不改 Embedding。只在 `query.py` 里维护一个消息列表，把最近的对话历史带上：

```python
# 核心逻辑：
messages = [
    {"role": "system", "content": "你是个人知识库助手..."},
    # 带上最近 N 轮对话
    *chat_history[-6:],   # 最近 3 轮（6 条消息）
    {"role": "user", "content": f"资料：{context}\n\n问题：{question}"},
]
```

不使用 LlamaIndex 的 ChatMemory（太重），手写一个轻量的消息管理，你更能理解原理。

#### ✅ Day 1 完成标志
- [ ] 能多轮追问（"上一个呢""再解释一下"）
- [ ] AI 不丢失上下文
- [ ] 代码改动不超过 30 行

---

### Day 2：打造 Chat 前端

**目标**：做一个漂亮的聊天页面，接入 `/ask` API

```
📄 本日会创建的文件：
   D:\learn\chat\chat\index.html  ← 纯前端 Chat 页面
```

#### 为什么不用 Streamlit

Streamlit 适合 Demo 和内部工具，但真正的聊天产品需要一个更像"聊天软件"的界面。用纯 HTML 写的好处：
- 零依赖，不需要装任何前端框架
- 部署简单，双击 `index.html` 就能用
- 面试时可以说"前端是我手写的"

#### 功能清单

- ⌨️ 消息气泡（用户/AI 双色区分）
- 📎 来源引用（点击展开）
- ⏳ 加载动画（等待 AI 回复时）
- 🔄 清空对话按钮
- 🎨 支持亮色/暗色主题

#### 架构

```
浏览器 (index.html)
    │  fetch POST http://localhost:8000/ask
    ▼
FastAPI (api.py)  ← 需要加 CORS 支持
    │
    ▼
混合检索 → DeepSeek → 返回回答
```

#### ✅ Day 2 完成标志
- [ ] 浏览器打开 `index.html`，输入问题能获得回答
- [ ] 回答附带来源引用
- [ ] 对话历史在页面中正确显示
- [ ] API 开启了 CORS，前端能跨域调用

---

### Day 3：Agent 工具调用 —— 让 AI 自己决定查不查知识库

**目标**：AI 自己判断要不要检索，而不是所有问题都查

```
现在（无脑检索）：
  🙋 你好 → 🔍 检索"你好" → 找不到 → 不知道     ← 不该查
  🙋 今天天气怎么样 → 🔍 检索 → 找不到 → 不知道    ← 不该查

改后（Agent 判断）：
  🙋 你好 → 🤖 你好！有什么可以帮你的？           ← 不查知识库
  🙋 今天天气怎么样 → 🤖 这个需要联网搜索才能回答   ← 不查知识库
  🙋 define是什么意思 → 🔍 检索 → ✅ 查到         ← 查知识库
```

#### 实现方式

不使用 LlamaIndex 的 Agent（太重，依赖多），我们手写一个轻量级路由：

```python
def should_search(question: str, llm) -> bool:
    """让 LLM 判断这个问题是否需要查知识库"""
    response = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[{
            "role": "system",
            "content": "判断以下问题是否需要查询个人知识库。知识库包含考研英语词汇。只需要回答 YES 或 NO。"
        }, {
            "role": "user",
            "content": question
        }],
        max_tokens=5,
    )
    return "YES" in response.choices[0].message.content.upper()
```

流程：

```
用户提问 → LLM 判断 → YES → 混合检索 → 生成回答
                    → NO  → 直接回答（不查知识库）
```

#### ✅ Day 3 完成标志
- [ ] "你好" 不触发检索
- [ ] "define是什么意思" 触发检索
- [ ] 知识库相关的问题正确检索
- [ ] 闲聊问题直接回答，不浪费检索时间

---

### Day 4：Agent 工具调用（进阶）—— 多工具并行

**目标**：给 AI 配多个工具，它自己选

```
工具列表：
  🔍 search_knowledge_base  — 查你的词汇书
  🧮 calculate              — 算数学
  ⏰ get_current_time       — 看时间
  💬 chat                   — 直接聊天
```

#### 实现方式

不再手写路由，用 LlamaIndex 的 `FunctionTool`：

```python
from llama_index.core.tools import FunctionTool

def search_kb(query: str) -> str:
    """Search the vocabulary knowledge base"""
    nodes = smart_retrieve(query)
    return "\n\n".join([n.node.get_text() for n in nodes])

search_tool = FunctionTool.from_defaults(
    fn=search_kb,
    name="search_knowledge_base",
    description="搜考研英语词汇知识库，查单词释义和用法",
)

agent = ReActAgent.from_tools([search_tool, ...], llm=llm, verbose=True)
```

这一步比 Day 3 更深入——你真正理解了 Agent 的 Tool Calling 机制。

#### ✅ Day 4 完成标志
- [ ] Agent 至少有 3 个工具
- [ ] AI 能自己选择用哪个工具
- [ ] 工具调用日志清晰可读

---

### Day 5：支持更多文档格式

**目标**：不只 PDF，Word 和网页也能入库

```
📄 本日会修改的文件：
   D:\learn\chat\ai-assistant\ingest.py  ← 升级支持多种格式
```

#### 新增支持的格式

| 格式 | 库 | 用途 |
|------|-----|------|
| `.docx` | python-docx | Word 文档 |
| `.html` | beautifulsoup4 | 网页 |
| `.py` `.js` `.java` | 纯文本 | 代码（大块分 chunk） |
| `.txt` | 已有 | 纯文本 |

#### 实现方式

在 `ingest.py` 里加一个格式路由：

```python
def load_file(filepath):
    ext = filepath.suffix.lower()
    if ext == '.pdf':
        return load_pdf(filepath)      # fitz
    elif ext == '.docx':
        return load_docx(filepath)     # python-docx
    elif ext in ('.html', '.htm'):
        return load_html(filepath)     # beautifulsoup
    else:
        return load_text(filepath)     # 纯文本
```

#### ✅ Day 5 完成标志
- [ ] 能导入 Word 文档
- [ ] 能导入网页
- [ ] 代码文件不按 512 字分块（用代码专用分块策略）

---

### Day 6：RAG 质量评估

**目标**：量化你的 RAG 系统到底有多好

#### 为什么要评估

面试时你说"我的 RAG 系统回答准确"，面试官问"准确率是多少？"——你不能说"感觉还行"。

#### 评估方法

用 RAGAS 框架跑一组标准测试：

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# 准备测试数据（至少 10 组问答）
test_data = [
    {"question": "define是什么意思", "ground_truth": "定义"},
    {"question": "vanish是什么意思", "ground_truth": "突然消失"},
    ...
]

# 跑评估
result = evaluate(test_data, metrics=[faithfulness, answer_relevancy, context_precision])
print(result)  # → 每个指标一个 0~1 的分数
```

| 指标 | 含义 |
|------|------|
| Faithfulness（忠实度） | AI 回答是否完全基于检索到的资料，有没有编造 |
| Answer Relevancy（回答相关性） | 回答是否切题 |
| Context Precision（上下文精确度） | 检索到的片段是否真的有用 |

#### ✅ Day 6 完成标志
- [ ] 至少 15 组标注问答
- [ ] RAGAS 三个指标都有分数
- [ ] 知道哪个环节是瓶颈（检索？生成？）

---

### Day 7：总结 + 面试准备

**目标**：把这两周的成果整理成面试素材

#### 产出清单

1. **更新 README**（加入 v2.0 新功能）
2. **写一篇技术博客草稿**（或面试自述稿）

```
面试自述模板：

我搭建了一个个人知识库 AI 助手，技术栈是 LlamaIndex + ChromaDB + DeepSeek。
第一周跑通了完整的 RAG 链路：文档入库、向量检索、大模型生成、Web 展示。
遇到的主要挑战是 PDF 中文解析和词汇表的语义检索失效问题——
我通过改用 PyMuPDF 和实现关键词全库直搜解决了。
第二周我给系统加上了多轮对话记忆、Agent 自主判断能力、
以及 RAGAS 量化评估，使系统从"能查到"升级为"能对话"。
最终回答准确率达到 XX%，检索召回率 XX%。
```

3. **更新简历项目描述**
4. **Git 打 tag `v2.0`**

#### ✅ Day 7 完成标志
- [ ] README 更新
- [ ] 面试自述稿完成（2~3 分钟版本）
- [ ] Git tag v2.0
- [ ] 明确第三阶段方向

---

## 第二周结束后你的产出

```
✅ 多轮对话记忆（能追问、能理解指代）
✅ 纯前端 Chat 页面（零依赖，好看）
✅ Agent 自主判断（该查才查，不浪费）
✅ 多文档格式支持（Word + 网页 + 代码）
✅ RAGAS 质量评估（有数据，不是感觉）
✅ 面试自述稿
✅ Git tag v2.0
```

## 简历升级版

> **个人知识库 AI 助手**（v2.0）
> 基于 LlamaIndex + ChromaDB + DeepSeek 构建的 RAG 问答系统。实现混合检索（关键词全库直搜 + 语义向量）、多轮对话记忆、Agent 自主工具调用。支持 PDF/Word/HTML 多格式文档导入，回答附带来源溯源。通过 RAGAS 评估框架量化系统质量，检索准确率 XX%，回答忠实度 XX%。

---

## Windows 新增命令速查

| 你要做的事 | 命令 |
|-----------|------|
| 启动 API（后台） | `start uvicorn api:app --port 8000` |
| 杀掉 8000 端口 | `netstat -ano \| findstr :8000` 然后 `taskkill /PID 进程号 /F` |
| 查看 CPU/内存 | `tasklist \| findstr python` |
| 测试 API | `curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"test\"}"` |

---

> **核心心态**：第一周是"能跑"，第二周是"能打"。面试官不会问你用了什么框架，会问你遇到过什么问题、怎么解决的、效果提升了多少。这一周积累的就是这些素材。
