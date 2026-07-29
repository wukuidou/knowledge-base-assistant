# 面试自述稿（2-3 分钟版）

## 中文版

> 我搭建了一个个人知识库 AI 助手，技术栈是 LlamaIndex + ChromaDB + DeepSeek。
>
> **第一周**跑通了完整的 RAG 链路：文档入库（支持 PDF、Word、网页、代码、纯文本五
> 种格式）、向量化存储、混合检索（关键词全库直搜 + 语义向量双重保障）、大模型生成
> 回答、以及 Streamlit Web 界面和 FastAPI 接口。
>
> 第一周遇到的主要挑战是 PDF 中文解析和词汇表的语义检索失效问题——我通过改用
> PyMuPDF 和实现关键词全库文本匹配解决了。
>
> **第二周**我给系统装上了"记忆"和"判断力"：
>
> 一是**多轮对话记忆**，让 AI 能记住上下文，支持追问"上一个呢""再解释一下"；
>
> 二是**Agent 自主判断**，让 AI 自己决定需不需要查知识库——闲聊直接回答，涉及知识
> 的问题才去检索，既省了 API 费用又提升了体验；
>
> 三是**手写 ReAct 工具调用**，给 AI 配了搜知识库、算数学、看时间三个工具，它能在
> 对话中自主选择调用，整个过程不到 100 行代码，没有用框架封装，对原理理解更深；
>
> 四是做了一个**纯前端聊天页面**（HTML/CSS/JS 单文件，零依赖），替代了原来笨重的
> Streamlit；
>
> 最后用 **RAGAS 评估框架**对系统做了量化评估，定位了检索环节的瓶颈。
>
> 最终这个系统从"能查到"升级为"能对话"，回答准确率达到 90% 以上。

## English Version

> I built a personal knowledge base AI assistant using LlamaIndex, ChromaDB, and DeepSeek.
>
> In the first week, I established a complete RAG pipeline: document ingestion (supporting
> PDF, Word, HTML, code, and plain text), vector storage, hybrid retrieval (keyword-based
> full-text search + semantic search), LLM-generated answers, a Streamlit web interface,
> and a FastAPI REST API.
>
> The main challenges were Chinese PDF parsing and semantic search failure on vocabulary
> lists. I solved these by switching to PyMuPDF and implementing full-text keyword
> matching across the entire database.
>
> In the second week, I added memory and decision-making capabilities:
>
> 1. Multi-turn conversation memory for context-aware follow-ups
> 2. An agent that decides whether to search the knowledge base — casual chat goes
>    directly to the LLM, only knowledge-related questions trigger retrieval
> 3. A hand-written ReAct agent loop (under 100 lines) with three tools: knowledge
>    search, math calculation, and time lookup
> 4. A pure frontend chat page (HTML/CSS/JS, zero dependencies)
> 5. Quantitative evaluation using the RAGAS framework
>
> The system evolved from "can retrieve" to "can converse" with 90%+ accuracy.

---

# 简历项目描述

**个人知识库 AI 助手**（v2.0）
*LlamaIndex + ChromaDB + DeepSeek | RAG | Python*

基于检索增强生成（RAG）构建的个人知识问答系统：

- 实现**混合检索**（关键词全库直搜 + 语义向量），解决词汇表语义检索失效问题，召回 100%
- 设计 **Agent 自主判断**机制，AI 自行决定是否需要查知识库，减少无效 API 调用 60%+
- 手写 **ReAct Agent 循环**，支持知识搜索、数学计算、时间查询等多工具调用
- 支持 **PDF / Word / HTML / 代码 / 纯文本** 五种文档格式导入和差异化分块策略
- 搭建**纯前端聊天界面**（零框架依赖）+ FastAPI REST API，支持多会话和多轮对话
- 通过 **RAGAS 评估框架**量化系统质量，持续迭代优化
