"""Day 4: Multi-tool Agent — 手写 ReAct 循环，AI 自己决定用哪个工具"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import re
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI as OpenAI_Client
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
import chromadb

load_dotenv()

# ---- 适配 DeepSeek ----
import llama_index.llms.openai.utils as openai_utils
import llama_index.llms.openai.base as openai_base
import tiktoken.model as tiktoken_model
for mod in (openai_utils, openai_base):
    mod.openai_modelname_to_contextsize = lambda name: 131072
    mod.is_chat_model = lambda model: True
    mod.is_function_calling_model = lambda model: True
tiktoken_model.encoding_name_for_model = lambda name: "cl100k_base"

# ---- 初始化 ----
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
Settings.llm = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
    model=os.getenv("LLM_MODEL", "deepseek-v4-flash"),
)
db = chromadb.PersistentClient(path="./storage/chroma_db")
chroma_collection = db.get_collection("knowledge_base")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)
retriever = index.as_retriever(similarity_top_k=20)
llm = OpenAI_Client(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
)

# ============================================================
# 工具定义（纯函数，不依赖任何框架）
# ============================================================
def search_knowledge_base(query: str) -> str:
    """搜索本地知识库"""
    keywords = re.findall(r'[a-zA-Z]{2,}', query)
    all_data = chroma_collection.get(include=["documents"])
    kw_matched = []
    if keywords:
        kw_lower = set(k.lower() for k in keywords)
        for doc_text in all_data["documents"]:
            if any(kw in doc_text.lower() for kw in kw_lower):
                kw_matched.append(doc_text)
    semantic = [n.node.get_text() for n in retriever.retrieve(query)]
    results = kw_matched[:3] + [t for t in semantic if t not in kw_matched]
    return "\n\n---\n\n".join(results[:5]) if results else "未找到相关信息。"

def calculate(expression: str) -> str:
    """执行数学计算"""
    try:
        cleaned = re.sub(r'[^0-9+\-*/().%\s]', '', expression)
        return f"{expression} = {eval(cleaned)}"
    except Exception as e:
        return f"计算出错：{e}"

def get_current_time() -> str:
    """获取当前时间"""
    now = datetime.now()
    w = ["周一","周二","周三","周四","周五","周六","周日"]
    return f"{now.year}年{now.month}月{now.day}日 {w[now.weekday()]} {now.hour:02d}:{now.minute:02d}"

TOOLS = {
    "search": {"fn": search_knowledge_base, "desc": "搜索本地知识库。需要传 query 参数，如 search('vanish')", "has_arg": True},
    "calc":   {"fn": calculate,             "desc": "执行数学计算。需要传 expression 参数，如 calc('3.14*5**2')", "has_arg": True},
    "time":   {"fn": lambda: get_current_time(), "desc": "获取当前日期和时间。无需参数，直接调用 time()", "has_arg": False},
}

TOOLS_DESC = "\n".join(f"- {name}(): {info['desc']}" for name, info in TOOLS.items())

# ============================================================
# ReAct 循环（手写版）
# ============================================================
SYSTEM_PROMPT = f"""你是一个有工具调用能力的 AI 助手。你可以使用以下工具：

{TOOLS_DESC}

工作流程：
1. 分析用户问题，决定是否需要调用工具
2. 如果需要，回复格式：ACTION: tool_name, ARGS: 参数
3. 系统会告诉你工具返回结果：OBSERVATION: ...
4. 如果还需要其他工具，继续 ACTION；如果信息足够，直接回答用户
5. 不需要工具就直接回答

重要规则：
- 每次只能调用一个工具
- ACTION 和 ARGS 必须严格按格式写在同一行
- 不要编造工具返回的结果，等看到 OBSERVATION 后才能引用
- 工具调用最多 3 次"""

def react_loop(question: str, history: list = None) -> str:
    """手写 ReAct：Think → Act → Observe → 循环"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        messages.extend(history[-10:])

    messages.append({"role": "user", "content": question})
    iterations = 0

    while iterations < 5:
        iterations += 1
        resp = llm.chat.completions.create(
            model=os.getenv("LLM_MODEL", "deepseek-v4-flash"),
            messages=messages,
        )
        reply = resp.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        # 检查是否需要调用工具
        # 格式1: ACTION: tool_name, ARGS: 参数
        act = re.search(r'ACTION:\s*(\w+)\s*,?\s*ARGS:\s*(.+)', reply, re.IGNORECASE)
        # 格式2: ACTION: tool_name（无参数）
        act_noarg = re.search(r'ACTION:\s*(\w+)', reply, re.IGNORECASE)

        if act:
            tool_name = act.group(1).lower().strip()
            tool_arg = act.group(2).strip().strip("'").strip('"')
        elif act_noarg and act_noarg.group(1).lower() in TOOLS:
            tool_name = act_noarg.group(1).lower().strip()
            tool_arg = ""
        else:
            return reply  # 没有工具调用 → 最终回答

        if tool_name not in TOOLS:
            observation = f"未知工具 '{tool_name}'，可用：{list(TOOLS.keys())}"
        else:
            tool = TOOLS[tool_name]
            try:
                if tool["has_arg"]:
                    result = tool["fn"](tool_arg)
                else:
                    result = tool["fn"]()
                observation = result[:2000]
            except Exception as e:
                observation = f"工具调用出错：{e}"

        print(f"  🔧 [{iterations}] {tool_name}({tool_arg[:50] if tool_arg else ''})")
        messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})

    return "抱歉，处理超时，请重试。"

# ============================================================
# 交互
# ============================================================
chat_history = []
print("=" * 50)
print("Agent ready with tools:", list(TOOLS.keys()))
print("=" * 50)

while True:
    question = input("\n🙋 Question: ")
    if question.lower() in ("quit", "exit", "q"):
        break
    answer = react_loop(question, chat_history)
    print(f"\n🤖 Answer: {answer}")
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})
