import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import re
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

# ---- 初始化引擎 ----
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

llm_client = OpenAI_Client(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
)

# ---- Agent 判断：是否需要查知识库 ----
def should_search(question: str) -> bool:
    if re.findall(r'[a-zA-Z]{2,}', question):
        return True
    resp = llm_client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "deepseek-v4-flash"),
        messages=[{
            "role": "system",
            "content": (
                "判断用户问题是否需要查询本地知识库来回答。"
                "需要：问题涉及具体知识、事实、概念、定义、数据。"
                "不需要：闲聊、问候、你是谁、今天天气、纯计算。"
                "只回复 YES 或 NO。"
            ),
        }, {"role": "user", "content": question}],
        max_tokens=5,
        temperature=0,
    )
    return "YES" in resp.choices[0].message.content.strip().upper()

# ---- 混合检索 ----
def smart_retrieve(question: str):
    keywords = re.findall(r'[a-zA-Z]{2,}', question)
    semantic_nodes = retriever.retrieve(question)

    if keywords:
        all_data = chroma_collection.get(include=["documents", "metadatas"])
        kw_lower = set(k.lower() for k in keywords)
        kw_matched = []
        for i, doc_text in enumerate(all_data["documents"]):
            if any(kw in doc_text.lower() for kw in kw_lower):
                kw_matched.append((i, doc_text, all_data["metadatas"][i]))

        if kw_matched:
            from llama_index.core.schema import NodeWithScore, TextNode
            result = []
            seen = set()
            for idx, text, meta in kw_matched:
                if text[:100] not in seen:
                    seen.add(text[:100])
                    node = TextNode(text=text, metadata=meta or {})
                    result.append(NodeWithScore(node=node, score=1.0))
            for n in semantic_nodes:
                if n.node.get_text()[:100] not in seen:
                    seen.add(n.node.get_text()[:100])
                    result.append(n)
            return result[:5]
        return semantic_nodes[:5]
    return semantic_nodes[:5]

# ---- FastAPI ----
app = FastAPI(title="Knowledge Base RAG API", version="1.0")

# CORS — 允许前端跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- 会话记忆（按 session_id 存储）----
chat_sessions: dict[str, list[dict]] = {}

def get_history(session_id: str) -> list[dict]:
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    return chat_sessions[session_id]

def update_history(session_id: str, question: str, answer: str):
    history = get_history(session_id)
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    # 只保留最近 5 轮
    if len(history) > 10:
        chat_sessions[session_id] = history[-10:]


class Question(BaseModel):
    question: str
    session_id: str = "default"
    verbose: bool = False


class Answer(BaseModel):
    question: str
    answer: str
    sources: list[str] = []


@app.get("/health")
def health():
    return {"status": "ok", "chunks": chroma_collection.count(), "sessions": len(chat_sessions)}


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    chat_sessions.pop(session_id, None)
    return {"status": "cleared"}


@app.post("/ask", response_model=Answer)
def ask(payload: Question):
    history = get_history(payload.session_id)
    nodes = []

    # ---- Agent 判断：需要查知识库吗？ ----
    if should_search(payload.question):
        # 追问时借用历史关键词
        search_question = payload.question
        if not re.findall(r'[a-zA-Z]{2,}', payload.question) and history:
            for msg in reversed(history):
                if msg["role"] == "user":
                    prev_kw = re.findall(r'[a-zA-Z]{2,}', msg["content"])
                    if prev_kw:
                        search_question = f"{' '.join(prev_kw)} {payload.question}"
                        break

        nodes = smart_retrieve(search_question)
        context = "\n\n---\n\n".join([n.node.get_text() for n in nodes])
        messages = [
            {"role": "system", "content": (
                "你是个人知识库助手。请基于以下资料回答问题。"
                "如果资料中确实没有相关信息，请诚实说不知道。"
            )},
            *history[-10:],
            {"role": "user", "content": f"资料：\n{context}\n\n问题：{payload.question}"},
        ]
    else:
        # 不需要查库 — 直接聊天
        messages = [
            {"role": "system", "content": "你是个人知识库助手。用友好简洁的方式回答用户。"},
            *history[-10:],
            {"role": "user", "content": payload.question},
        ]

    response = llm_client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "deepseek-v4-flash"),
        messages=messages,
        stream=False,
    )

    answer_text = response.choices[0].message.content
    update_history(payload.session_id, payload.question, answer_text)

    sources = []
    if payload.verbose and nodes:
        sources = [n.node.get_text()[:150] for n in nodes]

    return Answer(
        question=payload.question,
        answer=answer_text,
        sources=sources,
    )
