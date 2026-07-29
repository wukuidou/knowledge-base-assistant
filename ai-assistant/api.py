import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import re
import time
import threading
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI as OpenAI_Client
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
import chromadb

load_dotenv()

# ---- 导入文档解析器 ----
from ingest import load_file, chunk_documents

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
chroma_collection = db.get_or_create_collection("knowledge_base")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
retriever = index.as_retriever(similarity_top_k=20)

# 并发锁
_sessions_lock = threading.Lock()
_db_lock = threading.Lock()

llm_client = OpenAI_Client(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
)

# ---- Agent 判断：是否需要查知识库 ----
# 明显不需要查库的模式（问候/闲聊/纯计算）
_CHAT_PATTERNS = re.compile(
    r'^(你好|hi|hello|hey|早|晚上好|下午好|再见|bye|谢谢|thank|'
    r'你是谁|你叫什么|你能做什么|介绍一下自己|'
    r'\d+[\+\-\*\/]\d+|今天天气|现在几点了|'
    r'帮我写|帮我翻译|讲个笑话|聊天)\b',
    re.IGNORECASE,
)


def should_search(question: str) -> bool:
    # 1. 含英文关键词 → 极有可能需要查资料
    if re.findall(r'[a-zA-Z]{2,}', question):
        return True
    # 2. 明显是闲聊 → 不需要查
    if _CHAT_PATTERNS.match(question.strip()):
        return False
    # 3. 拿不准 → 让 LLM 判断
    try:
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
        content = (resp.choices[0].message.content or "NO").strip().upper()
        return "YES" in content
    except Exception:
        return True  # LLM 挂了也继续查库，兜底

# ---- 混合检索 ----
def smart_retrieve(question: str):
    keywords = re.findall(r'[a-zA-Z]{2,}', question)
    semantic_nodes = retriever.retrieve(question)

    if keywords and chroma_collection.count() <= 500:
        # 只在小规模库上做全量关键词扫描
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
            for idx, text, meta in kw_matched[:20]:  # 最多取 20 条
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
chat_sessions: dict[str, dict] = {}  # session_id -> {"history": [...], "last_access": float}

SESSION_TTL = 3600  # 1 小时不活跃自动清理


def _cleanup_sessions():
    """清理过期会话"""
    now = time.time()
    stale = [
        sid for sid, v in chat_sessions.items()
        if now - v.get("last_access", now) > SESSION_TTL
    ]
    for sid in stale:
        chat_sessions.pop(sid, None)


def get_history(session_id: str) -> list[dict]:
    with _sessions_lock:
        _cleanup_sessions()
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {"history": [], "last_access": time.time()}
        entry = chat_sessions[session_id]
        entry["last_access"] = time.time()
        return list(entry["history"])

def update_history(session_id: str, question: str, answer: str):
    with _sessions_lock:
        entry = chat_sessions.get(session_id, {"history": [], "last_access": time.time()})
        entry["history"].append({"role": "user", "content": question})
        entry["history"].append({"role": "assistant", "content": answer or "(empty response)"})
        if len(entry["history"]) > 10:
            entry["history"] = entry["history"][-10:]
        entry["last_access"] = time.time()
        chat_sessions[session_id] = entry


class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000)
    session_id: str = "default"
    verbose: bool = False


class Answer(BaseModel):
    question: str
    answer: str
    sources: list[str] = []


class FileInfo(BaseModel):
    name: str
    size: int
    type: str


class UploadResult(BaseModel):
    success: list[FileInfo] = []
    errors: list[str] = []
    total_chunks_added: int = 0


# ---- 数据目录 ----
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


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


@app.on_event("shutdown")
def shutdown():
    """优雅关闭：清理 ChromaDB 连接"""
    try:
        db._client.close()
    except Exception:
        pass


# ---- 文件上传 & 文档管理 ----

MAX_FILES = 10
MAX_SIZE = 50 * 1024 * 1024  # 50MB

# 后台处理状态: filename -> "pending" | "parsing" | "chunking" | "indexing" | "done" | "error: ..."
processing_status: dict[str, str] = {}
_status_lock = threading.Lock()


def _process_files(filepaths: list[Path]):
    """后台任务：解析 → 分块 → 向量化 → 入库"""
    for fp in filepaths:
        name = fp.name
        with _status_lock:
            processing_status[name] = "parsing"
        try:
            docs = load_file(fp)
            if not docs:
                with _status_lock:
                    processing_status[name] = "error: empty or unparseable"
                continue

            with _status_lock:
                processing_status[name] = "chunking"
            nodes = chunk_documents(docs)

            if nodes:
                with _status_lock:
                    processing_status[name] = "indexing"
                with _db_lock:
                    VectorStoreIndex(
                        nodes=nodes,
                        storage_context=storage_context,
                        show_progress=False,
                    )
                with _status_lock:
                    processing_status[name] = "done"
            else:
                with _status_lock:
                    processing_status[name] = "error: no chunks generated"
        except Exception as e:
            with _status_lock:
                processing_status[name] = f"error: {e}"


@app.post("/upload", response_model=UploadResult)
async def upload_files(files: list[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    """上传文档 — 先存盘立即返回，后台异步建索引"""
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"最多同时上传 {MAX_FILES} 个文件")
    if not files:
        raise HTTPException(400, "请选择至少一个文件")

    result = UploadResult()
    saved_paths: list[Path] = []

    for file in files:
        if not file.filename:
            result.errors.append("(未知文件): 文件名为空")
            continue

        content = await file.read()
        if len(content) > MAX_SIZE:
            result.errors.append(f"{file.filename}: 超过 50MB 限制")
            continue
        if len(content) == 0:
            result.errors.append(f"{file.filename}: 空文件")
            continue

        safe_name = Path(file.filename).name
        dest = DATA_DIR / safe_name
        if dest.exists():
            stem, ext = dest.stem, dest.suffix
            counter = 1
            while dest.exists():
                dest = DATA_DIR / f"{stem}_{counter}{ext}"
                counter += 1

        dest.write_bytes(content)
        saved_paths.append(dest)

        with _status_lock:
            processing_status[dest.name] = "pending"

        result.success.append(FileInfo(
            name=dest.name,
            size=len(content),
            type=dest.suffix.lower() or "text",
        ))

    # 后台异步处理（解析 + 分块 + 向量化）
    if saved_paths and background_tasks:
        background_tasks.add_task(_process_files, saved_paths)

    return result


@app.get("/upload/status")
def upload_status():
    """查询后台处理状态"""
    with _status_lock:
        return dict(processing_status)


@app.get("/documents")
def list_documents():
    """列出 data/ 下所有文档"""
    files = []
    for f in sorted(DATA_DIR.rglob("*")):
        if f.is_file() and not f.name.startswith("."):
            files.append(FileInfo(
                name=f.name,
                size=f.stat().st_size,
                type=f.suffix.lower() or "text",
            ))
    return {"files": files, "total": len(files)}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """删除文档及其向量数据"""
    safe_name = Path(filename).name
    filepath = DATA_DIR / safe_name

    if not filepath.exists():
        raise HTTPException(404, f"文件不存在: {safe_name}")

    # 拒绝删除正在处理的文件
    with _status_lock:
        st = processing_status.get(safe_name, "")
    if st and st not in ("done",) and not st.startswith("error"):
        raise HTTPException(409, f"文件正在处理中 ({st})，请稍后再试")

    # 从向量库删除
    source_path = str(filepath.resolve())
    deleted_count = 0
    try:
        results = chroma_collection.get(
            where={"source": source_path},
            include=["documents"],
        )
        if results["ids"]:
            chroma_collection.delete(ids=results["ids"])
            deleted_count = len(results["ids"])
    except Exception:
        # ChromaDB 可能不支持 where 过滤，尝试全量匹配
        all_data = chroma_collection.get(include=["documents", "metadatas"])
        ids_to_delete = []
        for i, meta in enumerate(all_data.get("metadatas", []) or []):
            if meta and meta.get("source") == source_path:
                ids_to_delete.append(all_data["ids"][i])
        if ids_to_delete:
            chroma_collection.delete(ids=ids_to_delete)
            deleted_count = len(ids_to_delete)

    # 删除文件
    filepath.unlink()

    # 清理状态
    with _status_lock:
        processing_status.pop(safe_name, None)

    return {"status": "deleted", "filename": safe_name, "chunks_removed": deleted_count}
