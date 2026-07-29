import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import re
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

# 1. Embedding
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5",
)

# 2. LLM
Settings.llm = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
    model=os.getenv("LLM_MODEL", "deepseek-v4-flash"),
)

# 3. 加载索引
db = chromadb.PersistentClient(path="./storage/chroma_db")
chroma_collection = db.get_collection("knowledge_base")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)

# 4. 检索器：多拿一些候选（20 个），后面再精选
retriever = index.as_retriever(similarity_top_k=20)

# 5. DeepSeek 客户端
deepseek_client = OpenAI_Client(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
)

# 6. 关键词提取 + 混合检索（全库搜索！）
def extract_keywords(question):
    """从问题中提取英文单词"""
    return re.findall(r'[a-zA-Z]{2,}', question)

def should_search(question: str) -> bool:
    """判断是否需要查知识库：有英文单词直接查，纯中文让 LLM 判断"""
    if extract_keywords(question):
        return True
    resp = deepseek_client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "deepseek-v4-flash"),
        messages=[{
            "role": "system",
            "content": (
                "判断用户问题是否需要查询本地知识库来回答。"
                "需要：问题涉及具体知识、事实、概念、定义、数据。"
                "不需要：闲聊、问候、你是谁、今天天气、纯计算。"
                "只回复 YES 或 NO。"
            ),
        }, {
            "role": "user", "content": question,
        }],
        max_tokens=5,
        temperature=0,
    )
    return "YES" in resp.choices[0].message.content.strip().upper()

def smart_retrieve(question: str):
    """混合检索：关键词全库直搜 + 语义兜底"""
    keywords = extract_keywords(question)
    semantic_nodes = retriever.retrieve(question)

    if keywords:
        all_data = chroma_collection.get(include=["documents", "metadatas"])
        kw_matched_texts = []
        kw_lower = set(k.lower() for k in keywords)
        for i, doc_text in enumerate(all_data["documents"]):
            if any(kw in doc_text.lower() for kw in kw_lower):
                kw_matched_texts.append((i, doc_text, all_data["metadatas"][i]))

        if kw_matched_texts:
            from llama_index.core.schema import NodeWithScore, TextNode
            kw_nodes = []
            seen_texts = set()
            for idx, text, meta in kw_matched_texts:
                if text[:100] not in seen_texts:
                    seen_texts.add(text[:100])
                    node = TextNode(text=text, metadata=meta or {})
                    kw_nodes.append(NodeWithScore(node=node, score=1.0))
            for n in semantic_nodes:
                if n.node.get_text()[:100] not in seen_texts:
                    seen_texts.add(n.node.get_text()[:100])
                    kw_nodes.append(n)
            return kw_nodes[:5]
        else:
            return semantic_nodes[:5]
    else:
        return semantic_nodes[:5]

# 7. 交互循环（带对话记忆）
chat_history = []  # 存储最近的消息对 [{"role": "user", ...}, {"role": "assistant", ...}, ...]
print("AI Assistant started! (type 'quit' to exit)")
print("-" * 40)
while True:
    question = input("\nQuestion: ")
    if question.lower() in ("quit", "exit", "q"):
        break

    # ---- Agent 判断：这个问题需要查知识库吗？ ----
    need_search = should_search(question)
    if need_search:
        # 检索增强：追问时借用历史中的关键词
        search_question = question
        if not extract_keywords(question) and chat_history:
            for msg in reversed(chat_history):
                if msg["role"] == "user":
                    prev_kw = extract_keywords(msg["content"])
                    if prev_kw:
                        search_question = f"{' '.join(prev_kw)} {question}"
                        break

        nodes = smart_retrieve(search_question)
        print(f"\n🔍 [SEARCH] Retrieved {len(nodes)} chunks")
        for i, node in enumerate(nodes):
            kw_match = any(k in node.node.get_text().lower() for k in extract_keywords(search_question))
            marker = " [KW-MATCH]" if kw_match else ""
            print(f"  {i+1}.{marker} {node.node.get_text()[:100]}...")

        context = "\n\n---\n\n".join([n.node.get_text() for n in nodes])
        messages = [
            {"role": "system", "content": (
                "你是个人知识库助手。请基于以下资料回答问题。"
                "资料是从你的考研词汇书中检索到的相关片段。"
                "如果资料中明确包含了用户问的单词，请给出该单词的释义和用法。"
                "如果资料中确实没有相关信息，请诚实说不知道。"
            )},
            *chat_history[-10:],
            {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
        ]
    else:
        # 不需要查库 — 直接聊天
        print(f"\n💬 [CHAT] No search needed")
        messages = [
            {"role": "system", "content": "你是个人知识库助手。用友好简洁的方式回答用户。"},
            *chat_history[-10:],
            {"role": "user", "content": question},
        ]

    response = deepseek_client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "deepseek-v4-flash"),
        messages=messages,
        stream=False,
    )
    answer = response.choices[0].message.content
    print(f"\nAnswer: {answer}")

    # ---- 更新对话历史 ----
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})
