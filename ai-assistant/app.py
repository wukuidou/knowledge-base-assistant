import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import re
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI as OpenAI_Client
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
import chromadb

load_dotenv()

# ---- 页面配置 ----
st.set_page_config(
    page_title="Personal Knowledge Base",
    page_icon="📚",
    layout="wide",
)

# ---- 适配 DeepSeek ----
import llama_index.llms.openai.utils as openai_utils
import llama_index.llms.openai.base as openai_base
import tiktoken.model as tiktoken_model

for mod in (openai_utils, openai_base):
    mod.openai_modelname_to_contextsize = lambda name: 131072
    mod.is_chat_model = lambda model: True
    mod.is_function_calling_model = lambda model: True
tiktoken_model.encoding_name_for_model = lambda name: "cl100k_base"

# ---- 初始化引擎（缓存，只加载一次）----
@st.cache_resource
def init_engine():
    # Embedding
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5",
    )
    # LLM
    Settings.llm = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        model=os.getenv("LLM_MODEL", "deepseek-v4-flash"),
    )
    # 向量库
    db = chromadb.PersistentClient(path="./storage/chroma_db")
    chroma_collection = db.get_collection("knowledge_base")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    # DeepSeek 客户端
    llm_client = OpenAI_Client(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
    )

    return index, llm_client, chroma_collection


index, llm_client, collection = init_engine()
retriever = index.as_retriever(similarity_top_k=20)


# ---- 混合检索（全库直搜）----
def smart_retrieve(question):
    keywords = re.findall(r'[a-zA-Z]{2,}', question)
    semantic_nodes = retriever.retrieve(question)

    if keywords:
        # 全库文本直搜（不走语义）
        all_data = collection.get(include=["documents", "metadatas"])
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


# ---- UI ----
st.title("📚 Personal Knowledge Base AI Assistant")
st.markdown("Ask questions based on your documents — answers with source citations.")

# 侧边栏
with st.sidebar:
    st.header("Knowledge Base")
    st.metric("Total Chunks", collection.count())

    st.divider()
    st.subheader("Documents")
    import glob
    files = glob.glob("data/**/*", recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    for f in files:
        st.write(f"- {f}")

    st.divider()
    st.caption("Retrieval: keyword-match + semantic hybrid")

# 对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for i, src in enumerate(msg["sources"]):
                    st.caption(f"Source {i+1}")
                    st.text(src[:300])

# 输入
if prompt := st.chat_input("Ask a question..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            nodes = smart_retrieve(prompt)

            context = "\n\n---\n\n".join([n.node.get_text() for n in nodes])
            messages = [
                {"role": "system", "content": (
                    "You are a personal knowledge base assistant. "
                    "Answer based on the provided materials. "
                    "If the materials contain the word the user is asking about, "
                    "give its definition and usage. "
                    "If the information is not in the materials, say so honestly."
                )},
                {"role": "user", "content": f"Materials:\n{context}\n\nQuestion: {prompt}"},
            ]

            response = llm_client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "deepseek-v4-flash"),
                messages=messages,
                stream=False,
            )
            answer = response.choices[0].message.content

            st.markdown(answer)

            # 来源
            sources = [n.node.get_text() for n in nodes]
            with st.expander(f"Sources ({len(sources)} chunks)"):
                for i, src in enumerate(sources):
                    st.caption(f"Source {i+1}")
                    st.text(src[:300])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
