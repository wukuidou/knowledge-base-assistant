import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

load_dotenv()

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5",
)

db = chromadb.PersistentClient(path="./storage/chroma_db")
chroma_collection = db.get_collection("knowledge_base")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)

retriever = index.as_retriever(similarity_top_k=5)

print("==== RAG Retrieval Diagnostic ====")

test_questions = [
    "realm是什么意思",
    "realm",
    "有多少个重点单词",
    "考研英语核心词有哪些",
]

for q in test_questions:
    print(f"\n---")
    print(f"Question: {q}")
    nodes = retriever.retrieve(q)
    print(f"Retrieved {len(nodes)} chunks:\n")
    for i, node in enumerate(nodes):
        score = node.score if hasattr(node, 'score') else 'N/A'
        text = node.node.get_text()
        if isinstance(score, float):
            print(f"  [{i+1}] score={score:.4f}")
        else:
            print(f"  [{i+1}]")
        print(f"      text: {text[:250]}")
        fname = node.node.metadata.get('file_name', 'unknown')
        print(f"      source: {fname}")
        print()
