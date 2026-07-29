import chromadb

db = chromadb.PersistentClient(path="./storage/chroma_db")
col = db.get_collection("knowledge_base")

print(f"文档数量: {col.count()}")

# 获取 embeddings 维度
result = col.get(limit=1, include=["embeddings"])
if result["embeddings"] is not None and len(result["embeddings"]) > 0:
    print(f"向量维度: {len(result['embeddings'][0])}")
else:
    print("向量维度: 无法获取")
