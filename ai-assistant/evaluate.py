"""
Day 6: RAG 质量评估
- 如果 RAGAS 装上了 → 用 RAGAS 跑三个指标
- 如果 RAGAS 没装上 → 用 LLM 自己打分（手写评估）
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import re
import json
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

MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")


# ============================================================
# 1. 测试数据（20 组问答，基于知识库中的考研词汇）
# ============================================================
TEST_CASES = [
    # 词汇释义类（应该在知识库中）
    {"question": "define是什么意思", "ground_truth": "v. 下定义，n. 定义"},
    {"question": "vanish是什么意思", "ground_truth": "v. 突然消失"},
    {"question": "identify是什么意思", "ground_truth": "v. 识别，确认，鉴定"},
    {"question": "statement是什么意思", "ground_truth": "n. 陈述，声明"},
    {"question": "productivity是什么意思", "ground_truth": "n. 生产力"},
    {"question": "generate是什么意思", "ground_truth": "v. 产生，生成"},
    {"question": "release是什么意思", "ground_truth": "v./n. 释放，发布"},
    {"question": "manufacture是什么意思", "ground_truth": "v. 制造，生产"},
    {"question": "reflect是什么意思", "ground_truth": "v. 反映，反射，思考"},
    {"question": "contribute是什么意思", "ground_truth": "v. 贡献，投稿，促成"},
    {"question": "recognize是什么意思", "ground_truth": "v. 认出，承认，识别"},
    {"question": "motivate是什么意思", "ground_truth": "v. 激发，促使，成为动机"},
    {"question": "calculate是什么意思", "ground_truth": "v. 计算"},
    {"question": "locate是什么意思", "ground_truth": "v. 位于，确定位置"},
    {"question": "extract是什么意思", "ground_truth": "v. 提取，摘录"},
    {"question": "specific是什么意思", "ground_truth": "adj. 具体的，特定的，明确的"},
    {"question": "illustrate是什么意思", "ground_truth": "v. 说明，阐明，加插图"},
    {"question": "publication是什么意思", "ground_truth": "n. 出版，出版物，发表"},
    {"question": "shelter是什么意思", "ground_truth": "v. 庇护，遮蔽 n. 避难所"},
    {"question": "measure是什么意思", "ground_truth": "v. 测量 n. 措施，尺寸"},
]

# ============================================================
# 2. 核心函数：检索 + 回答
# ============================================================
def smart_retrieve(question: str):
    """混合检索"""
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


def ask_rag(question: str):
    """单轮 RAG 问答"""
    nodes = smart_retrieve(question)
    context = "\n\n---\n\n".join([n.node.get_text() for n in nodes])
    messages = [
        {"role": "system", "content": "你是个人知识库助手。请基于以下资料回答问题。如果资料中没有相关信息，请诚实说不知道。"},
        {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
    ]
    resp = llm_client.chat.completions.create(
        model=MODEL, messages=messages, stream=False,
    )
    return resp.choices[0].message.content, [n.node.get_text()[:200] for n in nodes]


# ============================================================
# 3. 评估函数
# ============================================================
def faithfulness_score(question, answer, context):
    """忠实度：回答是否基于资料，没有编造"""
    prompt = f"""判断以下 AI 回答是否完全基于给定的资料，没有编造内容。

资料：
{context[:1500]}

问题：{question}
回答：{answer}

请评分（0~1），并给出简短理由。
格式：SCORE: 0.X
理由：..."""
    resp = llm_client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=200,
    )
    text = resp.choices[0].message.content
    score = re.search(r'SCORE:\s*(0?\.[0-9]+|1\.0)', text)
    return float(score.group(1)) if score else 0.5, text


def relevance_score(question, answer):
    """回答相关性：回答是否切题"""
    prompt = f"""判断以下回答是否准确回应了用户问题。

问题：{question}
回答：{answer}

请评分（0~1），0=完全不相关，1=精准回答。
格式：SCORE: 0.X
理由：..."""
    resp = llm_client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=200,
    )
    text = resp.choices[0].message.content
    score = re.search(r'SCORE:\s*(0?\.[0-9]+|1\.0)', text)
    return float(score.group(1)) if score else 0.5, text


def context_precision_score(question, context):
    """上下文精确度：检索到的片段是否真的有用"""
    prompt = f"""判断以下检索到的资料片段是否与问题相关，是否有助于回答问题。

问题：{question}

资料片段（前500字）：
{context[:1500]}

请评分（0~1），0=全部无关，1=全部有用。
格式：SCORE: 0.X
理由：..."""
    resp = llm_client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=200,
    )
    text = resp.choices[0].message.content
    score = re.search(r'SCORE:\s*(0?\.[0-9]+|1\.0)', text)
    return float(score.group(1)) if score else 0.5, text


# ============================================================
# 4. 跑评估
# ============================================================
def run_evaluation():
    print("=" * 60)
    print(f"RAG 质量评估 — {len(TEST_CASES)} 组测试")
    print("=" * 60)

    results = []
    faithfulness_scores = []
    relevance_scores = []
    precision_scores = []

    for i, tc in enumerate(TEST_CASES):
        q = tc["question"]
        gt = tc["ground_truth"]
        print(f"\n[{i+1}/{len(TEST_CASES)}] {q}")
        print(f"    标准答案: {gt}")

        # 问 RAG
        answer, contexts = ask_rag(q)
        print(f"    AI 回答: {answer[:80]}...")
        print(f"    检索到 {len(contexts)} 个片段")

        # 打分
        print("    评分中...")
        f_score, f_reason = faithfulness_score(q, answer, "\n".join(contexts))
        r_score, r_reason = relevance_score(q, answer)
        p_score, p_reason = context_precision_score(q, "\n".join(contexts))

        faithfulness_scores.append(f_score)
        relevance_scores.append(r_score)
        precision_scores.append(p_score)

        results.append({
            "question": q,
            "ground_truth": gt,
            "answer": answer,
            "faithfulness": f_score,
            "answer_relevancy": r_score,
            "context_precision": p_score,
        })

        print(f"    Faithfulness={f_score:.2f} | Relevancy={r_score:.2f} | Precision={p_score:.2f}")

    # 汇总
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)
    avg_f = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_r = sum(relevance_scores) / len(relevance_scores)
    avg_p = sum(precision_scores) / len(precision_scores)

    print(f"  Faithfulness (忠实度)    : {avg_f:.3f}  — 回答是否基于资料")
    print(f"  Answer Relevancy (相关性): {avg_r:.3f}  — 回答是否切题")
    print(f"  Context Precision (精确度): {avg_p:.3f}  — 检索是否命中")

    print(f"\n  综合得分: {(avg_f + avg_r + avg_p) / 3:.3f}")

    # 分析瓶颈
    print("\n  --- 瓶颈分析 ---")
    scores = [
        ("忠实度 (Faithfulness)", avg_f, "回答编造内容" if avg_f < 0.7 else "回答可靠"),
        ("相关性 (Relevancy)", avg_r, "回答跑题" if avg_r < 0.7 else "回答切题"),
        ("精确度 (Precision)", avg_p, "检索不精准" if avg_p < 0.7 else "检索准确"),
    ]
    for name, score, bad_msg in sorted(scores, key=lambda x: x[1]):
        status = bad_msg if score < 0.7 else "良好"
        print(f"    {name}: {score:.3f} → {status}")

    # 保存结果
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "num_cases": len(TEST_CASES),
            "summary": {
                "faithfulness": round(avg_f, 3),
                "answer_relevancy": round(avg_r, 3),
                "context_precision": round(avg_p, 3),
            },
            "details": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存到 eval_results.json")


if __name__ == "__main__":
    run_evaluation()
