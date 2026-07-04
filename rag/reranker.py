import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.documents import Document
from typing import List, Optional
from config import RERANK_MODEL_NAME


# 全局缓存，避免每次调用都重复加载模型
_TOKENIZER: Optional[AutoTokenizer] = None
_MODEL: Optional[AutoModelForSequenceClassification] = None
_DEVICE: Optional[str] = None

#设备选择cpu/gpu
def get_device() -> str:

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

#加载重排模型（只加载一次）
def load_reranker_model():

    global _TOKENIZER, _MODEL, _DEVICE

    if _TOKENIZER is None or _MODEL is None:
        _DEVICE = get_device()

        _TOKENIZER = AutoTokenizer.from_pretrained(RERANK_MODEL_NAME)
        _MODEL = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL_NAME)
        _MODEL.to(_DEVICE)
        _MODEL.eval()

    return _TOKENIZER, _MODEL, _DEVICE

#计算query与文档分数
def compute_rerank_scores(query: str, docs: List[Document]) -> List[float]:

    if not docs:
        return []

    tokenizer, model, device = load_reranker_model()

    #将query与文档构建成pairs
    pairs = [
        [query, doc.page_content]
        for doc in docs
    ]

    #不计算梯度，只预测
    with torch.no_grad():
        #文字转模型输入
        inputs = tokenizer(
            pairs,
            padding=True,   #自动补0，让所有句子长度一致
            truncation=True,    #句子太长就截断
            max_length=512,     #最大长度为512
            return_tensors="pt" #返回pytorch张量格式
        )

        #把数据搬到 GPU / CPU 上
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        #把模型计算的原始分数拉平成一维列表，然后转为浮点型，在cpu上进行计算，最终返回Python列表形式
        scores = outputs.logits.view(-1).float().cpu().tolist()

    return scores

#对文档重排
def rerank_docs(
    query: str,
    docs: List[Document],
    top_k: int = 3
) -> List[Document]:
    """
    使用 BGE Reranker 对候选 docs 进行重排序。
    """
    if not docs:
        return []

    scores = compute_rerank_scores(query, docs)

    #文档 + 分数 配对
    scored_docs = []

    for doc, score in zip(docs, scores):
        doc.metadata["rerank_score"] = float(score)
        scored_docs.append((score, doc))

    #按分数从高到低排序
    scored_docs.sort(
        key=lambda x: x[0],
        reverse=True
    )

    #只取前 top_k 篇
    reranked_docs = [
        doc for score, doc in scored_docs[:top_k]
    ]

    return reranked_docs


if __name__ == "__main__":
    from rag.hybrid_retriever import hybrid_retrieve_docs

    test_queries = [
        "RAG 的核心流程是什么？",
        "BM25 和 BGE 的区别是什么？",
        "Chroma 的作用是什么？",
        "Tool Calling 是什么？"
    ]

    for query in test_queries:
        print("=" * 80)
        print("Query:", query)

        candidate_docs = hybrid_retrieve_docs(
            query=query,
            bm25_k=8,
            vector_k=8,
            candidate_k=10,
            final_k=10,
            use_rerank=False
        )

        reranked_docs = rerank_docs(
            query=query,
            docs=candidate_docs,
            top_k=3
        )

        for i, doc in enumerate(reranked_docs, start=1):
            print(f"\n--- Rerank Top {i} ---")
            print("Rerank Score:", doc.metadata.get("rerank_score"))
            print("Hybrid Score:", doc.metadata.get("hybrid_score"))
            print("Source:", doc.metadata.get("source"))
            print("Content:", doc.page_content[:300])