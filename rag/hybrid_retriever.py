from typing import List
from rag.reranker import rerank_docs
from langchain_core.documents import Document
from rag.bm25_retriever import bm25_retrieve_docs
from rag.retriever import retrieve_docs

#私有函数，目的是：确保相同文档不会被重复计算分数
def _doc_key(doc: Document) -> str:
    """
    用 source + content 作为去重 key。
    """
    source = doc.metadata.get("source", "")
    content = doc.page_content.strip()
    return f"{source}::{content}"

def merge_and_deduplicate_docs(
    bm25_docs: List[Document],
    vector_docs: List[Document],
    top_k: int = 5,
    bm25_weight: float = 1.0,
    vector_weight: float = 1.5,
    rrf_k: int = 60
) -> List[Document]:
    """
    使用加权 RRF 融合 BM25 与向量检索结果。

    RRF 思想：
    排名越靠前，得分越高。
    如果同一个 chunk 同时被 BM25 和 Vector 召回，分数会累加。
    """

    doc_score_map = {}  #存每个文档的总得分
    doc_map = {}        #村每个文档的完整内容


    '''
    遍历 BM25 / Vector 结果：
    给文档生成唯一身份证 key
    按排名给它算一个分数
    如果是新文档：先登记一下
    把分数加到它的总分里
    '''

    # 1. 加入 BM25 结果
    #start=1,给所有文档进行排名
    for rank, doc in enumerate(bm25_docs, start=1):

        key = _doc_key(doc)

        #RRF打分
        score = bm25_weight / (rrf_k + rank)

        if key not in doc_score_map:
            # 若文档第一次出现则分数初始化为0
            doc_score_map[key] = 0
            doc_map[key] = doc

        doc_score_map[key] += score

    # 2. 加入 Vector 结果
    for rank, doc in enumerate(vector_docs, start=1):
        key = _doc_key(doc)

        score = vector_weight / (rrf_k + rank)

        if key not in doc_score_map:
            doc_score_map[key] = 0
            doc_map[key] = doc

        doc_score_map[key] += score

    # 3. 按融合分数排序
    sorted_keys = sorted(
        #把所有文档的唯一 key 拿出来，变成一个列表
        doc_score_map.keys(),
        #按key 对应的总分来排序
        key=lambda x: doc_score_map[x],
        #降序排序
        reverse=True
    )

    # 4. 返回 Top-K 文档
    final_docs = []

    for key in sorted_keys[:top_k]:
        #取出完整文档
        doc = doc_map[key]
        #给文档metadata增加score信息
        doc.metadata["hybrid_score"] = doc_score_map[key]
        #将更新的信息放入新列表
        final_docs.append(doc)

    return final_docs

def hybrid_retrieve_docs(
    query: str,
    bm25_k: int = 8,
    vector_k: int = 8,
    candidate_k: int = 10,
    final_k: int = 3,
    use_rerank: bool = True     #重排序开关
) -> List[Document]:
    """
    BM25 + Chroma 混合检索。
    可选是否启用模型版 Reranker。
    """

    # 1. BM25 检索
    bm25_docs = bm25_retrieve_docs(
        query=query,
        top_k=bm25_k
    )

    # 2. 向量检索
    vector_docs = retrieve_docs(
        query=query,
        k=vector_k
    )

    # 3. 候选融合去重
    candidate_docs = merge_and_deduplicate_docs(
        bm25_docs=bm25_docs,
        vector_docs=vector_docs,
        top_k=candidate_k,
        bm25_weight=1.0,
        vector_weight=1.5
    )

    # 4. 模型重排序
    if use_rerank:
        #创建排序实例
        final_docs = rerank_docs(
            query=query,
            docs=candidate_docs,
            top_k=final_k
        )
    else:
        final_docs = candidate_docs[:final_k]

    return final_docs

if __name__ == "__main__":
    test_queries = [
        "RAG 的核心流程是什么？",
        "BM25 和 BGE 的区别是什么？",
        "Chroma 的作用是什么？",
        "Tool Calling 是什么？"
    ]

    for query in test_queries:
        print("=" * 80)
        print("Query:", query)

        results = hybrid_retrieve_docs(
            query=query,
            bm25_k=8,
            vector_k=8,
            candidate_k=10,
            final_k=3,
            use_rerank=True
        )

        for i, doc in enumerate(results, start=1):
            print(f"\n--- Final Top {i} ---")
            print("Rerank Score:", doc.metadata.get("rerank_score"))
            print("Hybrid Score:", doc.metadata.get("hybrid_score"))
            print("Source:", doc.metadata.get("source"))
            print("Content:", doc.page_content[:300])