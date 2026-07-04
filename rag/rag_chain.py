from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import OPENAI_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL
from rag.hybrid_retriever import hybrid_retrieve_docs


def format_docs(docs) -> str:
    """
    把检索到的文档拼接成 context 文本。
    """
    if not docs:
        return ""

    parts = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "未知来源")
        content = doc.page_content.strip()

        parts.append(
            f"[文档{i}]\n来源: {source}\n内容:\n{content}"
        )

    return "\n\n".join(parts)


def build_sources(docs) -> list:
    """
    从 docs 中提取 sources，并做去重。
    """
    sources = []
    seen_files = set()

    for doc in docs:
        source_file = doc.metadata.get("source", "")
        content = doc.page_content.strip()

        if source_file not in seen_files:
            sources.append({
                "file": source_file,
                "content": content
            })
            seen_files.add(source_file)

    return sources


def rag_answer(question: str) -> dict:
    """
    正式 RAG 问答函数：
    - 使用 BM25 + Chroma 混合召回
    - 使用模型版 Reranker 重排序
    - 返回 answer 和 sources
    """
    # 1. 检索相关文档
    docs = hybrid_retrieve_docs(
        query=question,
        bm25_k=8,
        vector_k=8,
        candidate_k=10,
        final_k=3,
        use_rerank=True
    )

    # 2. 如果一个候选都没有，直接拒答
    if not docs:
        return {
            "answer": "资料中未提供相关信息。",
            "sources": []
        }

    # 3. 构造 context
    context = format_docs(docs)

    # 4. 初始化模型
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0
    )

    # 5. Prompt
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
你是一个严格的知识库问答助手。

你只能根据给定的 context 回答问题。

规则：
1. 如果 context 中能直接回答问题，就基于 context 简洁、准确回答。
2. 如果 context 中没有直接包含问题答案，必须只回答：
   “资料中未提供相关信息。”
3. 禁止使用你自己的参数知识补充回答。
4. 禁止根据常识、经验或外部知识扩展回答。
5. 禁止编造 context 中不存在的内容。
6. 回答尽量简洁清楚。
"""
        ),
        (
            "human",
            "问题：{question}\n\n参考资料：\n{context}"
        )
    ])

    # 6. 调用模型
    chain = prompt | llm
    response = chain.invoke({
        "question": question,
        "context": context
    })

    answer = response.content.strip()

    # 7. 无资料问题：清空 sources
    if "资料中未提供相关信息" in answer:
        return {
            "answer": "资料中未提供相关信息。",
            "sources": []
        }

    # 8. 构造 sources
    sources = build_sources(docs)

    return {
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":
    test_questions = [
        "RAG 的核心流程是什么？",
        "BM25 和 BGE 的区别是什么？",
        "Chroma 的作用是什么？",
        "Tool Calling 是什么？",
        "DeepSpeed ZeRO-3 是什么？"
    ]

    for question in test_questions:
        print("=" * 80)
        print("问题：", question)

        result = rag_answer(question)

        print("回答：")
        print(result["answer"])

        print("来源：")
        for source in result["sources"]:
            print("-", source.get("file", "未知来源"))