import re
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DOCS_DIR


#固定切块大小与重复大小
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# 创建全局缓存，避免每次查询都重新构建 BM25
_BM25_RETRIEVER: Optional[BM25Retriever] = None

#分词函数
def tokenize_text(text: str) -> List[str]:

    text = text.lower()

    tokens = re.findall(
        r"[a-zA-Z0-9_\-\.]+|[\u4e00-\u9fff]",
        text
    )

    return tokens

#文档加载-->Dict[Document]
def load_raw_documents() -> List[Document]:
    """
    从 data/docs 读取 md/txt 文档。
    """
    docs_dir = Path(DOCS_DIR)

    documents = []

    for file_path in docs_dir.glob("*"):
        if file_path.suffix.lower() not in [".md", ".txt"]:
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": str(file_path)
                }
            )
        )

    return documents

#文档切分函数
def split_documents(documents: List[Document]) -> List[Document]:
   #创建切分实例
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    #返回切分结果
    return splitter.split_documents(documents)

#构建BM25检索器
def build_bm25_retriever(top_k: int = 5) -> BM25Retriever:
   #获取原始文档-->Dict[Document]
    raw_docs = load_raw_documents()

    if not raw_docs:
        raise ValueError("data/docs 中没有可用于 BM25 检索的文档")

    #获取切分内容对象
    chunks = split_documents(raw_docs)

    #创建BM25检索实例
    retriever = BM25Retriever.from_documents(
        documents=chunks,
        preprocess_func=tokenize_text
    )
    #设置 BM25 检索器的 k 参数
    retriever.k = top_k

    #返回检索实例
    return retriever

#获取BM25检索器，用于全局缓存机制
def get_bm25_retriever(top_k: int = 5) -> BM25Retriever:

    global _BM25_RETRIEVER

    if _BM25_RETRIEVER is None:
        _BM25_RETRIEVER = build_bm25_retriever(top_k=top_k)

    _BM25_RETRIEVER.k = top_k

    return _BM25_RETRIEVER

#BM25检索执行函数
def bm25_retrieve_docs(query: str, top_k: int = 5) -> List[Document]:

    #创建检索器对象
    retriever = get_bm25_retriever(top_k=top_k)
    #创建检索内容对象
    docs = retriever.invoke(query)

    return docs


if __name__ == "__main__":
    test_queries = [
        "BM25 和 BGE 的区别是什么？",
        "RAG 的核心流程是什么？",
        "Chroma 的作用是什么？",
        "Tool Calling 是什么？"
    ]

    for query in test_queries:
        print("=" * 80)
        print("Query:", query)

        results = bm25_retrieve_docs(query, top_k=5)

        for i, doc in enumerate(results, start=1):
            print(f"\n--- Top {i} ---")
            print("Source:", doc.metadata.get("source"))
            print("Content:", doc.page_content[:300])