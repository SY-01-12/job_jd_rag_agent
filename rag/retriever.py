#向量库获取
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from config import CHROMA_DIR, EMBEDDING_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL


def get_vector_retriever(k: int = 4):
    """
    获取 Chroma 向量检索器。
    """
    chroma_path = Path(CHROMA_DIR)

    if not chroma_path.exists() or not any(chroma_path.iterdir()):
        raise RuntimeError(
            f"未找到 Chroma 知识库：{chroma_path}。"
            f"请先运行：python rag/build_index.py"
        )

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 未配置，无法使用 Embedding 检索。")

    embedding_kwargs = {
        "model": EMBEDDING_MODEL,
        "api_key": OPENAI_API_KEY,
    }

    if OPENAI_BASE_URL:
        embedding_kwargs["base_url"] = OPENAI_BASE_URL

    embeddings = OpenAIEmbeddings(**embedding_kwargs)

    vectorstore = Chroma(
        persist_directory=str(chroma_path),
        embedding_function=embeddings,
    )

    return vectorstore.as_retriever(search_kwargs={"k": k})

#文档检索函数
def retrieve_docs(query:str,k:int=3):

    '''
    db.as_retriever()：将 Chroma 向量库包装为 LangChain 检索器对象
    search_kwargs={"k": k}：设置检索参数，指定每次查询返回相似度最高的 k 条结果
    '''
    #创建检索器对象
    retriever = get_vector_retriever(k=k)
    #创建检索内容对象
    docs = retriever.invoke(query)

    return docs

#测试
if __name__ == "__main__":
    query = "RAG 的核心流程是什么？"
    result = retrieve_docs(query)

    print(f"找到{len(result)}个相关文档\n")
    for i,chunk in enumerate(result):
        print(f"{i+1}. {chunk.page_content}")
        print(f"   来源: {chunk.metadata.get('source','未知来源')}")
        print()