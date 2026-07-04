from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from rag.rag_chain import rag_answer


# 用于临时保存最近一次 RAG 的 sources
_LAST_RAG_SOURCES = []

#定义Pydantic 数据模型 RagSearchInput
class RagSearchInput(BaseModel):
    question: str = Field(
        description="用户提出的技术知识问题，例如：RAG 的核心流程是什么？"
    )

#每次 Agent 执行前清空旧的 RAG sources
def reset_last_rag_sources():

    global _LAST_RAG_SOURCES
    _LAST_RAG_SOURCES = []

#获取最近一次 rag_search 工具产生的 sources。
def get_last_rag_sources() -> list:

    return _LAST_RAG_SOURCES


#基于本地知识库进行 RAG 问答。
#只把 answer 返回给 Agent，sources 单独保存，避免 sources 混入 answer 文本。
def rag_search_func(question: str) -> str:

    global _LAST_RAG_SOURCES

    try:
        #执行 RAG（检索增强生成）问答流程-->dict
        result = rag_answer(question)

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        # 如果资料中没有答案，则清空 sources
        if "资料中未提供" in answer:
            _LAST_RAG_SOURCES = []
        else:
            _LAST_RAG_SOURCES = sources

        return answer

    except Exception as e:
        _LAST_RAG_SOURCES = []
        return f"RAG 检索失败：{str(e)}"

#创建StructureTool实例
rag_search = StructuredTool.from_function(
    func=rag_search_func,
    name="rag_search",
    description=(
        "用于基于本地知识库回答技术概念问题。"
        "适合回答 RAG、LangChain、BM25、BGE、Chroma、FastAPI、Tool Calling、Agent 等技术问题。"
        "不用于数学计算，不用于读取文件，不用于岗位能力差距分析。"
        "输入参数 question 是用户的技术问题。"
        "如果知识库没有相关内容，严格返回“资料中未提供相关信息”。不要额外添加其他的信息"
    ),
    args_schema=RagSearchInput,
)


if __name__ == "__main__":
    test_questions = [
        "RAG 的核心流程是什么？",
        "DeepSpeed ZeRO-3 是什么？"
    ]

    for q in test_questions:
        print("=" * 60)
        print("问题：", q)

        reset_last_rag_sources()

        answer = rag_search.invoke({
            "question": q
        })

        print("回答：")
        print(answer)

        print("sources：")
        print(get_last_rag_sources())