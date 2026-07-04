from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL
from tools.calculator_tool import calculator_func
from tools.file_tool import read_file_func
from tools.keyword_tool import extract_keywords
from tools.gap_analysis_tool import analyze_skill_gap_func
from tools.rag_search_tool import rag_search


def build_tool_agent():


    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0
    )

    tools = [
        calculator_func,
        read_file_func,
        extract_keywords,
        analyze_skill_gap_func,
        rag_search,
    ]

    system_prompt = """
                    你是一个岗位 JD 分析与技术问答助手。
                    
                    你可以使用以下工具：
                    1. calculator：处理数学计算问题
                    2. read_file：读取本地文件内容
                    3. extract_keywords：提取岗位或文本中的技术关键词
                    4. analyze_skill_gap：分析岗位要求与用户技能之间的差距
                    5. rag_search：回答知识库中的技术问题
                    
                    工具选择规则：
                    - 数学计算问题，使用 calculator
                    - 读取本地文件内容，使用 read_file
                    - 提取技术关键词，使用 extract_keywords
                    - 分析岗位能力差距，使用 analyze_skill_gap
                    - 技术知识问答，使用 rag_search
                    
                    重要约束：
                    - 如果用户问题明显需要调用工具，优先调用工具，不要凭空作答
                    - 如果能力差距分析缺少必要参数，要先提示用户补充岗位要求或已有技能
                    - 回答尽量准确、简洁
                    """

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

    return agent


def run_tool_agent(user_input: str) -> dict:
    """
    执行 Agent，并返回结果。
    """
    agent = build_tool_agent()

    result = agent.invoke({  # type: ignore
        "messages": [{"role": "user", "content": user_input}]
    })
    return result


if __name__ == "__main__":
    test_questions = [
        "23 * 17 等于多少？",
        "请读取 data/files/jd_agent_intern.txt 的内容",
        "提取这段 JD 的技术关键词：熟悉 Python、FastAPI、LangChain、RAG、Dify。",
        "请分析这个岗位和我的技能差距。岗位要求：熟悉 Python、FastAPI、LangChain、RAG、Chroma、Dify、Docker。我的技能：Python、FastAPI、LangChain、RAG。",
        "RAG 的核心流程是什么？"
    ]

    for question in test_questions:
        print("=" * 60)
        print("用户问题：", question)

        result = run_tool_agent(question)

        print("Agent 返回：")
        print(result)