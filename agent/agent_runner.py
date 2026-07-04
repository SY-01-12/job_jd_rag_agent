from agent.build_agent import run_tool_agent
from tools.rag_search_tool import get_last_rag_sources, reset_last_rag_sources

DIRECT_OUTPUT_TOOLS = {"read_file", "calculator"}


def parse_tool_steps_from_messages(messages: list) -> list[dict]:
    """
    从 LangGraph Agent 的 messages 列表中提取工具调用信息。
    """
    tool_steps = []

    for msg in messages:
        # 检查是否有 tool_calls（AIMessage）
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get('name', '')
                tool_args = tool_call.get('args', {})

                tool_steps.append({
                    "tool_name": tool_name,
                    "tool_input": tool_args,
                    "observation": "",  # 暂时为空，后续会从 ToolMessage 补充
                    "observation_preview": ""
                })

        # 检查是否是 ToolMessage（工具返回结果）
        elif hasattr(msg, 'name') and hasattr(msg, 'content'):
            # 这是一个工具响应消息
            if tool_steps:
                # 将 observation 补充到最后一个工具步骤
                last_step = tool_steps[-1]
                observation_text = str(msg.content)
                last_step["observation"] = observation_text
                last_step["observation_preview"] = observation_text[:300]

    return tool_steps


def choose_final_answer(tool_steps: list[dict], llm_output: str) -> str:
    """
    决定最终返回给前端的 answer。

    规则：
    1. 如果最后一次工具调用属于确定性工具（如 read_file、calculator），
       则直接返回工具 observation，不再使用 LLM 总结后的 output。
    2. 其他情况继续返回 LLM output。
    """
    if not tool_steps:
        return llm_output

    last_step = tool_steps[-1]
    last_tool = last_step.get("tool_name")
    last_observation = last_step.get("observation", "")

    if last_tool in DIRECT_OUTPUT_TOOLS and last_observation.strip():
        return last_observation

    return llm_output


def print_agent_log(user_input: str, tool_steps: list[dict], answer: str, sources: list):
    print("[User]", user_input)

    if tool_steps:
        print("[Tool]", [step["tool_name"] for step in tool_steps])

        for step in tool_steps:
            print("[Args]", step["tool_input"])
            print("[Observation_preview]", step["observation_preview"])
    else:
        print("[Tool]", ["none"])
        print("[Args]", {})
        print("[Observation_preview]", "")

    print("[Answer Length]", len(answer))
    print("[Source Count]", len(sources))
    print("-" * 60)


def run_agent(user_input: str) -> dict:
    try:
        # 每次执行前清空旧 RAG sources，避免上一轮 sources 污染本轮
        reset_last_rag_sources()

        result = run_tool_agent(user_input)

        # 从 messages 中提取最终答案
        messages = result.get("messages", [])
        if not messages:
            return {
                "answer": "Agent 未返回任何消息",
                "tool_used": "none",
                "sources": []
            }

        # 最后一条消息通常是最终的 AIMessage
        final_message = messages[-1]
        llm_output = getattr(final_message, 'content', '')

        # 提取工具调用步骤
        tool_steps = parse_tool_steps_from_messages(messages)
        used_tools = [step["tool_name"] for step in tool_steps if step["tool_name"]]

        if used_tools:
            tool_used = ",".join(used_tools)
        else:
            tool_used = "none"

        # 根据工具类型决定最终 answer
        answer = choose_final_answer(tool_steps, llm_output)

        # 只有调用了 rag_search，才返回 RAG sources
        if "rag_search" in used_tools:
            sources = get_last_rag_sources()
        else:
            sources = []

        print_agent_log(
            user_input=user_input,
            tool_steps=tool_steps,
            answer=answer,
            sources=sources
        )

        return {
            "answer": answer,
            "tool_used": tool_used,
            "sources": sources
        }

    except Exception as e:
        return {
            "answer": f"Agent 执行失败：{str(e)}",
            "tool_used": "error",
            "sources": []
        }


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

        result = run_agent(question)

        print("使用工具：", result["tool_used"])
        print("回答：")
        print(result["answer"])
        print("来源：")
        print(result["sources"])
