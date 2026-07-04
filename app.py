from agent.agent_runner import run_agent
def print_result(result:dict):
    #打印工具名称
    print("使用工具：", result["tool_used"])
    print("回答：", result["answer"])
    #尝试打印来源
    if result['sources']:
        for source in result['sources']:
            print("- 来源：", source['file'])


def main():
    print("-----------------欢迎使用AI Agent-----------------")
    print("输入 exit / quit / q 退出")

    while True:
        user_input = input("请输入问题：").strip()

        if user_input == "exit" or user_input == "quit" or user_input == "q":
            print("bye")
            break
        elif not user_input:
            print("请重新输入")
            continue

        response = run_agent(user_input)
        print_result(response)


if __name__ == "__main__":
    main()