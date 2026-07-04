# LangChain 工程开发知识文档


## 1. LangChain 的定位

LangChain 是一个用于开发大模型应用的框架。它可以帮助开发者组织 Prompt、调用大模型、定义工具、构建 Agent、连接向量数据库并实现 RAG 应用。对于 AI Agent 应用开发实习，LangChain 的价值在于把大模型调用、工具调用、检索问答和流程编排统一到一个工程结构中。

直接调用大模型 API 只能完成“输入文本、输出文本”的基础功能。而 LangChain 提供了 PromptTemplate、ChatPromptTemplate、MessagesPlaceholder、Tool、Retriever、VectorStore、OutputParser、AgentExecutor 等组件，使开发者更方便地搭建完整应用。

在实际项目中，不能只背概念，而要能说明它在系统流程中的位置。一个可讲清楚的回答通常包括三点：它是什么、解决什么问题、如何在项目中落地。对于 AI Agent 应用开发实习，面试官更关注候选人是否理解工具调用、RAG 检索、向量库、Prompt 约束和接口化服务之间的关系，而不是只会说某个框架名称。


## 2. PromptTemplate 与 ChatPromptTemplate

PromptTemplate 用于普通文本模板。它可以把变量填入模板，例如 skill 和 goal。这样比直接用 f-string 拼接更规范，也便于复用。ChatPromptTemplate 用于构造多角色消息，例如 system、human、assistant。它适合聊天模型，因为聊天模型天然接受消息列表。

在 Agent 或 RAG 项目中，常用 ChatPromptTemplate。system 消息用于设置角色和规则，human 消息用于放用户问题，MessagesPlaceholder 用于插入历史对话。

在实际项目中，不能只背概念，而要能说明它在系统流程中的位置。一个可讲清楚的回答通常包括三点：它是什么、解决什么问题、如何在项目中落地。对于 AI Agent 应用开发实习，面试官更关注候选人是否理解工具调用、RAG 检索、向量库、Prompt 约束和接口化服务之间的关系，而不是只会说某个框架名称。


## 3. MessagesPlaceholder 的作用

MessagesPlaceholder 是 ChatPromptTemplate 中的一个占位符，用于动态插入历史消息。它常用于多轮对话、带记忆的 Agent 和对话式 RAG。例如系统消息是“你是一个 AI Agent 学习助手”，然后插入历史对话，再插入当前用户问题。这样模型在回答当前问题时可以看到最近几轮上下文。

在项目里，历史可以手动用 chat_history 列表维护。每轮对话结束后，把 HumanMessage 和 AIMessage 加入列表，再只保留最近 2 轮，也就是 4 条消息。这样相当于 WindowBuffer，能避免上下文过长。

在实际项目中，不能只背概念，而要能说明它在系统流程中的位置。一个可讲清楚的回答通常包括三点：它是什么、解决什么问题、如何在项目中落地。对于 AI Agent 应用开发实习，面试官更关注候选人是否理解工具调用、RAG 检索、向量库、Prompt 约束和接口化服务之间的关系，而不是只会说某个框架名称。


## 4. LCEL 链式写法

LCEL 是 LangChain Expression Language。它支持使用管道写法把 Prompt、Model 和 Parser 连接起来，例如：prompt | model | parser。这种写法比旧式 LLMChain 更灵活，也更适合当前版本 LangChain。

一个典型的 LCEL 流程是：PromptTemplate 构造输入，大模型生成输出，StrOutputParser 提取文本结果。对于结构化输出，也可以使用 JsonOutputParser 或者手动 json.loads 解析模型返回内容。

在实际项目中，不能只背概念，而要能说明它在系统流程中的位置。一个可讲清楚的回答通常包括三点：它是什么、解决什么问题、如何在项目中落地。对于 AI Agent 应用开发实习，面试官更关注候选人是否理解工具调用、RAG 检索、向量库、Prompt 约束和接口化服务之间的关系，而不是只会说某个框架名称。


## 5. Tool 的定义

Tool 是 Agent 可以调用的外部能力。一个 Tool 应该有清晰的函数名、输入参数、函数说明和稳定返回值。例如 calculator 工具接收 expression 字符串，返回计算结果；read_file 工具接收 file_path，返回文件内容；extract_keywords 工具接收 text，返回技术关键词列表。

工具描述非常重要，因为模型会根据工具说明判断什么时候调用工具。如果工具说明模糊，模型可能选择错误工具或生成错误参数。

在实际项目中，不能只背概念，而要能说明它在系统流程中的位置。一个可讲清楚的回答通常包括三点：它是什么、解决什么问题、如何在项目中落地。对于 AI Agent 应用开发实习，面试官更关注候选人是否理解工具调用、RAG 检索、向量库、Prompt 约束和接口化服务之间的关系，而不是只会说某个框架名称。


## 6. Agent 的基本流程

Agent 的基本流程是：用户输入问题，模型判断是否需要工具，如果需要则选择工具并生成参数，程序执行工具，工具结果返回给模型或直接返回给用户。复杂 Agent 可以循环多次调用工具，简单项目中可以只做一次工具判断。

在岗位 JD 分析 Agent 中，可以把工具判断单独做成 tool_decision.py。该模块让模型只输出 JSON，例如 need_tool、tool_name 和 arguments。Agent Runner 接收这个 JSON 后，根据 tool_name 调用 calculator、read_file、extract_keywords、rag_search 或 none 分支。

在实际项目中，不能只背概念，而要能说明它在系统流程中的位置。一个可讲清楚的回答通常包括三点：它是什么、解决什么问题、如何在项目中落地。对于 AI Agent 应用开发实习，面试官更关注候选人是否理解工具调用、RAG 检索、向量库、Prompt 约束和接口化服务之间的关系，而不是只会说某个框架名称。


## 7. Retriever 与 RAG Chain

LangChain 的 RAG 应用通常包括 Document Loader、Text Splitter、Embedding、VectorStore、Retriever 和 RAG Chain。Document Loader 加载文档，Text Splitter 切块，Embedding 生成向量，VectorStore 保存向量，Retriever 返回相关文档，RAG Chain 把文档拼成上下文并调用大模型回答。

RAG Chain 的返回结果最好包含 answer 和 sources。answer 是模型生成的回答，sources 是检索到的文档来源和片段内容。这样用户可以知道答案依据来自哪个文件。

在实际项目中，不能只背概念，而要能说明它在系统流程中的位置。一个可讲清楚的回答通常包括三点：它是什么、解决什么问题、如何在项目中落地。对于 AI Agent 应用开发实习，面试官更关注候选人是否理解工具调用、RAG 检索、向量库、Prompt 约束和接口化服务之间的关系，而不是只会说某个框架名称。


## 8. 项目文件职责

一个简洁的 LangChain 项目可以分为 tools、rag、agent 三个目录。tools 放 calculator_tool.py、file_tool.py、keyword_tool.py；rag 放 build_index.py、retriever.py、rag_chain.py；agent 放 tool_decision.py 和 agent_runner.py；app.py 是命令行入口。

每个文件的职责应该清晰。build_index.py 只负责构建向量库；retriever.py 只负责检索；rag_chain.py 只负责知识库问答；tool_decision.py 只负责工具选择；agent_runner.py 只负责调度；app.py 只负责接收用户输入和打印结果。

在实际项目中，不能只背概念，而要能说明它在系统流程中的位置。一个可讲清楚的回答通常包括三点：它是什么、解决什么问题、如何在项目中落地。对于 AI Agent 应用开发实习，面试官更关注候选人是否理解工具调用、RAG 检索、向量库、Prompt 约束和接口化服务之间的关系，而不是只会说某个框架名称。
