# 测试结果记录

## 项目信息

项目名称：基于 LangChain 的岗位 JD 分析 RAG Agent  
测试方式：命令行交互测试  
入口文件：app.py  

---

## 1. 计算工具测试

问题：23 * 17 等于多少？  
期望工具：calculator  
实际工具：calculator  
结果：391  
状态：通过  

---

## 2. 文件读取测试

问题：请读取 data/files/jd_agent_intern.txt 的内容  
期望工具：read_file  
实际工具：read_file  
结果：成功读取岗位 JD 文件内容  
状态：通过  

---

## 3. 技术关键词提取测试

问题：提取这段 JD 的技术关键词：熟悉 Python、FastAPI、LangChain、RAG、向量数据库，有 Dify 或 Coze 经验优先。  
期望工具：extract_keywords  
实际工具：extract_keywords  
结果：成功提取 Python、FastAPI、LangChain、RAG、向量数据库、Dify、Coze  
状态：通过  

---

## 4. RAG 问答测试：RAG 核心流程

问题：RAG 的核心流程是什么？  
期望工具：rag_search  
实际工具：rag_search  
结果：成功基于知识库回答 RAG 离线阶段和在线阶段  
引用来源：data/docs/rag_retrieval_pipeline.md  
状态：通过  

---

## 5. RAG 问答测试：LangChain 作用

问题：LangChain 在 Agent 开发里有什么作用？  
期望工具：rag_search  
实际工具：rag_search  
结果：成功基于知识库回答 LangChain 在 Prompt、Tool、Retriever、RAG 和 Agent 编排中的作用  
引用来源：data/docs/langchain_engineering.md  
状态：通过  

---

## 6. RAG 问答测试：BM25 与 BGE 区别

问题：BM25 和 BGE 的区别是什么？  
期望工具：rag_search  
实际工具：rag_search  
结果：成功回答 BM25 偏关键词匹配，BGE 偏语义向量检索  
引用来源：data/docs/retrieval_models_bm25_bge_faiss_chroma_reranker.md  
状态：通过  

---

## 7. 普通对话测试

问题：你好  
期望工具：none  
实际工具：none  
结果：正常返回问候语  
状态：通过  

---

## 8. 退出测试

操作：输入 exit / quit / q  
结果：程序正常退出  
状态：通过  

---

# 总体结论

本项目已完成命令行版闭环，支持 calculator、read_file、extract_keywords、rag_search、none 五类分支，能够完成岗位 JD 文件读取、技术关键词提取、知识库 RAG 问答、引用来源返回和普通对话。

## FastAPI 接口测试

### 1. 健康检查接口

接口：GET /health  
状态：通过  

### 2. Agent 总控接口

接口：POST /agent/chat  
测试问题：23 * 17 等于多少？  
预期工具：calculator  
实际工具：calculator  
状态：通过  

测试问题：RAG 的核心流程是什么？  
预期工具：rag_search  
实际工具：rag_search  
状态：通过  

### 3. RAG 问答接口

接口：POST /rag/chat  
测试问题：BM25 和 BGE 的区别是什么？  
结果：成功返回 answer 和 sources  
状态：通过  

测试问题：DeepSpeed ZeRO-3 是什么？  
结果：资料中未提供相关信息，sources 为空  
状态：通过  

### 4. JD 关键词提取接口

接口：POST /jd/extract  
测试文本：熟悉 Python、FastAPI、LangChain、RAG、向量数据库，有 Dify 或 Coze 经验优先。  
结果：成功返回 keywords 数组和 count  
状态：通过  