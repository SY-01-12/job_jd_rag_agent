# 基于混合检索与工具调用的岗位 JD 智能问答 Agent

> 面向岗位 JD 解读、技术知识问答与技能分析场景的本地 RAG Agent 项目。系统使用 LangChain Agent 进行任务路由，结合 BM25、Chroma 向量检索、加权 RRF 融合与 BGE Reranker，实现可追溯的知识库问答；同时提供命令行入口和 FastAPI Web 服务。

## 项目概述

本项目用于处理岗位 JD 分析和技术知识问答两类任务：

* 对岗位 JD、简历或技术文本进行关键词提取与技能分析；
* 基于本地技术知识库回答 RAG、LangChain、Agent、检索模型等问题，并返回资料来源。

```text
用户问题
   │
   ├── Agent 路由
   │      ├── 计算工具
   │      ├── 本地文件读取
   │      ├── 技术关键词提取
   │      ├── 岗位技能差距分析
   │      └── RAG 检索问答
   │
   └── RAG 分支
          ├── BM25 关键词召回
          ├── Chroma 向量语义召回
          ├── 加权 RRF 融合与去重
          ├── BGE Reranker 重排序
          ├── 基于上下文的 LLM 回答
          └── 返回 answer 与 sources
```

## 核心特性

* Agent 工具路由；
* BM25 与 Chroma 混合检索；
* 加权 Reciprocal Rank Fusion 融合；
* `BAAI/bge-reranker-base` 本地重排序；
* 基于检索上下文的受限回答；
* 返回来源文档和内容片段；
* 命令行与 FastAPI Web 双入口；
* 支持 `txt`、`md`、`json`、`pdf` 文件读取。

## 技术栈

| 模块            | 技术                                                |
| ------------- | ------------------------------------------------- |
| 后端服务          | FastAPI、Uvicorn、Jinja2                            |
| Agent 编排      | LangChain `create_agent`、Tool Calling             |
| 模型与 Embedding | OpenAI-compatible API、ChatOpenAI、OpenAIEmbeddings |
| 向量数据库         | Chroma                                            |
| 关键词检索         | BM25、rank-bm25                                    |
| 文本切分          | RecursiveCharacterTextSplitter                    |
| 重排序           | PyTorch、Transformers、BAAI/bge-reranker-base       |
| 参数校验          | Pydantic                                          |
| 前端            | HTML、CSS、JavaScript                               |

## 快速开始

### 1. 创建环境

```bash
conda create -n job_jd_rag_py310 python=3.10 -y
conda activate job_jd_rag_py310
```

### 2. 安装依赖

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

如需读取 PDF，额外安装：

```bash
python -m pip install pypdf
```

### 3. 配置 `.env`

将项目中的 `.env.txt` 复制或重命名为 `.env`。

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://your-compatible-api-base-url/v1
OPENAI_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small

RERANK_TOP_K=3
RERANK_TIMEOUT=30
RERANK_MAX_DOCS=50
```

不要将真实 API Key 提交到 Git。

### 4. 构建向量索引

```bash
python rag/build_index.py
```

知识库文档默认位于：

```text
data/docs/
```

向量库默认保存至：

```text
db/chroma/
```

### 5. 启动项目

命令行模式：

```bash
python app.py
```

Web 服务模式：

```bash
python -m uvicorn main:app --reload
```

启动后访问：

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/health
```

## Agent 工具

| 工具                  | 功能                    | 示例                                       |
| ------------------- | --------------------- | ---------------------------------------- |
| `calculator`        | 基础数学计算                | `23 * 17 等于多少？`                          |
| `read_file`         | 读取 txt、md、json、pdf 文件 | `请读取 data/files/jd_agent_intern.txt 的内容` |
| `extract_keywords`  | 提取 JD 或简历中的技术关键词      | `提取这段 JD 的技术关键词`                         |
| `analyze_skill_gap` | 分析岗位技能与用户技能差距         | `岗位要求 Python、RAG；我会 Python，差哪些？`         |
| `rag_search`        | 基于知识库回答技术问题           | `BM25 和 BGE 的区别是什么？`                     |

## RAG 检索流程

### 两路召回

* BM25：适合关键词、技术名词、缩写和精确匹配；
* Chroma 向量检索：适合语义相近但表达不同的问题。

### RRF 融合

```text
score(doc) = BM25_weight / (rrf_k + rank_bm25)
           + Vector_weight / (rrf_k + rank_vector)
```

当前参数：

```text
BM25 weight   = 1.0
Vector weight = 1.5
rrf_k         = 60
```

### 重排序

融合候选文档后，系统使用：

```text
BAAI/bge-reranker-base
```

对 query-document 相关性进行重排序，得到最终 Top-K 文档。

### 无资料拒答

若知识库中没有可用于回答的问题，系统返回：

```text
资料中未提供相关信息。
```

并清空来源列表，避免返回无关文档。

## API 简要说明

### Agent 总控接口

```http
POST /agent/chat
```

请求：

```json
{
  "question": "RAG 的核心流程是什么？"
}
```

响应：

```json
{
  "answer": "...",
  "tool_used": "rag_search",
  "sources": [
    {
      "file": "data/docs/rag_retrieval_pipeline.md",
      "content": "..."
    }
  ]
}
```

### 独立 RAG 接口

```http
POST /rag/chat
```

```json
{
  "question": "BM25 和 BGE 的区别是什么？"
}
```

### JD 关键词提取接口

```http
POST /jd/extract
```

```json
{
  "text": "熟悉 Python、FastAPI、LangChain、RAG、向量数据库，有 Dify 或 Coze 经验优先。"
}
```

### JD 文本上传接口

```http
POST /jd/upload
```

仅支持上传 `.txt` 文件，上传文件保存至：

```text
data/uploads/
```

## 注意事项

1. 必须从项目根目录启动命令。
2. `.env.txt` 不会被 `load_dotenv()` 自动加载，实际配置文件必须命名为 `.env`。
3. 更换 Embedding 模型或知识库文档后，应删除旧的 `db/chroma/` 并重新构建索引。
4. 首次运行 Reranker 会下载模型文件。
5. PDF 文件读取依赖 `pypdf`。
6. 当前 `/jd/gap_analysis` 接口的请求参数与底层函数仍需统一，建议暂时通过 `/agent/chat` 调用技能差距分析能力。
7. 项目输出仅用于技术辅助分析，不应作为实际招聘决策依据。
