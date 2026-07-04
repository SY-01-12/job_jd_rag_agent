import json
from pydantic import Field,BaseModel
from langchain_core.tools import tool,StructuredTool

#定义Pydantic 数据模型 KeyWordInput
class KeyWordInput(BaseModel):
    text: str = Field(
        description='需要提取技术关键词的 JD / 简历 / 文本内容。例如：从Python、Java中提取Python'
    )

#关键词提取函数
def extract_keywords_func(text:str)->str:

    #创建关键词列表
    keys_list = ['Python','FastAPI','LangChain','LangGraph','LlamaIndex','Dify',
                 'Coze','RAG','BM25','BGE','FAISS','Chroma','Milvus','Reranker',
                 'Embedding','Prompt','Tool Calling','Function Calling','ReAct',
                 'OpenAI API','通义千问','DeepSeek','Docker','Linux','Git','Postman',
                 'Swagger','MySQL','Redis','向量数据库','大模型 API']

    #去除原始文本前后空白
    text = text.strip()

    res_list = []
    #遍历关键字列表
    for key in keys_list:
        if key in text:
            if key not in res_list:
                res_list.append(key)
            else:
                continue

    #返回json形式
    if res_list:
        result = {
            'keywords':res_list,
            'count':len(res_list)
        }
        return json.dumps(result,ensure_ascii=False)    #将包含关键词列表和数量的字典结果转换为 JSON 格式字符串返回。

    else:
        result = {
            'keywords':[],
            'count':0,
            "message":"未提取到明确技术关键词"
        }
        return json.dumps(result,ensure_ascii=False)

#创建StructureTool实例
extract_keywords = StructuredTool.from_function(
    func = extract_keywords_func,
    name = 'extract_keywords',
    description = (
        '用于从岗位 JD、简历文本或一段描述中提取技术关键词。'
        '适合提取 Python、FastAPI、LangChain、RAG、Chroma、Dify、Docker 等技能词。'
        '不用于回答技术概念问题。'
        '不用于做岗位能力差距分析。'
        '输入参数 text 是待分析文本。'
    ),
    args_schema=KeyWordInput
)
#测试
if __name__ == "__main__":
    print(extract_keywords.invoke('熟悉 Python、FastAPI、LangChain、RAG、向量数据库，有 Dify 或 Coze 经验优先。'))