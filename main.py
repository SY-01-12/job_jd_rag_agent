import json

from fastapi import FastAPI, UploadFile, File, Path
from config import UPLOAD_DIR
from agent.agent_runner import  run_agent
from rag.rag_chain import rag_answer
from schemas import (ChatRequest, ChatResponse, RagResponse, RagRequest,
                     KeywordResponse, KeywordRequest,UploadResponse,
                     GapAnalysisResponse,GapAnalysisRequest)
from tools.gap_analysis_tool import analyze_skill_gap_func
from tools.keyword_tool import extract_keywords_func

from tools.keyword_tool import extract_keywords
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from fastapi.templating import Jinja2Templates
from config import check_runtime_config





app = FastAPI(
    title='Job JD RAG Agent',
    description='基于LangChain的岗位JD分析RAG Agent接口服务',
    version='1.0.0'
)


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def startup_event():
    warnings = check_runtime_config()
    if warnings:
        print("启动配置警告：")
        for warning in warnings:
            print(f"- {warning}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    warnings = check_runtime_config()
    return {
        "status": "ok" if not warnings else "warning",
        "warnings": warnings
    }

@app.post("/agent/chat",response_model=ChatResponse)
async def agent_chat(req:ChatRequest):
    result = run_agent(req.question)
    return result

@app.post("/rag/chat",response_model=RagResponse)
async def rag_chat(req:RagRequest):
    result = rag_answer(req.question)
    return result

@app.post('/jd/extract',response_model=KeywordResponse)
async def jd_extract(req:KeywordRequest):
    result = extract_keywords.invoke({'text':req.text})

    if isinstance(result,str):
        result = json.loads(result)

    return result

@app.post('/jd/upload',response_model=UploadResponse)
async def upload_jd(file: UploadFile = File(...)):  #file是必填参数
    print("收到上传请求:", file.filename)
    UPLOAD_DIR.mkdir(parents=True,exist_ok=True)    #如果不存在就创建，如果存在不报错，父目录不存在也一起创建

    if not file.filename.endswith('.txt'):  #判断文件是否以.txt结尾
        return {
            'filename':file.filename,
            'file_path':'',
            'size':0,
            'message':"仅支持上传.txt文件"
        }

    save_path = UPLOAD_DIR / file.filename

    content = await file.read()     #异步读取上传文件的二进制内容

    with open(save_path,'wb') as f :    #wb是写入二进制
        f.write(content)

    return {
        'filename':file.filename,
        'file_path':str(save_path),
        'size':len(content),
        'message':'上传成功'
    }


@app.post('/jd/gap_analysis',response_model = GapAnalysisResponse )
async def gap_skills(req:GapAnalysisRequest):
    # 1. 先从 JD 文本中提取技能关键词
    kw_result = extract_keywords_func(req.text_jd)
    kw_data = json.loads(kw_result)
    jd_skills = kw_data.get("keywords", [])

    if not jd_skills:
        return GapAnalysisResponse(
            matched_skills=[],
            missing_skills=req.user_skills,
            suggestion="未能从 JD 文本中提取到技术关键词，请检查输入内容。"
        )

    # 2. 直接调用函数（非 .invoke），传入正确的参数
    result = analyze_skill_gap_func(
        jd_skills=jd_skills,
        user_skills=req.user_skills
    )

    # 3. 字段映射：summary -> suggestion
    return GapAnalysisResponse(
        matched_skills=result.get("matched_skills", []),
        missing_skills=result.get("missing_skills", []),
        suggestion=result.get("summary", "")
    )

