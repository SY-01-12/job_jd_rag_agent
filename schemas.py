from pydantic import BaseModel

class SourceItem(BaseModel):
    file: str
    content: str

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    tool_used: str
    sources: list[SourceItem]

class RagRequest(BaseModel):
    question: str

class RagResponse(BaseModel):
    answer: str
    sources: list[SourceItem]

class KeywordRequest(BaseModel):
    text: str

class KeywordResponse(BaseModel):
    keywords: list[str]
    count: int

class UploadResponse(BaseModel):
    filename: str
    file_path: str
    size: int
    message: str

class GapAnalysisRequest(BaseModel):
    text_jd: str
    user_skills: list[str]

class GapAnalysisResponse(BaseModel):
    matched_skills: list[str]
    missing_skills: list[str]
    suggestion: str
