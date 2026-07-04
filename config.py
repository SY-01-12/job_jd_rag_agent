import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# =========================
# 项目根目录
# =========================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
UPLOAD_DIR = DATA_DIR / "uploads"

DB_DIR = BASE_DIR / "db"
CHROMA_DIR = DB_DIR / "chroma"


# =========================
# LLM 配置
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
RERANK_MODEL_NAME="BAAI/bge-reranker-base" #BGE本地模型

# =========================
# RAG 配置
# =========================

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "8"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "8"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))


# =========================
# 初始化目录
# =========================

def ensure_dirs():
    """
    确保必要目录存在。
    """
    for path in [DATA_DIR, DOCS_DIR, UPLOAD_DIR, DB_DIR, CHROMA_DIR]:
        path.mkdir(parents=True, exist_ok=True)


# =========================
# 配置检查
# =========================

def check_runtime_config() -> list[str]:
    """
    返回当前配置存在的问题，不直接抛异常，方便 health 接口展示。
    """
    warnings = []

    if not OPENAI_API_KEY:
        warnings.append("OPENAI_API_KEY 未配置，LLM 和 Embedding 相关功能可能无法使用。")

    if not DOCS_DIR.exists():
        warnings.append(f"文档目录不存在：{DOCS_DIR}")

    if not CHROMA_DIR.exists():
        warnings.append(f"Chroma 目录不存在：{CHROMA_DIR}，请先运行 rag/build_index.py 构建知识库。")

    return warnings


ensure_dirs()