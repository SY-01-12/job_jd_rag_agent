from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


class ReadFileInput(BaseModel):
    file_path: str = Field(..., description="要读取的文件路径，支持 txt、md、json、pdf")


ALLOWED_SUFFIXES = {".txt", ".md", ".json", ".pdf"}


def _read_pdf(path: Path) -> str:
    """
    读取 PDF 文本内容。
    """
    if PdfReader is None:
        return "读取失败：当前环境未安装 pypdf，请先执行 pip install pypdf"

    reader = PdfReader(str(path))
    texts = []

    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
            if page_text.strip():
                texts.append(f"\n--- Page {i + 1} ---\n{page_text}")
        except Exception:
            texts.append(f"\n--- Page {i + 1} ---\n该页解析失败")

    content = "\n".join(texts).strip()

    if not content:
        return "读取失败：PDF 未提取到有效文本，可能是扫描版 PDF"

    return content


def read_file_func(file_path: str) -> str:
    """
    读取本地文件内容。
    """
    try:
        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            return f"读取失败：文件不存在：{path}"

        if not path.is_file():
            return f"读取失败：路径不是文件：{path}"

        suffix = path.suffix.lower()

        if suffix not in ALLOWED_SUFFIXES:
            return f"读取失败：暂不支持 {suffix} 文件。当前支持：txt、md、json、pdf"

        if suffix == ".pdf":
            content = _read_pdf(path)
        else:
            content = path.read_text(encoding="utf-8", errors="ignore")

        max_chars = 5000
        if len(content) > max_chars:
            return content[:max_chars] + f"\n\n[内容过长，已截断，仅显示前 {max_chars} 字符]"

        return content

    except Exception as e:
        return f"读取失败：{str(e)}"


read_file_tool = StructuredTool.from_function(
    func=read_file_func,
    name="read_file",
    description="读取本地文件内容，支持 txt、md、json、pdf。",
    args_schema=ReadFileInput,
)