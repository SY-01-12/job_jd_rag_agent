import glob
import os.path
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DOCS_DIR, EMBEDDING_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, CHROMA_DIR

#文档加载函数
def load_markdown(docs_path):
    files = []
    #判断路径是否存在
    if os.path.exists(docs_path):
        for file_path in glob.glob(os.path.join(docs_path, "*.md")):
            #创建文件加载器实例
            loader = TextLoader(
                file_path,
                encoding='utf-8'
            )
            #创建文件加载对象
            documents = loader.load()
            #将所有文档平铺添加到 files
            files.extend(documents)

    return files

#向量索引函数
def build_vector_index(
        docs_path = DOCS_DIR,
        db_path = CHROMA_DIR,
        chunk_size = 500,
        chunk_overlap = 80
):
    #获取文件对象
    documents = load_markdown(docs_path)

    print(f"加载了{len(documents)}个文档")

    if not documents:
        raise ValueError("未加载到Markdown文档")

    #切分文档实例
    documents_split = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    #切分文档对象
    chunks = documents_split.split_documents(documents)

    print(f"切分为{len(chunks)}个文本块")

    #获取嵌入模型
    emb_model = OpenAIEmbeddings(
        model = EMBEDDING_MODEL,
        api_key = OPENAI_API_KEY,
        base_url = OPENAI_BASE_URL,
        check_embedding_ctx_length=False,
        tiktoken_enabled = False,
        chunk_size=10
    )

    #写入Chroma向量库
    db = Chroma.from_documents(
        documents=chunks,
        embedding=emb_model,
        persist_directory=str(db_path)
    )

    print(f"向量数据库已保存到：{db_path}")

    return db

#进行测试
if __name__ == '__main__':
    build_vector_index()