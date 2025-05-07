
import os
import time
import logging
import requests
import nbformat
import ast
from typing import List, Tuple
from google.cloud import aiplatform
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.vectorstores import FAISS
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
# ===== Configuration =====
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'nse-gcp-ema-con-b372e-npd-1-fbfbb1d6bd61.json'
EMBEDDING_MODEL = "text-embedding-005"
LLM_MODEL = "gemini-1.5-pro-002"
# ===== Initialization =====
def init_vertex_ai():
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    return VertexAI(
        model_name=LLM_MODEL,
        max_output_tokens=2048,
        temperature=0.1,
        verbose=False,
    )

# ===== GitHub Crawl & Extract =====
IGNORE_LIST = ["__init__.py"]

# ===== Helper: AST-based chunking =====
def split_by_ast_functions(
    code: str,
    url: str,
    fallback_splitter: RecursiveCharacterTextSplitter = None
) -> List[Document]:
    """
    Parse code into an AST and emit each top-level function/class as its own Document.
    If fallback_splitter is provided, further split any oversized chunks.
    """
    docs: List[Document] = []
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Extract source segment for the node
            src = ast.get_source_segment(code, node)
            meta = {"url": url, "name": node.name, "lineno": node.lineno}
            docs.append(Document(page_content=src, metadata=meta))
    # If no functions or classes found, index entire file
    if not docs:
        docs.append(Document(page_content=code, metadata={"url": url}))
    # Fallback: split large chunks further
    if fallback_splitter:
        final: List[Document] = []
        for doc in docs:
            if len(doc.page_content) > fallback_splitter._chunk_size:
                # Use recursive splitter on this single document
                final.extend(fallback_splitter.split_documents([doc]))
            else:
                final.append(doc)
        return final
    return docs
def crawl_github_repo(url: str, token: str) -> List[str]:
    """
    Recursively crawl a GitHub repo and return list of .py and .ipynb file URLs.
    """
    api_base = f"https://api.github.com/repos/{url}/contents"
    headers = {"Authorization": f"Bearer {token}"}
    def _crawl(api_url: str) -> List[str]:
        res = requests.get(api_url, headers=headers)
        res.raise_for_status()
        items = res.json()
        files = []
        for item in items:
            if item["type"] == "file" and item["name"].endswith((".py", ".ipynb")) and item["name"] not in IGNORE_LIST:
                files.append(item["html_url"])
            elif item["type"] == "dir" and not item["name"].startswith('.'):
                files.extend(_crawl(item["url"]))
                time.sleep(0.1)
        return files
    return _crawl(api_base)

def extract_code(url: str) -> str:
    """
    Download raw content and extract code from .py or .ipynb
    """
    raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    text = requests.get(raw_url).text
    if url.endswith(".ipynb"):
        nb = nbformat.reads(text, as_version=nbformat.NO_CONVERT)
        code = []
        for cell in nb.cells:
            if cell.cell_type == "code":
                code.append(cell.source)
        return "\n".join(code)
    return text
# ===== Embeddings & Retriever =====
class RateLimiter:
    def __init__(self, qpm: int):
        self.period = 60.0 / qpm
        self.last = time.time()
    def wait(self):
        elapsed = time.time() - self.last
        if elapsed < self.period:
            time.sleep(self.period - elapsed)
        self.last = time.time()
class CustomEmbeddings(VertexAIEmbeddings):
    """Embeddings with rate limiting via extra attributes."""
    class Config:
        extra = "allow"
    def __init__(
        self,
        client,
        requests_per_minute: int,
        batch_size: int,
        model_name: str,
        **kwargs
    ):
        super().__init__(client=client, model_name=model_name, **kwargs)
        # Attach limiter and batch size dynamically
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.batch_size = batch_size
    def embed_documents(self, texts: List[str], batch_size: int | None = None) -> List[List[float]]:
        docs = texts.copy()
        results = []
        while docs:
            batch = docs[: self.batch_size]
            docs = docs[self.batch_size :]
            chunk = self.client.get_embeddings(batch)
            results.extend([r.values for r in chunk])
            self.rate_limiter.wait()
        return results
# ===== QA Pipeline with AST-based chunking =====
class GitHubQA:
    def __init__(self, repo: str):
        self.repo = repo
        self.llm = init_vertex_ai()
        self.embeddings = CustomEmbeddings(
            client=self.llm.client,
            requests_per_minute=100,
            batch_size=5,
            model_name=EMBEDDING_MODEL
        )
        self.prompt = PromptTemplate(
            template="""
            You are a proficient python developer. Use the context to answer.
            Question:
            {question}
            Context:
            {context}
            Answer with concise, correct code.
            """,
            input_variables=["context", "question"]
        )
        self._prepare()
    def _prepare(self):
        # 1) Crawl and extract raw code into Document objects
        files = crawl_github_repo(self.repo, GITHUB_TOKEN)
        docs = [Document(page_content=extract_code(url), metadata={"url": url, "index": idx})
                for idx, url in enumerate(files)]
        # 2) Set up a secondary splitter for oversized AST chunks
        secondary_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=2000,
            chunk_overlap=200
        )
        # 3) Apply AST-based splitting per file
        self.chunks: List[Document] = []
        for doc in docs:
            self.chunks.extend(
                split_by_ast_functions(
                    code=doc.page_content,
                    url=doc.metadata["url"],
                    fallback_splitter=secondary_splitter
                )
            )
        # 4) Build FAISS index with AST-aware chunks
        db = FAISS.from_documents(self.chunks, self.embeddings)
        self.retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        self.qa = RetrievalQA.from_llm(
            llm=self.llm,
            prompt=self.prompt,
            retriever=self.retriever,
            return_source_documents=True,
        )

    def list_chunks(self, snippet_chars: int = None):
        """
        Log each chunk's metadata and optionally the first `snippet_chars` characters of its content.
        If snippet_chars is None, logs full content.
        """
        log_file = "chunks.log"
        logger = logging.getLogger("ChunkLogger")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(handler)
        if not hasattr(self, "chunks") or not self.chunks:
            logger.info("[INFO] No chunks available. Did you run `_prepare()`?")
            return
        for i, chunk in enumerate(self.chunks):
            meta = chunk.metadata
            content = chunk.page_content.replace('\n', ' ')
            if snippet_chars:
                content = content[:snippet_chars]
            logger.info(f"--- Chunk {i} [{meta.get('name', 'module')} @ line {meta.get('lineno', '?')}] ({meta['url']})")
            logger.info(f"{content}...\n")


    def get_answer(self, question: str) -> Tuple[str, List[str]]:
        result = self.qa.invoke({"query": question})
        return (
            result.get("result"),
            [doc.metadata.get("url") for doc in result.get("source_documents", [])]
        )
    
# ===== Usage Example =====
# if __name__ == "__main__":
#     qa = GitHubQA(repo="rampal-punia/yolov8-streamlit-detection-tracking")
#     qa.list_chunks(snippet_chars=None)  # see your chunks
#     ans, srcs = qa.get_answer("What does this repo consist of?")
#     print("||||||||||||||||||||||||||||||||||||||||||||||||||||")
#     print(ans)
#     print(srcs)
