import os
import time
import logging
import requests
import nbformat
import ast
from typing import List, Tuple
from google.cloud import aiplatform, bigquery
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.vectorstores import FAISS
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from google.api_core.exceptions import GoogleAPIError, NotFound # Import specific exceptions

# ===== Configuration =====
logging.info("Loading configuration...")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
BQ_DATASET = os.getenv("BQ_DATASET")
BQ_TABLE = os.getenv("BQ_TABLE")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'nse-gcp-ema-con-b372e-npd-1-fbfbb1d6bd61.json'
EMBEDDING_MODEL = "text-embedding-005"
LLM_MODEL = "gemini-1.5-pro-002"
logging.info(f"Configuration loaded: PROJECT_ID={PROJECT_ID}, LOCATION={LOCATION}, BQ_DATASET={BQ_DATASET}, BQ_TABLE={BQ_TABLE}")

# ===== Logging Setup =====
# Chunk-specific logger
tmp_log_file = "chunks.log"
chunk_logger = logging.getLogger("ChunkLogger")
chunk_logger.setLevel(logging.INFO)
# Use 'a' for append mode if you want to keep logs across runs, 'w' for overwrite
chunk_handler = logging.FileHandler(tmp_log_file, mode='a')
chunk_handler.setFormatter(logging.Formatter('%(message)s'))
# Prevent adding handlers multiple times if script is re-run in interactive session
if not chunk_logger.handlers:
    chunk_logger.addHandler(chunk_handler)

# General logger to file
logging.basicConfig(filename='logging.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.info("General logging setup complete.")

# ===== Initialization =====
def init_vertex_ai():
    """Initializes Vertex AI SDK and returns a VertexAI LLM instance."""
    logging.info(f"Initializing Vertex AI with project='{PROJECT_ID}' and location='{LOCATION}'...")
    try:
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        llm = VertexAI(
            model_name=LLM_MODEL,
            max_output_tokens=2048,
            temperature=0.1,
            verbose=False,
        )
        logging.info(f"Vertex AI initialized successfully. Using model: {LLM_MODEL}")
        return llm
    except GoogleAPIError as e:
        logging.error(f"Failed to initialize Vertex AI: {e}", exc_info=True)
        raise # Re-raise the exception to stop execution if initialization fails

# ===== GitHub Crawl & Extract =====
IGNORE_LIST = ["__init__.py"]

def crawl_github_repo(url: str, token: str) -> List[str]:
    """Crawls a GitHub repository and returns a list of Python and Jupyter file URLs."""
    logging.info(f"Starting GitHub crawl for repo: {url}")
    api_base = f"https://api.github.com/repos/{url}/contents"
    headers = {"Authorization": f"Bearer {token}"}
    files = []

    def _crawl(api_url: str):
        logging.debug(f"Crawling URL: {api_url}")
        try:
            res = requests.get(api_url, headers=headers)
            res.raise_for_status() # Raise an exception for bad status codes
            items = res.json()
            for item in items:
                if item["type"] == "file" and item["name"].endswith((".py", ".ipynb")) and item["name"] not in IGNORE_LIST:
                    logging.info(f"Found relevant file: {item['html_url']}")
                    files.append(item["html_url"])
                elif item["type"] == "dir" and not item["name"].startswith('.'):
                    logging.info(f"Entering directory: {item['path']}")
                    _crawl(item["url"])
                    time.sleep(0.1) # Respect API rate limits
        except requests.exceptions.RequestException as e:
            logging.error(f"Error during GitHub API crawl for {api_url}: {e}", exc_info=True)
        except Exception as e:
            logging.error(f"An unexpected error occurred during crawl for {api_url}: {e}", exc_info=True)


    _crawl(api_base)
    logging.info(f"Finished GitHub crawl. Found {len(files)} files.")
    return files

def extract_code(url: str) -> str:
    """Extracts code content from a given GitHub file URL."""
    logging.info(f"Extracting code from: {url}")
    raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    try:
        res = requests.get(raw_url)
        res.raise_for_status()
        text = res.text
        if url.endswith(".ipynb"):
            logging.info(f"Processing Jupyter notebook: {url}")
            nb = nbformat.reads(text, as_version=nbformat.NO_CONVERT)
            code = [cell.source for cell in nb.cells if cell.cell_type == "code"]
            extracted_code = "\n".join(code)
            logging.info(f"Extracted {len(code)} code cells from notebook.")
            return extracted_code
        else:
            logging.info(f"Extracted code from Python file: {url}")
            return text
    except requests.exceptions.RequestException as e:
        logging.error(f"Error extracting code from {url}: {e}", exc_info=True)
        return "" # Return empty string or handle error as appropriate
    except Exception as e:
        logging.error(f"An unexpected error occurred during code extraction from {url}: {e}", exc_info=True)
        return ""

# ===== Helper: AST-based chunking =====
def split_by_ast_functions(code: str, url: str, fallback_splitter: RecursiveCharacterTextSplitter = None) -> List[Document]:
    """Splits code into documents based on AST function/class definitions, with a fallback splitter."""
    logging.info(f"Attempting AST-based chunking for {url}...")
    docs: List[Document] = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree): # Use ast.walk to traverse all nodes
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                try:
                    src = ast.get_source_segment(code, node)
                    if src is not None:
                        doc = Document(page_content=src, metadata={"url": url, "name": getattr(node, 'name', 'anonymous'), "lineno": node.lineno})
                        # --- MODIFIED LINE BELOW ---
                        chunk_logger.info(
                            f"AST Chunk | File: {url} | Name: {doc.metadata['name']} | Line: {doc.metadata['lineno']} | Length: {len(doc.page_content)}\n"
                            f"Content:\n{doc.page_content}\n--- End Chunk ---\n"
                        )
                        # --- END MODIFIED LINE ---
                        docs.append(doc)
                except ValueError as e:
                    logging.warning(f"Could not get source segment for a node in {url}: {e}")
                except Exception as e:
                    logging.error(f"An unexpected error occurred during AST parsing for a node in {url}: {e}", exc_info=True)

        if not docs:
            logging.info(f"No function/class definitions found in {url}. Adding entire code as one document.")
            full_doc = Document(page_content=code, metadata={"url": url})
            chunk_logger.info(f"Full File Chunk | File: {url} | Length: {len(full_doc.page_content)}")
            docs.append(full_doc)

        if fallback_splitter:
            logging.info(f"Applying fallback splitter for large chunks in {url}...")
            out = []
            for i, d in enumerate(docs):
                if len(d.page_content) > fallback_splitter._chunk_size:
                    logging.info(f"Chunk {i+1} from {url} is large ({len(d.page_content)} chars). Splitting...")
                    split_docs = fallback_splitter.split_documents([d])
                    logging.info(f"Split into {len(split_docs)} smaller chunks.")
                    for j, split_doc in enumerate(split_docs):
                        chunk_logger.info(f"Fallback Chunk {i+1}-{j+1} | File: {url} | Length: {len(split_doc.page_content)}")
                    # Corrected indentation here:
                    out.extend(split_docs)
                else:
                    out.append(d)
            logging.info(f"Fallback splitting complete for {url}. Total chunks: {len(out)}")
            return out # This return is inside the try block now
        else:
            # If no fallback splitter, return the AST-generated docs (or full file doc)
            return docs # This return is inside the try block now

    # Corrected indentation for these except blocks:
    except SyntaxError as e:
        logging.error(f"Syntax error parsing file {url} for AST chunking: {e}", exc_info=True)
        # If AST parsing fails, fall back to simple recursive splitting if available
        if fallback_splitter:
            logging.info(f"AST parsing failed for {url}, falling back to recursive splitting.")
            fallback_docs = fallback_splitter.split_documents([Document(page_content=code, metadata={"url": url})])
            for i, fb_doc in enumerate(fallback_docs):
                 # Corrected indentation here:
                 chunk_logger.info(f"Syntax Error Fallback Chunk {i+1} | File: {url} | Length: {len(fb_doc.page_content)}")
            return fallback_docs
        else:
            logging.error(f"AST parsing failed for {url} and no fallback splitter provided.", exc_info=True)
            return [] # Return empty list if cannot parse or fallback
    except Exception as e:
        logging.error(f"An unexpected error occurred during AST chunking for {url}: {e}", exc_info=True)
        # Also fall back if any other unexpected error occurs during AST processing
        if fallback_splitter:
            logging.info(f"Unexpected AST error for {url}, falling back to recursive splitting.")
            fallback_docs = fallback_splitter.split_documents([Document(page_content=code, metadata={"url": url})])
            for i, fb_doc in enumerate(fallback_docs):
                 # Corrected indentation here:
                 chunk_logger.info(f"Error Fallback Chunk {i+1} | File: {url} | Length: {len(fb_doc.page_content)}")
            return fallback_docs
        else:
            logging.error(f"Unexpected AST error for {url} and no fallback splitter provided.", exc_info=True)
            return []

    # If we reach here, it means AST parsing succeeded but no fallback was needed/provided,
    # and no errors occurred. So, return the docs generated by AST.
    # This return statement needs to be outside the try/except block for the case
    # where try succeeds and fallback_splitter is None.
    return docs

# ===== Rate-limited Embeddings =====
class RateLimiter:
    """Simple rate limiter."""
    def __init__(self, qpm: int):
        self.period = 60.0 / qpm
        self.last = time.time()
        logging.info(f"Rate limiter initialized with {qpm} queries per minute (period={self.period:.2f}s).")

    def wait(self):
        """Waits if necessary to respect the rate limit."""
        elapsed = time.time() - self.last
        if elapsed < self.period:
            sleep_duration = self.period - elapsed
            logging.debug(f"Rate limiting: Sleeping for {sleep_duration:.2f}s")
            time.sleep(sleep_duration)
        self.last = time.time()

class CustomEmbeddings(VertexAIEmbeddings):
    """Custom VertexAIEmbeddings with built-in rate limiting and batching."""
    class Config:
        extra = "allow"

    def __init__(self, client, requests_per_minute: int, batch_size: int, model_name: str, **kwargs):
        logging.info(f"Initializing CustomEmbeddings with model='{model_name}', batch_size={batch_size}, qpm={requests_per_minute}.")
        super().__init__(client=client, model_name=model_name, **kwargs)
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str], batch_size: int | None = None) -> List[List[float]]:
        """Embeds a list of documents in batches with rate limiting."""
        effective_batch_size = batch_size if batch_size is not None else self.batch_size
        logging.info(f"Embedding {len(texts)} documents in batches of {effective_batch_size}...")
        docs = texts.copy()
        results = []
        total_embedded = 0
        batch_count = 0
        while docs:
            batch_count += 1
            batch = docs[:effective_batch_size]
            docs = docs[effective_batch_size:]
            logging.debug(f"Processing embedding batch {batch_count} with {len(batch)} documents.")
            try:
                chunk = self.client.get_embeddings(batch)
                results.extend([r.values for r in chunk])
                total_embedded += len(batch)
                logging.debug(f"Successfully embedded batch {batch_count}. Total embedded: {total_embedded}")
                self.rate_limiter.wait()
            except GoogleAPIError as e:
                logging.error(f"Google API error during embedding batch {batch_count}: {e}", exc_info=True)
                # Depending on the error, you might want to retry or break
                raise # Re-raise for now
            except Exception as e:
                 logging.error(f"An unexpected error occurred during embedding batch {batch_count}: {e}", exc_info=True)
                 raise # Re-raise for now

        logging.info(f"Finished embedding {total_embedded} documents.")
        return results

# ===== QA Pipeline with FAISS & BigQuery storage =====
class GitHubQA:
    """QA system for a GitHub repository using LangChain, FAISS, and BigQuery."""
    def __init__(self, repo: str):
        self.repo = repo
        logging.info(f"Initializing GitHubQA for repository: {self.repo}")
        self.llm = init_vertex_ai()
        self.embeddings = CustomEmbeddings(
            client=self.llm.client,
            requests_per_minute=100, # Example rate limit
            batch_size=5, # Example batch size
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
        logging.info("Prompt template defined.")
        self._prepare()
        logging.info("GitHubQA initialization complete.")

    def _prepare(self):
        """Prepares the data by crawling, extracting, chunking, embedding, and indexing."""
        logging.info(f"Starting data preparation for repo: {self.repo}")
        files = crawl_github_repo(self.repo, GITHUB_TOKEN)
        docs = [Document(page_content=extract_code(url), metadata={"url": url, "index": idx})
                for idx, url in enumerate(files) if extract_code(url)] # Only process if extraction was successful and returned content

        splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=2000, chunk_overlap=200)
        logging.info(f"Initialized text splitter (lang=Python, chunk_size=2000, chunk_overlap=200).")
        self.chunks = []
        logging.info(f"Starting chunking of {len(docs)} documents...")
        for d in docs:
            self.chunks.extend(split_by_ast_functions(d.page_content, d.metadata["url"], fallback_splitter=splitter))
        logging.info(f"Finished chunking. Total chunks created: {len(self.chunks)}")

        # Build FAISS index for retrieval
        logging.info("Building FAISS index from chunks...")
        if not self.chunks:
            logging.warning("No chunks available to build FAISS index.")
            self.retriever = None # Or handle accordingly
        else:
            try:
                db = FAISS.from_documents(self.chunks, self.embeddings)
                self.retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
                logging.info(f"FAISS index built successfully. Retriever configured with k=5.")
            except Exception as e:
                logging.error(f"Error building FAISS index: {e}", exc_info=True)
                self.retriever = None # Ensure retriever is None if building fails
                # Optionally re-raise or continue depending on desired behavior


        # Optional: store chunks & embeddings in BigQuery for archival
        logging.info(f"Attempting to store chunks and embeddings in BigQuery table: {PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}")
        if not self.chunks:
             logging.info("No chunks to store in BigQuery.")
        else:
            try:
                client = bigquery.Client(project=PROJECT_ID)
                table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
                rows = []
                # Pre-calculate embeddings for BigQuery storage to avoid re-embedding
                logging.info("Generating embeddings for BigQuery storage...")
                chunk_texts = [chunk.page_content for chunk in self.chunks]
                chunk_embeddings = self.embeddings.embed_documents(chunk_texts)
                logging.info(f"Generated embeddings for {len(chunk_embeddings)} chunks.")

                for idx, chunk in enumerate(self.chunks):
                    # Use the pre-calculated embedding
                    embedding = chunk_embeddings[idx] if idx < len(chunk_embeddings) else None # Should always exist if chunk_embeddings size matches chunks
                    if embedding is None:
                        logging.warning(f"Embedding not found for chunk index {idx}. Skipping BigQuery insert for this chunk.")
                        continue # Skip this row if embedding is missing

                    rows.append({
                        "chunk_content": chunk.page_content,
                        "embeddings": embedding,
                        "url": chunk.metadata.get("url"),
                        "name": chunk.metadata.get("name"),
                        "lineno": chunk.metadata.get("lineno"),
                         # Add other metadata fields if needed
                    })
                if rows:
                    logging.info(f"Inserting {len(rows)} rows into BigQuery table {table_id}...")
                    errors = client.insert_rows_json(table_id, rows)
                    if errors:
                        logging.error(f"Errors occurred during BigQuery insertion: {errors}")
                    else:
                        logging.info("BigQuery insertion successful.")
                else:
                    logging.info("No rows to insert into BigQuery.")

            except NotFound as e:
                logging.error(f"BigQuery table or dataset not found: {e}", exc_info=True)
            except GoogleAPIError as e:
                logging.error(f"Google API error during BigQuery insertion: {e}", exc_info=True)
            except Exception as e:
                logging.error(f"An unexpected error occurred during BigQuery insertion: {e}", exc_info=True)


        # Initialize QA chain
        if self.retriever:
            logging.info("Initializing RetrievalQA chain...")
            self.qa = RetrievalQA.from_llm(llm=self.llm, prompt=self.prompt, retriever=self.retriever, return_source_documents=True)
            logging.info("RetrievalQA chain initialized.")
        else:
            logging.warning("Retriever not available. Cannot initialize RetrievalQA chain.")
            self.qa = None # Ensure qa is None if retriever is not available


    def get_answer(self, question: str) -> Tuple[str, List[str]]:
        """Gets an answer to a question using the QA chain."""
        logging.info(f"Received question: {question}")
        if not self.qa:
            logging.error("QA chain is not initialized. Cannot answer question.")
            return "Error: QA system not ready.", []

        try:
            result = self.qa.invoke({"query": question})
            answer = result.get("result")
            source_docs = result.get("source_documents", [])
            sources = [doc.metadata.get("url") for doc in source_docs]

            logging.info("Answer generated.")
            logging.info(f"Sources retrieved: {sources}")
            return answer, sources
        except Exception as e:
            logging.error(f"An error occurred while getting an answer for question '{question}': {e}", exc_info=True)
            return f"Error: Failed to get an answer. {e}", []


if __name__ == "__main__":
    logging.info("Starting main execution block.")
    repo_to_process = "github/codespaces-jupyter"
    logging.info(f"Processing repository: {repo_to_process}")

    if not GITHUB_TOKEN:
        logging.error("GITHUB_TOKEN environment variable not set. Cannot proceed.")
    else:
        try:
            qa = GitHubQA(repo=repo_to_process)
            if qa.qa: # Check if QA chain was successfully initialized
                question = "What does this repo consist of?"
                logging.info(f"Asking question: {question}")
                ans, srcs = qa.get_answer(question)
                print("\n--- Answer ---")
                print(ans)
                print("\n--- Sources ---")
                print(srcs)
                logging.info("Question answered and results printed.")
            else:
                logging.error("QA system failed to initialize. Skipping question.")

        except Exception as e:
            logging.critical(f"A critical error occurred during main execution: {e}", exc_info=True)

    logging.info("Main execution block finished.")