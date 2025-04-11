import os
import shutil
import re
import tempfile
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredFileLoader,
    UnstructuredImageLoader,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool, BaseTool
from langchain import hub
import easyocr
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Q&A Agent API",
    description="API to upload documents, index them, and ask questions using a Gemini-powered agent.",
)

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.1, 
        google_api_key=google_api_key,
    )
    logger.info("ChatGoogleGenerativeAI initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}")
    raise

try:
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2" 
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
    logger.info(f"HuggingFaceEmbeddings initialized with model: {embedding_model_name}")
except Exception as e:
    logger.error(f"Failed to initialize HuggingFaceEmbeddings: {e}")
    raise

FAISS_INDEX_PATH = "faiss_index_gemini" 

try:
    if os.path.exists(FAISS_INDEX_PATH):
        logger.info(f"Loading existing FAISS index from {FAISS_INDEX_PATH}")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embedding,
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS index loaded successfully.")
    else:
        logger.info(f"No existing FAISS index found at {FAISS_INDEX_PATH}. Creating new index.")
        vectorstore = FAISS.from_texts(["dummy initialization text"], embedding)
        vectorstore.save_local(FAISS_INDEX_PATH) 
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embedding,
            allow_dangerous_deserialization=True
        )
        logger.info("New FAISS index created and saved.")
except Exception as e:
    logger.error(f"Failed to initialize or load FAISS vector store: {e}")
    raise

try:
    ocr_reader = easyocr.Reader(['en']) 
    logger.info("EasyOCR reader initialized.")
except Exception as e:
    logger.error(f"Failed to initialize EasyOCR: {e}")
    ocr_reader = None

def extract_text_from_image_easyocr(file_path: str) -> str:
    """Extracts text from an image using EasyOCR."""
    if ocr_reader:
        try:
            results = ocr_reader.readtext(file_path, detail=0, paragraph=True)
            return "\n".join(results)
        except Exception as e:
            logger.error(f"Error during EasyOCR processing for {file_path}: {e}")
            return ""
    else:
        logger.warning("EasyOCR not available, cannot extract text from image.")
        return ""

def redact_pii(text: str) -> str:
    """Redacts common PII patterns (email, phone, potential IDs)."""
    patterns = {
        r"[\w\.-]+@[\w\.-]+\.\w+": "[REDACTED EMAIL]",
        r"\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b": "[REDACTED PHONE]",
        r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b": "[REDACTED SSN/ID]", # Example SSN-like
        r"\b\d{12,19}\b": "[REDACTED NUMBER]", # Example: Credit Card, Aadhaar etc. - adjust length as needed
        # more patterns as required
    }
    redacted_text = text
    for pattern, replacement in patterns.items():
        redacted_text = re.sub(pattern, replacement, redacted_text)
    return redacted_text

# --- Document Processing Endpoint ---

@app.post("/upload", summary="Upload & Process Document")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a file (PDF, TXT, DOCX, JPG, PNG), extracts text,
    redacts PII, chunks the text, and adds it to the vector store.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name 
        logger.info(f"File '{file.filename}' saved temporarily to '{temp_path}'")

        extracted_text = ""
        file_ext = os.path.splitext(file.filename)[1].lower()

        try:
            if file_ext == ".pdf":
                loader = PyPDFLoader(temp_path)
            elif file_ext in ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'):
                # UnstructuredImageLoader uses OCR internally (tesseract by default)
                loader = UnstructuredImageLoader(temp_path, mode="single")
            else:
                loader = UnstructuredFileLoader(temp_path, mode="single")

            logger.info(f"Loading document with {type(loader).__name__}")
            docs = loader.load()
            extracted_text = "\n".join([d.page_content for d in docs])
            logger.info(f"Successfully extracted text from {file.filename}. Length: {len(extracted_text)}")

        except Exception as e:
            logger.error(f"Error loading/extracting text from {file.filename}: {e}")

            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(status_code=500, detail=f"Failed to process file {file.filename}: {e}")

        logger.info("Applying PII redaction...")
        redacted_text = redact_pii(extracted_text)
        text_splitter = CharacterTextSplitter(
            separator="\n\n", 
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        split_texts = text_splitter.split_text(redacted_text)
        logger.info(f"Split redacted text into {len(split_texts)} chunks.")

        if not split_texts:
             logger.warning(f"No text chunks generated for {file.filename}. File might be empty or processing failed.")
             if os.path.exists(temp_path):
                 os.remove(temp_path)
             return {"message": f"File {file.filename} processed, but no text chunks were generated to index."}

        logger.info("Adding text chunks to FAISS vector store...")
        vectorstore.add_texts(split_texts)

        logger.info(f"Saving updated FAISS index to {FAISS_INDEX_PATH}...")
        vectorstore.save_local(FAISS_INDEX_PATH)
        logger.info("FAISS index saved successfully.")

        return {"message": f"File '{file.filename}' processed, redacted, and indexed successfully. Chunks added: {len(split_texts)}."}

    except Exception as e:
        logger.exception(f"An unexpected error occurred during file upload processing for {file.filename}: {e}") # Log traceback
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Temporary file {temp_path} cleaned up.")
            except OSError as e:
                logger.error(f"Error deleting temporary file {temp_path}: {e}")

        await file.close()


# --- Agent Setup ---

class VectorSearchTool(BaseTool):
    name: str = "search_indexed_documents"
    description: str = (
    "Searches and retrieves relevant excerpts from the previously uploaded and indexed documents. "
    "Use this tool whenever the user asks a question, requests a summary, or needs information "
    "that might be contained within the indexed documents. Input should be the user's request or question."
)
    vector_store: FAISS

    def _run(self, query: str) -> str:
        """Synchronous execution for searching relevant documents."""
        logger.info(f"Executing VectorSearchTool with query: '{query}'")
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            results = retriever.get_relevant_documents(query)
            if not results:
                return "No relevant documents found in the knowledge base for this query."
            return "\n\n---\n\n".join([f"Source Document Chunk:\n{doc.page_content}" for doc in results])
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            return "Error occurred while searching the document knowledge base."

    async def _arun(self, query: str) -> str:
        """Asynchronous execution (optional, depends on agent/executor)."""
        logger.info(f"Executing VectorSearchTool (async wrapper) with query: '{query}'")
        return self._run(query)

vector_search_tool = VectorSearchTool(vector_store=vectorstore)

tools: List[BaseTool] = [vector_search_tool]

try:

    prompt = hub.pull("hwchase17/react")
    logger.info("Pulled react prompt from LangChain Hub.")
except Exception as e:
    logger.error(f"Could not pull prompt from LangChain Hub: {e}. Using a default fallback (if defined) or failing.")
    raise

try:
    agent = create_react_agent(llm, tools, prompt)
    logger.info("ReAct agent created successfully.")
except Exception as e:
    logger.error(f"Failed to create ReAct agent: {e}")
    raise

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors="Check your output and make sure it conforms!", 
    max_iterations=5 # Prevent potential infinite loops
)
logger.info("AgentExecutor created successfully.")



@app.get("/query", summary="Query the Agent")
async def query_agent(q: str):
    """
    Sends a query to the LangChain agent. The agent will use the
    VectorSearchTool to find relevant information in the indexed documents
    and generate an answer using the Gemini LLM.
    """
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' cannot be empty.")

    logger.info(f"Received query: '{q}'")
    try:
        response = await agent_executor.ainvoke({"input": q})
        answer = response.get("output", "No specific answer found.")
        logger.info(f"Agent generated answer: '{answer}'")
        return {"query": q, "answer": answer}
    except Exception as e:
        logger.exception(f"Error during agent execution for query '{q}': {e}") # Log full traceback
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {e}")

@app.get("/", summary="API Health Check")
async def read_root():
    """Basic health check endpoint."""
    return {"status": "API is running"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server with uvicorn...")
    # Reload=True is useful for development, disable for production
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)