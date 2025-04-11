# Agentic-rag
---

# 🧠 Gemini-Powered Document Q&A API

A FastAPI-based system to **upload, index, and query documents** using Google Gemini LLM, LangChain agents, and FAISS vector search. Includes support for **OCR, PII redaction**, and **multi-format document handling** (PDFs, images, etc.).

---

## 🚀 Features

- 📄 Upload and process documents (PDF, TXT, DOCX, PNG, JPG, etc.)
- 🔍 Index content with FAISS and HuggingFace embeddings
- 🧠 Query documents using a Gemini-powered ReAct Agent
- 🧾 Automatic **OCR** (via EasyOCR and LangChain Unstructured)
- ❌ **PII redaction** (emails, phone numbers, SSNs, etc.)
- 🧰 Tool-enhanced LangChain Agent with custom `VectorSearchTool`
- ⚙️ Production-ready FastAPI server

---

## 📦 Tech Stack

- **FastAPI** – API Framework
- **LangChain** – Agents, Embeddings, Tools, Loaders
- **Gemini (Google Generative AI)** – LLM backend
- **FAISS** – Vector similarity search
- **HuggingFace Transformers** – Sentence embeddings
- **EasyOCR** – OCR for image text extraction
- **dotenv + logging** – Environment and debugging support

---

```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

---

## ▶️ Running the Server

```bash
uvicorn main:app --reload
```

Visit the docs at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📂 API Endpoints

### `/upload` (POST)
Upload a document. Supported formats: `.pdf`, `.txt`, `.docx`, `.jpg`, `.png`, etc.  
- Redacts PII  
- Chunks and indexes text into FAISS

### `/query?q=your+question` (GET)
Ask a question and get an intelligent answer based on your uploaded documents.

### `/` (GET)
Health check.

---

## 🤖 Example Use Case

> Upload patient health records, legal documents, research papers, or invoices. Then ask:
>
> *“Summarize key findings from the research”*  
> *“What is the total invoice amount?”*  
> *“List medications mentioned in the report”*

---

## 📌 TODO

- [ ] Add authentication for secure endpoints
- [ ] Dockerize the app
- [ ] UI frontend for document upload + query

