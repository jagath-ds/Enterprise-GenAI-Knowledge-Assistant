# 🤖 Enterprise GenAI Knowledge Assistant

> A production-grade **Retrieval-Augmented Generation (RAG)** system that lets organizations query internal documents using natural language — with zero hallucination and enterprise-level security.

---

## 📌 Overview

The Enterprise GenAI Knowledge Assistant transforms unstructured enterprise documents into a conversational AI interface. Built on RAG architecture, it ensures every response is **grounded in your actual documents** — not model hallucinations.

---

## ❗ Problem Statement

Organizations struggle with knowledge retrieval because:

- Information is scattered across hundreds of documents
- Traditional keyword search lacks semantic understanding
- Public AI tools risk data leakage
- General-purpose LLMs hallucinate facts not present in source material

---

## 💡 Solution

This system implements a full RAG pipeline:

1. Ingests enterprise documents (PDF, DOCX, HTML, TXT)
2. Extracts and chunks text intelligently
3. Generates semantic embeddings
4. Stores and indexes them in a vector database
5. On query — retrieves the most relevant chunks
6. Feeds context to an LLM to generate a grounded answer

Responses are **strictly derived from your uploaded documents**.

---

## 🏗️ Architecture

```
Document Upload → Text Extraction → Chunking → Embedding Generation
                                                        ↓
                                              Vector Database (FAISS / ChromaDB)
                                                        ↓
User Query → Semantic Retrieval (Top-K) → LLM (Mistral / Phi3:mini) → Grounded Answer
```

---

## 🧠 Key Features

- ✅ Retrieval-Augmented Generation (RAG)
- ✅ Semantic search using dense embeddings
- ✅ Context-aware, source-grounded answers
- ✅ Hallucination control
- ✅ Chat-based UI
- ✅ Scalable and modular architecture
- ✅ Enterprise data stays local / private
- ✅ Runs fully offline using Ollama

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React.js, Tailwind CSS |
| Backend | Python, FastAPI |
| Embeddings | BGE |
| LLMs | Mistral, Phi3:mini (via Ollama) |
| Vector DB | FAISS, ChromaDB |

---

## 📂 Supported Document Types

- PDF (`.pdf`)
- Word Documents (`.docx`)
- HTML (`.html`)
- Plain Text (`.txt`)

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- [Ollama](https://ollama.com) — for running LLMs locally

---

### Step 1: Install Ollama & Pull Models

Download and install Ollama from [https://ollama.com/download](https://ollama.com/download), then pull the required models:

```bash
ollama pull mistral
ollama pull phi3:mini
```

> ⚠️ **Important:** You must start the Ollama server before running the project.

```bash
ollama serve
```

Keep this running in a terminal throughout your session.

---

### Step 2: Clone the Repository

```bash
git clone <repo url>
cd Enterprise-GenAI-Knowledge-Assistant
```

---

### Step 3: Backend Setup

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### Step 4: Frontend Setup

```bash
cd frontend
npm install
```

---

## ▶️ Running the Application

Make sure `ollama serve` is running first, then:

```bash
# Terminal 1 — Start the backend
cd backend
uvicorn app.main:app --reload

# Terminal 2 — Start the frontend
cd frontend
npm run dev
```

The app will be available at `http://localhost:5173` (or as shown in your terminal).  
Backend API docs are available at `http://localhost:8000/docs`.

---

## 🔄 Workflow

```
1. Upload documents via the UI
2. System extracts, chunks, and embeds the text
3. Embeddings are stored in the vector database
4. Ask a question in the chat interface
5. Relevant chunks are retrieved semantically
6. LLM generates an answer grounded in those chunks
```

---

## 🧪 Testing

- Retrieval accuracy testing
- Hallucination / faithfulness testing
- API testing (Swagger UI available at `/docs`)
- UI/UX validation
- Performance benchmarking

---

## 🔮 Roadmap

- [ ] Multi-language document support
- [ ] Advanced RBAC and access control
- [ ] Real-time document sync
- [ ] Hybrid search (keyword + semantic)
- [ ] Fine-tuned enterprise LLM integration

---

## 👥 Team

- Jagath Jyothis T S
- Kiran Krishna P
- Devadath Madhusoodanan
- Unnikrishnan Thriuvara Variyath
- T P Devanarayanan

---

## 📜 License

This project is for academic and enterprise prototype purposes.
