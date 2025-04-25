# AI Medibot

AI Medibot is an intelligent medical assistant powered by Retrieval-Augmented Generation (RAG). It allows users to ask questions through a Streamlit-based UI, referencing a set of textbooks to generate accurate, explainable, and citation-backed answers using a local language model and vector database.

## 🧠 Key Features

- **Natural Language Q&A Interface**: Ask medical questions via a user-friendly Streamlit interface.
- **RAG Pipeline**: Utilizes local `ollama` with the DeepSeek R1 model for efficient and private inference.
- **Textbook-Centric Reasoning**: Answers include:
  - Model's reasoning
  - Referenced textbook name
  - Page number(s) of source
- **Fallback Handling**: Gracefully responds with _"I don't know"_ when no valid answer is found.
- **Local Embedding and Storage**: PDF textbooks are embedded using Nomic and stored in a local ChromaDB instance for fast retrieval.

---

## 📁 Project Structure

| File               | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `frontend.py`      | Streamlit UI for user input and displaying responses from the AI assistant. |
| `rag_pipeline.py`  | Implements the RAG pipeline with ollama + DeepSeek R1 and ChromaDB retrieval.|
| `vector_database.py` | Handles PDF chunking, embedding (Nomic), and vector storage in ChromaDB.    |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Streamlit
- `ollama` with DeepSeek R1 model installed
- `chromadb`
- `nomic` for embedding

### Installation

```bash
pip install -r requirements.txt
