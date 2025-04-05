from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import re 

# Load environment variables
load_dotenv()

# === Configuration ===
DB_PATH = "vectorstore/db_faiss"

# === Step 1: Load FAISS DB ===
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True}
)

if os.path.exists(DB_PATH):
    print("✅ FAISS DB found, loading...")
    faiss_db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    raise ValueError("❌ FAISS database not found! Run `create_vector_db.py` first.")

# === Step 2: Setup LLM (DeepSeek via Ollama) ===
llm_model = ChatOllama(model="deepseek-r1:1.5b")

# === Step 3: Retrieval Function ===
def retrieve_docs(query):
    return faiss_db.similarity_search(query)

def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

# === Step 4: Prompt Template ===
custom_prompt_template = """
Use only the information provided in the context below to answer the user's question.
If the answer is not in the context, say "I don't know."

Question: {question}
Context: {context}
Answer:
"""


def parse_response(response_text):
    """Split model response into <think> and actual answer"""
    think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
    think = think_match.group(1).strip() if think_match else ""
    answer = response_text.replace(think_match.group(0), "").strip() if think_match else response_text.strip()
    return think, answer

def answer_query(query):
    retrieved_docs = retrieve_docs(query)
    context = get_context(retrieved_docs)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | llm_model
    response = chain.invoke({"question": query, "context": context})

    # Parse the content
    think, answer = parse_response(response.content)
        # Collect sources
    sources = []
    for doc in retrieved_docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "n/a")
        sources.append(f"{src} (Page {page})")

    return think, answer, sources

# === Step 5: Run ===
if __name__ == "__main__":
    question = "tell me something about the author"
    think, answer, sources = answer_query(question)
    print("🧠 AI Answer:", answer)
    print("🧩 Model reasoning (think):", think)
    print("📚 Sources:")
    for src in sources:
        print("  -", src)