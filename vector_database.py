from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import os

# Config
PDF_DIR = r'C:/D/Other/AI PROJECTS/AI Hedge fund/medibot-ai/pdfs-test' #MENTION LOCAL PATH FOR PDF DIRECTORY
FAISS_DB_PATH = "vectorstore/db_faiss" #MENTION LOCAL PATH FOR VECTOR DB

# Step 0: Check if FAISS DB already exists
if os.path.exists(FAISS_DB_PATH):
    print(f"✅ FAISS database already exists at '{FAISS_DB_PATH}'")
    print("ℹ️  Skipping embedding and indexing. Delete the folder if you want to rebuild.")
    exit()

# Step 1: Load PDFs
def load_pdfs(directory):
    documents = []
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    print(f"📁 Found {len(pdf_files)} PDF files to process\n")

    for filename in pdf_files:
        file_path = os.path.join(directory, filename)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"📄 Processing: {filename} ({file_size_mb:.1f} MB)")

        try:
            pdf_loader = PyPDFLoader(file_path)
            file_docs = pdf_loader.load()

            # Add metadata to each document
            for doc in file_docs:
                doc.metadata["source"] = filename
                doc.metadata["file_path"] = file_path

            print(f"✓ Extracted {len(file_docs)} pages from {filename}\n")
            documents.extend(file_docs)
        except Exception as e:
            print(f"❌ Error processing {filename}: {str(e)}\n")

    return documents

# Step 2: Chunk Documents
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

# Step 3: Get Embedding Model
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={"trust_remote_code": True}
    )

# Step 4: Create and Save FAISS DB
def build_vector_db(chunks, embeddings):
    batch_size = 100
    total_chunks = len(chunks)
    faiss_db = None

    print("🚀 Starting FAISS indexing...\n")
    for i in tqdm(range(0, total_chunks, batch_size), desc="🔧 Indexing chunks"):
        batch = chunks[i:i + batch_size]

        if faiss_db is None:
            faiss_db = FAISS.from_documents(batch, embedding=embeddings)
        else:
            faiss_db.add_documents(batch)

    print("\n💾 Saving the FAISS database...")
    faiss_db.save_local(FAISS_DB_PATH)
    print(f"✅ FAISS database saved to '{FAISS_DB_PATH}'\n")

def main():
    documents = load_pdfs(PDF_DIR)
    print(f"📚 Total PDF pages loaded: {len(documents)}")

    chunks = create_chunks(documents)
    print(f"✂️ Total chunks created: {len(chunks)}")

    embeddings = get_embedding_model()
    build_vector_db(chunks, embeddings)

if __name__ == "__main__":
    main()