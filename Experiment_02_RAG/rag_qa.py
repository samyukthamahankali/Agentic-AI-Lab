import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENT_PATH = "data/knowledge.txt"
VECTOR_DATABASE = "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"


# ============================================================
# 1. LOAD DOCUMENT
# ============================================================

print("=" * 65)
print("          RAG-BASED QUESTION ANSWERING SYSTEM")
print("=" * 65)

print("\n[1] Loading document...")

loader = TextLoader(
    DOCUMENT_PATH,
    encoding="utf-8"
)

documents = loader.load()

print(f"Loaded {len(documents)} document(s).")


# ============================================================
# 2. SPLIT DOCUMENT INTO CHUNKS
# ============================================================

print("\n[2] Splitting document into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} text chunks.")


# ============================================================
# 3. CREATE EMBEDDINGS
# ============================================================

print("\n[3] Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ============================================================
# 4. CREATE VECTOR DATABASE
# ============================================================

print("\n[4] Creating vector database...")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=VECTOR_DATABASE
)

print("Vector database created successfully.")


# ============================================================
# 5. CREATE RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# ============================================================
# 6. INITIALIZE LOCAL LLM
# ============================================================

print("\n[5] Loading Llama 3.2...")

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0
)

print("LLM loaded.")


# ============================================================
# 7. ASK QUESTION
# ============================================================

question = input("\nEnter your question: ")


# ============================================================
# 8. RETRIEVE RELEVANT DOCUMENTS
# ============================================================

print("\n[6] Retrieving relevant information...")

retrieved_documents = retriever.invoke(question)

print(f"Retrieved {len(retrieved_documents)} relevant chunks.")


# ============================================================
# 9. CREATE CONTEXT
# ============================================================

context = "\n\n".join(
    document.page_content
    for document in retrieved_documents
)


# ============================================================
# 10. GENERATE ANSWER
# ============================================================

print("\n[7] Generating answer...")

prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I don't have enough information in the provided document."

Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
"""

response = llm.invoke(prompt)


# ============================================================
# 11. DISPLAY ANSWER
# ============================================================

print("\n" + "=" * 65)
print("ANSWER")
print("=" * 65)

print(response.content)


# ============================================================
# 12. DISPLAY SOURCES
# ============================================================

print("\n" + "=" * 65)
print("RETRIEVED SOURCES")
print("=" * 65)

for index, document in enumerate(retrieved_documents, start=1):

    print(f"\nSource {index}:")
    print("-" * 65)
    print(document.page_content[:300])


print("\n" + "=" * 65)
print("RAG EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 65)