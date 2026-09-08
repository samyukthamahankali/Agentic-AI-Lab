import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

PDF_FILE = "document.pdf"
VECTOR_DB_DIR = "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"


# ============================================================
# 1. LOAD PDF
# ============================================================

def load_pdf():

    if not os.path.exists(PDF_FILE):
        print(f"\nERROR: {PDF_FILE} not found.")
        print("Place your PDF in this folder and rename it to document.pdf")
        return None

    print("\n[1] Loading PDF...")

    loader = PyPDFLoader(PDF_FILE)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages.")

    return documents


# ============================================================
# 2. SPLIT DOCUMENT
# ============================================================

def split_documents(documents):

    print("\n[2] Splitting document into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    return chunks


# ============================================================
# 3. CREATE VECTOR DATABASE
# ============================================================

def create_vector_database(chunks):

    print("\n[3] Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("[4] Creating vector database...")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )

    print("Vector database created successfully.")

    return vector_db


# ============================================================
# 4. CREATE LLM
# ============================================================

def create_llm():

    print("\n[5] Loading Llama 3.2...")

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0
    )

    return llm


# ============================================================
# 5. DOCUMENT QUESTION ANSWERING
# ============================================================

def ask_question(vector_db, llm, question):

    print("\n[6] Retrieving relevant information...")

    retrieved_docs = vector_db.similarity_search(
        question,
        k=4
    )

    # Combine retrieved content
    context_parts = []

    for doc in retrieved_docs:

        page_number = doc.metadata.get("page", 0) + 1

        context_parts.append(
            f"[Page {page_number}]\n{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    print("[7] Generating answer...")

    prompt = f"""
You are a reliable Document Question Answering Agent.

Your job is to answer questions using ONLY the information
contained in the provided document context.

IMPORTANT RULES:

1. Use the document context to answer the question.
2. For broad questions such as:
   - What is the main topic?
   - What are the key points?
   - Summarize the document.
   analyze all relevant information in the retrieved context.
3. Do NOT invent facts that are not supported by the document.
4. If the document does not contain the answer, say:
   "I could not find this information in the document."
5. Keep the answer clear and concise.
6. Do not mention information from outside the document.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content, retrieved_docs


# ============================================================
# 6. DISPLAY SOURCES
# ============================================================

def display_sources(retrieved_docs):

    pages = []

    for doc in retrieved_docs:

        page = doc.metadata.get("page", 0) + 1

        if page not in pages:
            pages.append(page)

    pages.sort()

    print("\nSOURCES:")
    print("-" * 60)

    for page in pages:
        print(f"Page {page}")

    print("-" * 60)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print("ASSIGNMENT 2.1 - DOCUMENT Q&A AGENT")
    print("=" * 60)

    # Load document
    documents = load_pdf()

    if documents is None:
        return

    # Split document
    chunks = split_documents(documents)

    # Create vector database
    vector_db = create_vector_database(chunks)

    # Create LLM
    llm = create_llm()

    print("\n" + "=" * 60)
    print("DOCUMENT Q&A READY")
    print("Type 'exit' to stop.")
    print("=" * 60)

    while True:

        question = input("\nAsk a question: ")

        if question.lower().strip() == "exit":

            print("\nExiting Document Q&A Agent.")
            break

        if not question.strip():
            continue

        answer, retrieved_docs = ask_question(
            vector_db,
            llm,
            question
        )

        print("\nANSWER:")
        print("-" * 60)
        print(answer)
        print("-" * 60)

        display_sources(retrieved_docs)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()