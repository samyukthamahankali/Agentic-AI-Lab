"""
======================================================================
EXPERIMENT 12 - MINI PROJECT
RAG COMPONENT
======================================================================

Project      : CyberShield AI
Course       : Applied Agentic AI
University   : Malla Reddy University
Domain       : Cybersecurity

Component:
Retrieval-Augmented Generation (RAG)

Workflow:
Documents
    ↓
Text Loading
    ↓
Text Splitting
    ↓
Embeddings
    ↓
Chroma Vector Database
    ↓
Similarity Retrieval
    ↓
Relevant Cybersecurity Context
======================================================================
"""

import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ======================================================================
# CONFIGURATION
# ======================================================================

KNOWLEDGE_BASE_DIR = "./knowledge_base"

VECTOR_DB_DIR = "./data/chroma_db"

COLLECTION_NAME = "cybershield_knowledge"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ======================================================================
# LOAD DOCUMENTS
# ======================================================================

def load_documents():

    documents = []

    print("\n" + "=" * 70)
    print("STEP 1 - LOADING CYBERSECURITY KNOWLEDGE BASE")
    print("=" * 70)

    for filename in os.listdir(KNOWLEDGE_BASE_DIR):

        if filename.endswith(".txt"):

            file_path = os.path.join(
                KNOWLEDGE_BASE_DIR,
                filename
            )

            print(f"\nLoading: {filename}")

            loader = TextLoader(
                file_path,
                encoding="utf-8"
            )

            file_documents = loader.load()

            documents.extend(file_documents)

    print(
        f"\nTotal documents loaded: "
        f"{len(documents)}"
    )

    return documents


# ======================================================================
# SPLIT DOCUMENTS
# ======================================================================

def split_documents(documents):

    print("\n" + "=" * 70)
    print("STEP 2 - SPLITTING DOCUMENTS")
    print("=" * 70)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"\nTotal chunks created: "
        f"{len(chunks)}"
    )

    return chunks


# ======================================================================
# CREATE EMBEDDINGS
# ======================================================================

def create_embeddings():

    print("\n" + "=" * 70)
    print("STEP 3 - CREATING EMBEDDINGS")
    print("=" * 70)

    print(
        f"\nEmbedding model: "
        f"{EMBEDDING_MODEL}"
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print(
        "\nEmbeddings initialized successfully."
    )

    return embeddings


# ======================================================================
# CREATE VECTOR DATABASE
# ======================================================================

def create_vector_database(
    chunks,
    embeddings
):

    print("\n" + "=" * 70)
    print("STEP 4 - CREATING CHROMA VECTOR DATABASE")
    print("=" * 70)

    os.makedirs(
        VECTOR_DB_DIR,
        exist_ok=True
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_DB_DIR
    )

    print(
        "\nChroma vector database created successfully."
    )

    print(
        f"Vector database location:\n"
        f"{VECTOR_DB_DIR}"
    )

    return vector_store


# ======================================================================
# RETRIEVAL TEST
# ======================================================================

def test_retrieval(vector_store):

    print("\n" + "=" * 70)
    print("STEP 5 - TESTING RETRIEVAL")
    print("=" * 70)

    test_queries = [

        "What are common phishing indicators?",

        "What should be done when ransomware is detected?",

        "What is the purpose of a firewall?",

        "What are the stages of incident response?"
    ]

    for query in test_queries:

        print("\n" + "-" * 70)

        print(
            f"QUERY:\n{query}"
        )

        results = vector_store.similarity_search(
            query,
            k=2
        )

        print(
            "\nRETRIEVED DOCUMENTS:"
        )

        for index, document in enumerate(
            results,
            start=1
        ):

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            print(
                f"\n[{index}] Source: "
                f"{os.path.basename(source)}"
            )

            print(
                document.page_content[:400]
            )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("\n")
    print("=" * 70)
    print("       CYBERSHIELD AI - RAG COMPONENT")
    print("=" * 70)

    print(
        "\nProject: Cybersecurity Multi-Agent AI"
    )

    print(
        "Component: Retrieval-Augmented Generation"
    )

    # Step 1
    documents = load_documents()

    if not documents:

        print(
            "\nERROR: No knowledge-base documents found."
        )

        return

    # Step 2
    chunks = split_documents(
        documents
    )

    # Step 3
    embeddings = create_embeddings()

    # Step 4
    vector_store = create_vector_database(
        chunks,
        embeddings
    )

    # Step 5
    test_retrieval(
        vector_store
    )

    print("\n")
    print("=" * 70)
    print("RAG COMPONENT COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        "\nKnowledge Base:"
        f" {len(documents)} documents"
    )

    print(
        "Text Chunks:"
        f" {len(chunks)}"
    )

    print(
        f"Vector Database:"
        f" {VECTOR_DB_DIR}"
    )

    print(
        "\nStatus: SUCCESS"
    )

    print("\n" + "=" * 70)


# ======================================================================
# PROGRAM ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()