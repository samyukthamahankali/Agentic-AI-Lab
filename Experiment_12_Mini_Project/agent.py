"""
======================================================================
EXPERIMENT 12 - MINI PROJECT
CYBERSHIELD AI - SECURITY ANALYSIS AGENT
======================================================================

Course      : Applied Agentic AI
University  : Malla Reddy University
Domain      : Cybersecurity

Architecture:

User Query
     |
     v
Cybersecurity Agent
     |
     +----------------------+
     |                      |
     v                      v
    RAG                   Tools
     |                      |
     v                      v
 Chroma DB          Security Analysis
     |                      |
     +----------+-----------+
                |
                v
            Llama 3.2
                |
                v
       Final Security Report

======================================================================
"""

import json
import re

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from rag import (
    create_embeddings,
    load_documents,
    split_documents,
    create_vector_database
)

from tools import (
    analyze_phishing,
    analyze_password,
    analyze_ip_address,
    classify_severity
)


# ======================================================================
# CONFIGURATION
# ======================================================================

MODEL_NAME = "llama3.2:3b"


# ======================================================================
# INITIALIZE LLM
# ======================================================================

def create_llm():

    print("\n")
    print("=" * 70)
    print("INITIALIZING LLM")
    print("=" * 70)

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )

    print(
        f"\nModel: {MODEL_NAME}"
    )

    print(
        "LLM initialized successfully."
    )

    return llm


# ======================================================================
# INITIALIZE RAG
# ======================================================================

def initialize_rag():

    print("\n")
    print("=" * 70)
    print("INITIALIZING RAG")
    print("=" * 70)

    documents = load_documents()

    chunks = split_documents(
        documents
    )

    embeddings = create_embeddings()

    vector_store = create_vector_database(
        chunks,
        embeddings
    )

    print(
        "\nRAG initialized successfully."
    )

    return vector_store


# ======================================================================
# RETRIEVE CONTEXT
# ======================================================================

def retrieve_context(
    vector_store,
    query
):

    results = vector_store.similarity_search(
        query,
        k=3
    )

    context_parts = []

    for document in results:

        context_parts.append(
            document.page_content
        )

    return "\n\n".join(
        context_parts
    )


# ======================================================================
# TOOL SELECTION
# ======================================================================

def select_tool(query):

    query_lower = query.lower()

    # --------------------------------------------------------------
    # Phishing
    # --------------------------------------------------------------

    if any(
        keyword in query_lower
        for keyword in [
            "phishing",
            "email",
            "suspicious message",
            "suspicious link",
            "verify my account",
            "verify your account",
            "password request"
        ]
    ):

        return "phishing"

    # --------------------------------------------------------------
    # Password
    # --------------------------------------------------------------

    if any(
        keyword in query_lower
        for keyword in [
            "password strength",
            "password score",
            "password security",
            "passcode"
        ]
    ):

        return "password"

    # --------------------------------------------------------------
    # IP
    # --------------------------------------------------------------

    if (
        re.search(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            query
        )
        or
        any(
            keyword in query_lower
            for keyword in [
                "ip address",
                "ip classification",
                "network address"
            ]
        )
    ):

        return "ip"

    # --------------------------------------------------------------
    # Severity
    # --------------------------------------------------------------

    if any(
        keyword in query_lower
        for keyword in [
            "severity",
            "ransomware",
            "malware",
            "incident",
            "trojan",
            "data breach",
            "account takeover"
        ]
    ):

        return "severity"

    return "none"


# ======================================================================
# EXTRACT IP ADDRESS
# ======================================================================

def extract_ip(query):

    match = re.search(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        query
    )

    if match:

        return match.group(0)

    return None


# ======================================================================
# RUN SECURITY TOOL
# ======================================================================

def run_tool(
    tool_name,
    query
):

    # --------------------------------------------------------------
    # Phishing
    # --------------------------------------------------------------

    if tool_name == "phishing":

        return analyze_phishing(
            query
        )

    # --------------------------------------------------------------
    # Password
    # --------------------------------------------------------------

    if tool_name == "password":

        return analyze_password(
            query
        )

    # --------------------------------------------------------------
    # IP
    # --------------------------------------------------------------

    if tool_name == "ip":

        ip_address = extract_ip(
            query
        )

        if ip_address:

            return analyze_ip_address(
                ip_address
            )

        return {
            "tool": "IP Address Analyzer",
            "error": "No valid IP address found in query."
        }

    # --------------------------------------------------------------
    # Severity
    # --------------------------------------------------------------

    if tool_name == "severity":

        return classify_severity(
            query
        )

    # --------------------------------------------------------------
    # No specialized tool
    # --------------------------------------------------------------

    return {
        "tool": "No Specialized Tool",
        "message": "No specialized security tool was required."
    }


# ======================================================================
# GENERATE FINAL RESPONSE
# ======================================================================

def generate_response(
    llm,
    query,
    context,
    tool_result
):

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are CyberShield AI, a defensive cybersecurity analysis assistant.

Your purpose is to help users understand cybersecurity threats and
respond safely to security incidents.

Use the supplied knowledge-base context and security-tool result.

IMPORTANT RULES:

1. Provide defensive cybersecurity guidance.
2. Do not invent facts.
3. Prefer the supplied knowledge-base information.
4. Clearly explain the detected risk.
5. Provide practical defensive recommendations.
6. If ransomware or malware is involved, provide safe incident-response
   guidance such as isolation, reporting, evidence preservation,
   investigation, containment, and recovery.
7. Do not provide offensive instructions for attacking systems.
8. Do not claim that you performed actions you did not perform.
9. If information is insufficient, clearly say so.
10. Keep the answer concise and structured.

KNOWLEDGE BASE CONTEXT:

{context}

SECURITY TOOL RESULT:

{tool_result}
"""
            ),
            (
                "human",
                "{query}"
            )
        ]
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "tool_result": json.dumps(
                tool_result,
                indent=2
            ),
            "query": query
        }
    )

    return response.content


# ======================================================================
# COMPLETE AGENT
# ======================================================================

def run_agent(
    llm,
    vector_store,
    query
):

    print("\n")
    print("=" * 70)
    print("CYBERSHIELD AI - AGENT")
    print("=" * 70)

    print(
        f"\nUser Query:\n{query}"
    )

    # --------------------------------------------------------------
    # RAG
    # --------------------------------------------------------------

    print(
        "\n[1] Retrieving cybersecurity knowledge..."
    )

    context = retrieve_context(
        vector_store,
        query
    )

    print(
        "Knowledge retrieved successfully."
    )

    # --------------------------------------------------------------
    # TOOL SELECTION
    # --------------------------------------------------------------

    print(
        "\n[2] Selecting security tool..."
    )

    tool_name = select_tool(
        query
    )

    print(
        f"Selected tool: {tool_name}"
    )

    # --------------------------------------------------------------
    # TOOL EXECUTION
    # --------------------------------------------------------------

    print(
        "\n[3] Executing security tool..."
    )

    tool_result = run_tool(
        tool_name,
        query
    )

    print(
        "Tool execution completed."
    )

    print(
        "\nTool Result:"
    )

    print(
        json.dumps(
            tool_result,
            indent=2
        )
    )

    # --------------------------------------------------------------
    # LLM RESPONSE
    # --------------------------------------------------------------

    print(
        "\n[4] Generating LLM response..."
    )

    final_response = generate_response(
        llm,
        query,
        context,
        tool_result
    )

    print(
        "\nFINAL SECURITY ANALYSIS"
    )

    print(
        "-" * 70
    )

    print(
        final_response
    )

    print(
        "-" * 70
    )

    return final_response


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("\n")
    print("=" * 70)
    print("       CYBERSHIELD AI - SECURITY AGENT")
    print("=" * 70)

    print(
        "\nProject Components:"
    )

    print(
        "✓ Llama 3.2"
    )

    print(
        "✓ Retrieval-Augmented Generation"
    )

    print(
        "✓ Chroma Vector Database"
    )

    print(
        "✓ Security Tools"
    )

    print(
        "✓ Agentic Decision Making"
    )

    # --------------------------------------------------------------
    # Initialize LLM
    # --------------------------------------------------------------

    llm = create_llm()

    # --------------------------------------------------------------
    # Initialize RAG
    # --------------------------------------------------------------

    vector_store = initialize_rag()

    # --------------------------------------------------------------
    # Test Queries
    # --------------------------------------------------------------

    test_queries = [

        (
            "I received an urgent email asking me to verify my "
            "account and provide my password. Is this phishing?"
        ),

        (
            "What should I do if ransomware has encrypted files "
            "on an employee workstation?"
        ),

        (
            "What is the security classification of IP address "
            "192.168.1.10?"
        )
    ]

    # --------------------------------------------------------------
    # Run agent
    # --------------------------------------------------------------

    for query in test_queries:

        run_agent(
            llm,
            vector_store,
            query
        )

    # --------------------------------------------------------------
    # Final
    # --------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CYBERSHIELD SECURITY AGENT COMPLETED")
    print("=" * 70)

    print(
        "\nStatus: SUCCESS"
    )

    print(
        "\nArchitecture:"
    )

    print(
        "User → Agent → RAG → Tool → Llama 3.2 → Security Response"
    )

    print(
        "\n" + "=" * 70
    )


# ======================================================================
# PROGRAM ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()