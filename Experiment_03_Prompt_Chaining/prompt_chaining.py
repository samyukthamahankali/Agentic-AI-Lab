from langchain_ollama import ChatOllama


# ============================================================
# EXPERIMENT 3
# PROMPT CHAINING FOR SUMMARIZATION
# ============================================================


# ============================================================
# 1. INITIALIZE LOCAL LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# ============================================================
# 2. INPUT TEXT
# ============================================================

text = """
Artificial Intelligence is transforming many industries by enabling
machines to perform tasks that traditionally required human intelligence.
AI systems are used in healthcare, finance, education, transportation,
cybersecurity, and many other fields.

Large Language Models are a type of AI model designed to understand and
generate human language. These models are trained on large collections
of text and can perform tasks such as summarization, translation,
question answering, and content generation.

Agentic AI extends the capabilities of language models by allowing systems
to make decisions, use tools, maintain context, and perform multiple steps
to accomplish a goal. An AI agent may interact with databases, APIs,
search engines, or other software tools.

Retrieval-Augmented Generation is another important technique used in
modern AI applications. RAG retrieves relevant information from external
knowledge sources and provides that information to a language model as
context before generating an answer.

The combination of LLMs, RAG, tools, memory, and planning can be used
to create powerful AI agents capable of solving complex real-world tasks.
However, security, privacy, reliability, and responsible AI practices
must be considered when deploying these systems.
"""


# ============================================================
# 3. PROMPT CHAIN — STEP 1
# EXTRACT KEY POINTS
# ============================================================

print("=" * 70)
print("        PROMPT CHAINING FOR SUMMARIZATION")
print("=" * 70)

print("\n[STEP 1] Extracting key points...")

prompt_1 = f"""
Read the following text and extract the most important key points.

Return 5 to 7 concise bullet points.

Do not add information that is not present in the text.

Text:
{text}

Key Points:
"""

response_1 = llm.invoke(prompt_1)

key_points = response_1.content

print("\nKey Points:")
print("-" * 70)
print(key_points)


# ============================================================
# 4. PROMPT CHAIN — STEP 2
# CREATE SUMMARY
# ============================================================

print("\n[STEP 2] Creating summary from key points...")

prompt_2 = f"""
Create a clear and concise summary using ONLY the key points below.

The summary should be approximately 100 words.

Do not introduce new information.

Key Points:
{key_points}

Summary:
"""

response_2 = llm.invoke(prompt_2)

summary = response_2.content

print("\nSummary:")
print("-" * 70)
print(summary)


# ============================================================
# 5. PROMPT CHAIN — STEP 3
# CREATE FINAL STRUCTURED OUTPUT
# ============================================================

print("\n[STEP 3] Creating final structured output...")

prompt_3 = f"""
Convert the following summary into a structured study note.

Use exactly these sections:

1. Main Topic
2. Key Concepts
3. Applications
4. Important Considerations

Keep the information concise.

Do not add information that is not present in the summary.

Summary:
{summary}

Structured Study Note:
"""

response_3 = llm.invoke(prompt_3)

final_output = response_3.content


# ============================================================
# 6. DISPLAY FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL STRUCTURED OUTPUT")
print("=" * 70)

print(final_output)

print("\n" + "=" * 70)
print("PROMPT CHAINING EXPERIMENT COMPLETED")
print("=" * 70)