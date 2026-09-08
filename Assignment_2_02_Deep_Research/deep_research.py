import requests
from bs4 import BeautifulSoup
from langchain_ollama import ChatOllama


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "llama3.2:3b"
MAX_SOURCES = 6

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0
)


# ============================================================
# 1. RESEARCH PLANNER AGENT
# ============================================================

def create_research_plan(topic):

    prompt = f"""
You are a research planning agent.

Research topic:
{topic}

Create exactly 4 focused research questions.

Cover:
1. Background and definition
2. Technologies, methods, or mechanisms
3. Benefits and applications
4. Challenges, risks, and future trends

Return only the numbered questions.
"""

    response = llm.invoke(prompt)

    return response.content


# ============================================================
# 2. WIKIPEDIA SEARCH
# ============================================================

def search_wikipedia(query):

    url = "https://en.wikipedia.org/w/rest.php/v1/search/page"

    headers = {
        "User-Agent": "DeepResearchAgent/1.0 StudentProject"
    }

    params = {
        "q": query,
        "limit": 5
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        results = []

        for page in data.get("pages", []):

            title = page.get("title", "")
            description = page.get("description", "")
            excerpt = page.get("excerpt", "")

            if not title:
                continue

            page_url = (
                "https://en.wikipedia.org/wiki/"
                + title.replace(" ", "_")
            )

            results.append({
                "title": title,
                "url": page_url,
                "description": description,
                "snippet": excerpt
            })

        return results

    except Exception as e:

        print(f"Wikipedia search error: {e}")

        return []


# ============================================================
# 3. EXTRACT WEBPAGE CONTENT
# ============================================================

def extract_page_text(title):

    page_name = title.replace(" ", "_")

    url = (
        "https://en.wikipedia.org/wiki/"
        + page_name
    )

    headers = {
        "User-Agent": "DeepResearchAgent/1.0 StudentProject"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove unnecessary elements
        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "table",
            "sup"
        ]):
            tag.decompose()

        paragraphs = soup.find_all("p")

        text = "\n".join(
            p.get_text(" ", strip=True)
            for p in paragraphs
        )

        return text[:7000]

    except Exception as e:

        print(
            f"Could not read '{title}': {e}"
        )

        return ""


# ============================================================
# 4. SOURCE COLLECTION AND FILTERING
# ============================================================

def collect_sources(topic):

    # Targeted cybersecurity queries
    queries = [
        f"{topic} cybersecurity",
        f"{topic} cyber attack detection",
        f"{topic} threat detection incident response",
        f"{topic} security risks challenges",
        f"{topic} cybersecurity applications"
    ]

    sources = []

    # Keywords used to filter irrelevant search results
    relevant_keywords = [
        "cyber",
        "security",
        "computer security",
        "information security",
        "cybersecurity",
        "malware",
        "phishing",
        "intrusion",
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "network security",
        "threat",
        "vulnerability"
    ]

    for query in queries:

        print(f"\nSearching Wikipedia: {query}")

        results = search_wikipedia(query)

        for result in results:

            title = result["title"]
            title_lower = title.lower()

            # Check whether the title is relevant
            is_relevant = any(
                keyword in title_lower
                for keyword in relevant_keywords
            )

            if not is_relevant:
                continue

            # Avoid duplicate sources
            existing_titles = [
                source["title"].lower()
                for source in sources
            ]

            if title_lower not in existing_titles:

                sources.append(result)

                print(
                    f"  + Selected: {title}"
                )

            if len(sources) >= MAX_SOURCES:
                break

        if len(sources) >= MAX_SOURCES:
            break

    return sources


# ============================================================
# 5. SOURCE ANALYSIS AGENT
# ============================================================

def analyze_sources(topic, sources):

    source_material = ""

    for index, source in enumerate(sources, 1):

        source_material += f"""
SOURCE {index}

Title:
{source['title']}

URL:
{source['url']}

Description:
{source['description']}

Content:
{source.get('content', '')}

--------------------------------------------------
"""

    prompt = f"""
You are an expert cybersecurity research analyst.

Research topic:
{topic}

Analyze the following research sources:

{source_material}

Create an evidence-based analysis containing:

1. Key Findings
2. Technologies and Methods
3. Benefits and Applications
4. Cybersecurity Challenges and Risks
5. Future Trends

Rules:

- Use only information supported by the supplied sources.
- Do not invent statistics.
- Do not invent companies or technologies.
- Do not make unsupported claims.
- If information is unavailable, say:
  "Not found in the collected sources."
- Focus specifically on cybersecurity.
"""

    response = llm.invoke(prompt)

    return response.content


# ============================================================
# 6. FINAL REPORT GENERATOR
# ============================================================

def generate_report(
    topic,
    research_questions,
    analysis,
    sources
):

    references = ""

    for index, source in enumerate(sources, 1):

        references += (
            f"{index}. {source['title']}\n"
            f"   {source['url']}\n\n"
        )

    prompt = f"""
You are a professional cybersecurity research report writer.

Research Topic:
{topic}

Research Questions:
{research_questions}

Research Analysis:
{analysis}

Sources:
{references}

Create a clear academic-style research report.

Use exactly this structure:

# Research Report: {topic}

## 1. Executive Summary

## 2. Research Questions

## 3. Key Findings

## 4. AI Technologies and Methods in Cybersecurity

## 5. Benefits and Applications

## 6. Cybersecurity Challenges and Risks

## 7. Future Trends

## 8. Conclusion

## 9. References

Rules:

- Focus specifically on cybersecurity.
- Use only information supported by the research.
- Do not invent statistics or facts.
- Keep the language suitable for a college assignment.
- Clearly explain the role of AI in cybersecurity.
- Include all provided URLs in the References section.
"""

    response = llm.invoke(prompt)

    report = response.content

    # Ensure references are included
    if "## 9. References" not in report:

        report += (
            "\n\n## 9. References\n\n"
            + references
        )

    return report


# ============================================================
# 7. MAIN DEEP RESEARCH WORKFLOW
# ============================================================

def deep_research(topic):

    print("\n" + "=" * 70)
    print("DEEP RESEARCH AGENT")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1 - RESEARCH PLANNING
    # --------------------------------------------------------

    print("\n[1/5] Creating research plan...")

    research_questions = create_research_plan(topic)

    print("\nResearch Plan:")
    print(research_questions)

    # --------------------------------------------------------
    # STEP 2 - SOURCE SEARCH
    # --------------------------------------------------------

    print("\n[2/5] Searching and filtering research sources...")

    sources = collect_sources(topic)

    print(
        f"\nCollected {len(sources)} relevant sources."
    )

    if not sources:

        print(
            "\nNo relevant sources were found."
        )

        return

    print("\nSelected Sources:")

    for index, source in enumerate(sources, 1):

        print(
            f"{index}. {source['title']}"
        )

    # --------------------------------------------------------
    # STEP 3 - CONTENT EXTRACTION
    # --------------------------------------------------------

    print("\n[3/5] Reading research sources...")

    valid_sources = []

    for index, source in enumerate(
        sources,
        1
    ):

        print(
            f"Reading source "
            f"{index}/{len(sources)}: "
            f"{source['title']}"
        )

        content = extract_page_text(
            source["title"]
        )

        if content:

            source["content"] = content

            valid_sources.append(source)

    sources = valid_sources

    print(
        f"\nSuccessfully read "
        f"{len(sources)} sources."
    )

    if not sources:

        print(
            "\nNo readable source content found."
        )

        return

    # --------------------------------------------------------
    # STEP 4 - SOURCE ANALYSIS
    # --------------------------------------------------------

    print("\n[4/5] Analyzing research sources...")

    analysis = analyze_sources(
        topic,
        sources
    )

    print(
        "\nResearch analysis completed."
    )

    # --------------------------------------------------------
    # STEP 5 - REPORT GENERATION
    # --------------------------------------------------------

    print(
        "\n[5/5] Generating final research report..."
    )

    report = generate_report(
        topic,
        research_questions,
        analysis,
        sources
    )

    print("\n")
    print("=" * 70)
    print("FINAL RESEARCH REPORT")
    print("=" * 70)

    print(report)

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    filename = "research_report.txt"

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    print("\n" + "=" * 70)
    print(
        f"Report saved successfully as: "
        f"{filename}"
    )
    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print("\nDeep Research Agent")
    print("Powered by Ollama + Llama 3.2")

    topic = input(
        "\nEnter a research topic: "
    ).strip()

    if not topic:

        print(
            "\nPlease enter a valid research topic."
        )

    else:

        deep_research(topic)
