
"""
============================================================
EXPERIMENT 7 - DEEP RESEARCH AGENT WORKFLOW
============================================================

Course       : Applied Agentic AI
University   : Malla Reddy University
Model        : Llama 3.2:3b via Ollama

Objective:
Implement planning + reflection for content generation.

Workflow:

Research Topic
      |
      v
Planning Agent
      |
      v
Research/Drafting Agent
      |
      v
Reflection Agent
      |
      v
Revision Agent
      |
      v
Final Research Report

Concepts Demonstrated:
- Task decomposition
- Prompt chaining
- Planning
- LLM-based content generation
- Reflection
- Iterative refinement
- Agent workflow
============================================================
"""

import requests


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

# Keep the timeout reasonable for a local 3B model.
OLLAMA_TIMEOUT = 180


# ============================================================
# OLLAMA LLM FUNCTION
# ============================================================

def ask_llama(prompt):
    """
    Sends a prompt to Llama 3.2 through Ollama.
    """

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,

        # Keep generation short for faster execution.
        "options": {
            "temperature": 0.3,
            "num_predict": 350
        }
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT
        )

        response.raise_for_status()

        result = response.json()

        answer = result.get(
            "response",
            ""
        ).strip()

        if not answer:

            return "ERROR: Ollama returned an empty response."

        return answer


    except requests.exceptions.ConnectionError:

        return (
            "ERROR: Could not connect to Ollama.\n"
            "Make sure Ollama is running."
        )


    except requests.exceptions.Timeout:

        return (
            "ERROR: Ollama request timed out."
        )


    except Exception as error:

        return f"ERROR: {error}"


# ============================================================
# ERROR CHECK
# ============================================================

def is_error(response):

    return response.startswith("ERROR:")


# ============================================================
# STAGE 1 - PLANNING AGENT
# ============================================================

def planning_agent(topic):

    print("\n")
    print("=" * 70)
    print("STAGE 1 - PLANNING AGENT")
    print("=" * 70)

    print("\nCreating research plan...")

    prompt = f"""
You are a planning agent.

Topic:
{topic}

Create exactly 5 short research questions.

Cover:
- Definition
- Key concepts
- Applications
- Advantages and limitations
- Future scope

Return only the numbered questions.
"""

    plan = ask_llama(prompt)

    print("\nResearch Plan:")
    print(plan)

    return plan


# ============================================================
# STAGE 2 - RESEARCH / DRAFTING AGENT
# ============================================================

def research_agent(topic, research_plan):

    print("\n")
    print("=" * 70)
    print("STAGE 2 - RESEARCH / DRAFTING AGENT")
    print("=" * 70)

    print("\nGenerating concise research draft...")

    prompt = f"""
You are a research drafting agent.

Topic:
{topic}

Research Plan:
{research_plan}

Write a concise college-level research draft.

Use these sections:

1. Introduction
2. Key Concepts
3. Applications
4. Advantages and Limitations
5. Challenges and Future Scope
6. Conclusion

Keep each section short.

Use simple academic language.

Do not invent statistics or fake sources.

Return only the draft.
"""

    draft = ask_llama(prompt)

    print("\nFirst Draft:")
    print(draft)

    return draft


# ============================================================
# STAGE 3 - REFLECTION AGENT
# ============================================================

def reflection_agent(topic, draft):

    print("\n")
    print("=" * 70)
    print("STAGE 3 - REFLECTION AGENT")
    print("=" * 70)

    print("\nEvaluating the draft...")

    prompt = f"""
You are a reflection agent.

Topic:
{topic}

Draft:
{draft}

Evaluate the draft.

Return exactly:

QUALITY:
GOOD or NEEDS IMPROVEMENT

STRENGTHS:
- Two short points

WEAKNESSES:
- Two short points

IMPROVEMENTS:
- Two short points

Do not rewrite the draft.
"""

    reflection = ask_llama(prompt)

    print("\nReflection:")
    print(reflection)

    return reflection


# ============================================================
# STAGE 4 - REVISION AGENT
# ============================================================

def revision_agent(topic, draft, reflection):

    print("\n")
    print("=" * 70)
    print("STAGE 4 - REVISION AGENT")
    print("=" * 70)

    print("\nImproving the draft using reflection...")

    prompt = f"""
You are a revision agent.

Topic:
{topic}

Draft:
{draft}

Reflection:
{reflection}

Improve the draft using the reflection.

Keep the report concise.

Use these sections:

1. Introduction
2. Key Concepts
3. Applications
4. Advantages and Limitations
5. Challenges and Future Scope
6. Conclusion

Use clear academic language.

Do not invent statistics or fake sources.

Return only the improved report.
"""

    final_report = ask_llama(prompt)

    print("\nFinal Improved Report:")
    print(final_report)

    return final_report


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("        EXPERIMENT 7 - DEEP RESEARCH AGENT")
    print("=" * 70)

    print("\nCourse: Applied Agentic AI")
    print("University: Malla Reddy University")
    print("Model:", MODEL_NAME)

    print("\nObjective:")
    print(
        "Implement planning and reflection for "
        "content generation using an agentic workflow."
    )


    # --------------------------------------------------------
    # GET RESEARCH TOPIC
    # --------------------------------------------------------

    topic = input(
        "\nEnter a research topic: "
    ).strip()


    if not topic:

        print(
            "\nERROR: Research topic cannot be empty."
        )

        return


    print("\nSelected Topic:")
    print(topic)


    # --------------------------------------------------------
    # STAGE 1
    # --------------------------------------------------------

    research_plan = planning_agent(topic)


    if is_error(research_plan):

        print("\nWorkflow stopped.")
        print("Planning Agent failed.")

        return


    # --------------------------------------------------------
    # STAGE 2
    # --------------------------------------------------------

    draft = research_agent(
        topic,
        research_plan
    )


    if is_error(draft):

        print("\nWorkflow stopped.")
        print("Research/Drafting Agent failed.")

        return


    # --------------------------------------------------------
    # STAGE 3
    # --------------------------------------------------------

    reflection = reflection_agent(
        topic,
        draft
    )


    if is_error(reflection):

        print("\nWorkflow stopped.")
        print("Reflection Agent failed.")

        return


    # --------------------------------------------------------
    # STAGE 4
    # --------------------------------------------------------

    final_report = revision_agent(
        topic,
        draft,
        reflection
    )


    if is_error(final_report):

        print("\n")
        print("=" * 70)
        print("DEEP RESEARCH WORKFLOW FAILED")
        print("=" * 70)

        print("\nRevision Agent failed.")

        print("\nCompleted Stages:")
        print("1. Planning Agent       - Completed")
        print("2. Research/Draft Agent - Completed")
        print("3. Reflection Agent     - Completed")
        print("4. Revision Agent       - Failed")

        return


    # --------------------------------------------------------
    # FINAL SUCCESS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("DEEP RESEARCH WORKFLOW COMPLETED")
    print("=" * 70)

    print("\nWorkflow Stages:")
    print("1. Planning Agent       - Completed")
    print("2. Research/Draft Agent - Completed")
    print("3. Reflection Agent     - Completed")
    print("4. Revision Agent       - Completed")

    print("\nResearch Topic:")
    print(topic)

    print("\nFinal Status: SUCCESS")

    print("\n" + "=" * 70)
    print("Experiment 7 Completed Successfully")
    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()