from langchain_ollama import ChatOllama
from datetime import datetime


MODEL_NAME = "llama3.2:3b"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0
)


# =========================================================
# AGENT 1: RESEARCH AGENT
# =========================================================

def research_agent(topic):
    prompt = f"""
You are the Research Agent in a multi-agent cybersecurity system.

Research and explain the following topic:

{topic}

Provide:
1. Definition
2. Background
3. Important technologies or methods
4. Main applications
5. Important facts

Focus only on cybersecurity-related information.
Do not invent statistics or sources.

Return a concise research summary.
"""

    response = llm.invoke(prompt)
    return response.content


# =========================================================
# AGENT 2: SECURITY ANALYST AGENT
# =========================================================

def security_analyst_agent(topic, research):
    prompt = f"""
You are the Security Analyst Agent.

Analyze the research provided by another AI agent.

TOPIC:
{topic}

RESEARCH AGENT OUTPUT:
{research}

Your task:

1. Identify cybersecurity benefits.
2. Identify possible threats and risks.
3. Identify vulnerabilities or limitations.
4. Explain how organizations can use the technology safely.
5. Recommend security controls or mitigation strategies.

Do not invent information that is not reasonably supported by the research.

Return a structured security analysis.
"""

    response = llm.invoke(prompt)
    return response.content


# =========================================================
# AGENT 3: REPORT AGENT
# =========================================================

def report_agent(topic, research, security_analysis):
    prompt = f"""
You are the Report Agent in a multi-agent cybersecurity system.

Create a professional final report by combining the outputs
from the Research Agent and Security Analyst Agent.

TOPIC:
{topic}

RESEARCH AGENT:
{research}

SECURITY ANALYST AGENT:
{security_analysis}

Create the final report using exactly this structure:

# Multi-Agent Cybersecurity Report: {topic}

## 1. Executive Summary

## 2. Research Findings

## 3. Cybersecurity Applications

## 4. Security Benefits

## 5. Threats and Risks

## 6. Limitations

## 7. Recommended Security Controls

## 8. Final Assessment

Make the report clear, concise and technically accurate.
Do not invent statistics, references, or facts.
"""

    response = llm.invoke(prompt)
    return response.content


# =========================================================
# MAIN MULTI-AGENT WORKFLOW
# =========================================================

def main():

    print("=" * 70)
    print("MULTI-AGENT CYBERSECURITY COLLABORATION SYSTEM")
    print("Powered by Ollama + Llama 3.2")
    print("=" * 70)

    topic = input("\nEnter a cybersecurity topic: ").strip()

    if not topic:
        print("Please enter a valid topic.")
        return

    # -----------------------------------------------------
    # AGENT 1
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("[AGENT 1] RESEARCH AGENT")
    print("=" * 70)

    print("\nResearch Agent is analyzing the topic...")

    research = research_agent(topic)

    print("\nResearch Agent completed its task.")

    print("\n--- RESEARCH OUTPUT ---")
    print(research)

    # -----------------------------------------------------
    # AGENT 2
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("[AGENT 2] SECURITY ANALYST AGENT")
    print("=" * 70)

    print("\nSecurity Analyst Agent is analyzing the research...")

    security_analysis = security_analyst_agent(
        topic,
        research
    )

    print("\nSecurity Analyst Agent completed its task.")

    print("\n--- SECURITY ANALYSIS ---")
    print(security_analysis)

    # -----------------------------------------------------
    # AGENT 3
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("[AGENT 3] REPORT AGENT")
    print("=" * 70)

    print("\nReport Agent is combining the agent outputs...")

    final_report = report_agent(
        topic,
        research,
        security_analysis
    )

    print("\nReport Agent completed its task.")

    # -----------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL MULTI-AGENT REPORT")
    print("=" * 70)

    print(final_report)

    # -----------------------------------------------------
    # SAVE REPORT
    # -----------------------------------------------------

    filename = "multi_agent_report.txt"

    with open(filename, "w", encoding="utf-8") as file:

        file.write("MULTI-AGENT CYBERSECURITY REPORT\n")
        file.write("=" * 70 + "\n")
        file.write(f"Generated: {datetime.now()}\n")
        file.write(f"Topic: {topic}\n\n")

        file.write("\n" + "=" * 70 + "\n")
        file.write("AGENT 1 - RESEARCH AGENT\n")
        file.write("=" * 70 + "\n")
        file.write(research)

        file.write("\n\n" + "=" * 70 + "\n")
        file.write("AGENT 2 - SECURITY ANALYST AGENT\n")
        file.write("=" * 70 + "\n")
        file.write(security_analysis)

        file.write("\n\n" + "=" * 70 + "\n")
        file.write("AGENT 3 - REPORT AGENT\n")
        file.write("=" * 70 + "\n")
        file.write(final_report)

    print("\n" + "=" * 70)
    print(f"Report saved successfully as: {filename}")
    print("=" * 70)


if __name__ == "__main__":
    main()