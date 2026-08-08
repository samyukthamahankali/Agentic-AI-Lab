from langchain_ollama import ChatOllama


# ============================================================
# MULTI-AGENT SDR SYSTEM
# ============================================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.2
)


# ============================================================
# AGENT 1 — LEAD RESEARCH AGENT
# ============================================================

def lead_research_agent(lead):

    prompt = f"""
You are the Lead Research Agent in a multi-agent SDR system.

Analyze the following potential customer:

Name: {lead['name']}
Company: {lead['company']}
Industry: {lead['industry']}
Role: {lead['role']}
Company Size: {lead['company_size']}
Interest: {lead['interest']}

Provide a concise analysis containing:

1. Potential business need
2. Why this person may be interested
3. Possible pain point
4. Recommended product/service angle

Do not invent specific facts about the person or company.
Use only the information provided.
"""

    response = llm.invoke(prompt)

    return response.content


# ============================================================
# AGENT 2 — QUALIFICATION AGENT
# ============================================================

def qualification_agent(lead, research):

    prompt = f"""
You are the Lead Qualification Agent.

Your job is to determine whether a potential customer is
worth pursuing.

Lead information:

Name: {lead['name']}
Company: {lead['company']}
Industry: {lead['industry']}
Role: {lead['role']}
Company Size: {lead['company_size']}
Interest: {lead['interest']}

Research Agent Analysis:
{research}

Evaluate:

1. Relevance
2. Potential need
3. Decision-making authority
4. Buying interest
5. Overall qualification

Classify the lead as:

QUALIFIED
or
NOT QUALIFIED

Then provide a short explanation.

Do not invent facts.
"""

    response = llm.invoke(prompt)

    return response.content


# ============================================================
# AGENT 3 — EMAIL AGENT
# ============================================================

def email_agent(lead, research, qualification):

    prompt = f"""
You are the Email Outreach Agent in a multi-agent SDR system.

Create a professional and personalized outreach email.

Lead:
Name: {lead['name']}
Company: {lead['company']}
Industry: {lead['industry']}
Role: {lead['role']}
Interest: {lead['interest']}

Research:
{research}

Qualification:
{qualification}

Requirements:

- Write a short professional email.
- Mention the lead's likely business need.
- Do not invent personal information.
- Do not make false claims.
- Include a clear call to action.
- Provide a subject line.
"""

    response = llm.invoke(prompt)

    return response.content


# ============================================================
# SUPERVISOR / WORKFLOW
# ============================================================

def run_multi_agent_system(lead):

    print("\n" + "=" * 70)
    print("MULTI-AGENT SDR SYSTEM")
    print("=" * 70)

    # --------------------------------------------------------
    # Agent 1
    # --------------------------------------------------------

    print("\n[AGENT 1] Lead Research Agent")
    print("-" * 70)

    research = lead_research_agent(lead)

    print(research)


    # --------------------------------------------------------
    # Agent 2
    # --------------------------------------------------------

    print("\n[AGENT 2] Lead Qualification Agent")
    print("-" * 70)

    qualification = qualification_agent(
        lead,
        research
    )

    print(qualification)


    # --------------------------------------------------------
    # Check qualification
    # --------------------------------------------------------

    if "NOT QUALIFIED" in qualification.upper():

        print("\nLead is not qualified.")
        print("Email generation skipped.")

        return


    # --------------------------------------------------------
    # Agent 3
    # --------------------------------------------------------

    print("\n[AGENT 3] Email Outreach Agent")
    print("-" * 70)

    email = email_agent(
        lead,
        research,
        qualification
    )

    print(email)


    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MULTI-AGENT WORKFLOW COMPLETED")
    print("=" * 70)


# ============================================================
# SAMPLE LEAD
# ============================================================

lead = {
    "name": "Rahul Sharma",
    "company": "TechNova Solutions",
    "industry": "Software Development",
    "role": "Technology Manager",
    "company_size": "100-250 employees",
    "interest": "AI automation for internal business workflows"
}


# ============================================================
# START SYSTEM
# ============================================================

if __name__ == "__main__":

    run_multi_agent_system(lead)