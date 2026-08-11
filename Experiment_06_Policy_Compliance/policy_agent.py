
"""
============================================================
EXPERIMENT 6 - POLICY COMPLIANCE AGENT
============================================================

Course       : Applied Agentic AI
University   : Malla Reddy University
Model        : Llama 3.2:3b via Ollama

Objective:
Build a Policy Compliance Agent that evaluates synthetic
employee requests using rule-based policy evaluation and
LLM-generated explanations.

Architecture:

Synthetic Employee Request
            |
            v
     Rule-Based Engine
            |
            v
    Compliance Decision
            |
            +----------------+
            |                |
       COMPLIANT        NON-COMPLIANT
            |                |
            +--------+-------+
                     |
                     v
              Llama 3.2
                     |
                     v
              Explanation
============================================================
"""

import json
import re
import requests


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


# ============================================================
# POLICY RULES
# ============================================================

POLICY_RULES = {

    "leave_policy": {
        "maximum_consecutive_days": 10,
        "description": (
            "Employees cannot request more than "
            "10 consecutive leave days."
        )
    },

    "work_from_home_policy": {
        "maximum_days_per_week": 3,
        "description": (
            "Employees can work from home for a maximum "
            "of 3 days per week."
        )
    },

    "expense_policy": {
        "maximum_reimbursement": 50000,
        "description": (
            "Individual business expense reimbursement "
            "cannot exceed ₹50,000."
        )
    },

    "data_access_policy": {
        "restricted_data": [
            "salary records",
            "employee passwords",
            "customer personal data",
            "financial records"
        ],
        "description": (
            "Employees cannot access restricted confidential "
            "information without authorization."
        )
    },

    "software_installation_policy": {
        "requires_approval": True,
        "description": (
            "Employees must obtain IT approval before "
            "installing unauthorized software."
        )
    }
}


# ============================================================
# SYNTHETIC EMPLOYEE REQUESTS
# ============================================================

SYNTHETIC_REQUESTS = [

    {
        "employee": "Rahul",
        "department": "Engineering",
        "request": (
            "I would like to take 5 consecutive days "
            "of annual leave."
        )
    },

    {
        "employee": "Priya",
        "department": "Marketing",
        "request": (
            "I would like to work from home for "
            "4 days this week."
        )
    },

    {
        "employee": "Arjun",
        "department": "Finance",
        "request": (
            "I need reimbursement of ₹25,000 "
            "for a business trip."
        )
    },

    {
        "employee": "Sneha",
        "department": "HR",
        "request": (
            "I need access to employee salary records "
            "to check confidential payroll information."
        )
    },

    {
        "employee": "Kiran",
        "department": "Engineering",
        "request": (
            "I want to install a new development tool "
            "on my company laptop without contacting IT."
        )
    },

    {
        "employee": "Anjali",
        "department": "Sales",
        "request": (
            "I would like to take 15 consecutive "
            "days of leave."
        )
    },

    {
        "employee": "Vikram",
        "department": "Finance",
        "request": (
            "I need reimbursement of ₹75,000 "
            "for a business conference."
        )
    }
]


# ============================================================
# RULE-BASED POLICY EVALUATION
# ============================================================

def rule_based_check(request):

    """
    Deterministic policy evaluation.

    This function is the authoritative compliance checker.
    """

    text = request["request"].lower()

    violations = []


    # --------------------------------------------------------
    # LEAVE POLICY
    # --------------------------------------------------------

    if "leave" in text:

        numbers = re.findall(r"\d+", text)

        if numbers:

            days = int(numbers[0])

            maximum = (
                POLICY_RULES["leave_policy"]
                ["maximum_consecutive_days"]
            )

            if days > maximum:

                violations.append(
                    f"Leave request exceeds the maximum "
                    f"allowed limit of {maximum} consecutive days."
                )


    # --------------------------------------------------------
    # WORK FROM HOME POLICY
    # --------------------------------------------------------

    if "work from home" in text or "wfh" in text:

        numbers = re.findall(r"\d+", text)

        if numbers:

            days = int(numbers[0])

            maximum = (
                POLICY_RULES["work_from_home_policy"]
                ["maximum_days_per_week"]
            )

            if days > maximum:

                violations.append(
                    f"Work-from-home request exceeds the maximum "
                    f"limit of {maximum} days per week."
                )


    # --------------------------------------------------------
    # EXPENSE POLICY
    # --------------------------------------------------------

    if "reimbursement" in text:

        numbers = re.findall(r"[\d,]+", text)

        if numbers:

            amount_text = numbers[0].replace(",", "")

            try:

                amount = int(amount_text)

                maximum = (
                    POLICY_RULES["expense_policy"]
                    ["maximum_reimbursement"]
                )

                if amount > maximum:

                    violations.append(
                        f"Requested reimbursement of "
                        f"₹{amount:,} exceeds the maximum "
                        f"allowed amount of ₹{maximum:,}."
                    )

            except ValueError:

                pass


    # --------------------------------------------------------
    # RESTRICTED DATA POLICY
    # --------------------------------------------------------

    restricted_data = (
        POLICY_RULES["data_access_policy"]
        ["restricted_data"]
    )

    for data_type in restricted_data:

        if data_type in text:

            violations.append(
                f"Request involves restricted information: "
                f"{data_type}."
            )


    # --------------------------------------------------------
    # SOFTWARE INSTALLATION POLICY
    # --------------------------------------------------------

    if "install" in text and "without" in text:

        violations.append(
            "Software installation requires "
            "prior IT approval."
        )


    return violations


# ============================================================
# LLM EXPLANATION AGENT
# ============================================================

def ask_llama(employee_request, violations):

    """
    Uses Llama 3.2 to generate an explanation.

    IMPORTANT:
    Llama does NOT determine the final compliance status.
    The deterministic rule engine is the authority.
    """

    if violations:

        final_status = "NON-COMPLIANT"

        violation_text = "\n".join(
            f"- {violation}"
            for violation in violations
        )

    else:

        final_status = "COMPLIANT"

        violation_text = "None"


    policy_text = json.dumps(
        POLICY_RULES,
        indent=2,
        ensure_ascii=False
    )


    prompt = f"""
You are an AI assistant explaining a company's
policy compliance decision.

The deterministic policy engine has already made
the final decision.

FINAL POLICY STATUS:
{final_status}

DETECTED POLICY VIOLATIONS:
{violation_text}

COMPANY POLICY RULES:
{policy_text}

EMPLOYEE REQUEST:
{employee_request}

Your task is ONLY to explain the decision.

Do NOT change the compliance status.

Do NOT invent any policies.

Return exactly:

STATUS: {final_status}
VIOLATION: None or the detected policy violation
REASON: A short and clear explanation
"""


    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }


    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        result = response.json()

        return result.get(
            "response",
            "No explanation received."
        )


    except requests.exceptions.ConnectionError:

        return (
            "ERROR: Could not connect to Ollama.\n"
            "Please make sure Ollama is running."
        )


    except requests.exceptions.Timeout:

        return (
            "ERROR: Ollama request timed out."
        )


    except Exception as error:

        return f"ERROR: {error}"


# ============================================================
# POLICY COMPLIANCE AGENT
# ============================================================

def compliance_agent(employee_request):

    print("\n" + "=" * 70)
    print("POLICY COMPLIANCE AGENT")
    print("=" * 70)


    # --------------------------------------------------------
    # Display Request
    # --------------------------------------------------------

    print("\nEmployee Request:")

    print(
        f"Employee: {employee_request['employee']}"
    )

    print(
        f"Department: {employee_request['department']}"
    )

    print(
        f"Request: {employee_request['request']}"
    )


    # --------------------------------------------------------
    # STEP 1 - RULE-BASED EVALUATION
    # --------------------------------------------------------

    violations = rule_based_check(
        employee_request
    )


    print("\n[1] RULE-BASED EVALUATION")


    if violations:

        final_status = "NON-COMPLIANT"

        print(
            "Result: NON-COMPLIANT"
        )

        for violation in violations:

            print(
                f"Violation: {violation}"
            )

    else:

        final_status = "COMPLIANT"

        print(
            "Result: COMPLIANT"
        )

        print(
            "No policy violation detected."
        )


    # --------------------------------------------------------
    # STEP 2 - LLM EXPLANATION
    # --------------------------------------------------------

    print("\n[2] LLM AGENT EXPLANATION")


    formatted_request = (
        f"Employee: {employee_request['employee']}\n"
        f"Department: {employee_request['department']}\n"
        f"Request: {employee_request['request']}"
    )


    llm_result = ask_llama(
        formatted_request,
        violations
    )


    print(llm_result)


    # --------------------------------------------------------
    # FINAL AUTHORITATIVE RESULT
    # --------------------------------------------------------

    print("\n[3] FINAL AUTHORITATIVE DECISION")

    print(
        f"FINAL STATUS: {final_status}"
    )


    if violations:

        print(
            "FINAL DECISION: Request violates "
            "one or more company policies."
        )

    else:

        print(
            "FINAL DECISION: Request complies "
            "with all checked company policies."
        )


    return {

        "request": employee_request,

        "rule_violations": violations,

        "final_status": final_status,

        "llm_result": llm_result
    }


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n")

    print("=" * 70)

    print(
        "        EXPERIMENT 6 - POLICY COMPLIANCE AGENT"
    )

    print("=" * 70)


    print(
        "\nCourse: Applied Agentic AI"
    )

    print(
        "University: Malla Reddy University"
    )

    print(
        "Model:",
        MODEL_NAME
    )


    print("\nObjective:")

    print(
        "Evaluate synthetic employee requests using "
        "rule-based policies and an LLM agent."
    )


    print(
        "\nSynthetic Requests Available:",
        len(SYNTHETIC_REQUESTS)
    )


    # --------------------------------------------------------
    # PROCESS ALL SYNTHETIC REQUESTS
    # --------------------------------------------------------

    results = []


    for index, request in enumerate(
        SYNTHETIC_REQUESTS,
        start=1
    ):

        print("\n")

        print("#" * 70)

        print(
            f"TEST CASE {index}"
        )

        print("#" * 70)


        result = compliance_agent(
            request
        )


        results.append(result)


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n\n")

    print("=" * 70)

    print(
        "FINAL COMPLIANCE SUMMARY"
    )

    print("=" * 70)


    compliant = 0

    non_compliant = 0


    for index, result in enumerate(
        results,
        start=1
    ):

        if result["final_status"] == "COMPLIANT":

            compliant += 1

        else:

            non_compliant += 1


        print(
            f"Test Case {index}: "
            f"{result['final_status']}"
        )


    print(
        "\nTotal Requests:",
        len(results)
    )

    print(
        "Compliant:",
        compliant
    )

    print(
        "Non-Compliant:",
        non_compliant
    )


    print("\n" + "=" * 70)

    print(
        "Experiment 6 Completed Successfully"
    )

    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()