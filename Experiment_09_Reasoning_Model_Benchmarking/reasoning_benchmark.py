"""
============================================================
EXPERIMENT 9 - REASONING MODEL BENCHMARKING
============================================================

Course       : Applied Agentic AI
University   : Malla Reddy University
Model        : llama3.2:3b via Ollama

Objective:
Compare the quality of LLM responses using different
prompting strategies.

Prompting Strategies:
1. Direct Prompt
2. Structured Reasoning Prompt
3. Few-Shot Prompt

============================================================
"""

import requests
import time


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"
TIMEOUT = 180


# ============================================================
# OLLAMA FUNCTION
# ============================================================

def ask_llama(prompt):

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 250
        }
    }

    try:

        start_time = time.time()

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=TIMEOUT
        )

        response.raise_for_status()

        result = response.json()

        answer = result.get(
            "response",
            ""
        ).strip()

        end_time = time.time()

        response_time = round(
            end_time - start_time,
            2
        )

        return answer, response_time

    except requests.exceptions.ConnectionError:

        return (
            "ERROR: Could not connect to Ollama. "
            "Make sure Ollama is running.",
            0
        )

    except requests.exceptions.Timeout:

        return (
            "ERROR: Ollama request timed out.",
            0
        )

    except Exception as error:

        return (
            f"ERROR: {error}",
            0
        )


# ============================================================
# BENCHMARK QUESTIONS
# ============================================================

QUESTIONS = [

    {
        "question":
        "A company has 100 employees. "
        "60 use Python, 50 use Java, and 20 use both. "
        "How many employees use at least one of these languages?",

        "answer":
        "90"
    },

    {
        "question":
        "A cybersecurity team detects 120 alerts. "
        "25 percent are false positives. "
        "How many alerts are genuine security alerts?",

        "answer":
        "90"
    },

    {
        "question":
        "A server processes 500 requests per minute. "
        "If the processing capacity increases by 20 percent, "
        "how many requests per minute can it process?",

        "answer":
        "600"
    }

]


# ============================================================
# PROMPT STRATEGIES
# ============================================================

def direct_prompt(question):

    return f"""
Answer the following question.

Question:
{question}

Give the final answer clearly and briefly.
"""


def structured_prompt(question):

    return f"""
You are a reasoning assistant.

Solve the following problem carefully.

Follow these steps:

1. Identify the important information.
2. Determine the appropriate calculation or reasoning.
3. Perform the calculation.
4. Verify the result.
5. Give the final answer.

Question:
{question}

Keep the explanation concise.
"""


def few_shot_prompt(question):

    return f"""
You are a reasoning assistant.

Use the following examples to understand the expected
reasoning style.

Example 1:
Question:
A box contains 10 red balls and 5 blue balls.
How many balls are there in total?

Answer:
10 + 5 = 15.
Final answer: 15.

Example 2:
Question:
A system receives 200 requests and 10 percent fail.
How many requests succeed?

Answer:
10 percent of 200 = 20.
200 - 20 = 180.
Final answer: 180.

Now solve this question using the same style.

Question:
{question}

Give the calculation and final answer.
"""


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_response(response, expected_answer):

    response_lower = response.lower()

    expected_lower = expected_answer.lower()

    if expected_lower in response_lower:

        return "CORRECT"

    return "CHECK"


# ============================================================
# RUN STRATEGY
# ============================================================

def run_strategy(strategy_name, prompt_function):

    print("\n")
    print("=" * 70)
    print(strategy_name)
    print("=" * 70)

    results = []

    for index, item in enumerate(
        QUESTIONS,
        start=1
    ):

        question = item["question"]

        expected_answer = item["answer"]

        print("\n")
        print("-" * 70)

        print(
            f"QUESTION {index}"
        )

        print("-" * 70)

        print("\nQuestion:")
        print(question)

        prompt = prompt_function(
            question
        )

        answer, response_time = ask_llama(
            prompt
        )

        evaluation = evaluate_response(
            answer,
            expected_answer
        )

        print("\nModel Response:")
        print(answer)

        print(
            f"\nExpected Answer: "
            f"{expected_answer}"
        )

        print(
            f"Evaluation: "
            f"{evaluation}"
        )

        print(
            f"Response Time: "
            f"{response_time} seconds"
        )

        results.append(
            {
                "question": index,
                "evaluation": evaluation,
                "time": response_time
            }
        )

    return results


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    direct_results,
    structured_results,
    few_shot_results
):

    print("\n")
    print("=" * 70)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 70)

    strategies = [
        (
            "Direct Prompt",
            direct_results
        ),
        (
            "Structured Reasoning Prompt",
            structured_results
        ),
        (
            "Few-Shot Prompt",
            few_shot_results
        )
    ]

    for name, results in strategies:

        correct = sum(
            1
            for result in results
            if result["evaluation"] == "CORRECT"
        )

        total = len(results)

        accuracy = round(
            (correct / total) * 100,
            2
        )

        average_time = round(
            sum(
                result["time"]
                for result in results
            ) / total,
            2
        )

        print("\n")
        print(f"Strategy: {name}")
        print(f"Correct: {correct}/{total}")
        print(f"Accuracy: {accuracy}%")
        print(
            f"Average Response Time: "
            f"{average_time} seconds"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("     EXPERIMENT 9 - REASONING MODEL BENCHMARKING")
    print("=" * 70)

    print("\nCourse: Applied Agentic AI")
    print("University: Malla Reddy University")
    print("Model:", MODEL_NAME)

    print("\nObjective:")
    print(
        "Compare outputs across different "
        "prompting strategies."
    )

    print("\nPrompting Strategies:")
    print("1. Direct Prompt")
    print("2. Structured Reasoning Prompt")
    print("3. Few-Shot Prompt")

    # --------------------------------------------------------
    # DIRECT PROMPT
    # --------------------------------------------------------

    direct_results = run_strategy(
        "1. DIRECT PROMPT",
        direct_prompt
    )

    # --------------------------------------------------------
    # STRUCTURED PROMPT
    # --------------------------------------------------------

    structured_results = run_strategy(
        "2. STRUCTURED REASONING PROMPT",
        structured_prompt
    )

    # --------------------------------------------------------
    # FEW-SHOT PROMPT
    # --------------------------------------------------------

    few_shot_results = run_strategy(
        "3. FEW-SHOT PROMPT",
        few_shot_prompt
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_summary(
        direct_results,
        structured_results,
        few_shot_results
    )

    print("\n")
    print("=" * 70)
    print("EXPERIMENT 9 COMPLETED")
    print("=" * 70)

    print("\nWorkflow:")
    print("1. Test Questions          - Loaded")
    print("2. Direct Prompt           - Evaluated")
    print("3. Structured Prompt       - Evaluated")
    print("4. Few-Shot Prompt         - Evaluated")
    print("5. Results Comparison      - Completed")

    print("\nFinal Status: SUCCESS")

    print("\n" + "=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()