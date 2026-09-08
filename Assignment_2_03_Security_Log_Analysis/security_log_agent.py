from langchain_ollama import ChatOllama
from datetime import datetime


MODEL_NAME = "llama3.2:3b"


# ---------------------------------------------------------
# AI SECURITY LOG ANALYZER
# ---------------------------------------------------------

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0
)


def analyze_security_logs(log_data):
    prompt = f"""
You are an expert Cybersecurity Log Analysis Agent.

Analyze the following security logs carefully.

Your task is to:

1. Identify suspicious or malicious activities.
2. Identify the possible type of threat or attack.
3. Assign a severity level:
   - LOW
   - MEDIUM
   - HIGH
   - CRITICAL
4. Explain why the activity is suspicious.
5. Identify affected systems, users, IP addresses, or resources.
6. Provide recommended mitigation steps.
7. Clearly separate genuine security concerns from normal activity.

Do NOT invent information that is not present in the logs.

Return the analysis using this structure:

SECURITY LOG ANALYSIS
=====================

1. THREAT SUMMARY
- Threat:
- Severity:
- Confidence:

2. SUSPICIOUS ACTIVITIES
- Activity 1:
- Activity 2:
- Activity 3:

3. ATTACK TYPE
- Type:
- Explanation:

4. AFFECTED RESOURCES
- IP addresses:
- Users:
- Systems/Resources:

5. EVIDENCE FROM LOGS
- Evidence 1:
- Evidence 2:

6. RECOMMENDED MITIGATION
- Action 1:
- Action 2:
- Action 3:

7. FINAL SECURITY ASSESSMENT
Provide a short overall assessment.

SECURITY LOGS:
----------------
{log_data}
"""

    response = llm.invoke(prompt)
    return response.content


# ---------------------------------------------------------
# SAMPLE SECURITY LOGS
# ---------------------------------------------------------

sample_logs = """
2026-09-08 09:15:21 INFO User login successful username=admin source_ip=192.168.1.20

2026-09-08 09:17:42 WARNING Multiple failed login attempts
username=admin source_ip=185.220.101.45 attempts=8

2026-09-08 09:18:03 WARNING Multiple failed login attempts
username=admin source_ip=185.220.101.45 attempts=15

2026-09-08 09:18:27 ALERT Account temporarily locked
username=admin source_ip=185.220.101.45

2026-09-08 09:20:14 INFO User login successful
username=admin source_ip=192.168.1.20

2026-09-08 09:22:51 WARNING Large outbound data transfer
source_ip=192.168.1.20 destination=185.220.101.45
data_transferred=850MB

2026-09-08 09:23:10 ALERT Suspicious executable detected
file=update_service.exe
path=C:\\Users\\admin\\AppData\\Temp\\update_service.exe
"""



# ---------------------------------------------------------
# MAIN AGENT WORKFLOW
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("SECURITY LOG / ALERT ANALYSIS AGENT")
    print("Powered by Ollama + Llama 3.2")
    print("=" * 70)

    print("\n[1/3] Loading security logs...")

    print(f"Log entries loaded successfully.")

    print("\n[2/3] Analyzing security events with AI...")

    analysis = analyze_security_logs(sample_logs)

    print("Security analysis completed.")

    print("\n[3/3] Generating security assessment...")

    print("\n" + "=" * 70)
    print("FINAL SECURITY ANALYSIS")
    print("=" * 70)

    print(analysis)

    # Save report
    filename = "security_analysis_report.txt"

    with open(filename, "w", encoding="utf-8") as file:
        file.write("SECURITY LOG / ALERT ANALYSIS REPORT\n")
        file.write("=" * 70 + "\n")
        file.write(f"Generated: {datetime.now()}\n\n")
        file.write(analysis)

    print("\n" + "=" * 70)
    print(f"Report saved successfully as: {filename}")
    print("=" * 70)


if __name__ == "__main__":
    main()