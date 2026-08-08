import sqlite3
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent


DB_NAME = "company.db"


# ============================================================
# 1. CREATE DATABASE
# ============================================================

def setup_database():

    connection = sqlite3.connect(DB_NAME)

    cursor = connection.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        salary INTEGER NOT NULL,
        experience INTEGER NOT NULL
    )
    """)

    cursor.execute("DELETE FROM employees")

    employees = [
        (1, "Rahul", "IT", 60000, 2),
        (2, "Priya", "HR", 50000, 3),
        (3, "Arjun", "IT", 75000, 5),
        (4, "Sneha", "Finance", 65000, 4),
        (5, "Kiran", "IT", 55000, 1),
        (6, "Ananya", "HR", 58000, 4)
    ]

    cursor.executemany("""
    INSERT INTO employees
    (id, name, department, salary, experience)
    VALUES (?, ?, ?, ?, ?)
    """, employees)

    connection.commit()
    connection.close()


# ============================================================
# 2. DATABASE SCHEMA TOOL
# ============================================================

@tool
def get_database_schema() -> str:
    """
    Returns the exact schema of the employees database.
    """

    return """
Table: employees

Columns:
- id: employee ID
- name: employee name
- department: department name
- salary: annual salary
- experience: years of experience

IMPORTANT:
Use ONLY these column names.
"""


# ============================================================
# 3. SQL EXECUTION TOOL
# ============================================================

@tool
def execute_sql(query: str) -> str:
    """
    Executes a read-only SQLite SELECT query.
    """

    query = query.strip().rstrip(";")

    query_upper = query.upper()

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not query_upper.startswith("SELECT"):
        return "ERROR: Only SELECT queries are allowed."

    forbidden_commands = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "REPLACE"
    ]

    for command in forbidden_commands:

        if command in query_upper:
            return f"ERROR: {command} operation is not allowed."

    # --------------------------------------------------------
    # Open a NEW connection inside the tool
    # --------------------------------------------------------

    try:

        connection = sqlite3.connect(DB_NAME)

        cursor = connection.cursor()

        cursor.execute(query)

        rows = cursor.fetchall()

        if not rows:

            connection.close()

            return "No records found."

        column_names = [
            description[0]
            for description in cursor.description
        ]

        result = []

        result.append("Columns: " + ", ".join(column_names))

        for row in rows:

            result.append(str(row))

        connection.close()

        return "\n".join(result)

    except sqlite3.Error as error:

        return f"SQL Error: {error}"


# ============================================================
# 4. INITIALIZE LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# ============================================================
# 5. CREATE AGENT
# ============================================================

tools = [
    get_database_schema,
    execute_sql
]

agent = create_agent(

    model=llm,

    tools=tools,

    system_prompt="""
You are an intelligent SQL database agent.

You answer questions about the employees database.

DATABASE SCHEMA:

Table: employees

Columns:
id
name
department
salary
experience

IMPORTANT:
Never invent column names.

Use ONLY:
id, name, department, salary, experience.

WORKFLOW:

1. Understand the user's question.

2. If necessary, use get_database_schema.

3. Generate a valid SQLite SELECT query.

4. Use execute_sql to execute the query.

5. Analyze the returned database result.

6. If there is an SQL error, correct the query and try again.

7. Give a concise final answer.

SECURITY:

- Only SELECT queries are allowed.
- Never INSERT data.
- Never UPDATE data.
- Never DELETE data.
- Never DROP tables.
- Never ALTER the database.
- Never invent database information.
"""
)


# ============================================================
# 6. MAIN PROGRAM
# ============================================================

def main():

    setup_database()

    print("=" * 65)
    print("             SQL AGENT WITH TOOL USE")
    print("=" * 65)

    print("\nDatabase: company.db")
    print("Table: employees")

    print("\nAvailable columns:")
    print("id, name, department, salary, experience")

    print("\nExample questions:")
    print("1. Who has the highest salary?")
    print("2. How many employees work in IT?")
    print("3. What is the average salary of IT employees?")
    print("4. Who has more than 3 years of experience?")

    question = input("\nEnter your question: ")

    print("\nAgent is working...")
    print("-" * 65)

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )

    # ========================================================
    # DISPLAY AGENT EXECUTION
    # ========================================================

    print("\n" + "=" * 65)
    print("AGENT EXECUTION")
    print("=" * 65)

    for message in result["messages"]:

        message_type = type(message).__name__

        print(f"\n[{message_type}]")

        if hasattr(message, "content") and message.content:

            print(message.content)

        if hasattr(message, "tool_calls") and message.tool_calls:

            for tool_call in message.tool_calls:

                print("\nTool selected:")
                print("Tool:", tool_call["name"])
                print("Arguments:", tool_call["args"])

    # ========================================================
    # FINAL ANSWER
    # ========================================================

    print("\n" + "=" * 65)
    print("FINAL ANSWER")
    print("=" * 65)

    final_message = result["messages"][-1]

    print(final_message.content)

    print("\n" + "=" * 65)
    print("SQL AGENT EXPERIMENT COMPLETED")
    print("=" * 65)


# ============================================================
# 7. START
# ============================================================

if __name__ == "__main__":
    main()