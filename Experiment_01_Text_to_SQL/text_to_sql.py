import sqlite3
import re
from langchain_ollama import ChatOllama


# ============================================================
# 1. CREATE DATABASE
# ============================================================

DB_NAME = "company.db"

connection = sqlite3.connect(DB_NAME)
cursor = connection.cursor()

# Create employees table
cursor.execute("""
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    salary INTEGER NOT NULL,
    experience INTEGER NOT NULL
)
""")

# Remove old data so every run starts clean
cursor.execute("DELETE FROM employees")

# Insert sample employee data
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


# ============================================================
# 2. DATABASE SCHEMA
# ============================================================

schema = """
Table: employees

Columns:
- id: employee ID
- name: employee name
- department: department name
- salary: annual salary
- experience: years of experience
"""


# ============================================================
# 3. INITIALIZE LOCAL LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# ============================================================
# 4. FUNCTION TO GENERATE SQL
# ============================================================

def generate_sql(question):

    prompt = f"""
You are an SQL expert.

Convert the user's question into a SQLite SQL query.

Database schema:
{schema}

Rules:
1. Generate only ONE SQL query.
2. Use only the employees table.
3. Do not use INSERT, UPDATE, DELETE, DROP, ALTER, or CREATE.
4. Only generate SELECT queries.
5. Do not explain the query.
6. Return only the SQL query.

User question:
{question}
"""

    response = llm.invoke(prompt)

    sql = response.content.strip()

    # Remove Markdown SQL code fences if the model adds them
    sql = re.sub(r"```sql", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"```", "", sql)

    # Find the SELECT statement
    match = re.search(
        r"(SELECT\s+.*?)(?:;|$)",
        sql,
        flags=re.IGNORECASE | re.DOTALL
    )

    if match:
        sql = match.group(1).strip()

    return sql


# ============================================================
# 5. SAFETY CHECK
# ============================================================

def is_safe_sql(sql):

    sql_upper = sql.upper().strip()

    dangerous_commands = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "REPLACE"
    ]

    if not sql_upper.startswith("SELECT"):
        return False

    for command in dangerous_commands:
        if re.search(r"\b" + command + r"\b", sql_upper):
            return False

    return True


# ============================================================
# 6. EXECUTE SQL
# ============================================================

def execute_sql(sql):

    try:

        cursor.execute(sql)

        results = cursor.fetchall()

        return results

    except sqlite3.Error as error:

        return f"SQL Error: {error}"


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

print("=" * 60)
print("       TEXT-TO-SQL WORKFLOW")
print("=" * 60)

print("\nDatabase: company.db")
print("Table: employees")

print("\nAvailable columns:")
print("id, name, department, salary, experience")

print("\nExample questions:")
print("1. Who has the highest salary?")
print("2. How many employees are in IT?")
print("3. What is the average salary in IT?")
print("4. List employees with more than 3 years experience.")

question = input("\nEnter your question: ")


# Generate SQL
print("\nGenerating SQL using Llama 3.2...")

sql_query = generate_sql(question)

print("\nGenerated SQL:")
print("-" * 60)
print(sql_query)
print("-" * 60)


# Safety validation
if not is_safe_sql(sql_query):

    print("\nERROR: Unsafe SQL query detected.")
    connection.close()
    exit()


# Execute query
print("\nExecuting query...")

result = execute_sql(sql_query)


# Display result
print("\nQuery Result:")
print("-" * 60)

if isinstance(result, str):

    print(result)

elif len(result) == 0:

    print("No records found.")

else:

    for row in result:
        print(row)


print("-" * 60)

print("\nExperiment completed successfully.")

# Close database
connection.close()