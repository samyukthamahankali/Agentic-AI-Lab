# Experiment 12 - Mini Project

## Cybersecurity Multi-Agent AI System

### Course
Applied Agentic AI

### University
Malla Reddy University

### Domain
Cybersecurity

---

## Objective

Develop an end-to-end agentic AI cybersecurity system integrating:

- Large Language Model
- Retrieval-Augmented Generation (RAG)
- Chroma Vector Database
- Security Analysis Tools
- Agentic Decision Making
- Local LLM using Ollama

---

## Architecture

User
  |
  v
Agent
  |
  +----> RAG
  |       |
  |       v
  |   Chroma Vector Database
  |       |
  |       v
  |   Cybersecurity Knowledge
  |
  +----> Security Tools
  |       |
  |       +---- Phishing Analyzer
  |       +---- Password Strength Analyzer
  |       +---- IP Address Analyzer
  |       +---- Incident Severity Classifier
  |
  v
Llama 3.2
  |
  v
Security Response

---

## Technologies Used

- Python 3.12.5
- Llama 3.2:3b
- Ollama
- LangChain
- ChromaDB
- Sentence Transformers
- Hugging Face
- Streamlit
- PyTorch
- Transformers

---

## RAG Pipeline

1. Load cybersecurity documents
2. Split documents into chunks
3. Generate embeddings
4. Store embeddings in ChromaDB
5. Retrieve relevant cybersecurity knowledge
6. Provide retrieved context to the agent

---

## Security Tools

### 1. Phishing Analyzer

Detects suspicious phrases such as:

- urgent
- immediately
- verify your account
- password
- click here
- login
- confirm your password

Returns a risk level and detected indicators.

### 2. Password Strength Analyzer

Evaluates password strength based on:

- Length
- Uppercase characters
- Lowercase characters
- Numbers
- Special characters

### 3. IP Address Analyzer

Classifies IP addresses as:

- PRIVATE
- PUBLIC
- LOOPBACK
- INVALID

### 4. Incident Severity Classifier

Classifies cybersecurity incidents as:

- LOW
- MEDIUM
- HIGH

---

## Example Queries

### Phishing

"I received an urgent email asking me to verify my account and provide my password. Is this phishing?"

### Ransomware

"What should I do if ransomware has encrypted files on an employee workstation?"

### IP Analysis

"What is the security classification of IP address 192.168.1.10?"

---

## Example Results

### Phishing

Risk indicators are detected and the agent recommends avoiding suspicious links and protecting credentials.

### Ransomware

The system identifies ransomware as a HIGH severity incident and provides safe incident-response guidance.

### IP Address

192.168.1.10

Classification:

PRIVATE

---

## Knowledge Base

The system uses four cybersecurity knowledge documents:

- phishing.txt
- malware.txt
- network_security.txt
- incident_response.txt

---

## Running the Project

Activate the virtual environment:

```cmd
C:\Agentic_AI_Lab\venv\Scripts\activate.bat