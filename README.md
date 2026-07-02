# ShopMate RAG App, an LLM Observability Pipeline

An instrumented Retrieval-Augmented Generation (RAG) agent pipeline built with Python tracking integrations and monitored using Langfuse Cloud Analytics. Runs using Groq API inference execution models.

## Installation & Local Replication

### 1. Prerequisites & Environment Initialization
Ensure that Python 3.11+ is downloaded and the `uv` tool runner set up on the local machine. Initialize the environment:
```powershell
uv sync
```

### 2. Configure Your Secure Keys
Create a local `.env` configuration file in the project root directory. Do **not** commit this file to public version tracking:
```text
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_HOST="https://langfuse.com"
GROQ_API_KEY="gsk_..."
```

### 3. Run the Automated Tracking Routine
Execute the main test harness using `uv run` to guarantee isolated dependency alignment. This automatically routes 20 sequential analytical operations straight into the cloud dashboard:
```powershell
uv run python src/main.py
```

### 4. Code Constraints & Production Export
To extract production dependencies into a standard python package list, use the package builder command:
```powershell
uv pip freeze > requirements.txt
```
