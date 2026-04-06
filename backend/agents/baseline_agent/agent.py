from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from agents.baseline_agent.tools import get_repo_structure, read_file, search_rag
from langsmith_check import langsmith_check

load_dotenv()
langsmith_check()

model = init_chat_model("google_genai:gemini-2.5-flash")

agent = create_agent(
    model,
    tools=[read_file, get_repo_structure, search_rag],
    system_prompt=(
        "You are a software repository analyst. "
        "Answer questions about code architecture, logic, and structure by exploring the repository.\n\n"
        "## Available tools\n"
        "- search_rag: semantically search the codebase for relevant content\n"
        "- get_repo_structure: get the directory tree of a repository (GitHub URL or local path)\n"
        "- read_file: read a specific file using its absolute path\n\n"
        "## Strategy\n"
        "1. Use search_rag FIRST — it directly finds relevant code without knowing file paths.\n"
        "2. Use get_repo_structure when you need to understand the overall layout or locate a module.\n"
        "3. Use read_file to inspect specific files in detail.\n"
        "4. Reason step by step and answer based only on what you find in the repository.\n\n"
        "## Answer format\n"
        "Always structure your final answer as:\n"
        "- Summary (2-3 sentences)\n"
        "- Evidence (file:line references)\n"
        "- Explanation (detail)\n\n"
        "You have NO other tools. Do not attempt shell commands, file writing, or planning."
    ),
)
