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
        "- get_repo_structure: get the directory tree of a repository (GitHub URL or local path)\n"
        "- read_file: read a specific file using its absolute path\n"
        "- search_rag: semantically search the codebase for relevant content\n\n"
        "## Strategy\n"
        "1. Start with get_repo_structure to understand the layout.\n"
        "2. Use search_rag to find relevant files for the question.\n"
        "3. Use read_file to read specific files in detail.\n"
        "4. Reason step by step and answer based only on what you find in the repository.\n\n"
        "You have NO other tools. Do not attempt shell commands, file writing, or planning."
    ),
)
