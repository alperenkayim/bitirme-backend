from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from agents.deep_agent.tools import (
    clone_and_read_repo,
    find_local_repos,
    read_local_repo,
    read_repo_file,
)
from langsmith_check import langsmith_check

load_dotenv()
langsmith_check()

model = init_chat_model("google_genai:gemini-2.5-flash")

agent = create_deep_agent(
    model=model,
    tools=[find_local_repos, clone_and_read_repo, read_local_repo, read_repo_file],
    system_prompt=(
        "You are an expert software repository analyst. "
        "When the user asks which repositories exist on their computer, use find_local_repos. "
        "When given a GitHub URL, use clone_and_read_repo to fetch the full file tree and source files. "
        "When given a local filesystem path, use read_local_repo instead. "
        "Use read_repo_file to read a specific file by its absolute path.\n\n"
        "## Filesystem tool rules\n"
        "IMPORTANT: The built-in ls, read_file, write_file tools operate on a virtual workspace, "
        "NOT on the real filesystem. For reading real files on disk, always use read_repo_file "
        "with the absolute path. Never call the built-in read_file with a /Users/... path.\n\n"
        "## Analysis strategy\n"
        "1. Start with clone_and_read_repo or read_local_repo to get the file tree and key file contents.\n"
        "2. Use read_repo_file for specific files that need deeper inspection.\n"
        "3. Save every file you read to the Virtual Filesystem so sub-agents can reuse it without re-reading.\n\n"
        "## Sub-agent delegation\n"
        "The only valid subagent name is 'general-purpose'.\n"
        "OPEN a sub-agent for: 'Where' questions (multi-file search), 'How' questions (cross-file implementation), architecture questions.\n"
        "DO NOT open a sub-agent for: 'What is X' questions (single file or docstring is enough), 'Why' questions, greetings, clarifications.\n\n"
        "## Answer format\n"
        "Always structure your final answer as:\n"
        "- Summary (2-3 sentences)\n"
        "- Evidence (file:line references)\n"
        "- Explanation (detail)"
    ),
)
