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
        "## Sub-agent usage\n"
        "The only valid subagent name is 'general-purpose'. "
        "Only delegate to it for complex, multi-step repository analysis tasks. "
        "For simple conversational messages (greetings, short questions, clarifications), "
        "respond directly without calling any subagent or tool."
    ),
)
