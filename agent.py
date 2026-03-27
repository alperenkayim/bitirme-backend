from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from dotenv import load_dotenv
import os
import tempfile
import git

load_dotenv()

# To switch providers, change the model string:
#   Google AI Studio : "google_genai:gemini-2.5-flash"
#   OpenRouter       : set OPENAI_API_KEY + OPENAI_BASE_URL=https://openrouter.ai/api/v1
#                      then use "openai:<model-name>"
model = init_chat_model("google_genai:gemini-2.5-flash")

# Cache cloned repos so we don't re-clone on follow-up questions
_cloned_repos: dict[str, str] = {}

TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".txt", ".json",
    ".yaml", ".yml", ".toml", ".cfg", ".ini", ".sh", ".env.example",
    ".html", ".css", ".rs", ".go", ".java", ".cpp", ".c", ".h",
}
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}
MAX_FILE_BYTES = 3_000  # per file
MAX_TOTAL_CHARS = 25_000  # total output cap


@tool
def clone_and_read_repo(repo_url: str) -> str:
    """Clone a public GitHub repository and return its file tree plus contents of key source files.
    Use this whenever the user provides a GitHub URL and wants to understand the codebase."""
    if repo_url in _cloned_repos:
        tmp = _cloned_repos[repo_url]
    else:
        tmp = tempfile.mkdtemp()
        try:
            git.Repo.clone_from(repo_url, tmp, depth=1)
        except Exception as e:
            return f"Failed to clone {repo_url}: {e}"
        _cloned_repos[repo_url] = tmp

    # Build file tree
    tree_lines = []
    file_paths = []
    for root, dirs, files in os.walk(tmp):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        level = root.replace(tmp, "").count(os.sep)
        folder = os.path.basename(root) or os.path.basename(tmp)
        tree_lines.append("  " * level + folder + "/")
        for f in files:
            tree_lines.append("  " * (level + 1) + f)
            full = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            if ext in TEXT_EXTENSIONS:
                file_paths.append(full)

    tree = "\n".join(tree_lines[:400])

    # Read key files
    file_contents = []
    total_chars = 0
    for path in file_paths:
        if total_chars >= MAX_TOTAL_CHARS:
            break
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read(MAX_FILE_BYTES)
            rel = path.replace(tmp, "").lstrip(os.sep)
            snippet = f"\n{'='*60}\n# {rel}\n{'='*60}\n{content}"
            file_contents.append(snippet)
            total_chars += len(snippet)
        except Exception:
            continue

    return (
        f"Repository cloned: {repo_url}\n\n"
        f"FILE TREE:\n{tree}\n\n"
        f"FILE CONTENTS (key files):\n{''.join(file_contents)}"
    )


@tool
def read_repo_file(file_path: str) -> str:
    """Read a specific file from a previously cloned repository.
    Use the absolute path returned by clone_and_read_repo."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read(MAX_FILE_BYTES * 2)
    except Exception as e:
        return f"Error reading {file_path}: {e}"


agent = create_deep_agent(
    model=model,
    tools=[clone_and_read_repo, read_repo_file],
    system_prompt=(
        "You are an expert software repository analyst. "
        "When given a GitHub URL, use clone_and_read_repo to fetch the full file tree and source files. "
        "Then analyze the code directly from the returned content — do NOT try to use ls or read_file on repo paths, "
        "use read_repo_file instead if you need a specific file. "
        "Answer questions about architecture, code structure, entry points, and logic. "
        "Break down complex questions into subtasks and delegate to sub-agents when needed."
    ),
)
