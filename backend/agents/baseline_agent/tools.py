"""Tools for the Baseline (ReAct-style) Agent.

Strictly limited to the three tools defined in the SWE-QA paper:
  1. read_file
  2. get_repo_structure
  3. search_rag

No shell access, no planning, no filesystem writing, no extra utilities.
"""

import fnmatch
import logging
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import settings
from shared.repo_utils import build_file_tree, clone_repo, read_text_file

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 12_000

# In-memory RAG index cache keyed by repo path
_rag_indexes: dict[str, FAISS] = {}

# Separate embeddings for indexing vs. querying — different task_type yields better retrieval
_doc_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    task_type="RETRIEVAL_DOCUMENT",
)
_query_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    task_type="RETRIEVAL_QUERY",
)


def _get_or_build_index(repo_path: str) -> FAISS:
    """Build (or return cached) a FAISS index for all text files in repo_path."""
    if repo_path in _rag_indexes:
        return _rag_indexes[repo_path]

    _, file_paths = build_file_tree(repo_path)
    docs: list[Document] = []
    for path in file_paths:
        content = read_text_file(path, MAX_FILE_BYTES)
        if content.strip():
            rel = str(Path(path).relative_to(repo_path))
            docs.append(Document(page_content=content, metadata={"source": rel, "abs_path": path}))

    index = FAISS.from_documents(docs, _doc_embeddings)
    _rag_indexes[repo_path] = index
    return index


def _resolve_repo(repo_url_or_path: str) -> tuple[str, None] | tuple[None, str]:
    """Return local path for a URL (clones if needed) or a local path as-is."""
    if repo_url_or_path.startswith("http://") or repo_url_or_path.startswith("https://"):
        return clone_repo(repo_url_or_path)
    if Path(repo_url_or_path).is_dir():
        return repo_url_or_path, None
    return None, f"Path not found: {repo_url_or_path}"


@tool
def read_file(file_path: str) -> str:
    """Read the contents of a specific file in the repository.
    Provide the absolute path to the file."""
    if not Path(file_path).is_file():
        return f"File not found: {file_path}"
    return read_text_file(file_path, MAX_FILE_BYTES)


@tool
def get_repo_structure(repo_url_or_path: str) -> str:
    """Returns the file and directory structure of the target repository.
    Use this to understand the repo layout and locate relevant modules
    before diving into specific files. Output is depth-limited and
    filters out build artifacts, caches, and binary files.
    Returns a summary of file types at the top.
    """
    local_path, error = _resolve_repo(repo_url_or_path)
    if error:
        return error

    repo_name = Path(local_path).name
    tree_lines, file_paths = _build_filtered_tree(local_path)

    # File type summary
    py_count = sum(1 for p in file_paths if p.endswith(".py") and "test" not in p.lower())
    test_count = sum(1 for p in file_paths if "test" in p.lower() and p.endswith(".py"))
    config_count = sum(
        1 for p in file_paths
        if Path(p).suffix.lower() in {".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
    )
    other_count = len(file_paths) - py_count - test_count - config_count

    summary = (
        f"=== Repository: {repo_name} ===\n"
        f"Python files: {py_count} | Test files: {test_count} | "
        f"Config files: {config_count} | Other: {other_count}\n"
        f"{'─' * 40}"
    )
    tree_str = "\n".join(tree_lines)
    return f"{summary}\n{tree_str}"


def _build_filtered_tree(root_path: str) -> tuple[list[str], list[str]]:
    """Walk the directory up to settings.repo_max_depth, filtering ignored patterns.

    Returns (tree_lines, list_of_file_paths).
    """
    root = Path(root_path)
    tree_lines: list[str] = []
    file_paths: list[str] = []
    ignore = settings.repo_ignore_patterns
    max_depth = settings.repo_max_depth

    def _is_ignored(name: str) -> bool:
        return any(fnmatch.fnmatch(name, pat) for pat in ignore)

    def _walk(current: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(current.iterdir(), key=lambda e: (e.is_file(), e.name))
        except PermissionError:
            return

        dirs = [e for e in entries if e.is_dir() and not _is_ignored(e.name)]
        files = [e for e in entries if e.is_file() and not _is_ignored(e.name)]

        indent = "  " * depth
        for d in dirs:
            tree_lines.append(f"{indent}{d.name}/")
            if depth < max_depth:
                _walk(d, depth + 1)
            else:
                # Count hidden items beyond max depth
                try:
                    n = sum(1 for _ in d.iterdir())
                    tree_lines.append(f"{indent}  ... ({n} more items)")
                except PermissionError:
                    pass

        for f in files:
            tree_lines.append(f"{indent}{f.name}  [path: {f}]")
            file_paths.append(str(f))

    _walk(root, 0)
    return tree_lines, file_paths


@tool
def search_rag(query: str, repo_url_or_path: str) -> str:
    """Semantic search over the repository codebase using RAG.
    Use this FIRST before read_file — it efficiently finds relevant
    code snippets across the entire repo without knowing exact file paths.
    Best for: finding function implementations, class definitions,
    understanding patterns used across multiple files.
    Input: a natural language query describing what you're looking for.
    Returns: the most relevant code chunks with file paths and line numbers.
    """
    logger.info("search_rag called | query='%s' | k=%d", query[:80], settings.rag_k)

    local_path, error = _resolve_repo(repo_url_or_path)
    if error:
        return error

    try:
        index = _get_or_build_index(local_path)
    except Exception as e:
        return f"Failed to build search index: {e}"

    # Retrieve with scores so we can filter
    scored = index.similarity_search_with_score(
        query, k=settings.rag_k,
        embedding=_query_embeddings.embed_query(query),
    )

    # Filter below threshold, but keep top 2 as safety fallback
    above = [(doc, score) for doc, score in scored if score >= settings.rag_score_threshold]
    results = above if above else scored[:2]

    logger.info(
        "search_rag returned %d chunks | scores: %s",
        len(results),
        [round(score, 3) for _, score in results],
    )

    if not results:
        return "No relevant results found."

    parts = []
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] {source} (score={score:.3f})\n{'-'*40}\n{doc.page_content}")
    return "\n\n".join(parts)
