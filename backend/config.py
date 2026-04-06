"""Centralized settings for the sweqa-codebase-chatbot backend.

Import the global singleton:
    from config import settings
"""

from dataclasses import dataclass, field


@dataclass
class Settings:
    """All tuneable parameters in one place — no pydantic required."""

    # LangSmith
    langsmith_project: str = "sweqa-codebase-chatbot"

    # RAG parameters
    rag_k: int = 8
    rag_chunk_size: int = 1500
    rag_chunk_overlap: int = 200
    rag_score_threshold: float = 0.3

    # Repo structure tool
    repo_max_depth: int = 4
    repo_ignore_patterns: list = field(default_factory=lambda: [
        ".git", "__pycache__", "node_modules", ".venv", "*.pyc",
        "*.egg-info", "dist", "build", ".pytest_cache",
    ])

    # Agent limits
    baseline_max_iterations: int = 10
    deep_agent_max_iterations: int = 20


# Global singleton — import this everywhere
settings = Settings()
