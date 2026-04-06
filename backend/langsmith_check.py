"""LangSmith connection check.

Importable as a module (call langsmith_check()) or runnable standalone.
Prints a clear status message — never crashes the application.
"""

import os
import sys


def _print(msg: str) -> None:
    """Print with UTF-8 encoding, falling back to ASCII-safe output."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


def langsmith_check() -> bool:
    """Verify LangSmith connectivity and print a status message.

    Returns True if the connection is alive, False otherwise.
    """
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    project = os.getenv("LANGCHAIN_PROJECT", "sweqa-codebase-chatbot")
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower()

    if not api_key or api_key == "<I_WILL_FILL_THIS>":
        # Disable tracing at runtime so the SDK doesn't spam 403 errors
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        _print(
            f"[WARN] LangSmith | project='{project}' | "
            "LANGCHAIN_API_KEY is missing or not set -- tracing disabled.\n"
            "  -> Set LANGCHAIN_API_KEY in your .env file to enable tracing."
        )
        return False

    if tracing != "true":
        _print(
            f"[WARN] LangSmith | project='{project}' | "
            "LANGCHAIN_TRACING_V2 is not 'true' -- tracing disabled."
        )
        return False

    try:
        from langsmith import Client
        client = Client(api_key=api_key)
        # A lightweight call to verify the key is valid
        list(client.list_projects(limit=1))
        _print(f"[OK] LangSmith | project='{project}' | tracing enabled and connection verified.")
        return True
    except Exception as exc:
        _print(
            f"[FAIL] LangSmith | project='{project}' | "
            f"Connection failed: {exc}\n"
            "  -> Check your LANGCHAIN_API_KEY and network access."
        )
        return False


if __name__ == "__main__":
    langsmith_check()
