"""Smoke test script: runs 5 SWE-QA sample questions through one or both agents.

Usage:
    python run_test.py                # runs both agents
    python run_test.py --agent baseline
    python run_test.py --agent deep
"""

import argparse
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# Disable tracing early if API key is missing — suppresses background 403 spam
from langsmith_check import langsmith_check  # noqa: E402
langsmith_check()

SAMPLE_QUESTIONS: list[dict[str, str]] = [
    {
        "id": "q1",
        "type": "what",
        "text": "What are the main modules and their responsibilities in this repository?",
    },
    {
        "id": "q2",
        "type": "why",
        "text": "Why does the error handling use this specific pattern across the codebase?",
    },
    {
        "id": "q3",
        "type": "where",
        "text": "Where is the main entry point for the application defined?",
    },
    {
        "id": "q4",
        "type": "how",
        "text": "How does the authentication flow work from request to response?",
    },
    {
        "id": "q5",
        "type": "arch",
        "text": "Explain the overall architecture and design patterns used in this repository.",
    },
]


def _run_agent(
    agent_graph,  # CompiledStateGraph
    question: dict[str, str],
    agent_label: str,
    logger,
) -> dict[str, object]:
    """Run one question through an agent graph, record metrics, and return a result dict."""
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import HumanMessage

    class _ToolTracker(BaseCallbackHandler):
        """Callback that forwards tool invocations to the MetricsLogger."""

        def on_tool_start(
            self,
            serialized: dict,
            input_str: str,
            **kwargs: object,
        ) -> None:
            """Record a tool call when the tool starts."""
            tool_name = serialized.get("name", "unknown")
            logger.log_tool_call(tool_name, input_str[:200])

    logger.start_query(question["id"], agent_label, question["text"])
    answer = ""
    success = True

    try:
        result = agent_graph.invoke(
            {"messages": [HumanMessage(content=question["text"])]},
            config={"recursion_limit": 50, "callbacks": [_ToolTracker()]},
        )
        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            answer = getattr(last, "content", str(last))
        else:
            answer = str(result)
    except Exception as exc:
        answer = f"ERROR: {exc}"
        success = False

    logger.end_query(answer, success=success)
    return {
        "id": question["id"],
        "type": question["type"],
        "agent": agent_label,
        "tool_calls": logger._records[-1].tool_call_count,
        "duration": logger._records[-1].duration_seconds,
        "success": success,
    }


def _print_table(rows: list[dict[str, object]]) -> None:
    """Print a comparison table of baseline vs deep agent results (ASCII-safe)."""
    by_id: dict[str, dict[str, dict]] = {}
    for r in rows:
        qid = str(r["id"])
        agent = str(r["agent"])
        by_id.setdefault(qid, {})
        by_id[qid][agent] = r
        by_id[qid]["type"] = str(r["type"])

    sep = "+" + "-" * 5 + "+" + "-" * 6 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 17 + "+"
    header_row = (
        f"| {'ID':<3} | {'Type':<4} | {'Baseline':<12} | {'Deep Agent':<12} | {'Winner':<15} |"
    )
    sub_row = (
        f"| {'':<3} | {'':<4} | {'calls/time':<12} | {'calls/time':<12} | {'(by tool count)':<15} |"
    )
    print(sep)
    print(header_row)
    print(sub_row)
    print(sep)

    for qid, data in sorted(by_id.items()):
        qtype = data["type"]
        baseline = data.get("baseline")
        deep = data.get("deep")

        def _fmt(r: dict | None) -> str:
            if r is None:
                return "n/a".ljust(12)
            return f"{r['tool_calls']} / {r['duration']:.1f}s".ljust(12)

        def _winner(b: dict | None, d: dict | None) -> str:
            if b is None and d is None:
                return "n/a"
            if b is None:
                return "Deep"
            if d is None:
                return "Baseline"
            b_calls = int(b["tool_calls"])
            d_calls = int(d["tool_calls"])
            if b_calls < d_calls:
                return "Baseline"
            if d_calls < b_calls:
                return "Deep Agent"
            return "Tie"

        print(
            f"| {qid:<3} | {qtype:<4} | {_fmt(baseline)} | {_fmt(deep)} | {_winner(baseline, deep):<15} |"
        )

    print(sep)


def main() -> None:
    """Entry point: parse args, run agents, save results, print table."""
    parser = argparse.ArgumentParser(description="SWE-QA smoke test")
    parser.add_argument(
        "--agent",
        choices=["baseline", "deep", "both"],
        default="both",
        help="Which agent to run (default: both)",
    )
    args = parser.parse_args()

    from metrics_logger import MetricsLogger

    ml = MetricsLogger()
    all_rows: list[dict[str, object]] = []

    run_baseline = args.agent in ("baseline", "both")
    run_deep = args.agent in ("deep", "both")

    if run_baseline:
        print("Loading baseline agent...")
        from agents.baseline_agent.agent import agent as baseline_agent
        print("Baseline agent loaded.\n")

        for q in SAMPLE_QUESTIONS:
            print(f"[baseline] Running {q['id']}: {q['text'][:60]}...")
            row = _run_agent(baseline_agent, q, "baseline", ml)
            all_rows.append(row)
            ml.print_summary()
            print()

    if run_deep:
        print("Loading deep agent...")
        from agents.deep_agent.agent import agent as deep_agent
        print("Deep agent loaded.\n")

        for q in SAMPLE_QUESTIONS:
            print(f"[deep] Running {q['id']}: {q['text'][:60]}...")
            row = _run_agent(deep_agent, q, "deep", ml)
            all_rows.append(row)
            ml.print_summary()
            print()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/smoke_test_{timestamp}.json"
    ml.save_to_json(out_path)
    print(f"\nResults saved to: {out_path}")

    # Print comparison table
    print()
    _print_table(all_rows)


if __name__ == "__main__":
    main()
