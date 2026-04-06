"""Metrics logger for the SWE-QA research paper experiments.

Tracks per-query tool calls, durations, and results for both agents.
Usage:
    from metrics_logger import MetricsLogger
    ml = MetricsLogger()
    ml.start_query("q1", "baseline", "What does this repo do?")
    ml.log_tool_call("search_rag", "What does this repo do?")
    ml.end_query("It is a ...")
    ml.save_to_json()
    ml.print_summary()
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class _ToolCallRecord:
    """Single tool invocation record."""

    tool_name: str
    timestamp: str
    input_preview: str


@dataclass
class _QueryRecord:
    """All data collected for one agent run on one query."""

    query_id: str
    agent_type: str
    query_text: str
    tool_calls: list[_ToolCallRecord] = field(default_factory=list)
    tool_call_count: int = 0
    duration_seconds: float = 0.0
    final_answer_length: int = 0
    success: bool = True
    timestamp: str = ""
    _start_time: float = field(default=0.0, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "query_id": self.query_id,
            "agent_type": self.agent_type,
            "query_text": self.query_text,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "timestamp": tc.timestamp,
                    "input_preview": tc.input_preview,
                }
                for tc in self.tool_calls
            ],
            "tool_call_count": self.tool_call_count,
            "duration_seconds": round(self.duration_seconds, 3),
            "final_answer_length": self.final_answer_length,
            "success": self.success,
            "timestamp": self.timestamp,
        }


class MetricsLogger:
    """Tracks per-query metrics for both agents."""

    def __init__(self) -> None:
        """Initialise an empty logger."""
        self._records: list[_QueryRecord] = []
        self._current: _QueryRecord | None = None

    def start_query(self, query_id: str, agent_type: str, query: str) -> None:
        """Call this when a new query begins."""
        self._current = _QueryRecord(
            query_id=query_id,
            agent_type=agent_type,
            query_text=query,
            timestamp=datetime.now(timezone.utc).isoformat(),
            _start_time=time.perf_counter(),
        )

    def log_tool_call(self, tool_name: str, input_preview: str) -> None:
        """Call this each time a tool is invoked."""
        if self._current is None:
            return
        record = _ToolCallRecord(
            tool_name=tool_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_preview=input_preview[:200],
        )
        self._current.tool_calls.append(record)
        self._current.tool_call_count += 1

    def end_query(self, final_answer: str, success: bool = True) -> None:
        """Call this when the agent finishes. Calculates duration."""
        if self._current is None:
            return
        self._current.duration_seconds = time.perf_counter() - self._current._start_time
        self._current.final_answer_length = len(final_answer)
        self._current.success = success
        self._records.append(self._current)
        self._current = None

    def save_to_json(self, filepath: str = "results/metrics.json") -> None:
        """Append this session's metrics to a JSON file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        existing: list[dict[str, Any]] = []
        if os.path.isfile(filepath):
            try:
                with open(filepath, encoding="utf-8") as fh:
                    existing = json.load(fh)
            except (json.JSONDecodeError, OSError):
                existing = []

        existing.extend(r.to_dict() for r in self._records)

        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2, ensure_ascii=False)

    def print_summary(self) -> None:
        """Print a human-readable summary of the most recent completed query."""
        if not self._records:
            return
        r = self._records[-1]
        success_mark = "[OK]" if r.success else "[FAIL]"
        print(
            f"Query: {r.query_text[:60]}\n"
            f"Agent: {r.agent_type} | Tool calls: {r.tool_call_count} | "
            f"Duration: {r.duration_seconds:.1f}s | Success: {success_mark}"
        )
