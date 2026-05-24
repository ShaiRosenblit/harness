from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ToolResult:
    ok: bool
    content: str
    error: Optional[str] = None
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., ToolResult]
    # When True, calls to this tool are gated by the approval mechanism:
    # the worker blocks until the user explicitly /approve or /deny it.
    risky: bool = False


@dataclass
class Seat:
    """One running agent in the forest. Children spawned from this seat
    inherit the same config fields (model, system_prompt, tools, web,
    max_turns, max_depth, max_children, tool_timeout_s, web_*) at depth+1.
    Recursion is bounded by max_depth.

    Token/cost counters are observability only — not enforced.
    """
    id: str
    parent_id: Optional[str]
    depth: int
    model: str
    system_prompt: str
    tools: Tuple[str, ...]
    web: Tuple[str, ...]
    max_turns: int
    max_depth: int
    max_children: int
    tool_timeout_s: float
    web_max_results: int
    web_search_context_size: str
    provider: Tuple[str, ...] = ()
    history: list = field(default_factory=list)
    turns_used: int = 0
    child_count: int = 0
    halted: bool = False
    halt_reason: Optional[str] = None
    submit_result: Optional[str] = None
    tokens_prompt: int = 0
    tokens_completion: int = 0
    cost_usd: float = 0.0
    web_searches: int = 0


@dataclass(frozen=True)
class LogEntry:
    seq: int
    ts: float
    seat_id: str
    parent_id: Optional[str]
    type: str
    payload: dict


@dataclass(frozen=True)
class Agent:
    """An agent config — what model, what prompt, what capabilities, what
    structural limits. The harness mints a seat from this for a run.

    When a seat spawns, the child inherits this exact config (same model,
    same prompt, same tools, same web grants) at depth+1. There is no
    separate child template — children are copies of the parent. Recursion
    is bounded by max_depth.
    """
    model: str
    system_prompt: str
    tools: Tuple[str, ...]   # subset of: code_exec, submit, spawn
    max_turns: int = 12
    max_depth: int = 1
    max_children: int = 3
    tool_timeout_s: float = 15.0
    web: Tuple[str, ...] = ()   # subset of: search, fetch  (resolved by OpenRouter)
    web_max_results: int = 4
    web_search_context_size: str = "low"   # "low" | "medium" | "high"
    # OpenRouter provider slugs to restrict routing to (in preference order).
    # Empty = let OpenRouter pick. Useful when a model's tool-call template
    # is only parsed correctly by a subset of providers — e.g. Kimi K2.6
    # served by some providers leaks `<|tool_call_begin|>...` into text.
    provider: Tuple[str, ...] = ()
