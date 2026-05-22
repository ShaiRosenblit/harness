from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple


STARTER_PROMPT = (
    "You complete the given task using the available tools. "
    "You act by calling tools; you'll see each result and can continue. "
    "Call submit() with your answer when done."
)


@dataclass(frozen=True)
class Limits:
    max_turns: int
    max_depth: int
    max_children: int
    max_concurrent_seats: int
    tool_timeout_s: float


@dataclass
class Budget:
    usd_remaining: float


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
    requires_approval: bool
    handler: Callable[..., ToolResult]


@dataclass
class Seat:
    id: str
    parent_id: Optional[str]
    depth: int
    prompt: str
    granted_tools: Tuple[str, ...]
    model: str
    limits: Limits
    budget: Budget
    history: list = field(default_factory=list)
    turns_used: int = 0
    child_count: int = 0
    halted: bool = False
    halt_reason: Optional[str] = None
    submit_result: Optional[str] = None
    tokens_prompt: int = 0
    tokens_completion: int = 0
    # Web (server-side) tool grants. Subset of ("search", "fetch").
    # Resolved server-side by OpenRouter; the local adjudicator does not run them.
    web: Tuple[str, ...] = ()
    web_max_results: int = 4
    web_search_context_size: str = "low"  # "low" | "medium" | "high"
    web_searches: int = 0  # cumulative across this seat's turns


@dataclass(frozen=True)
class LogEntry:
    seq: int
    ts: float
    seat_id: str
    parent_id: Optional[str]
    type: str
    payload: dict


@dataclass(frozen=True)
class Policy:
    name: str
    model: str
    system_prompt: str
    tools: Tuple[str, ...]
    limits: Limits
    budget_usd: float
    child_policy: Optional["Policy"] = None
    # Server-side web tool grants (resolved by OpenRouter, not by us).
    # Subset of ("search", "fetch"). Empty tuple = no web access.
    web: Tuple[str, ...] = ()
    web_max_results: int = 4
    web_search_context_size: str = "low"  # "low" | "medium" | "high"
