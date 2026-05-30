from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


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


@dataclass
class BackgroundTask:
    """A spawn that runs concurrently with its parent instead of blocking it.

    The child runs its own driver loop on the background executor. When it
    finishes, a done-callback records `result` and pushes `id` into the
    parent seat's mailbox, so the parent's next turn surfaces it as a
    notification. `delivered` guards against double-delivery: whichever of
    the mailbox drain or an explicit await_task() fires first flips it, and
    the other side skips re-injecting the same result.
    """
    id: str
    child_id: str
    prompt: str
    future: Any                       # concurrent.futures.Future[ToolResult]
    result: Optional[ToolResult] = None
    done: bool = False
    delivered: bool = False


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
    autonomous: bool = False
    history: list = field(default_factory=list)
    turns_used: int = 0
    child_count: int = 0
    halted: bool = False
    halt_reason: Optional[str] = None
    submit_result: Optional[str] = None
    tokens_prompt: int = 0
    tokens_prompt_last: int = 0
    tokens_completion: int = 0
    cost_usd: float = 0.0
    web_searches: int = 0
    # Background spawns launched from this seat, keyed by task id. The
    # mailbox holds ids of tasks that have finished but not yet been
    # surfaced to the model; the driver drains it at the top of each turn.
    bg_tasks: Dict[str, "BackgroundTask"] = field(default_factory=dict)
    _mailbox: List[str] = field(default_factory=list)
    _mailbox_lock: threading.Lock = field(default_factory=threading.Lock)

    def push_mailbox(self, task_id: str) -> None:
        """Mark a finished background task for delivery. Called from the
        worker thread via the task's done-callback."""
        with self._mailbox_lock:
            self._mailbox.append(task_id)

    def drain_mailbox(self) -> List["BackgroundTask"]:
        """Pop all pending finished tasks that haven't been delivered yet.
        Flips each task's `delivered` flag so an explicit await_task() for
        the same id won't surface it a second time. Called from this seat's
        own driver thread at the top of a turn."""
        with self._mailbox_lock:
            ids = self._mailbox
            self._mailbox = []
        out: List["BackgroundTask"] = []
        for tid in ids:
            task = self.bg_tasks.get(tid)
            if task is not None and not task.delivered:
                task.delivered = True
                out.append(task)
        return out


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
    # When True, a text-only model response is treated as inner thinking
    # (loop continues) instead of an implicit submit. The only way to
    # surface text to the user becomes an explicit submit() call. Use
    # this for long-running autonomous loops that shouldn't halt the
    # moment the model emits prose.
    autonomous: bool = False
