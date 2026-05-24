from __future__ import annotations

import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from .log import Log
from .types import (
    Agent,
    Seat,
    ToolCall,
    ToolResult,
    ToolSpec,
)


# ---- Shared run context ----------------------------------------------------


@dataclass
class ApprovalRequest:
    """A pending request for the user to approve or deny a risky tool call.

    The worker thread blocks on `event`; the UI sets `decision` (to
    "approve" or "deny") and then `event.set()` to release it.
    `displayed` is a one-shot flag the UI flips when it first shows the
    request, so polling won't double-print.
    """
    id: str
    seat_id: str
    tool_name: str
    args: dict
    event: threading.Event
    decision: Optional[str] = None   # None | "approve" | "deny"
    displayed: bool = False


@dataclass
class RunCtx:
    log: Log
    workdir: Path
    agent: Agent
    live_seats: set = field(default_factory=set)
    _spawn_lock: threading.Lock = field(default_factory=threading.Lock)
    _executor: Optional[ThreadPoolExecutor] = None
    # Approval gate: pending and resolved requests for risky tool calls.
    _approvals: Dict[str, ApprovalRequest] = field(default_factory=dict)
    _approval_seq: int = 0
    _approvals_lock: threading.Lock = field(default_factory=threading.Lock)

    def executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=8, thread_name_prefix="harness-spawn"
            )
        return self._executor

    def request_approval(self, seat_id: str, tool_name: str, args: dict) -> ApprovalRequest:
        """Register a pending approval; caller blocks on req.event.wait()."""
        with self._approvals_lock:
            self._approval_seq += 1
            req = ApprovalRequest(
                id=f"a{self._approval_seq}",
                seat_id=seat_id,
                tool_name=tool_name,
                args=args,
                event=threading.Event(),
            )
            self._approvals[req.id] = req
        return req

    def resolve_approval(self, approval_id: str, decision: str) -> Optional[ApprovalRequest]:
        """Set the decision and release the waiting worker. Returns the
        request if it was pending; None if not found or already resolved."""
        if decision not in ("approve", "deny"):
            return None
        with self._approvals_lock:
            req = self._approvals.get(approval_id)
            if req is None or req.decision is not None:
                return None
            req.decision = decision
        req.event.set()
        return req

    def pending_undisplayed(self) -> list:
        """Return pending approvals that haven't been shown to the user
        yet, and mark them displayed so we don't re-show on next poll."""
        out = []
        with self._approvals_lock:
            for req in self._approvals.values():
                if req.decision is None and not req.displayed:
                    req.displayed = True
                    out.append(req)
        return out


# ---- Seat factory (shared by forest, session, spawn) ----------------------


def mint_seat(
    agent: Agent,
    seat_id: str,
    parent_id: Optional[str],
    depth: int,
    history: list,
) -> Seat:
    return Seat(
        id=seat_id,
        parent_id=parent_id,
        depth=depth,
        model=agent.model,
        system_prompt=agent.system_prompt,
        tools=agent.tools,
        web=agent.web,
        max_turns=agent.max_turns,
        max_depth=agent.max_depth,
        max_children=agent.max_children,
        tool_timeout_s=agent.tool_timeout_s,
        web_max_results=agent.web_max_results,
        web_search_context_size=agent.web_search_context_size,
        history=history,
    )


# ---- Registry --------------------------------------------------------------


REGISTRY: Dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> None:
    REGISTRY[spec.name] = spec


def get_specs(names) -> list:
    return [REGISTRY[n] for n in names if n in REGISTRY]


# ---- Adjudicator -----------------------------------------------------------


def adjudicate_and_run(seat: Seat, call: ToolCall, ctx: RunCtx) -> ToolResult:
    """Capability check → schema check → execute → log. All in one place."""
    # Capability check.
    if call.name not in seat.tools:
        ctx.log.write(
            seat,
            "denial",
            {"reason": "not_granted", "tool": call.name, "call_id": call.id},
        )
        return ToolResult(
            ok=False,
            content=f"tool not granted: {call.name}",
            error="not_granted",
        )

    spec = REGISTRY.get(call.name)
    if spec is None:
        ctx.log.write(
            seat,
            "denial",
            {"reason": "unknown_tool", "tool": call.name, "call_id": call.id},
        )
        return ToolResult(
            ok=False, content=f"unknown tool: {call.name}", error="unknown_tool"
        )

    # Shallow schema check: required keys present.
    required = (spec.parameters or {}).get("required", []) or []
    missing = [k for k in required if k not in (call.args or {})]
    if missing:
        ctx.log.write(
            seat,
            "denial",
            {
                "reason": "bad_args",
                "tool": call.name,
                "missing": missing,
                "call_id": call.id,
            },
        )
        return ToolResult(
            ok=False,
            content=f"missing required args: {missing}",
            error="bad_args",
        )

    ctx.log.write(
        seat,
        "tool_call",
        {"tool": call.name, "args": call.args, "call_id": call.id},
    )

    # Approval gate for risky tools. Blocks the worker thread until the
    # user types /approve <id> or /deny <id> in the UI.
    if spec.risky:
        req = ctx.request_approval(seat.id, call.name, call.args or {})
        ctx.log.write(
            seat,
            "approval_request",
            {"approval_id": req.id, "tool": call.name, "args": call.args, "call_id": call.id},
        )
        req.event.wait()   # ← blocks here until UI resolves
        ctx.log.write(
            seat,
            "approval_decision",
            {"approval_id": req.id, "decision": req.decision, "tool": call.name},
        )
        if req.decision != "approve":
            result = ToolResult(
                ok=False,
                content=f"denied by user (approval {req.id})",
                error="denied",
            )
            ctx.log.write(
                seat,
                "tool_result",
                {
                    "tool": call.name,
                    "call_id": call.id,
                    "ok": False,
                    "error": "denied",
                    "content": result.content,
                    "meta": {},
                },
            )
            return result

    try:
        result = spec.handler(seat, call.args, ctx)
    except subprocess.TimeoutExpired:
        result = ToolResult(ok=False, content="timeout", error="timeout")
    except Exception as e:
        result = ToolResult(ok=False, content=f"exception: {e!r}", error="exception")

    truncated = result.content if len(result.content) <= 16000 else (
        result.content[:16000] + "...[truncated]"
    )
    ctx.log.write(
        seat,
        "tool_result",
        {
            "tool": call.name,
            "call_id": call.id,
            "ok": result.ok,
            "error": result.error,
            "content": truncated,
            "meta": result.meta,
        },
    )

    return result


# ---- Built-in tools --------------------------------------------------------


def _h_code_exec(seat: Seat, args: dict, ctx: RunCtx) -> ToolResult:
    code = args.get("code", "")
    proc = subprocess.run(
        ["python3", "-c", code],
        cwd=str(ctx.workdir),
        capture_output=True,
        text=True,
        timeout=seat.tool_timeout_s,
    )
    out = (proc.stdout or "")[-8000:]
    err = (proc.stderr or "")[-2000:]
    content = f"stdout:\n{out}\nstderr:\n{err}\nexit: {proc.returncode}"
    return ToolResult(
        ok=(proc.returncode == 0),
        content=content,
        meta={"exit": proc.returncode},
    )


def _h_bash(seat: Seat, args: dict, ctx: RunCtx) -> ToolResult:
    cmd = args.get("command", "")
    proc = subprocess.run(
        ["bash", "-c", cmd],
        cwd=str(ctx.workdir),
        capture_output=True,
        text=True,
        timeout=seat.tool_timeout_s,
    )
    out = (proc.stdout or "")[-8000:]
    err = (proc.stderr or "")[-2000:]
    content = f"stdout:\n{out}\nstderr:\n{err}\nexit: {proc.returncode}"
    return ToolResult(
        ok=(proc.returncode == 0),
        content=content,
        meta={"exit": proc.returncode},
    )


def _h_submit(seat: Seat, args: dict, ctx: RunCtx) -> ToolResult:
    result = args.get("result", "")
    seat.submit_result = result if isinstance(result, str) else json.dumps(result)
    seat.halted = True
    seat.halt_reason = "submit"
    ctx.log.write(seat, "submit", {"result": seat.submit_result})
    ctx.log.write(seat, "halt", {"reason": "submit"})
    return ToolResult(ok=True, content=seat.submit_result, meta={"submitted": True})


def _h_spawn(seat: Seat, args: dict, ctx: RunCtx) -> ToolResult:
    """Spawn a same-kind child: a copy of the parent's seat config at
    depth+1, with a fresh history starting from `prompt`. Bounded by
    `max_depth` (recursion) and `max_children` (breadth per seat)."""
    prompt = args.get("prompt", "")

    new_depth = seat.depth + 1
    if new_depth > seat.max_depth:
        return ToolResult(
            ok=False,
            content=f"depth limit: depth {new_depth} > max {seat.max_depth}",
            error="depth_exceeded",
        )

    # Critical section: breadth, child_count, child id mint, live seats.
    with ctx._spawn_lock:
        if seat.child_count >= seat.max_children:
            if seat.max_children <= 0:
                msg = (
                    "spawn unavailable: this agent's max_children is 0. "
                    "Relaunch with --max-children N (and --max-depth N if it's 0)."
                )
            else:
                msg = (
                    f"breadth limit: this agent's max_children={seat.max_children}, "
                    f"already spawned {seat.child_count}. Relaunch with a higher "
                    f"--max-children if you need more siblings."
                )
            return ToolResult(ok=False, content=msg, error="breadth_exceeded")
        seat.child_count += 1
        child_id = f"{seat.id}.{seat.child_count}"
        ctx.live_seats.add(child_id)

    # Mint a child with the parent's exact config (capability attenuation
    # is automatic — child == parent).
    child = Seat(
        id=child_id,
        parent_id=seat.id,
        depth=new_depth,
        model=seat.model,
        system_prompt=seat.system_prompt,
        tools=seat.tools,
        web=seat.web,
        max_turns=seat.max_turns,
        max_depth=seat.max_depth,
        max_children=seat.max_children,
        tool_timeout_s=seat.tool_timeout_s,
        web_max_results=seat.web_max_results,
        web_search_context_size=seat.web_search_context_size,
        history=[{"role": "user", "content": prompt}],
    )

    ctx.log.write(
        seat,
        "spawn",
        {
            "child_id": child_id,
            "prompt": prompt,
            "depth": new_depth,
        },
    )

    from .driver import run_seat  # lazy to avoid cycle

    try:
        child_result = run_seat(child, ctx)
    finally:
        with ctx._spawn_lock:
            ctx.live_seats.discard(child_id)

    final = child.submit_result if child.submit_result is not None else child_result.content
    return ToolResult(
        ok=child_result.ok and child.halt_reason == "submit",
        content=final or "",
        meta={
            "child_id": child_id,
            "child_halt_reason": child.halt_reason,
        },
    )


# ---- Registration ----------------------------------------------------------


def register_builtins() -> None:
    register(
        ToolSpec(
            name="bash",
            description=(
                "Run an arbitrary shell command via `bash -c`. Returns "
                "stdout, stderr, and exit code. This tool is gated — each "
                "call requires explicit user approval before it runs. "
                "Prefer code_exec for Python work; use bash when you need "
                "the shell (file ops, installing packages, running other "
                "binaries, pipelines)."
            ),
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
            handler=_h_bash,
            risky=True,
        )
    )
    register(
        ToolSpec(
            name="code_exec",
            description=(
                "Execute Python source in a shared scratch directory. "
                "Returns stdout, stderr, and exit code."
            ),
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
            handler=_h_code_exec,
        )
    )
    register(
        ToolSpec(
            name="submit",
            description="End this seat and return the given result to the caller.",
            parameters={
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
            },
            handler=_h_submit,
        )
    )
    register(
        ToolSpec(
            name="spawn",
            description=(
                "Spawn a sub-agent to handle a focused sub-task. The child "
                "is a fresh copy of you (same model, prompt, tools, limits) "
                "at depth+1, running its own driver loop from a fresh "
                "history starting with `prompt`. Blocks until the child "
                "submits, then returns its submitted result to you. "
                "Bounded by your max_depth (recursion) and max_children "
                "(siblings per turn)."
            ),
            parameters={
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"],
            },
            handler=_h_spawn,
        )
    )


register_builtins()
