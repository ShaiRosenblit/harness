from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

from . import killswitch
from .log import Log
from .types import (
    Budget,
    Limits,
    Policy,
    Seat,
    ToolCall,
    ToolResult,
    ToolSpec,
)


# ---- Shared run context ----------------------------------------------------


@dataclass
class RunCtx:
    log: Log
    workdir: Path
    kill_path: Path
    policy: Policy  # root policy; carries child_policy template for spawn
    live_seats: set = field(default_factory=set)
    chat_mode: bool = False  # if True, driver yields on text-only responses


# ---- Registry --------------------------------------------------------------


REGISTRY: Dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> None:
    REGISTRY[spec.name] = spec


def get_specs(names) -> list:
    return [REGISTRY[n] for n in names if n in REGISTRY]


# ---- Adjudicator (the single chokepoint for tool execution) ---------------


def adjudicate_and_run(seat: Seat, call: ToolCall, ctx: RunCtx) -> ToolResult:
    """Brief rule #4 step 7 / rule #5. Failures log denial (or halt for quota)
    and short-circuit; successes fall through to handler execution."""

    # 1. Kill check (belt + braces; driver also checks top-of-turn).
    reason = killswitch.check(seat, ctx.kill_path)
    if reason is not None:
        seat.halted = True
        seat.halt_reason = reason
        ctx.log.write(seat, "halt", {"reason": reason, "where": "adjudicator"})
        return ToolResult(ok=False, content="halted", error=reason)

    # 2. Capability check.
    if call.name not in seat.granted_tools:
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

    # 3. Shallow schema check: required keys present.
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

    # 4. Approval gate (v1: flagged set is empty; mechanism present).
    if spec.requires_approval:
        ctx.log.write(
            seat,
            "denial",
            {"reason": "approval_required", "tool": call.name, "call_id": call.id},
        )
        return ToolResult(
            ok=False, content="approval required", error="approval_required"
        )

    # 5. Quota pre-check.
    if seat.budget.usd_remaining <= 0 or seat.turns_used >= seat.limits.max_turns:
        seat.halted = True
        seat.halt_reason = "quota"
        ctx.log.write(
            seat,
            "halt",
            {
                "reason": "quota",
                "budget_usd_remaining": seat.budget.usd_remaining,
                "turns_used": seat.turns_used,
                "max_turns": seat.limits.max_turns,
            },
        )
        return ToolResult(ok=False, content="quota exhausted", error="quota")

    # 6. Log tool_call.
    ctx.log.write(
        seat,
        "tool_call",
        {"tool": call.name, "args": call.args, "call_id": call.id},
    )

    # 7. Execute (handler honors timeout).
    try:
        result = spec.handler(seat, call.args, ctx)
    except subprocess.TimeoutExpired:
        result = ToolResult(ok=False, content="timeout", error="timeout")
    except Exception as e:
        result = ToolResult(ok=False, content=f"exception: {e!r}", error="exception")

    # 8. Log tool_result.
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
        timeout=seat.limits.tool_timeout_s,
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
    """Brief rule #8. Synchronous v1: parent blocks on child."""
    # Lazy import to avoid driver<->tools cycle.
    from .driver import run_seat  # noqa: WPS433

    child_policy = ctx.policy.child_policy
    if child_policy is None:
        return ToolResult(
            ok=False,
            content="spawn not configured (no child_policy)",
            error="spawn_unconfigured",
        )

    prompt = args.get("prompt", "")
    # Models sometimes echo OpenAI's internal "functions." namespace prefix
    # when referencing tools. Strip it so attenuation matches our bare names.
    # Also: web tools live in the seat's `web` field, NOT in `granted_tools`.
    # If the model puts "web_search" / "web_fetch" in the spawn tools list
    # (a natural mistake), silently drop them — web is inherited from the
    # child_policy.web attenuated by parent.web, not selected via spawn args.
    _WEB_NAMES = {"web_search", "web_fetch",
                  "openrouter:web_search", "openrouter:web_fetch"}

    def _normalize(t: str) -> str:
        if t.startswith("functions."):
            t = t[len("functions."):]
        return t
    _seen: set = set()
    requested_tools_list: list = []
    for raw in args.get("tools") or []:
        t = _normalize(raw)
        if t in _WEB_NAMES or t in _seen:
            continue
        _seen.add(t)
        requested_tools_list.append(t)
    requested_tools = tuple(requested_tools_list)
    requested_budget = float(args.get("budget_usd", 0.0) or 0.0)
    requested_turns = int(args.get("max_turns", child_policy.limits.max_turns) or
                          child_policy.limits.max_turns)

    # Depth limit.
    new_depth = seat.depth + 1
    if new_depth > seat.limits.max_depth:
        return ToolResult(
            ok=False,
            content=f"depth limit: depth {new_depth} > max {seat.limits.max_depth}",
            error="depth_exceeded",
        )

    # Breadth limit.
    if seat.child_count >= seat.limits.max_children:
        return ToolResult(
            ok=False,
            content=f"breadth limit: already spawned {seat.child_count}",
            error="breadth_exceeded",
        )

    # Concurrent seats forest-wide.
    if len(ctx.live_seats) >= seat.limits.max_concurrent_seats:
        return ToolResult(
            ok=False,
            content="concurrent-seat limit reached",
            error="concurrency_exceeded",
        )

    # Capability attenuation: requested ⊆ parent.granted_tools AND ⊆ child_policy.tools.
    parent_set = set(seat.granted_tools)
    child_menu = set(child_policy.tools)
    if not requested_tools:
        # Default to the full child menu intersected with parent.
        granted_for_child = tuple(t for t in child_policy.tools if t in parent_set)
    else:
        bad = [t for t in requested_tools if t not in parent_set or t not in child_menu]
        if bad:
            return ToolResult(
                ok=False,
                content=f"capability attenuation violated for: {bad}",
                error="attenuation",
            )
        granted_for_child = requested_tools

    # Budget conservation: clamp + debit parent.
    if requested_budget <= 0:
        return ToolResult(
            ok=False, content="budget_usd must be > 0", error="bad_args"
        )
    clamped = min(requested_budget, seat.budget.usd_remaining)
    if clamped <= 0:
        return ToolResult(
            ok=False,
            content="parent has no remaining budget to allocate",
            error="budget_exhausted",
        )
    seat.budget.usd_remaining -= clamped

    seat.child_count += 1
    child_id = f"{seat.id}.{seat.child_count}"
    child_limits = Limits(
        max_turns=min(requested_turns, child_policy.limits.max_turns),
        max_depth=child_policy.limits.max_depth,
        max_children=child_policy.limits.max_children,
        max_concurrent_seats=seat.limits.max_concurrent_seats,  # forest-wide
        tool_timeout_s=child_policy.limits.tool_timeout_s,
    )
    # Web attenuation: child can only get web modes the parent already has.
    child_web = tuple(w for w in child_policy.web if w in seat.web)

    child = Seat(
        id=child_id,
        parent_id=seat.id,
        depth=new_depth,
        prompt=child_policy.system_prompt,
        granted_tools=granted_for_child,
        model=child_policy.model,
        limits=child_limits,
        budget=Budget(usd_remaining=clamped),
        history=[{"role": "user", "content": prompt}],
        web=child_web,
        web_max_results=child_policy.web_max_results,
        web_search_context_size=child_policy.web_search_context_size,
    )

    ctx.log.write(
        seat,
        "spawn",
        {
            "child_id": child_id,
            "prompt": prompt,
            "granted_tools": list(granted_for_child),
            "web": list(child_web),
            "budget_usd": clamped,
            "depth": new_depth,
        },
    )

    ctx.live_seats.add(child_id)
    try:
        child_result = run_seat(child, ctx)
    finally:
        ctx.live_seats.discard(child_id)

    final = child.submit_result if child.submit_result is not None else child_result.content
    return ToolResult(
        ok=child_result.ok and child.halt_reason == "submit",
        content=final or "",
        meta={
            "child_id": child_id,
            "child_halt_reason": child.halt_reason,
            "child_budget_remaining": child.budget.usd_remaining,
        },
    )


# ---- Registration ----------------------------------------------------------


def register_builtins() -> None:
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
            requires_approval=False,
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
            requires_approval=False,
            handler=_h_submit,
        )
    )
    register(
        ToolSpec(
            name="spawn",
            description=(
                "Spawn a sub-agent with a focused prompt and a budget. Blocks "
                "until the sub-agent submits. Returns its submitted result.\n"
                "- `tools` lists LOCAL tool names only (e.g. code_exec, "
                "submit). DO NOT list web_search/web_fetch here — your child "
                "inherits web access automatically from its policy template, "
                "attenuated by yours. If you omit `tools`, the child gets the "
                "default local toolset for its role.\n"
                "- `budget_usd` is debited from YOUR remaining budget "
                "immediately, even if the child does not spend it all. Keep "
                "enough for your own follow-up turns after the child returns. "
                "A reasonable per-child allocation is 20–40% of your "
                "remaining budget; never give a child more than half."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "tools": {"type": "array", "items": {"type": "string"}},
                    "budget_usd": {"type": "number"},
                    "max_turns": {"type": "integer"},
                },
                "required": ["prompt", "budget_usd"],
            },
            requires_approval=False,
            handler=_h_spawn,
        )
    )


register_builtins()
