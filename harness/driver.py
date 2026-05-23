from __future__ import annotations

import json
from typing import List

from . import killswitch
from .tools import REGISTRY, RunCtx, adjudicate_and_run, get_specs
from .types import Seat, ToolCall, ToolResult


def _parse_tool_calls(model_tool_calls) -> List[ToolCall]:
    out: List[ToolCall] = []
    for tc in model_tool_calls:
        try:
            args = json.loads(tc.arguments_json or "{}")
            if not isinstance(args, dict):
                args = {}
        except json.JSONDecodeError:
            args = {}
        out.append(ToolCall(id=tc.id, name=tc.name, args=args))
    return out


def run_seat(seat: Seat, ctx: RunCtx) -> ToolResult:
    """The driver loop. Brief rule #4 — strict 10-step turn order."""
    # Lazy import keeps the dependency arrow driver -> model (not the reverse).
    from .model import call_model

    with ctx._spawn_lock:
        ctx.live_seats.add(seat.id)
    last_result = ToolResult(ok=False, content="no turns executed", error="empty")
    try:
        while not seat.halted:
            # 1. Kill check.
            reason = killswitch.check(seat, ctx.kill_path)
            if reason is not None:
                seat.halted = True
                seat.halt_reason = reason
                ctx.log.write(seat, "halt", {"reason": reason, "where": "driver_pre_turn"})
                return ToolResult(ok=False, content="halted", error=reason)

            # 2. Quota check.
            if (
                seat.budget.usd_remaining <= 0
                or seat.turns_used >= seat.limits.max_turns
            ):
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

            # 3 + 4 + 5. Build input + call model + log (handled by call_model).
            tool_specs = get_specs(seat.granted_tools)
            resp = call_model(seat, tool_specs, ctx.log)
            seat.turns_used += 1
            seat.history.append(resp.raw_assistant_message)

            # 6. Parse tool calls.
            calls = _parse_tool_calls(resp.tool_calls)

            if not calls:
                # Model produced an assistant message with no tool calls.
                if ctx.chat_mode:
                    # Chat mode: this is the agent's reply — yield to the user.
                    seat.halted = True
                    seat.halt_reason = "yield"
                    ctx.log.write(
                        seat,
                        "halt",
                        {"reason": "yield", "text": (resp.text or "")[:500]},
                    )
                    return ToolResult(
                        ok=True, content=(resp.text or ""), meta={"yield": True}
                    )
                # Task mode: nudge the model to actually call a tool.
                ctx.log.write(
                    seat,
                    "denial",
                    {"reason": "no_tool_call", "text": (resp.text or "")[:500]},
                )
                seat.history.append(
                    {
                        "role": "user",
                        "content": "Please call a tool. If you are done, call submit().",
                    }
                )
                continue

            # 7 + 8. Adjudicate + execute + log each tool call, append tool messages.
            #
            # Optimization: when the turn is an all-spawn batch (the common
            # fan-out pattern), run them concurrently and gather results in
            # the original order. Mixed batches stay serial to preserve the
            # model's likely intended ordering (e.g. code_exec → spawn).
            results: list = [None] * len(calls)
            if len(calls) > 1 and all(c.name == "spawn" for c in calls):
                futures = {
                    i: ctx.executor().submit(adjudicate_and_run, seat, c, ctx)
                    for i, c in enumerate(calls)
                }
                for i, fut in futures.items():
                    try:
                        results[i] = fut.result()
                    except Exception as e:
                        results[i] = ToolResult(
                            ok=False, content=f"worker exception: {e!r}",
                            error="exception",
                        )
            else:
                for i, call in enumerate(calls):
                    results[i] = adjudicate_and_run(seat, call, ctx)
                    if seat.halted:
                        break

            # Append tool messages in original order; stop on halt mid-loop.
            for call, result in zip(calls, results):
                if result is None:
                    break  # serial path broke early on halt
                last_result = result
                seat.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result.content,
                    }
                )
                if seat.halted:
                    break

            # 10. Loop until submit / halt.
        return last_result
    finally:
        with ctx._spawn_lock:
            ctx.live_seats.discard(seat.id)
