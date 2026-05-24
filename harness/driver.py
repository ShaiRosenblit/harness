from __future__ import annotations

import json
from typing import List

from .tools import RunCtx, adjudicate_and_run, get_specs
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
    """Per-seat driver loop.

    Chat-vs-task mode is derived: a seat without `submit` in its tools is
    a chat seat — it yields to its caller (the user or its parent) on a
    text-only model response. A seat with `submit` is a task seat — text-
    only responses are nudged with a reminder to call a tool.

    Halts:
      - max_turns:    seat.turns_used >= seat.max_turns
      - submit:       agent called submit (seat.submit_result populated)
      - yield:        chat-mode seat replied with text only
    """
    from .model import call_model

    chat_mode = "submit" not in seat.tools

    with ctx._spawn_lock:
        ctx.live_seats.add(seat.id)
    last_result = ToolResult(ok=False, content="no turns executed", error="empty")
    try:
        while not seat.halted:
            # 1. Turn cap.
            if seat.turns_used >= seat.max_turns:
                seat.halted = True
                seat.halt_reason = "max_turns"
                ctx.log.write(
                    seat,
                    "halt",
                    {
                        "reason": "max_turns",
                        "turns_used": seat.turns_used,
                        "max_turns": seat.max_turns,
                    },
                )
                return ToolResult(ok=False, content="max_turns exhausted", error="max_turns")

            # 2. Model call.
            tool_specs = get_specs(seat.tools)
            resp = call_model(seat, tool_specs, ctx.log)
            seat.turns_used += 1
            seat.history.append(resp.raw_assistant_message)

            calls = _parse_tool_calls(resp.tool_calls)

            if not calls:
                if chat_mode:
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

            # 3. Execute tool calls. Parallel-within-turn for all-spawn batches.
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

            for call, result in zip(calls, results):
                if result is None:
                    break
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
        return last_result
    finally:
        with ctx._spawn_lock:
            ctx.live_seats.discard(seat.id)
