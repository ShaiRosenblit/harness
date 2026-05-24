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
    from .compact import maybe_compact
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

            # 2. Compact history if the previous turn's prompt got large.
            # Done before the next model call (not right after the previous
            # one) so the full assistant+tool-result pairs are present in
            # history at the time of summarization — collapsing in the
            # middle would leave orphaned tool_call_ids.
            maybe_compact(seat, seat.tokens_prompt_last, ctx.log)

            # 3. Model call.
            tool_specs = get_specs(seat.tools)
            resp = call_model(seat, tool_specs, ctx.log)
            seat.turns_used += 1
            seat.history.append(resp.raw_assistant_message)

            calls = _parse_tool_calls(resp.tool_calls)

            if not calls:
                text = (resp.text or "").strip()
                if chat_mode:
                    seat.halted = True
                    seat.halt_reason = "yield"
                    ctx.log.write(
                        seat,
                        "halt",
                        {"reason": "yield", "text": text[:500]},
                    )
                    return ToolResult(
                        ok=True, content=text, meta={"yield": True}
                    )
                # Task mode + a real text response = the model effectively
                # answered without calling submit. Treat the text as an
                # implicit submit. This avoids the "agent calls a meaningless
                # tool just to satisfy the no_tool_call nudge" failure mode
                # we saw with chat-style messages ("hi", "what can you do?").
                #
                # Autonomous agents opt out: their prose is inner monologue,
                # and the only legitimate user-facing channel is an explicit
                # submit() call. Nudge them back into the loop.
                if text and seat.autonomous:
                    ctx.log.write(
                        seat,
                        "denial",
                        {"reason": "text_without_submit_autonomous",
                         "text_preview": text[:200]},
                    )
                    seat.history.append({
                        "role": "user",
                        "content": (
                            "Inner thinking noted but not delivered. Plain "
                            "text is your private monologue. To address the "
                            "Principal — escalations, status digests, "
                            "anything user-facing — call submit(<message>). "
                            "Otherwise continue with a tool call."
                        ),
                    })
                    continue
                if text:
                    seat.submit_result = text
                    seat.halted = True
                    seat.halt_reason = "submit"
                    ctx.log.write(
                        seat,
                        "submit",
                        {"result": text, "implicit": True},
                    )
                    ctx.log.write(seat, "halt", {"reason": "submit"})
                    return ToolResult(
                        ok=True, content=text,
                        meta={"submitted": True, "implicit": True},
                    )
                # Genuinely empty response — nudge.
                ctx.log.write(
                    seat,
                    "denial",
                    {"reason": "no_tool_call_empty"},
                )
                seat.history.append(
                    {
                        "role": "user",
                        "content": "Please call a tool or send a reply.",
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
