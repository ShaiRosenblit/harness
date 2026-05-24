from __future__ import annotations

from .log import Log
from .types import Seat


COMPACT_AT_TOKENS = 120_000
KEEP_ASSISTANT_TURNS = 4


_SUMMARIZER_PROMPT = (
    "You are summarizing a running agent's conversation history so the agent "
    "can continue its task with less context. Produce a dense summary that "
    "preserves: (1) the user's original goal, (2) key findings and decisions, "
    "(3) important file paths, identifiers, and values discovered, (4) open "
    "questions or next steps. Drop verbose tool output, dead-end exploration, "
    "and chit-chat. Return only the summary text — no preamble."
)


def _find_tail_start(history: list, keep_turns: int) -> int:
    """Index in `history` where the verbatim tail begins. The tail starts at
    the Kth-from-last assistant message so tool results that follow it
    aren't stranded from their `tool_call_id`. Returns 0 if there aren't
    enough assistant turns to slice a tail (caller should skip)."""
    asst_idx = [i for i, m in enumerate(history) if m.get("role") == "assistant"]
    if len(asst_idx) <= keep_turns:
        return 0
    return asst_idx[-keep_turns]


def _render_for_summary(messages: list) -> str:
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content") or ""
        if role == "assistant" and m.get("tool_calls"):
            calls = ", ".join(
                f"{tc['function']['name']}({tc['function']['arguments']})"
                for tc in m["tool_calls"]
            )
            lines.append(f"[assistant] {content}\n  tool_calls: {calls}")
        elif role == "tool":
            lines.append(f"[tool result] {content}")
        else:
            lines.append(f"[{role}] {content}")
    return "\n\n".join(lines)


def maybe_compact(seat: Seat, last_prompt_tokens: int, log: Log) -> None:
    """If the previous turn's prompt crossed the threshold, summarize the
    middle of `seat.history` while keeping the anchoring first user message
    and the last K assistant turns (with their tool results) verbatim.

    Layout after compaction:
        [first_user_msg?, {synthetic user "[earlier ... summarized]"}, *tail]

    Skipped silently when there's nothing to gain (not enough assistant
    turns to slice a tail, or no messages between anchor and tail)."""
    if last_prompt_tokens < COMPACT_AT_TOKENS:
        return
    if not seat.history:
        return

    tail_start = _find_tail_start(seat.history, KEEP_ASSISTANT_TURNS)
    if tail_start == 0:
        # Not enough assistant turns to safely carve a tail. Don't compact
        # — risk of breaking tool-pair invariants outweighs the savings.
        return

    # Anchor the original task verbatim if the first message is a user msg
    # and isn't already inside the tail.
    has_anchor = (
        seat.history[0].get("role") == "user" and tail_start > 0
    )
    anchor_offset = 1 if has_anchor else 0
    middle = seat.history[anchor_offset:tail_start]
    if not middle:
        return  # nothing in between to summarize

    tail = seat.history[tail_start:]
    transcript = _render_for_summary(middle)

    log.write(
        seat,
        "compact_request",
        {
            "prompt_tokens": last_prompt_tokens,
            "history_messages": len(seat.history),
            "middle_messages": len(middle),
            "tail_messages": len(tail),
            "anchored": has_anchor,
        },
    )

    from .model import _client  # local import to avoid a cycle at module load

    completion = _client().chat.completions.create(
        model=seat.model,
        messages=[
            {"role": "system", "content": _SUMMARIZER_PROMPT},
            {"role": "user", "content": transcript},
        ],
        extra_body={"usage": {"include": True}},
    )
    summary = completion.choices[0].message.content or ""

    usage = getattr(completion, "usage", None)
    cost = float(getattr(usage, "cost", 0.0) or 0.0)
    seat.cost_usd += cost

    summary_msg = {
        "role": "user",
        "content": f"[earlier conversation summarized]\n{summary}",
    }
    new_history: list = []
    if has_anchor:
        new_history.append(seat.history[0])
    new_history.append(summary_msg)
    new_history.extend(tail)

    old_count = len(seat.history)
    seat.history[:] = new_history

    log.write(
        seat,
        "compact_done",
        {
            "messages_before": old_count,
            "messages_after": len(new_history),
            "summary_chars": len(summary),
            "cost_usd": cost,
        },
    )
