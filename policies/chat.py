from __future__ import annotations

from harness.types import Limits, Policy


CHAT_PROMPT = (
    "You are an assistant in an interactive shell. The user will message "
    "you turn-by-turn. Reply concisely. You have a code_exec tool you can "
    "call to run Python in a scratch directory (use it when running code "
    "would help answer accurately, e.g. arithmetic, parsing, checking). "
    "When you have nothing else to do, just respond with text — that "
    "yields control back to the user."
)


POLICY = Policy(
    name="chat",
    model="openai/gpt-4o-mini",
    system_prompt=CHAT_PROMPT,
    tools=("code_exec",),   # no submit: chat doesn't end on its own
    limits=Limits(
        max_turns=20,        # per-message ceiling; ChatSession extends it each round
        max_depth=0,
        max_children=0,
        max_concurrent_seats=1,
        tool_timeout_s=15,
    ),
    budget_usd=2.00,         # multi-turn budget; session ends when this drains
)
