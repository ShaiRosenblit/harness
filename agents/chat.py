"""Conversational chat agent — talks to the user turn by turn.

No `submit` (yields on text-only replies). No `spawn` (chat children
would themselves be chat agents and never end). Has code_exec and web.
"""
from harness.types import Agent


PROMPT = (
    "You are an assistant in an interactive shell. The user messages you "
    "turn by turn; reply concisely. Tools available:\n"
    "  - code_exec — run Python in a scratch directory.\n"
    "  - web_search / web_fetch — live web access.\n"
    "When you have nothing more to do, respond with plain text — that "
    "yields control back to the user for the next message."
)


AGENT = Agent(
    model="moonshotai/kimi-k2.6",
    system_prompt=PROMPT,
    tools=("code_exec",),
    max_turns=20,
    max_depth=0,
    max_children=0,
    tool_timeout_s=30.0,
    web=("search", "fetch"),
)
