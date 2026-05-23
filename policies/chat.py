from __future__ import annotations

from harness.types import Limits, Policy


CHAT_PROMPT = (
    "You are an assistant in an interactive shell. The user messages you "
    "turn by turn; reply concisely. You have the following tools:\n"
    "  • code_exec — run Python in a scratch directory (arithmetic, "
    "parsing, file work, anything code can do better than guessing).\n"
    "  • web_search / web_fetch — live web access. Use whenever the answer "
    "could have changed recently (versions, prices, news, docs).\n"
    "  • spawn — delegate a focused sub-task to a fresh sub-agent and get "
    "back a single result. Useful for parallel research, or to isolate "
    "long/expensive work from your main context.\n"
    "  • submit — not normally needed in chat; leave it alone.\n"
    "When you have nothing more to do, respond with plain text — that "
    "yields control back to the user for the next message."
)


# Spawned sub-agents are one-shot researchers. They inherit code_exec +
# submit, plus web access attenuated from the chat seat.
CHILD = Policy(
    name="chat-child",
    model="moonshotai/kimi-k2.6",
    system_prompt=(
        "You are a research worker. Solve the focused task you were given "
        "and call submit(result) with a concise sourced answer. Use "
        "code_exec when running code helps; use web_search / web_fetch "
        "when current information is needed. Be fast and economical."
    ),
    tools=("code_exec", "submit"),   # no spawn at this level
    limits=Limits(
        max_turns=8,
        max_depth=0,
        max_children=0,
        max_concurrent_seats=1,
        tool_timeout_s=30,
    ),
    budget_usd=0.30,
    web=("search", "fetch"),
    web_max_results=4,
    web_search_context_size="low",
)


POLICY = Policy(
    name="chat",
    model="moonshotai/kimi-k2.6",
    system_prompt=CHAT_PROMPT,
    tools=("code_exec", "submit", "spawn"),   # everything local
    limits=Limits(
        max_turns=20,
        max_depth=1,
        max_children=4,
        max_concurrent_seats=4,
        tool_timeout_s=30,
    ),
    budget_usd=2.00,
    child_policy=CHILD,
    web=("search", "fetch"),    # everything web
    web_max_results=4,
    web_search_context_size="low",
)
