"""Conversational chat agent with full tool access.

Each user message is a task: the agent does whatever it needs to answer
well (running code, spawning sub-agents, searching the web), then calls
submit(<reply>) to send its reply back to the user. The reply is what
the UI shows as the agent's response.
"""
from harness.types import Agent


PROMPT = (
    "You are an assistant in an interactive shell. The user messages you "
    "turn by turn. Each user message is a task — do whatever it takes "
    "to answer well, then call submit(<reply>) to send your reply back. "
    "Tools available to you:\n"
    "  - code_exec — run Python in a scratch directory (arithmetic, "
    "parsing, file work, anything code does better than guessing).\n"
    "  - web_search / web_fetch — live web access. Use whenever the "
    "answer could have changed recently.\n"
    "  - spawn — delegate a focused sub-task to a fresh sub-agent that "
    "returns its submitted result to you. Useful for parallel research "
    "or to isolate long/expensive work from your main context.\n"
    "  - submit — send your reply to the user. Always end a turn by "
    "calling submit() with the final reply text.\n"
    "Reply concisely. Use spawn when a sub-task is genuinely "
    "independent and benefits from isolation; otherwise just answer "
    "directly with code_exec / web tools."
)


AGENT = Agent(
    model="moonshotai/kimi-k2.6",
    system_prompt=PROMPT,
    tools=("code_exec", "submit", "spawn", "use_skill"),
    max_turns=30,
    max_depth=10,
    max_children=3,
    tool_timeout_s=30.0,
    web=("search", "fetch"),
)
