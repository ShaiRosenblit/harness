"""Research policy: a single-seat agent with OpenRouter's server-side web
search and fetch enabled.

The agent has no code_exec and no spawn — its only local tool is submit.
The web tools are resolved entirely by OpenRouter; we never see them in
the local adjudicator path. Citations attached to the assistant's reply
are parsed into the log so we can audit which sources informed the answer.
"""
from __future__ import annotations

from harness.types import Limits, Policy


PROMPT = (
    "You are a research assistant. You have web_search and web_fetch tools "
    "(resolved server-side); the harness will not run them — you call them "
    "directly through the OpenRouter API and citations come back attached "
    "to your message. Search the live web when the user asks about "
    "anything that could have changed recently (versions, prices, news, "
    "documentation). When you have an answer, call submit(result) with a "
    "concise final answer that names the key sources you used."
)


POLICY = Policy(
    name="research",
    model="openai/gpt-4o-mini",
    system_prompt=PROMPT,
    tools=("submit",),
    limits=Limits(
        max_turns=6,
        max_depth=0,
        max_children=0,
        max_concurrent_seats=1,
        tool_timeout_s=30,
    ),
    budget_usd=0.50,
    web=("search", "fetch"),
    web_max_results=4,
    web_search_context_size="low",
)
