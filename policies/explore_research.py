"""explore_research — single-seat web research probe.
Run with the question as the user message, e.g.:
    /run explore_research find the current per-million-token input/output price
        of the 3 most popular commercial LLM APIs, with sources
"""
from harness.types import Policy, Limits

POLICY = Policy(
    name="explore_research",
    model="moonshotai/kimi-k2.6",
    system_prompt=(
        "You are a research agent. Investigate the task you are given using "
        "web search and web fetch.\n"
        "- Search first to find sources, then fetch only the most relevant pages.\n"
        "- Cross-check each fact against at least two sources; note disagreements.\n"
        "- Be economical: lean on search highlights; fetch a full page only when a "
        "snippet isn't enough. Tokens cost budget.\n"
        "- When you have a confident, sourced answer, call submit() with a concise "
        "summary and the source URLs. Don't keep searching once it's solid."
    ),
    tools=("code_exec", "submit"),
    limits=Limits(
        max_turns=15,
        max_depth=0,
        max_children=0,
        max_concurrent_seats=1,
        tool_timeout_s=30.0,
    ),
    budget_usd=1.00,
    web=("search", "fetch"),
    web_max_results=4,
    web_search_context_size="low",
)
