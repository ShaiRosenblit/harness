"""explore_delegation — coordinator that spawns web-enabled sub-agents.
Run with the comparison task as the user message, e.g.:
    /run explore_delegation compare the web-search APIs Tavily, Exa and Brave on
        price, free tier and main strengths; produce one comparison table

Exercises: spawn, tool + web attenuation, budget conservation, result collapse,
the seat tree.

ATTENUATION: a child can only hold tools/web its PARENT holds. The coordinator is
granted code_exec AND web even though its prompt only delegates — otherwise the
children couldn't inherit them. Keep them.
"""
from harness.types import Policy, Limits

CHILD = Policy(
    name="explore_delegation_child",
    model="moonshotai/kimi-k2.6",
    system_prompt=(
        "You are a research worker. Research EXACTLY the question you are given "
        "— nothing broader.\n"
        "- Use web search to find sources, then fetch only what you need.\n"
        "- Be fast and economical; your budget is small.\n"
        "- When you have a solid, sourced answer, call submit() with a concise "
        "result and the source URLs."
    ),
    tools=("code_exec", "submit"),  # no spawn
    limits=Limits(
        max_turns=8,
        max_depth=0,
        max_children=0,
        max_concurrent_seats=1,
        tool_timeout_s=30.0,
    ),
    budget_usd=0.30,
    web=("search", "fetch"),
    web_max_results=4,
    web_search_context_size="low",
)

POLICY = Policy(
    name="explore_delegation",
    model="moonshotai/kimi-k2.6",
    system_prompt=(
        "You are a research coordinator. You do NOT research directly — you delegate.\n"
        "- Break the task into independent sub-questions (one per item being compared).\n"
        "- For each, call spawn with a focused instruction naming exactly what to "
        "research. Each child has web access and a small budget, and CANNOT spawn.\n"
        "- Spawn one child per sub-question. If a child returns thin or failed results, "
        "you may re-spawn it once with a sharper instruction.\n"
        "- Each child returns one result. Collect them, synthesize a single "
        "consolidated answer (a comparison table works well), then call submit().\n"
        "- Stay within budget; don't spawn more children than there are sub-questions."
    ),
    tools=("code_exec", "submit", "spawn"),  # holds code_exec for attenuation
    limits=Limits(
        max_turns=20,
        max_depth=1,
        max_children=5,
        max_concurrent_seats=5,
        tool_timeout_s=30.0,
    ),
    budget_usd=2.00,
    child_policy=CHILD,
    web=("search", "fetch"),   # held for attenuation; prompt says delegate, not search
    web_max_results=4,
    web_search_context_size="low",
)
