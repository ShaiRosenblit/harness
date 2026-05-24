"""Research agent — single seat with web access and code_exec.

Has submit (must call to end). Has web_search and web_fetch. No spawn.
"""
from harness.types import Agent


PROMPT = (
    "You are a research agent. Investigate the task using web_search and "
    "web_fetch. Search first to find sources, fetch only the most "
    "relevant pages, cross-check facts against multiple sources. Use "
    "code_exec when parsing or arithmetic helps. When you have a sourced "
    "answer, call submit() with a concise summary and the source URLs."
)


AGENT = Agent(
    model="moonshotai/kimi-k2.6",
    system_prompt=PROMPT,
    tools=("code_exec", "submit"),
    max_turns=10,
    max_depth=0,
    max_children=0,
    tool_timeout_s=30.0,
    web=("search", "fetch"),
)
