"""One-shot task agent — must call submit() to finish.

Has code_exec, submit, and spawn. Children inherit this same config, so
they can spawn further; recursion is bounded by max_depth.
"""
from harness.types import Agent


PROMPT = (
    "You complete the given task using the available tools. You act by "
    "calling tools; you'll see each result and can continue. Use "
    "code_exec when running code helps. Use spawn to delegate a focused "
    "sub-task to a fresh sub-agent (its submitted result will come back "
    "to you). Call submit() with your final answer when done."
)


AGENT = Agent(
    model="moonshotai/kimi-k2.6",
    system_prompt=PROMPT,
    tools=("code_exec", "submit", "spawn", "use_skill"),
    max_turns=12,
    max_depth=2,
    max_children=3,
    tool_timeout_s=15.0,
)
