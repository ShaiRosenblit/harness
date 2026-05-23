"""One-shot task policy for `/run`.

The agent gets code_exec, submit, and spawn (one level deep, three children).
It is expected to finish by calling submit() with the answer; sub-agents may
only call code_exec and submit (capability attenuation — they cannot spawn).
"""
from __future__ import annotations

from harness.types import Limits, Policy


SYSTEM_PROMPT = (
    "You complete the given task using the available tools. You act by "
    "calling tools; you'll see each result and can continue. Use code_exec "
    "when running code helps. Use spawn to delegate a sub-task to a "
    "fresh sub-agent (its result will come back to you). Call submit() "
    "with your final answer when done."
)


CHILD = Policy(
    name="task-child",
    model="moonshotai/kimi-k2.6",
    system_prompt=SYSTEM_PROMPT,
    tools=("code_exec", "submit"),
    limits=Limits(
        max_turns=8,
        max_depth=0,
        max_children=0,
        max_concurrent_seats=1,
        tool_timeout_s=15,
    ),
    budget_usd=0.20,
)


POLICY = Policy(
    name="task",
    model="moonshotai/kimi-k2.6",
    system_prompt=SYSTEM_PROMPT,
    tools=("code_exec", "submit", "spawn"),
    limits=Limits(
        max_turns=12,
        max_depth=2,
        max_children=3,
        max_concurrent_seats=4,
        tool_timeout_s=15,
    ),
    budget_usd=2.00,
    child_policy=CHILD,
)
