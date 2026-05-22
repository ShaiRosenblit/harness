"""Deep-spawn variant of the task policy.

Same as `task` but the child policy ALSO grants `spawn`, so children can
mint grandchildren. The substrate's `max_depth=2` then caps the tree at
three levels (root → child → grandchild — grandchildren cannot spawn).
Child policies recurse: the harness uses `ctx.policy.child_policy` for
every spawn at any depth, so the template is the same at each level.
"""
from __future__ import annotations

from harness.types import Limits, Policy


SYSTEM_PROMPT = (
    "You complete the given task using the available tools. Use code_exec "
    "for running Python. Use spawn to delegate. When spawning, give each "
    "child a small slice of your budget. Call submit() with your final "
    "answer when done."
)


SUB = Policy(
    name="task-deep-sub",
    model="moonshotai/kimi-k2.6",
    system_prompt=SYSTEM_PROMPT,
    tools=("code_exec", "submit", "spawn"),   # spawn granted at every level
    limits=Limits(
        max_turns=10,
        max_depth=2,
        max_children=3,
        max_concurrent_seats=8,
        tool_timeout_s=15,
    ),
    budget_usd=0.50,
)


POLICY = Policy(
    name="task-deep",
    model="moonshotai/kimi-k2.6",
    system_prompt=SYSTEM_PROMPT,
    tools=("code_exec", "submit", "spawn"),
    limits=Limits(
        max_turns=15,
        max_depth=2,
        max_children=3,
        max_concurrent_seats=8,
        tool_timeout_s=15,
    ),
    budget_usd=5.00,
    child_policy=SUB,
)
