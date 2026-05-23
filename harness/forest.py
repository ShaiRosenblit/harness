from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .driver import run_seat
from .log import Log
from .tools import RunCtx
from .types import Budget, Policy, Seat, ToolResult


@dataclass
class ForestResult:
    root_seat: Seat
    final: ToolResult
    log_path: Path


def run_forest(
    policy: Policy,
    log_path: Path,
    workdir: Path,
    kill_path: Path,
    user_message: str,
) -> ForestResult:
    """Top-level entrypoint. Mints the root seat from the policy and runs it."""
    log_path = Path(log_path)
    workdir = Path(workdir)
    kill_path = Path(kill_path)
    workdir.mkdir(parents=True, exist_ok=True)

    log = Log(log_path)
    ctx = RunCtx(
        log=log,
        workdir=workdir,
        kill_path=kill_path,
        policy=policy,
    )

    root = Seat(
        id="s0",
        parent_id=None,
        depth=0,
        prompt=policy.system_prompt,
        granted_tools=policy.tools,
        model=policy.model,
        limits=policy.limits,
        budget=Budget(usd_remaining=policy.budget_usd),
        history=[{"role": "user", "content": user_message}],
        web=policy.web,
        web_max_results=policy.web_max_results,
        web_search_context_size=policy.web_search_context_size,
    )

    try:
        result = run_seat(root, ctx)
    finally:
        if ctx._executor is not None:
            ctx._executor.shutdown(wait=True)
        log.close()
    return ForestResult(root_seat=root, final=result, log_path=log_path)
