from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .driver import run_seat
from .log import Log
from .tools import RunCtx, mint_seat
from .types import Agent, Seat, ToolResult


@dataclass
class ForestResult:
    root_seat: Seat
    final: ToolResult
    log_path: Path


def run_forest(
    agent: Agent,
    log_path: Path,
    workdir: Path,
    user_message: str,
) -> ForestResult:
    """Top-level entrypoint: mint the root seat from `agent` and run it."""
    log_path = Path(log_path)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    log = Log(log_path)
    ctx = RunCtx(log=log, workdir=workdir, agent=agent)

    root = mint_seat(
        agent=agent,
        seat_id="s0",
        parent_id=None,
        depth=0,
        history=[{"role": "user", "content": user_message}],
    )

    try:
        result = run_seat(root, ctx)
    finally:
        if ctx._executor is not None:
            ctx._executor.shutdown(wait=True)
        log.close()
    return ForestResult(root_seat=root, final=result, log_path=log_path)
