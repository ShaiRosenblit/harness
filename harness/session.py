"""Long-lived chat session.

A ChatSession holds one persistent seat and one RunCtx. Each `send` call
appends a user message, runs the driver loop until the model produces a
text-only response (yield), and returns that reply.

Chat-vs-task mode is derived from the seat's capabilities: if `submit` is
not in `seat.tools`, the driver yields on text-only responses; otherwise
it nudges the model to call a tool.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .driver import run_seat
from .log import Log
from .tools import RunCtx, mint_seat
from .types import Agent, Seat


@dataclass
class ChatSession:
    seat: Seat
    ctx: RunCtx
    log_path: Path
    max_turns_per_round: int

    def send(self, user_text: str) -> str:
        """Send a user message; return the assistant's text reply."""
        self.seat.halted = False
        self.seat.halt_reason = None
        self.seat.history.append({"role": "user", "content": user_text})
        # Soft-extend turn ceiling each round so chat doesn't run out on turn N.
        self.seat.max_turns = self.seat.turns_used + self.max_turns_per_round
        result = run_seat(self.seat, self.ctx)
        return result.content if result.content is not None else ""

    def close(self) -> None:
        if self.ctx._executor is not None:
            self.ctx._executor.shutdown(wait=True)
            self.ctx._executor = None
        if self.ctx._bg_executor is not None:
            self.ctx._bg_executor.shutdown(wait=True)
            self.ctx._bg_executor = None
        self.ctx.log.close()


def start_chat(
    agent: Agent,
    log_path: Path,
    workdir: Path,
    auto_approve: bool = False,
) -> ChatSession:
    """Create a fresh chat session.

    `auto_approve=True` runs risky tools without prompting the user.
    Default off.
    """
    log_path = Path(log_path)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    log = Log(log_path)
    ctx = RunCtx(log=log, workdir=workdir, agent=agent, auto_approve=auto_approve)
    seat = mint_seat(agent=agent, seat_id="s0", parent_id=None, depth=0, history=[])
    return ChatSession(
        seat=seat,
        ctx=ctx,
        log_path=log_path,
        max_turns_per_round=agent.max_turns,
    )
