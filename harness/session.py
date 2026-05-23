"""Long-lived chat session over the harness substrate.

A ChatSession holds one persistent seat and one RunCtx. Each call to
``send(user_text)`` appends a user message, runs the driver loop until the
model produces a text-only response (chat_mode yield), and returns that
reply. The seat, history, budget, and log all persist across turns.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .driver import run_seat
from .log import Log
from .tools import RunCtx
from .types import Budget, Policy, Seat


@dataclass
class ChatSession:
    seat: Seat
    ctx: RunCtx
    log_path: Path

    def send(self, user_text: str) -> str:
        """Send a user message and return the assistant's text reply.

        Re-enters the driver loop with the seat refreshed (halted=False,
        new user message appended). The driver yields on chat_mode when
        the model produces a text-only response.
        """
        if self.seat.budget.usd_remaining <= 0:
            return "[session budget exhausted]"
        # Refresh seat for another turn.
        self.seat.halted = False
        self.seat.halt_reason = None
        self.seat.history.append({"role": "user", "content": user_text})
        # Soft-extend the per-turn budget so chat doesn't run out on turn 10.
        # We rely on dollar budget as the real ceiling.
        old_max = self.seat.limits.max_turns
        from dataclasses import replace
        self.seat.limits = replace(
            self.seat.limits,
            max_turns=self.seat.turns_used + old_max,
        )
        result = run_seat(self.seat, self.ctx)
        return result.content if result.content is not None else ""

    def close(self) -> None:
        if self.ctx._executor is not None:
            self.ctx._executor.shutdown(wait=True)
            self.ctx._executor = None
        self.ctx.log.close()


def start_chat(
    policy: Policy,
    log_path: Path,
    workdir: Path,
    kill_path: Path,
) -> ChatSession:
    """Create a fresh chat session under the given policy.

    The seat starts with an empty history (the policy's system_prompt is
    attached at model-call time, the same way it is for run_forest). The
    first ``session.send(user_text)`` triggers the first model call.
    """
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
        chat_mode=True,
    )
    seat = Seat(
        id="s0",
        parent_id=None,
        depth=0,
        prompt=policy.system_prompt,
        granted_tools=policy.tools,
        model=policy.model,
        limits=policy.limits,
        budget=Budget(usd_remaining=policy.budget_usd),
        history=[],
        web=policy.web,
        web_max_results=policy.web_max_results,
        web_search_context_size=policy.web_search_context_size,
    )
    return ChatSession(seat=seat, ctx=ctx, log_path=log_path)
