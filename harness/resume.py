"""Continue a stopped run.

Reads the run's log.jsonl, rebuilds the root seat's state (history +
counters), drops any orphaned trailing assistant tool_call that never
got its tool_result, then re-enters `run_seat` appending to the same
log. Lossy by design — if a turn was mid-flight when the run stopped,
the model redoes it from the previous known-consistent state.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

from .driver import run_seat
from .forest import ForestResult
from .log import Log
from .session import ChatSession
from .tools import RunCtx, mint_seat
from .types import Agent, Seat


META_FILENAME = "meta.json"


# ---- meta persistence -----------------------------------------------------


def write_meta(
    run_dir: Path,
    agent: Agent,
    kind: str,
    max_turns_per_round: int = 0,
) -> None:
    """Persist what /continue needs to rebuild the run: agent config and
    whether this was a one-shot forest or a long-lived chat."""
    data = {
        "kind": kind,
        "agent": asdict(agent),
        "max_turns_per_round": max_turns_per_round,
    }
    (run_dir / META_FILENAME).write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def _read_meta(run_dir: Path) -> dict:
    p = run_dir / META_FILENAME
    if not p.exists():
        raise FileNotFoundError(
            f"no {META_FILENAME} in {run_dir}. This run was started before "
            "/continue support landed, so the agent config wasn't saved. "
            "Only newer runs can be continued."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def _agent_from_meta(meta: dict) -> Agent:
    a = dict(meta["agent"])
    # JSON turns tuples into lists; the Agent dataclass expects tuples.
    a["tools"] = tuple(a.get("tools") or ())
    a["web"] = tuple(a.get("web") or ())
    return Agent(**a)


# ---- log → seat rehydration ----------------------------------------------


def _read_log(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _rehydrate_history(entries: list, seat_id: str) -> Optional[list]:
    """Rebuild the OpenAI-format message history for `seat_id`.

    Take the messages from the seat's LAST model_request (these are
    exactly what the driver fed the model on that turn — already
    reflecting any earlier compaction). Then walk events after it: if
    the matching model_response and ALL its tool_results landed,
    append that completed turn too. Otherwise drop the orphan and let
    the driver redo the model call from a known-consistent state.
    """
    last_req_idx = None
    for i in range(len(entries) - 1, -1, -1):
        e = entries[i]
        if e.get("seat_id") == seat_id and e.get("type") == "model_request":
            last_req_idx = i
            break
    if last_req_idx is None:
        return None

    msgs = (entries[last_req_idx].get("payload") or {}).get("messages") or []
    history = [m for m in msgs if m.get("role") != "system"]

    pending_asst: Optional[dict] = None
    expected_call_ids: list = []
    results_by_id: dict = {}

    for e in entries[last_req_idx + 1:]:
        if e.get("seat_id") != seat_id:
            continue
        t = e.get("type")
        p = e.get("payload") or {}
        if t == "model_request":
            # A new turn started — shouldn't happen for the LAST model_request,
            # but be defensive.
            break
        if t == "model_response":
            pending_asst = {"role": "assistant", "content": p.get("text")}
            tcs = p.get("tool_calls") or []
            if tcs:
                pending_asst["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc.get("arguments", "{}"),
                        },
                    }
                    for tc in tcs
                ]
                expected_call_ids = [tc["id"] for tc in tcs]
        elif t == "tool_result":
            cid = p.get("call_id")
            if cid:
                results_by_id[cid] = p.get("content", "")

    if pending_asst is None:
        return history

    if not expected_call_ids:
        # Text-only response — safe to keep; if the seat was a chat that
        # yielded, this is just the last assistant turn. If the model is
        # invoked again it'll see prior context.
        history.append(pending_asst)
        return history

    if all(cid in results_by_id for cid in expected_call_ids):
        history.append(pending_asst)
        for cid in expected_call_ids:
            history.append({
                "role": "tool",
                "tool_call_id": cid,
                "content": results_by_id[cid],
            })
        return history

    # Orphan: at least one tool_result is missing. Drop the assistant
    # message entirely so the model can produce a fresh turn from the
    # last consistent state. (Partial tool_results we'd keep would be
    # stranded without their assistant parent, breaking the OpenAI
    # tool_call_id pairing rule.)
    return history


def _rehydrate_counters(entries: list, seat_id: str, history: list) -> dict:
    """Token + cost counters sum the whole log (cheap and exact).
    `turns_used` and `child_count` come from the *rebuilt history* so they
    agree with what the driver will see — if a trailing assistant turn
    was dropped as an orphan, the counters don't double-count it."""
    tokens_prompt = 0
    tokens_prompt_last = 0
    tokens_completion = 0
    cost = 0.0
    web_searches = 0
    for e in entries:
        if e.get("seat_id") != seat_id:
            continue
        if e.get("type") != "model_response":
            continue
        p = e.get("payload") or {}
        u = p.get("usage") or {}
        tp = int(u.get("prompt_tokens") or 0)
        tokens_prompt += tp
        tokens_prompt_last = tp
        tokens_completion += int(u.get("completion_tokens") or 0)
        cost += float(u.get("usd") or 0.0)
        web_searches += len(p.get("citations") or [])

    # Derive from history so child_count matches whatever tool_calls
    # actually made it back in.
    turns = sum(1 for m in history if m.get("role") == "assistant")
    child_count = 0
    for m in history:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            if (tc.get("function") or {}).get("name") == "spawn":
                child_count += 1
    return dict(
        turns_used=turns,
        tokens_prompt=tokens_prompt,
        tokens_prompt_last=tokens_prompt_last,
        tokens_completion=tokens_completion,
        cost_usd=cost,
        web_searches=web_searches,
        child_count=child_count,
    )


def _build_root_seat(agent: Agent, entries: list) -> Seat:
    history = _rehydrate_history(entries, "s0") or []
    counters = _rehydrate_counters(entries, "s0", history)
    seat = mint_seat(
        agent=agent, seat_id="s0", parent_id=None, depth=0, history=history
    )
    for k, v in counters.items():
        setattr(seat, k, v)
    return seat


# ---- entry points ---------------------------------------------------------


def continue_forest(
    run_dir: Path,
    extra_user_message: Optional[str] = None,
    on_ctx: Optional[Callable[[RunCtx], None]] = None,
    auto_approve: bool = False,
) -> ForestResult:
    """Re-enter a one-shot run. Appends to the existing log. If
    `extra_user_message` is given, it's appended as a fresh user turn
    before the driver resumes."""
    run_dir = Path(run_dir)
    log_path = run_dir / "log.jsonl"
    workdir = run_dir / "wd"
    workdir.mkdir(parents=True, exist_ok=True)

    meta = _read_meta(run_dir)
    if meta.get("kind") != "forest":
        raise ValueError(
            f"{run_dir.name} is not a one-shot run (kind={meta.get('kind')!r}). "
            "Use /continue on the chat instead."
        )
    agent = _agent_from_meta(meta)

    entries = _read_log(log_path)
    log = Log(log_path)
    ctx = RunCtx(log=log, workdir=workdir, agent=agent, auto_approve=auto_approve)
    if on_ctx is not None:
        on_ctx(ctx)

    seat = _build_root_seat(agent, entries)
    if extra_user_message:
        seat.history.append({"role": "user", "content": extra_user_message})

    # Extend the per-seat turn cap so we don't immediately re-halt on max_turns.
    seat.max_turns = seat.turns_used + agent.max_turns

    log.write(
        seat,
        "resume",
        {
            "turns_so_far": seat.turns_used,
            "history_messages": len(seat.history),
            "extra_user": bool(extra_user_message),
        },
    )

    try:
        result = run_seat(seat, ctx)
    finally:
        if ctx._executor is not None:
            ctx._executor.shutdown(wait=True)
        log.close()
    return ForestResult(root_seat=seat, final=result, log_path=log_path)


def continue_chat(
    run_dir: Path,
    auto_approve: bool = False,
) -> ChatSession:
    """Re-attach to an existing chat. Returns a fresh ChatSession ready
    for more user messages; the next `send` runs against the rebuilt
    history and appends to the same log."""
    run_dir = Path(run_dir)
    log_path = run_dir / "log.jsonl"
    workdir = run_dir / "wd"
    workdir.mkdir(parents=True, exist_ok=True)

    meta = _read_meta(run_dir)
    if meta.get("kind") != "chat":
        raise ValueError(
            f"{run_dir.name} is not a chat run (kind={meta.get('kind')!r}). "
            "Use /continue on the one-shot run instead."
        )
    agent = _agent_from_meta(meta)
    max_turns_per_round = int(meta.get("max_turns_per_round") or agent.max_turns)

    entries = _read_log(log_path)
    log = Log(log_path)
    ctx = RunCtx(log=log, workdir=workdir, agent=agent, auto_approve=auto_approve)

    seat = _build_root_seat(agent, entries)
    log.write(
        seat,
        "resume",
        {
            "turns_so_far": seat.turns_used,
            "history_messages": len(seat.history),
        },
    )

    return ChatSession(
        seat=seat,
        ctx=ctx,
        log_path=log_path,
        max_turns_per_round=max_turns_per_round,
    )


def run_kind(run_dir: Path) -> Optional[str]:
    """Best-effort: read meta.json and return "chat"/"forest", or None
    if missing/unreadable. The UI uses this to dispatch /continue."""
    try:
        meta = _read_meta(Path(run_dir))
        k = meta.get("kind")
        return k if k in ("chat", "forest") else None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
