from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from .log import Log
from .types import (
    Agent,
    Seat,
    ToolCall,
    ToolResult,
    ToolSpec,
)


# ---- Shared run context ----------------------------------------------------


@dataclass
class ApprovalRequest:
    """A pending request for the user to approve or deny a risky tool call.

    The worker thread blocks on `event`; the UI sets `decision` (to
    "approve" or "deny") and then `event.set()` to release it.
    `displayed` is a one-shot flag the UI flips when it first shows the
    request, so polling won't double-print.
    """
    id: str
    seat_id: str
    tool_name: str
    args: dict
    event: threading.Event
    decision: Optional[str] = None   # None | "approve" | "deny"
    displayed: bool = False


@dataclass
class RunCtx:
    log: Log
    workdir: Path
    agent: Agent
    live_seats: set = field(default_factory=set)
    _spawn_lock: threading.Lock = field(default_factory=threading.Lock)
    _executor: Optional[ThreadPoolExecutor] = None
    # Approval gate: pending and resolved requests for risky tool calls.
    _approvals: Dict[str, ApprovalRequest] = field(default_factory=dict)
    _approval_seq: int = 0
    _approvals_lock: threading.Lock = field(default_factory=threading.Lock)

    def executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=8, thread_name_prefix="harness-spawn"
            )
        return self._executor

    def request_approval(self, seat_id: str, tool_name: str, args: dict) -> ApprovalRequest:
        """Register a pending approval; caller blocks on req.event.wait()."""
        with self._approvals_lock:
            self._approval_seq += 1
            req = ApprovalRequest(
                id=f"a{self._approval_seq}",
                seat_id=seat_id,
                tool_name=tool_name,
                args=args,
                event=threading.Event(),
            )
            self._approvals[req.id] = req
        return req

    def resolve_approval(self, approval_id: str, decision: str) -> Optional[ApprovalRequest]:
        """Set the decision and release the waiting worker. Returns the
        request if it was pending; None if not found or already resolved."""
        if decision not in ("approve", "deny"):
            return None
        with self._approvals_lock:
            req = self._approvals.get(approval_id)
            if req is None or req.decision is not None:
                return None
            req.decision = decision
        req.event.set()
        return req

    def pending_undisplayed(self) -> list:
        """Return pending approvals that haven't been shown to the user
        yet, and mark them displayed so we don't re-show on next poll."""
        out = []
        with self._approvals_lock:
            for req in self._approvals.values():
                if req.decision is None and not req.displayed:
                    req.displayed = True
                    out.append(req)
        return out


# ---- Seat factory (shared by forest, session, spawn) ----------------------


def mint_seat(
    agent: Agent,
    seat_id: str,
    parent_id: Optional[str],
    depth: int,
    history: list,
) -> Seat:
    return Seat(
        id=seat_id,
        parent_id=parent_id,
        depth=depth,
        model=agent.model,
        system_prompt=agent.system_prompt,
        tools=agent.tools,
        web=agent.web,
        max_turns=agent.max_turns,
        max_depth=agent.max_depth,
        max_children=agent.max_children,
        tool_timeout_s=agent.tool_timeout_s,
        web_max_results=agent.web_max_results,
        web_search_context_size=agent.web_search_context_size,
        provider=agent.provider,
        history=history,
    )


# ---- Registry --------------------------------------------------------------


REGISTRY: Dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> None:
    REGISTRY[spec.name] = spec


def get_specs(names) -> list:
    return [REGISTRY[n] for n in names if n in REGISTRY]


# ---- Adjudicator -----------------------------------------------------------


def adjudicate_and_run(seat: Seat, call: ToolCall, ctx: RunCtx) -> ToolResult:
    """Capability check → schema check → execute → log. All in one place."""
    # Capability check. seat.web (("search","fetch")) implicitly grants
    # web_search / web_fetch (which the driver injects into the model's
    # visible tool list); they don't appear in seat.tools.
    allowed = set(seat.tools)
    if "search" in seat.web:
        allowed.add("web_search")
    if "fetch" in seat.web:
        allowed.add("web_fetch")
    if call.name not in allowed:
        ctx.log.write(
            seat,
            "denial",
            {"reason": "not_granted", "tool": call.name, "call_id": call.id},
        )
        return ToolResult(
            ok=False,
            content=f"tool not granted: {call.name}",
            error="not_granted",
        )

    spec = REGISTRY.get(call.name)
    if spec is None:
        ctx.log.write(
            seat,
            "denial",
            {"reason": "unknown_tool", "tool": call.name, "call_id": call.id},
        )
        return ToolResult(
            ok=False, content=f"unknown tool: {call.name}", error="unknown_tool"
        )

    # Shallow schema check: required keys present.
    required = (spec.parameters or {}).get("required", []) or []
    missing = [k for k in required if k not in (call.args or {})]
    if missing:
        ctx.log.write(
            seat,
            "denial",
            {
                "reason": "bad_args",
                "tool": call.name,
                "missing": missing,
                "call_id": call.id,
            },
        )
        return ToolResult(
            ok=False,
            content=f"missing required args: {missing}",
            error="bad_args",
        )

    ctx.log.write(
        seat,
        "tool_call",
        {"tool": call.name, "args": call.args, "call_id": call.id},
    )

    # Approval gate for risky tools. Blocks the worker thread until the
    # user types /approve <id> or /deny <id> in the UI.
    if spec.risky:
        req = ctx.request_approval(seat.id, call.name, call.args or {})
        ctx.log.write(
            seat,
            "approval_request",
            {"approval_id": req.id, "tool": call.name, "args": call.args, "call_id": call.id},
        )
        req.event.wait()   # ← blocks here until UI resolves
        ctx.log.write(
            seat,
            "approval_decision",
            {"approval_id": req.id, "decision": req.decision, "tool": call.name},
        )
        if req.decision != "approve":
            result = ToolResult(
                ok=False,
                content=f"denied by user (approval {req.id})",
                error="denied",
            )
            ctx.log.write(
                seat,
                "tool_result",
                {
                    "tool": call.name,
                    "call_id": call.id,
                    "ok": False,
                    "error": "denied",
                    "content": result.content,
                    "meta": {},
                },
            )
            return result

    try:
        result = spec.handler(seat, call.args, ctx)
    except subprocess.TimeoutExpired:
        result = ToolResult(ok=False, content="timeout", error="timeout")
    except Exception as e:
        result = ToolResult(ok=False, content=f"exception: {e!r}", error="exception")

    truncated = result.content if len(result.content) <= 16000 else (
        result.content[:16000] + "...[truncated]"
    )
    ctx.log.write(
        seat,
        "tool_result",
        {
            "tool": call.name,
            "call_id": call.id,
            "ok": result.ok,
            "error": result.error,
            "content": truncated,
            "meta": result.meta,
        },
    )

    return result


# ---- Built-in tools --------------------------------------------------------


def _h_code_exec(seat: Seat, args: dict, ctx: RunCtx) -> ToolResult:
    code = args.get("code", "")
    proc = subprocess.run(
        ["python3", "-c", code],
        cwd=str(ctx.workdir),
        capture_output=True,
        text=True,
        timeout=seat.tool_timeout_s,
    )
    out = (proc.stdout or "")[-8000:]
    err = (proc.stderr or "")[-2000:]
    content = f"stdout:\n{out}\nstderr:\n{err}\nexit: {proc.returncode}"
    return ToolResult(
        ok=(proc.returncode == 0),
        content=content,
        meta={"exit": proc.returncode},
    )


def _h_bash(seat: Seat, args: dict, ctx: RunCtx) -> ToolResult:
    cmd = args.get("command", "")
    proc = subprocess.run(
        ["bash", "-c", cmd],
        cwd=str(ctx.workdir),
        capture_output=True,
        text=True,
        timeout=seat.tool_timeout_s,
    )
    out = (proc.stdout or "")[-8000:]
    err = (proc.stderr or "")[-2000:]
    content = f"stdout:\n{out}\nstderr:\n{err}\nexit: {proc.returncode}"
    return ToolResult(
        ok=(proc.returncode == 0),
        content=content,
        meta={"exit": proc.returncode},
    )


def _h_submit(seat: Seat, args: dict, ctx: RunCtx) -> ToolResult:
    result = args.get("result", "")
    seat.submit_result = result if isinstance(result, str) else json.dumps(result)
    seat.halted = True
    seat.halt_reason = "submit"
    ctx.log.write(seat, "submit", {"result": seat.submit_result})
    ctx.log.write(seat, "halt", {"reason": "submit"})
    return ToolResult(ok=True, content=seat.submit_result, meta={"submitted": True})


def _h_web_search(seat: Seat, args: dict, ctx: RunCtx) -> ToolResult:
    """Web search via OpenRouter's `web` plugin on a cheap side-call.

    OpenRouter's `openrouter:web_search` tool type is broken with
    several models (notably Kimi K2.6) — the model emits its native
    tool-call template into the visible content stream and the
    response never carries structured tool_calls back. The fix is to
    expose web_search as a *regular* function tool: the model calls
    it like any other function, we run the search ourselves by piggy-
    backing on OpenRouter's `plugins=[{id:"web"}]` against a small
    model, and return the cited results as a tool result.

    Cost-wise this is a single short cheap-model call per search."""
    from . import credentials
    from openai import OpenAI

    query = args.get("query", "")
    if not isinstance(query, str) or not query.strip():
        return ToolResult(ok=False, content="missing 'query'", error="bad_args")

    credentials.inject_env()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return ToolResult(
            ok=False, content="OPENROUTER_API_KEY not set", error="no_auth"
        )

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Search the web for: {query}\n\n"
                        "Reply with one short sentence per result and "
                        "nothing else. Do not add commentary."
                    ),
                }
            ],
            max_completion_tokens=600,
            extra_body={
                "plugins": [
                    {"id": "web", "max_results": seat.web_max_results}
                ],
                "usage": {"include": True},
            },
        )
    except Exception as e:
        return ToolResult(
            ok=False, content=f"web_search failed: {e!r}", error="search_failed"
        )

    msg = completion.choices[0].message
    summary = (msg.content or "").strip()
    raw_anns = getattr(msg, "annotations", None) or []
    results: list = []
    for a in raw_anns:
        ad = a.model_dump() if hasattr(a, "model_dump") else (a or {})
        if ad.get("type") != "url_citation":
            continue
        uc = ad.get("url_citation") or {}
        url = uc.get("url") or ""
        title = (uc.get("title") or "").strip()
        snippet = (uc.get("content") or "").strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:297] + "..."
        results.append(f"- {title or url}\n  {url}\n  {snippet}")

    seat.web_searches += 1
    parts = [f"web_search: {query}"]
    if summary:
        parts.append("\nSUMMARY:\n" + summary)
    if results:
        parts.append("\nRESULTS:\n" + "\n\n".join(results))
    elif not summary:
        parts.append("\n(no results)")
    return ToolResult(ok=True, content="\n".join(parts))


def _h_web_fetch(seat: Seat, args: dict, ctx: RunCtx) -> ToolResult:
    """Fetch the given URL and return the text content (HTML stripped).

    Caps the response at ~200 KB so a long page doesn't blow context."""
    url = args.get("url", "")
    if not isinstance(url, str) or not url.strip():
        return ToolResult(ok=False, content="missing 'url'", error="bad_args")
    if not (url.startswith("http://") or url.startswith("https://")):
        return ToolResult(
            ok=False, content="url must start with http:// or https://", error="bad_args"
        )

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; harness-bot/1.0; "
                "+https://github.com/anthropics/claude-code)"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=seat.tool_timeout_s) as r:
            raw = r.read(2_000_000)
            content_type = r.headers.get("Content-Type", "")
            final_url = r.url
    except Exception as e:
        return ToolResult(
            ok=False, content=f"fetch error: {e!r}", error="fetch_failed"
        )

    text = raw.decode(errors="replace")
    is_html = "html" in content_type.lower() or "<html" in text[:1000].lower()
    if is_html:
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"&quot;", '"', text)
        text = re.sub(r"&#39;", "'", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

    MAX = 200_000
    truncated = len(text) > MAX
    if truncated:
        text = text[:MAX] + "\n\n[... truncated]"
    header = f"URL: {final_url}\nContent-Type: {content_type}\nBytes: {len(raw)}{' (truncated)' if truncated else ''}\n\n"
    return ToolResult(ok=True, content=header + text)


def _h_spawn(seat: Seat, args: dict, ctx: RunCtx) -> ToolResult:
    """Spawn a same-kind child: a copy of the parent's seat config at
    depth+1, with a fresh history starting from `prompt`. Bounded by
    `max_depth` (recursion) and `max_children` (breadth per seat)."""
    prompt = args.get("prompt", "")

    new_depth = seat.depth + 1
    if new_depth > seat.max_depth:
        return ToolResult(
            ok=False,
            content=f"depth limit: depth {new_depth} > max {seat.max_depth}",
            error="depth_exceeded",
        )

    # Critical section: breadth, child_count, child id mint, live seats.
    with ctx._spawn_lock:
        if seat.child_count >= seat.max_children:
            if seat.max_children <= 0:
                msg = (
                    "spawn unavailable: this agent's max_children is 0. "
                    "Relaunch with --max-children N (and --max-depth N if it's 0)."
                )
            else:
                msg = (
                    f"breadth limit: this agent's max_children={seat.max_children}, "
                    f"already spawned {seat.child_count}. Relaunch with a higher "
                    f"--max-children if you need more siblings."
                )
            return ToolResult(ok=False, content=msg, error="breadth_exceeded")
        seat.child_count += 1
        child_id = f"{seat.id}.{seat.child_count}"
        ctx.live_seats.add(child_id)

    # Mint a child with the parent's exact config (capability attenuation
    # is automatic — child == parent).
    child = Seat(
        id=child_id,
        parent_id=seat.id,
        depth=new_depth,
        model=seat.model,
        system_prompt=seat.system_prompt,
        tools=seat.tools,
        web=seat.web,
        max_turns=seat.max_turns,
        max_depth=seat.max_depth,
        max_children=seat.max_children,
        tool_timeout_s=seat.tool_timeout_s,
        web_max_results=seat.web_max_results,
        web_search_context_size=seat.web_search_context_size,
        provider=seat.provider,
        history=[{"role": "user", "content": prompt}],
    )

    ctx.log.write(
        seat,
        "spawn",
        {
            "child_id": child_id,
            "prompt": prompt,
            "depth": new_depth,
        },
    )

    from .driver import run_seat  # lazy to avoid cycle

    try:
        child_result = run_seat(child, ctx)
    finally:
        with ctx._spawn_lock:
            ctx.live_seats.discard(child_id)

    final = child.submit_result if child.submit_result is not None else child_result.content
    return ToolResult(
        ok=child_result.ok and child.halt_reason == "submit",
        content=final or "",
        meta={
            "child_id": child_id,
            "child_halt_reason": child.halt_reason,
        },
    )


# ---- Registration ----------------------------------------------------------


def register_builtins() -> None:
    register(
        ToolSpec(
            name="bash",
            description=(
                "Run an arbitrary shell command via `bash -c`. Returns "
                "stdout, stderr, and exit code. This tool is gated — each "
                "call requires explicit user approval before it runs. "
                "Prefer code_exec for Python work; use bash when you need "
                "the shell (file ops, installing packages, running other "
                "binaries, pipelines)."
            ),
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
            handler=_h_bash,
            risky=True,
        )
    )
    register(
        ToolSpec(
            name="code_exec",
            description=(
                "Execute Python source in a shared scratch directory. "
                "Returns stdout, stderr, and exit code."
            ),
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
            handler=_h_code_exec,
        )
    )
    register(
        ToolSpec(
            name="submit",
            description="End this seat and return the given result to the caller.",
            parameters={
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
            },
            handler=_h_submit,
        )
    )
    register(
        ToolSpec(
            name="web_search",
            description=(
                "Search the web. Returns a short list of relevant results "
                "(title, url, snippet) — fetch the URLs that look most "
                "promising with web_fetch. Use specific queries with year "
                "and entity names for fresh results."
            ),
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=_h_web_search,
        )
    )
    register(
        ToolSpec(
            name="web_fetch",
            description=(
                "Fetch the body of a URL (HTML is tag-stripped). Capped at "
                "~200 KB. Use after web_search to read a specific source."
            ),
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            handler=_h_web_fetch,
        )
    )
    register(
        ToolSpec(
            name="spawn",
            description=(
                "Spawn a sub-agent to handle a focused sub-task. The child "
                "is a fresh copy of you (same model, prompt, tools, limits) "
                "at depth+1, running its own driver loop from a fresh "
                "history starting with `prompt`. Blocks until the child "
                "submits, then returns its submitted result to you. "
                "Bounded by your max_depth (recursion) and max_children "
                "(siblings per turn)."
            ),
            parameters={
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"],
            },
            handler=_h_spawn,
        )
    )


register_builtins()
