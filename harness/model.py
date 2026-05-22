from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional

from .log import Log
from .types import Seat, ToolSpec


# ---- Normalized response shape ---------------------------------------------


@dataclass(frozen=True)
class ModelToolCall:
    id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class ModelResponse:
    text: Optional[str]
    tool_calls: list
    usage_usd: float
    raw_assistant_message: dict   # OpenAI-shaped message dict, ready to append


# ---- Client setup ----------------------------------------------------------


_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI
        from . import credentials

        credentials.inject_env()
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. "
                "Run `python3 ui.py` and `/login <key>`, or export the variable."
            )
        _CLIENT = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return _CLIENT


def _extract_usage(usage: Any) -> dict:
    """Pull tokens + cost from an OpenRouter response.usage object.

    Returns ``{prompt_tokens, completion_tokens, total_tokens, usd, usd_source}``
    where any of the token counts may be absent if the provider didn't report
    them. ``usd_source`` is either ``"openrouter"`` (real cost from the
    provider) or ``"estimate"`` (1c flat fallback).
    """
    out: dict = {}
    if usage is None:
        return {"usd": 0.01, "usd_source": "estimate"}
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = getattr(usage, k, None)
        if v is not None:
            try:
                out[k] = int(v)
            except (TypeError, ValueError):
                pass
    cost = getattr(usage, "cost", None)
    if cost is not None:
        try:
            out["usd"] = float(cost)
            out["usd_source"] = "openrouter"
        except (TypeError, ValueError):
            pass
    if "usd" not in out:
        out["usd"] = 0.01
        out["usd_source"] = "estimate"
    return out


def _build_tools_param(tool_specs: List[ToolSpec]) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
            },
        }
        for s in tool_specs
    ]


def _append_web_tools(tools_param: list, seat: Seat) -> list:
    """If the seat is granted server-side web tools, append them in the format
    OpenRouter expects. The tool execution happens entirely on OpenRouter's
    side; we never see a corresponding tool_call in our adjudicator path."""
    if not seat.web:
        return tools_param
    out = list(tools_param)
    if "search" in seat.web:
        out.append({
            "type": "openrouter:web_search",
            "openrouter:web_search": {
                "max_results": seat.web_max_results,
                "search_context_size": seat.web_search_context_size,
            },
        })
    if "fetch" in seat.web:
        out.append({"type": "openrouter:web_fetch"})
    return out


def _extract_citations(msg: Any) -> list:
    """OpenRouter attaches url_citation annotations on the assistant message
    when its web tools fire. The openai SDK exposes them on .annotations
    (or via model_dump). Return a list of {url, title, content?} dicts."""
    raw_anns = getattr(msg, "annotations", None)
    if raw_anns is None:
        try:
            raw_anns = msg.model_dump().get("annotations")
        except Exception:
            raw_anns = None
    if not raw_anns:
        return []
    cites: list = []
    for a in raw_anns:
        if isinstance(a, dict):
            t = a.get("type")
            uc = a.get("url_citation") or {}
        else:
            t = getattr(a, "type", None)
            uc_obj = getattr(a, "url_citation", None)
            uc = uc_obj.model_dump() if uc_obj is not None and hasattr(uc_obj, "model_dump") else (uc_obj or {})
        if t == "url_citation":
            cites.append({
                "url": uc.get("url"),
                "title": uc.get("title"),
                "content": (uc.get("content") or "")[:500] or None,
                "start_index": uc.get("start_index"),
                "end_index": uc.get("end_index"),
            })
    return cites


# ---- The single chokepoint -------------------------------------------------


def call_model(seat: Seat, tool_specs: List[ToolSpec], log: Log) -> ModelResponse:
    """The ONE wrapper. Builds messages, attaches tool schemas, calls
    OpenRouter, logs request and response, debits the seat's budget."""
    messages = [{"role": "system", "content": seat.prompt}] + list(seat.history)
    local_tools = _build_tools_param(tool_specs)
    tools_param = _append_web_tools(local_tools, seat)
    tool_summary = [t["function"]["name"] for t in local_tools]
    web_summary = [t["type"] for t in tools_param if t.get("type", "").startswith("openrouter:")]
    log.write(
        seat,
        "model_request",
        {
            "model": seat.model,
            "messages": messages,
            "tools": tool_summary,
            "web_tools": web_summary,
        },
    )

    completion = _client().chat.completions.create(
        model=seat.model,
        messages=messages,
        tools=tools_param if tools_param else None,
        tool_choice="auto" if tools_param else None,
        # Ask OpenRouter to populate usage.cost with the actual billed cost
        # (token cost + web search cost, when web tools fired).
        extra_body={"usage": {"include": True}},
    )
    msg = completion.choices[0].message
    raw_msg: dict = {"role": "assistant", "content": msg.content}
    tcs: List[ModelToolCall] = []
    if getattr(msg, "tool_calls", None):
        raw_tcs = []
        for tc in msg.tool_calls:
            tcs.append(
                ModelToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments_json=tc.function.arguments or "{}",
                )
            )
            raw_tcs.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
            )
        raw_msg["tool_calls"] = raw_tcs

    usage = _extract_usage(getattr(completion, "usage", None))
    usd = usage["usd"]
    seat.budget.usd_remaining -= usd
    seat.tokens_prompt += int(usage.get("prompt_tokens", 0) or 0)
    seat.tokens_completion += int(usage.get("completion_tokens", 0) or 0)

    citations = _extract_citations(msg)
    if citations:
        # OpenRouter charges roughly $4 per 1,000 search results returned
        # (Exa pricing pass-through). usage.cost from OpenRouter should
        # already include this; we record the count for visibility.
        seat.web_searches += len(citations)
        # Preserve citations in the assistant history so the model can
        # reference them on follow-up turns.
        raw_msg["annotations"] = [
            {"type": "url_citation", "url_citation": c} for c in citations
        ]

    log.write(
        seat,
        "model_response",
        {
            "text": msg.content,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments_json}
                for tc in tcs
            ],
            "usage": usage,
            "citations": citations,
            "model": seat.model,
        },
    )
    return ModelResponse(
        text=msg.content, tool_calls=tcs, usage_usd=usd, raw_assistant_message=raw_msg
    )
