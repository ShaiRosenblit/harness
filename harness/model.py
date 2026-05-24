from __future__ import annotations

import datetime as _dt
import os
import time as _time
from dataclasses import dataclass
from typing import Any, List, Optional

from .log import Log
from .types import Seat, ToolSpec


def _harness_preamble() -> str:
    """Real-time context the model can rely on each turn: date, ISO timestamp,
    timezone. Prepended to the system prompt on every model_request so the
    agent never has to guess what 'today' is or how to date a search."""
    now = _dt.datetime.now().astimezone()
    return (
        f"[harness context]\n"
        f"  today: {now.date().isoformat()}\n"
        f"  now:   {now.isoformat(timespec='seconds')}\n"
        f"  tz:    {_time.tzname[_time.daylight]}\n"
        f"Use this date when reasoning about anything time-sensitive. "
        f"For time-sensitive web queries (latest, recent, current, "
        f"today's, this week's, this year's, etc.), include today's "
        f"year explicitly in the query so fresh results surface.\n"
    )


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
    """Pull tokens + cost from OpenRouter's response.usage."""
    out: dict = {}
    if usage is None:
        return {"usd": 0.0, "usd_source": "missing"}
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
        out["usd"] = 0.0
        out["usd_source"] = "missing"
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


def _extract_citations_from(raw_anns) -> list:
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
            uc = (uc_obj.model_dump() if uc_obj is not None and hasattr(uc_obj, "model_dump")
                  else (uc_obj or {}))
        if t == "url_citation":
            cites.append({
                "url": uc.get("url"),
                "title": uc.get("title"),
                "content": (uc.get("content") or "")[:500] or None,
                "start_index": uc.get("start_index"),
                "end_index": uc.get("end_index"),
            })
    return cites


def _extract_citations(msg: Any) -> list:
    raw_anns = getattr(msg, "annotations", None)
    if raw_anns is None:
        try:
            raw_anns = msg.model_dump().get("annotations")
        except Exception:
            raw_anns = None
    return _extract_citations_from(raw_anns)


# ---- The single chokepoint -------------------------------------------------


def call_model(seat: Seat, tool_specs: List[ToolSpec], log: Log) -> ModelResponse:
    """The ONE wrapper. Builds messages, attaches tool schemas (local + web),
    calls OpenRouter, logs request and response, and accumulates the seat's
    observability counters (cost_usd, tokens_*, web_searches). Counters are
    NOT enforced — `max_turns` and `max_depth`/`max_children` are the caps."""
    system_content = _harness_preamble() + "\n" + seat.system_prompt
    messages = [{"role": "system", "content": system_content}] + list(seat.history)
    tools_param = _build_tools_param(tool_specs)
    log.write(
        seat,
        "model_request",
        {
            "model": seat.model,
            "messages": messages,
            "tools": [t["function"]["name"] for t in tools_param],
            "provider_pref": list(seat.provider) if seat.provider else None,
        },
    )

    extra_body: dict = {"usage": {"include": True}}
    if seat.provider:
        # OpenRouter provider routing — restrict routing to the named
        # providers in preference order. Use this when a model's tool-call
        # template is only parsed correctly by a subset of providers (e.g.
        # some Kimi K2.6 providers leak raw `<|tool_call_begin|>` tokens
        # into the response content instead of returning structured
        # tool_calls). `allow_fallbacks=False` makes failures explicit
        # rather than silently routing to a broken provider.
        extra_body["provider"] = {
            "order": list(seat.provider),
            "allow_fallbacks": False,
        }

    # Stream the response. Non-streaming responses from OpenRouter are
    # padded with whitespace keepalive lines while the upstream provider
    # is generating; for slow/long requests this can result in a truncated
    # body that json.loads chokes on ("Expecting value: line N column 1").
    # In streaming mode those keepalives are proper SSE comments
    # (`: OPENROUTER PROCESSING`) that the SDK's SSE parser ignores.
    stream = _client().chat.completions.create(
        model=seat.model,
        messages=messages,
        tools=tools_param if tools_param else None,
        tool_choice="auto" if tools_param else None,
        stream=True,
        stream_options={"include_usage": True},
        extra_body=extra_body,
    )

    content_parts: list = []
    reasoning_parts: list = []
    tool_calls_by_index: dict = {}   # idx -> {"id", "name", "args_parts"}
    annotations_raw: list = []
    usage_obj: Any = None
    provider_used: Optional[str] = None
    final_msg_obj: Any = None        # for citation extraction fallback

    for chunk in stream:
        if not provider_used:
            # OpenRouter puts the actual upstream provider name in the
            # `provider` field of each chunk. Read it from model_extra so
            # we don't accidentally pick up an SDK-side default attribute.
            chunk_extras = getattr(chunk, "model_extra", None) or {}
            provider_used = chunk_extras.get("provider")
        if getattr(chunk, "usage", None):
            usage_obj = chunk.usage
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        # Some providers attach the final assembled message on the last
        # choice; keep a reference so we can re-use _extract_citations.
        msg_on_choice = getattr(choice, "message", None)
        if msg_on_choice is not None:
            final_msg_obj = msg_on_choice

        if getattr(delta, "content", None):
            content_parts.append(delta.content)

        # Reasoning (Kimi K2.6 et al.) arrives separately on delta.reasoning,
        # not in content. Capture it for the log; don't pass it back to the
        # caller (the driver only acts on content + tool_calls).
        extras = getattr(delta, "model_extra", None) or {}
        r = extras.get("reasoning")
        if r:
            reasoning_parts.append(r)

        # Annotations (web citations) may arrive on a delta, on a
        # constructed message, or as a top-level chunk attribute.
        for src in (delta, msg_on_choice):
            anns = getattr(src, "annotations", None) if src is not None else None
            if anns:
                for a in anns:
                    annotations_raw.append(a)

        for tc in getattr(delta, "tool_calls", None) or []:
            idx = getattr(tc, "index", 0) or 0
            entry = tool_calls_by_index.setdefault(
                idx, {"id": None, "name": None, "args_parts": []}
            )
            if getattr(tc, "id", None) and not entry["id"]:
                entry["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None) and not entry["name"]:
                    entry["name"] = fn.name
                if getattr(fn, "arguments", None):
                    entry["args_parts"].append(fn.arguments)

    content = "".join(content_parts) if content_parts else None
    raw_msg: dict = {"role": "assistant", "content": content}
    tcs: List[ModelToolCall] = []
    if tool_calls_by_index:
        raw_tcs = []
        for idx in sorted(tool_calls_by_index.keys()):
            e = tool_calls_by_index[idx]
            args_json = "".join(e["args_parts"]) or "{}"
            call_id = e["id"] or f"call_{idx}"
            name = e["name"] or ""
            tcs.append(ModelToolCall(id=call_id, name=name, arguments_json=args_json))
            raw_tcs.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args_json},
                }
            )
        raw_msg["tool_calls"] = raw_tcs

    usage = _extract_usage(usage_obj)
    seat.cost_usd += usage["usd"]
    seat.tokens_prompt += int(usage.get("prompt_tokens", 0) or 0)
    seat.tokens_completion += int(usage.get("completion_tokens", 0) or 0)

    citations = _extract_citations_from(annotations_raw)
    if not citations and final_msg_obj is not None:
        citations = _extract_citations(final_msg_obj)
    if citations:
        seat.web_searches += len(citations)
        raw_msg["annotations"] = [
            {"type": "url_citation", "url_citation": c} for c in citations
        ]

    reasoning_text = "".join(reasoning_parts) if reasoning_parts else None
    log.write(
        seat,
        "model_response",
        {
            "text": content,
            "reasoning": reasoning_text,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments_json}
                for tc in tcs
            ],
            "usage": usage,
            "citations": citations,
            "model": seat.model,
            "provider": provider_used,
        },
    )
    return ModelResponse(
        text=content, tool_calls=tcs, raw_assistant_message=raw_msg
    )
