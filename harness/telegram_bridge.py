"""Telegram bridge for the harness.

Runs a python-telegram-bot Application in a background asyncio loop so it
can live alongside the textual UI. Each allowed Telegram chat gets its own
ChatSession (independent of the UI's local chat) and its own run dir under
runs/. Plain text auto-starts a chat with the default agent. Slash commands
mirror the UI's: /chat, /end, /run, /agents, /status, /model, /approve,
/deny, /auto, /help.

Access is gated by a numeric chat-id allowlist (env
`TELEGRAM_ALLOWED_CHAT_IDS`, comma-separated). Following OpenClaw's
`allowFrom` pattern — required, no open-by-default. The bot refuses to
start without it.

While a turn runs (blocking) in a worker thread, an asyncio tail task
forwards tool_call / tool_result / spawn / submit / halt / denial log
entries and announces any pending approvals so the user can /approve or
/deny them from Telegram.
"""
from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import threading
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

from .forest import run_forest
from .session import ChatSession, start_chat
from .types import Agent


# --------------------------------------------------------------------------- #
# Per-chat state
# --------------------------------------------------------------------------- #


@dataclass
class BotChat:
    """Per-Telegram-chat state. Independent from the textual UI's chat."""
    telegram_chat_id: int
    agent_name: Optional[str] = None
    chat: Optional[ChatSession] = None
    run_path: Optional[Path] = None
    seen_seq: int = 0
    # Set during a /run; the bridge needs it to forward approvals & resolve
    # them via ctx.resolve_approval.
    run_ctx: object = None
    auto_approve: bool = False
    session_model: Optional[str] = None
    busy: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# --------------------------------------------------------------------------- #
# Bridge
# --------------------------------------------------------------------------- #


HELP_TEXT = (
    "harness telegram bridge\n\n"
    "just type a message — a chat with the chat agent starts.\n\n"
    "commands:\n"
    "  /chat [agent] [flags]  start a chat (default: chat)\n"
    "  /end                   end the active chat\n"
    "  /run <agent> [flags] <message>   one-shot task\n"
    "  /agents                list available agents\n"
    "  /status                auth, model, chat state\n"
    "  /model <id|->          session model override\n"
    "  /approve <id>          approve a risky tool call\n"
    "  /deny <id>             deny a risky tool call\n"
    "  /auto on|off           session-wide auto-approve\n"
    "  /help                  show this\n"
)


def _format_log_entry(e: dict) -> Optional[str]:
    """Render a JSONL entry for Telegram. Returns None for entries we skip
    (model_request / model_response — too noisy for chat)."""
    t = e["type"]
    p = e.get("payload", {}) or {}
    sid = e.get("seat_id") or "-"
    if t in ("model_request", "model_response"):
        return None
    if t == "tool_call":
        args = json.dumps(p.get("args"))
        if len(args) > 200:
            args = args[:200] + "…"
        return f"→ [{sid}] {p.get('tool')}  {args}"
    if t == "tool_result":
        glyph = "✓" if p.get("ok") else "✗"
        err = p.get("error")
        suffix = f"  err={err}" if err else ""
        return f"  {glyph} [{sid}] {p.get('tool')}{suffix}"
    if t == "spawn":
        return f"spawn → {p.get('child_id')} depth={p.get('depth')}"
    if t == "submit":
        r = p.get("result") or ""
        if len(r) > 400:
            r = r[:400] + "…"
        return f"submit [{sid}]: {r!r}"
    if t == "halt":
        return f"halt [{sid}] reason={p.get('reason')}"
    if t == "denial":
        return f"denial [{sid}] tool={p.get('tool')} reason={p.get('reason')}"
    return None


def _read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return out


TELEGRAM_MSG_LIMIT = 4000  # under the 4096 hard limit, leaving slack


def _chunk(text: str, limit: int = TELEGRAM_MSG_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    while text:
        parts.append(text[:limit])
        text = text[limit:]
    return parts


class TelegramBridge:
    def __init__(
        self,
        token: str,
        allowed_ids: set[int],
        runs_dir: Path,
        default_chat_agent: str,
        load_agent: Callable[[str], Agent],
        list_agents: Callable[[], list[str]],
        parse_overrides: Callable[[list[str]], tuple[dict, list[str]]],
        apply_overrides: Callable[[Agent, dict], Agent],
        on_log: Callable[[str], None],
    ) -> None:
        self.token = token
        self.allowed_ids = allowed_ids
        self.runs_dir = runs_dir
        self.default_chat_agent = default_chat_agent
        self.load_agent = load_agent
        self.list_agents = list_agents
        self.parse_overrides = parse_overrides
        self.apply_overrides = apply_overrides
        self.on_log = on_log  # write a line back to the textual UI

        self.chats: dict[int, BotChat] = {}
        self._auto_approve_session: bool = False
        self._stop_event: Optional[asyncio.Event] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._app = None

    # ---- lifecycle ----------------------------------------------------- #

    def start_in_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("bridge already running")
        self._thread = threading.Thread(
            target=self._thread_main, name="harness-telegram", daemon=True
        )
        self._thread.start()

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception:
            self.on_log(f"telegram bridge crashed:\n{traceback.format_exc()}")

    def stop(self) -> None:
        loop = self._loop
        stop = self._stop_event
        if loop is None or stop is None:
            return
        loop.call_soon_threadsafe(stop.set)
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._thread = None
        self._loop = None
        self._stop_event = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    async def _main(self) -> None:
        from telegram.ext import (
            ApplicationBuilder,
            CommandHandler,
            MessageHandler,
            filters,
        )

        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()

        app = ApplicationBuilder().token(self.token).build()
        self._app = app
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("help", self._cmd_help))
        app.add_handler(CommandHandler("agents", self._cmd_agents))
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("model", self._cmd_model))
        app.add_handler(CommandHandler("chat", self._cmd_chat))
        app.add_handler(CommandHandler("end", self._cmd_end))
        app.add_handler(CommandHandler("run", self._cmd_run))
        app.add_handler(CommandHandler("approve", self._cmd_approve))
        app.add_handler(CommandHandler("deny", self._cmd_deny))
        app.add_handler(CommandHandler("auto", self._cmd_auto))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text))

        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        self.on_log(f"telegram bridge listening · allowed_ids={sorted(self.allowed_ids)}")
        try:
            await self._stop_event.wait()
        finally:
            try:
                await app.updater.stop()
            except Exception:
                pass
            try:
                await app.stop()
            except Exception:
                pass
            try:
                await app.shutdown()
            except Exception:
                pass
            self.on_log("telegram bridge stopped")

    # ---- helpers ------------------------------------------------------- #

    def _allowed(self, update) -> bool:
        chat = update.effective_chat
        user = update.effective_user
        ids = {x for x in (chat.id if chat else None, user.id if user else None) if x is not None}
        return bool(ids & self.allowed_ids)

    def _state(self, chat_id: int) -> BotChat:
        s = self.chats.get(chat_id)
        if s is None:
            s = BotChat(telegram_chat_id=chat_id)
            self.chats[chat_id] = s
        return s

    async def _reply(self, update, text: str) -> None:
        for part in _chunk(text):
            await update.effective_chat.send_message(part)

    async def _send(self, chat_id: int, text: str) -> None:
        if self._app is None:
            return
        for part in _chunk(text):
            await self._app.bot.send_message(chat_id, part)

    # ---- handlers ------------------------------------------------------ #

    async def _cmd_start(self, update, context) -> None:
        if not self._allowed(update):
            await self._reply(update,
                f"not authorized. your chat id is {update.effective_chat.id}; "
                "add it to TELEGRAM_ALLOWED_CHAT_IDS.")
            return
        await self._reply(update, "harness bridge ready. " + HELP_TEXT)

    async def _cmd_help(self, update, context) -> None:
        if not self._allowed(update):
            return
        await self._reply(update, HELP_TEXT)

    async def _cmd_agents(self, update, context) -> None:
        if not self._allowed(update):
            return
        lines = []
        for name in self.list_agents():
            try:
                a = self.load_agent(name)
                bits = f"tools={list(a.tools)} model={a.model}"
                if a.web:
                    bits += f" web={list(a.web)}"
                lines.append(f"• {name}  {bits}")
            except Exception as e:
                lines.append(f"• {name}  (load error: {e})")
        await self._reply(update, "\n".join(lines) or "(no agents)")

    async def _cmd_status(self, update, context) -> None:
        if not self._allowed(update):
            return
        s = self._state(update.effective_chat.id)
        lines = []
        if s.chat is not None:
            seat = s.chat.seat
            lines.append(
                f"chat: active  agent={s.agent_name}  turns={seat.turns_used}  "
                f"tokens={seat.tokens_prompt}/{seat.tokens_completion}  "
                f"spent=${seat.cost_usd:.4f}"
            )
        else:
            lines.append("chat: inactive")
        if s.session_model:
            lines.append(f"session model: {s.session_model}")
        lines.append(f"auto-approve: {'ON' if s.auto_approve or self._auto_approve_session else 'off'}")
        await self._reply(update, "\n".join(lines))

    async def _cmd_model(self, update, context) -> None:
        if not self._allowed(update):
            return
        s = self._state(update.effective_chat.id)
        arg = (" ".join(context.args)).strip() if context.args else ""
        if not arg:
            await self._reply(update, f"current session model: {s.session_model or '(default)'}")
            return
        if arg == "-":
            s.session_model = None
            await self._reply(update, "session model cleared")
            return
        s.session_model = arg
        await self._reply(update, f"session model set to {arg}")

    async def _cmd_auto(self, update, context) -> None:
        if not self._allowed(update):
            return
        arg = (context.args[0].lower() if context.args else "")
        if arg in ("on", "true", "yes", "1"):
            self._auto_approve_session = True
            await self._reply(update, "⚠ session auto-approve ON for new runs/chats")
        elif arg in ("off", "false", "no", "0"):
            self._auto_approve_session = False
            await self._reply(update, "session auto-approve OFF")
        else:
            await self._reply(update,
                f"auto-approve: {'ON' if self._auto_approve_session else 'off'}\n"
                "usage: /auto on|off")

    async def _cmd_chat(self, update, context) -> None:
        if not self._allowed(update):
            return
        s = self._state(update.effective_chat.id)
        if s.chat is not None:
            await self._reply(update,
                f"chat already active (agent: {s.agent_name}). /end to stop it first.")
            return
        tokens = list(context.args or [])
        agent_name = self.default_chat_agent
        if tokens and not tokens[0].startswith("--"):
            agent_name = tokens[0]
            tokens = tokens[1:]
        try:
            overrides, leftover = self.parse_overrides(tokens)
        except ValueError as e:
            await self._reply(update, f"flag error: {e}")
            return
        if leftover:
            await self._reply(update, f"unexpected arg(s): {leftover}")
            return
        if agent_name not in self.list_agents():
            await self._reply(update,
                f"unknown agent: {agent_name} (known: {', '.join(self.list_agents())})")
            return
        per_run_auto = overrides.pop("auto-approve", None)
        auto = (self._auto_approve_session if per_run_auto is None
                else per_run_auto.lower() in ("1", "true", "yes", "on"))
        try:
            agent = self.load_agent(agent_name)
            agent = self.apply_overrides(agent, overrides)
            if s.session_model and "model" not in overrides:
                agent = replace(agent, model=s.session_model)
        except Exception as e:
            await self._reply(update, f"bad agent config: {e}")
            return

        ts = time.strftime("%Y%m%d-%H%M%S")
        run_dir = self.runs_dir / f"tg-chat-{agent_name}-{ts}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        s.chat = start_chat(
            agent=agent,
            log_path=run_dir / "log.jsonl",
            workdir=run_dir / "wd",
            auto_approve=auto,
        )
        s.agent_name = agent_name
        s.run_path = run_dir / "log.jsonl"
        s.seen_seq = 0
        s.auto_approve = auto
        s.run_ctx = s.chat.ctx
        note = "  (auto-approve ON)" if auto else ""
        await self._reply(update,
            f"chat started · agent={agent_name} · model={agent.model}{note}\n"
            f"tools={list(agent.tools)}"
            + (f"  web={list(agent.web)}" if agent.web else "")
        )

    async def _cmd_end(self, update, context) -> None:
        if not self._allowed(update):
            return
        s = self._state(update.effective_chat.id)
        if s.chat is None:
            await self._reply(update, "no chat active")
            return
        try:
            s.chat.close()
        except Exception:
            pass
        seat = s.chat.seat
        await self._reply(update,
            f"chat ended · turns={seat.turns_used} · spent=${seat.cost_usd:.4f}")
        s.chat = None
        s.agent_name = None
        s.run_path = None
        s.run_ctx = None

    async def _cmd_run(self, update, context) -> None:
        if not self._allowed(update):
            return
        s = self._state(update.effective_chat.id)
        if s.busy:
            await self._reply(update, "busy — wait for the current turn to finish")
            return
        tokens = list(context.args or [])
        if not tokens:
            await self._reply(update,
                f"usage: /run <agent> [flags] <message>\nagents: {', '.join(self.list_agents())}")
            return
        agent_name = tokens[0]
        if agent_name not in self.list_agents():
            await self._reply(update,
                f"unknown agent: {agent_name} (known: {', '.join(self.list_agents())})")
            return
        try:
            overrides, remaining = self.parse_overrides(tokens[1:])
        except ValueError as e:
            await self._reply(update, f"flag error: {e}")
            return
        message = " ".join(remaining).strip()
        if not message:
            await self._reply(update, "missing message")
            return
        per_run_auto = overrides.pop("auto-approve", None)
        auto = (self._auto_approve_session if per_run_auto is None
                else per_run_auto.lower() in ("1", "true", "yes", "on"))
        try:
            agent = self.load_agent(agent_name)
            agent = self.apply_overrides(agent, overrides)
            if s.session_model and "model" not in overrides:
                agent = replace(agent, model=s.session_model)
        except Exception as e:
            await self._reply(update, f"bad agent config: {e}")
            return

        ts = time.strftime("%Y%m%d-%H%M%S")
        run_dir = self.runs_dir / f"tg-{agent_name}-{ts}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "log.jsonl"
        s.run_path = log_path
        s.seen_seq = 0
        s.busy = True
        note = "  (auto-approve ON)" if auto else ""
        await self._reply(update,
            f"→ running {agent_name} · model={agent.model}{note}\n{run_dir}")

        chat_id = update.effective_chat.id

        def grab_ctx(ctx):
            s.run_ctx = ctx

        tail_task = asyncio.create_task(self._tail_loop(chat_id))
        try:
            res = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: run_forest(
                    agent=agent,
                    log_path=log_path,
                    workdir=run_dir / "wd",
                    user_message=message,
                    on_ctx=grab_ctx,
                    auto_approve=auto,
                ),
            )
            seat = res.root_seat
            summary = (
                f"done · submit={seat.submit_result!r} · halt={seat.halt_reason} · "
                f"turns={seat.turns_used} · tokens={seat.tokens_prompt}/{seat.tokens_completion} · "
                f"spent=${seat.cost_usd:.4f}"
            )
        except Exception:
            summary = f"FAILED:\n{traceback.format_exc()}"
        finally:
            tail_task.cancel()
            try:
                await tail_task
            except asyncio.CancelledError:
                pass
            await self._tail_once(chat_id)
            s.busy = False
            s.run_ctx = None
        await self._send(chat_id, summary)

    async def _cmd_approve(self, update, context) -> None:
        await self._resolve_approval(update, context, "approve")

    async def _cmd_deny(self, update, context) -> None:
        await self._resolve_approval(update, context, "deny")

    async def _resolve_approval(self, update, context, decision: str) -> None:
        if not self._allowed(update):
            return
        s = self._state(update.effective_chat.id)
        if not context.args:
            await self._reply(update, f"usage: /{decision} <approval-id>")
            return
        aid = context.args[0]
        ctx = s.run_ctx
        if ctx is None:
            await self._reply(update, "no active run with pending approvals")
            return
        req = ctx.resolve_approval(aid, decision)
        if req is None:
            await self._reply(update, f"no such pending approval: {aid}")
            return
        glyph = "✓ approved" if decision == "approve" else "✗ denied"
        await self._reply(update, f"{glyph}  {aid}  ({req.tool_name})")

    async def _on_text(self, update, context) -> None:
        if not self._allowed(update):
            return
        text = (update.message.text or "").strip()
        if not text:
            return
        chat_id = update.effective_chat.id
        s = self._state(chat_id)
        if s.chat is None:
            # Auto-start a default chat, mirroring ui.py's behavior.
            class _FakeCtx:
                args: list = []
            await self._cmd_chat(update, _FakeCtx())
            if s.chat is None:
                return
        async with s.lock:
            if s.busy:
                await self._reply(update, "busy — wait for the current turn to finish")
                return
            s.busy = True
            tail_task = asyncio.create_task(self._tail_loop(chat_id))
            try:
                reply = await asyncio.get_running_loop().run_in_executor(
                    None, s.chat.send, text
                )
            except Exception:
                reply = f"chat error:\n{traceback.format_exc()}"
            finally:
                tail_task.cancel()
                try:
                    await tail_task
                except asyncio.CancelledError:
                    pass
                await self._tail_once(chat_id)
                s.busy = False
        if reply:
            await self._send(chat_id, reply)

    # ---- live tail ----------------------------------------------------- #

    async def _tail_loop(self, chat_id: int) -> None:
        try:
            while True:
                await asyncio.sleep(0.7)
                await self._tail_once(chat_id)
        except asyncio.CancelledError:
            raise

    async def _tail_once(self, chat_id: int) -> None:
        s = self.chats.get(chat_id)
        if s is None or s.run_path is None:
            return
        entries = _read_log(s.run_path)
        new = [e for e in entries if e.get("seq", 0) > s.seen_seq]
        for e in new:
            s.seen_seq = max(s.seen_seq, e.get("seq", 0))
            line = _format_log_entry(e)
            if line:
                try:
                    await self._send(chat_id, line)
                except Exception:
                    pass
        # Pending approvals: announce them.
        ctx = s.run_ctx
        if ctx is None:
            return
        try:
            pending = ctx.pending_undisplayed()
        except Exception:
            return
        for req in pending:
            args = json.dumps(req.args, indent=2)
            if len(args) > 1500:
                args = args[:1500] + "…"
            msg = (
                f"⚠ APPROVAL NEEDED  id={req.id}  seat={req.seat_id}  tool={req.tool_name}\n"
                f"{args}\n"
                f"reply with /approve {req.id}  or  /deny {req.id}"
            )
            try:
                await self._send(chat_id, msg)
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Convenience: parse the env allowlist
# --------------------------------------------------------------------------- #


def parse_allowlist(raw: str) -> set[int]:
    """Parse `TELEGRAM_ALLOWED_CHAT_IDS` — comma-separated integer ids."""
    out: set[int] = set()
    for tok in (raw or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.add(int(tok))
        except ValueError:
            raise ValueError(f"bad chat id in TELEGRAM_ALLOWED_CHAT_IDS: {tok!r}")
    return out


def load_config() -> tuple[str, set[int]]:
    """Return (token, allowed_ids), reading the env first, then the
    persistent credentials store (`~/.config/harness/credentials.json`).
    Raises ValueError if either piece is missing — open-by-default is
    refused on purpose."""
    from . import credentials
    token = credentials.get_telegram_token() or ""
    if not token:
        raise ValueError(
            "no telegram bot token configured. "
            "use /telegram login <token> or export TELEGRAM_BOT_TOKEN."
        )
    allowed = set(credentials.get_telegram_allowed_ids())
    if not allowed:
        raise ValueError(
            "no allowed telegram chat ids configured. "
            "use /telegram allow <id>[,<id>...] or export TELEGRAM_ALLOWED_CHAT_IDS."
        )
    return token, allowed


# Back-compat alias for the original env-only loader.
read_env_config = load_config
