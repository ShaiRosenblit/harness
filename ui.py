#!/usr/bin/env python3
"""Interactive shell for the harness — a CLI-style UI built on Textual.

One scrolling output area on top, one prompt at the bottom. All actions are
slash-commands. Type /help for the list.

First run:  /login <key>     stores your OpenRouter key (0600) so future
                             launches and CLI tests skip authentication.

Then:       /run <policy> [--model M] <user message...>
"""
from __future__ import annotations

import json
import shlex
import shutil
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from harness import credentials  # noqa: E402
from harness.forest import run_forest  # noqa: E402
from harness.session import start_chat  # noqa: E402

from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual.widgets import Footer, Header, Input, RichLog  # noqa: E402


RUNS_DIR = ROOT / "runs"
POLICIES_DIR = ROOT / "policies"

DEFAULT_MODEL = "openai/gpt-4o-mini"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def list_policies() -> list[str]:
    return sorted(
        p.stem for p in POLICIES_DIR.glob("*.py") if not p.name.startswith("_")
    )


def list_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted(
        [p for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "log.jsonl").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def load_policy(module_name: str):
    import importlib
    mod = importlib.import_module(f"policies.{module_name}")
    return getattr(mod, "POLICY")


def with_model_override(policy, model: Optional[str]):
    """Return a copy of `policy` with model swapped (and child_policy too)."""
    if not model:
        return policy
    child = policy.child_policy
    if child is not None:
        child = replace(child, model=model)
    return replace(policy, model=model, child_policy=child)


def summarize_entry(e: dict) -> str:
    t = e["type"]
    p = e.get("payload", {}) or {}
    if t == "model_request":
        return f"[dim]model_request tools={p.get('tools')}[/dim]"
    if t == "model_response":
        tcs = p.get("tool_calls") or []
        u = p.get("usage") or {}
        suffix = ""
        if isinstance(u, dict):
            pt, ct = u.get("prompt_tokens"), u.get("completion_tokens")
            usd, src = u.get("usd"), u.get("usd_source")
            bits = []
            if pt is not None or ct is not None:
                bits.append(f"tok={pt or 0}/{ct or 0}")
            if usd is not None:
                bits.append(f"${usd:.5f}{'*' if src == 'estimate' else ''}")
            cites = p.get("citations") or []
            if cites:
                bits.append(f"+{len(cites)}cites")
            if bits:
                suffix = " [dim](" + " ".join(bits) + ")[/dim]"
        elif "usage_usd" in p:
            suffix = f" [dim](${p['usage_usd']:.5f}*)[/dim]"
        if tcs:
            return "[dim]model_response[/dim] tool_calls=[{}]{}".format(
                ", ".join(tc.get("name", "?") for tc in tcs), suffix
            )
        return f"[dim]model_response[/dim] text={(p.get('text') or '')[:80]!r}{suffix}"
    if t == "tool_call":
        return f"[b]→[/b] {p.get('tool')} [dim]{json.dumps(p.get('args'))[:120]}[/dim]"
    if t == "tool_result":
        ok = p.get("ok")
        glyph = "[green]✓[/green]" if ok else "[red]✗[/red]"
        return f"  {glyph} {p.get('tool')} [dim]err={p.get('error')}[/dim]"
    if t == "spawn":
        return f"[b magenta]spawn[/b magenta] → {p.get('child_id')}  budget=${p.get('budget_usd')}"
    if t == "submit":
        return f"[b green]submit[/b green] [italic]{(p.get('result') or '')[:120]!r}[/italic]"
    if t == "halt":
        return f"[red]halt[/red] reason={p.get('reason')}"
    if t == "denial":
        return f"[yellow]denial[/yellow] reason={p.get('reason')} tool={p.get('tool')}"
    return t


def read_log(path: Path) -> list[dict]:
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


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #


CSS = """
Screen { layout: vertical; }
#output { height: 1fr; padding: 0 1; }
#prompt { dock: bottom; }
"""


HELP_TEXT = """\
[b]how to use[/b]

  Just type a message and hit enter — a chat with the agent starts
  automatically. The agent can run Python via the [b]code_exec[/b] tool.
  Type [cyan]/end[/cyan] to end the chat.

[b]commands[/b]

  [cyan]/login[/cyan] [dim]<key>[/dim]         sign in (saves your OpenRouter API key, 0600)
  [cyan]/logout[/cyan]              clear the saved key
  [cyan]/status[/cyan]              auth, default model, chat state
  [cyan]/model[/cyan] [dim]<id|->[/dim]       set or clear the session model override
  [cyan]/policies[/cyan]            list available policies
  [cyan]/runs[/cyan]                list past runs
  [cyan]/view[/cyan] [dim]<run>[/dim]          replay a run's timeline
  [cyan]/tree[/cyan] [dim]<run>[/dim]          show a run's seat tree
  [cyan]/chat[/cyan] [dim][policy] [--model M][/dim]    start a chat (default policy: chat)
  [cyan]/end[/cyan]                 end the current chat
  [cyan]/run[/cyan]  [dim]<policy> [--model M] <message...>[/dim]   one-shot task
  [cyan]/clear[/cyan] · [cyan]/help[/cyan] · [cyan]/quit[/cyan]
"""


class HarnessApp(App):
    CSS = CSS
    TITLE = "harness"
    SUB_TITLE = "interactive agent harness"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("ctrl+l", "clear", "Clear"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._session_model: Optional[str] = None
        self._active_run: Optional[Path] = None
        self._active_seen_seq: int = 0
        self._is_running: bool = False
        self._chat = None  # active ChatSession or None
        self._chat_policy_name: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield RichLog(id="output", highlight=True, markup=True, wrap=True)
        yield Input(placeholder="type a message to chat, or /help for commands", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self._banner()
        if credentials.get_api_key():
            credentials.inject_env()
            self._line(
                f"[green]signed in[/green]  ·  model [b]{self._effective_model()}[/b]  "
                f"·  type a message to chat, or [cyan]/help[/cyan]"
            )
        else:
            self._line(
                "[yellow]not signed in[/yellow]  ·  start with [cyan]/login <key>[/cyan]  "
                "[dim](get one at https://openrouter.ai/keys)[/dim]"
            )
        self._line("")
        self.query_one("#prompt", Input).focus()
        self.set_interval(0.5, self._tail_active_run)

    # ---- output ---------------------------------------------------------- #

    def _line(self, text: str = "") -> None:
        self.query_one("#output", RichLog).write(text)

    def _echo(self, cmd: str) -> None:
        self._line(f"[dim]›[/dim] {cmd}")

    def _banner(self) -> None:
        out = self.query_one("#output", RichLog)
        out.write("[b]harness[/b]")
        out.write("[dim]" + "─" * 60 + "[/dim]")

    # ---- input dispatch ------------------------------------------------- #

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "prompt":
            return
        raw = (event.value or "").strip()
        event.input.value = ""
        if not raw:
            return
        self._echo(raw)
        try:
            self._dispatch(raw)
        except Exception:
            self._line(f"[red]error:[/red] {traceback.format_exc()}")

    def _dispatch(self, raw: str) -> None:
        if not raw.startswith("/"):
            # In an active chat, free text is the next user message.
            if self._chat is not None:
                self._send_chat(raw)
                return
            # Auto-start a chat on the first plain-text message.
            if not credentials.get_api_key():
                self._line("[red]not signed in.[/red]  use [cyan]/login <key>[/cyan] first.")
                return
            self._cmd_chat("")
            if self._chat is not None:
                self._send_chat(raw)
            return
        head, _, rest = raw.partition(" ")
        cmd = head[1:].lower()
        rest = rest.strip()

        handler = getattr(self, f"_cmd_{cmd}", None)
        if handler is None:
            self._line(f"[red]unknown command:[/red] {head}")
            return
        handler(rest)

    # ---- commands ------------------------------------------------------- #

    def _cmd_help(self, _rest: str) -> None:
        self._line(HELP_TEXT)

    def _cmd_quit(self, _rest: str) -> None:
        self.exit()

    def _cmd_exit(self, _rest: str) -> None:
        self.exit()

    def _cmd_clear(self, _rest: str) -> None:
        self.query_one("#output", RichLog).clear()
        self._banner()

    def _cmd_status(self, _rest: str) -> None:
        key = credentials.get_api_key()
        if key:
            self._line(f"auth: [green]signed in[/green]   key=[dim]{credentials.mask(key)}[/dim]")
        else:
            self._line("auth: [yellow]not signed in[/yellow]")
        self._line(f"default model: [b]{self._effective_model()}[/b]"
                   + ("  [dim](session override)[/dim]" if self._session_model else ""))
        if self._chat is not None:
            seat = self._chat.seat
            self._line(
                f"chat: [green]active[/green]   policy=[b]{self._chat_policy_name}[/b]   "
                f"turns=[b]{seat.turns_used}[/b]   "
                f"tokens=[b]{seat.tokens_prompt}[/b]/[b]{seat.tokens_completion}[/b] "
                f"[dim](prompt/completion)[/dim]   "
                f"budget=$[b]{seat.budget.usd_remaining:.4f}[/b]"
            )
        else:
            self._line("chat: [dim]inactive[/dim]")
        self._line(f"runs dir: [dim]{RUNS_DIR}[/dim]")
        self._line(f"credentials: [dim]{credentials.CREDENTIALS_PATH}[/dim]")

    def _cmd_login(self, rest: str) -> None:
        key = rest.strip()
        if not key:
            self._line("usage: [cyan]/login <openrouter-api-key>[/cyan]")
            self._line("[dim]paste your key right after /login. it's stored 0600 at "
                       f"{credentials.CREDENTIALS_PATH}.[/dim]")
            return
        if len(key) < 20:
            self._line("[red]that doesn't look like a real key (too short)[/red]")
            return
        try:
            path = credentials.save_api_key(key)
        except OSError as e:
            self._line(f"[red]failed to save:[/red] {e}")
            return
        credentials.inject_env()
        self._line(f"[green]✓ signed in[/green]   key=[dim]{credentials.mask(key)}[/dim]")
        self._line(f"[dim]saved to {path}[/dim]")

    def _cmd_logout(self, _rest: str) -> None:
        import os as _os
        credentials.clear_api_key()
        _os.environ.pop("OPENROUTER_API_KEY", None)
        self._line("[yellow]logged out[/yellow]  (saved key cleared)")

    def _cmd_model(self, rest: str) -> None:
        rest = rest.strip()
        if not rest:
            self._line(f"current: [b]{self._effective_model()}[/b]"
                       + ("  [dim](session override)[/dim]" if self._session_model else "  [dim](policy default)[/dim]"))
            self._line("usage: [cyan]/model openai/gpt-4o-mini[/cyan]   (or any OpenRouter model id)")
            self._line("       [cyan]/model -[/cyan]   to clear the override")
            return
        if rest == "-":
            self._session_model = None
            self._line("session model override cleared")
            return
        self._session_model = rest
        self._line(f"session model set to [b]{rest}[/b]")

    def _cmd_policies(self, _rest: str) -> None:
        for name in list_policies():
            try:
                p = load_policy(name)
                self._line(f"  [cyan]{name}[/cyan]  [dim]tools={list(p.tools)}  model={p.model}  budget=${p.budget_usd}[/dim]")
            except Exception as e:
                self._line(f"  [cyan]{name}[/cyan]  [red]load error: {e}[/red]")

    def _cmd_runs(self, _rest: str) -> None:
        runs = list_runs()
        if not runs:
            self._line("[dim](no runs yet)[/dim]")
            return
        for r in runs:
            mtime = datetime.fromtimestamp(r.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            self._line(f"  [cyan]{r.name}[/cyan]  [dim]{mtime}[/dim]")

    def _cmd_view(self, rest: str) -> None:
        run = self._resolve_run(rest)
        if run is None:
            return
        entries = read_log(run / "log.jsonl")
        if not entries:
            self._line("[dim](empty log)[/dim]")
            return
        for e in entries:
            self._render_entry(e)
        self._line(f"[dim]{len(entries)} entries from {run.name}[/dim]")

    def _cmd_tree(self, rest: str) -> None:
        run = self._resolve_run(rest)
        if run is None:
            return
        entries = read_log(run / "log.jsonl")
        self._line(build_tree(entries) or "(no seats)")

    def _cmd_run(self, rest: str) -> None:
        if self._is_running:
            self._line("[yellow]a run is already in progress; wait for it to finish[/yellow]")
            return
        if not rest:
            self._line("usage: [cyan]/run <policy> [--model M] <message...>[/cyan]")
            self._line(f"policies: {', '.join(list_policies())}")
            return
        try:
            tokens = shlex.split(rest)
        except ValueError as e:
            self._line(f"[red]bad quoting:[/red] {e}")
            return
        if not tokens:
            self._line("usage: [cyan]/run <policy> [--model M] <message...>[/cyan]")
            return
        policy_name = tokens[0]
        if policy_name not in list_policies():
            self._line(f"[red]unknown policy:[/red] {policy_name}  (known: {', '.join(list_policies())})")
            return

        # Parse --model override.
        i = 1
        model_override: Optional[str] = None
        while i < len(tokens) and tokens[i].startswith("--"):
            flag = tokens[i]
            if flag == "--model":
                if i + 1 >= len(tokens):
                    self._line("[red]--model needs a value[/red]")
                    return
                model_override = tokens[i + 1]
                i += 2
            else:
                self._line(f"[red]unknown flag:[/red] {flag}")
                return
        message = " ".join(tokens[i:]).strip()
        if not message:
            self._line("[red]missing user message[/red]")
            return

        if not credentials.get_api_key():
            self._line("[red]a run needs an API key.[/red]  use [cyan]/login <key>[/cyan] first.")
            return
        credentials.inject_env()

        policy = load_policy(policy_name)
        effective_model = model_override or self._session_model
        policy = with_model_override(policy, effective_model)

        ts = time.strftime("%Y%m%d-%H%M%S")
        run_dir = RUNS_DIR / f"{policy_name}-{ts}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        self._line(f"[b]→ running[/b] [cyan]{policy_name}[/cyan]  model=[b]{policy.model}[/b]  budget=$[b]{policy.budget_usd}[/b]")
        self._line(f"[dim]   {run_dir}[/dim]")
        self._active_run = run_dir / "log.jsonl"
        self._active_seen_seq = 0
        self._is_running = True
        self.run_worker(
            self._do_run(policy_name, policy, run_dir, message),
            exclusive=True,
            group="run",
            thread=True,
        )

    # ---- chat ----------------------------------------------------------- #

    def _cmd_chat(self, rest: str) -> None:
        if self._chat is not None:
            self._line(f"[yellow]chat already active[/yellow] (policy: {self._chat_policy_name}). "
                       "use [cyan]/end[/cyan] to end it first.")
            return
        try:
            tokens = shlex.split(rest) if rest else []
        except ValueError as e:
            self._line(f"[red]bad quoting:[/red] {e}")
            return
        policy_name = "chat"
        model_override: Optional[str] = None
        i = 0
        if tokens and not tokens[0].startswith("--"):
            policy_name = tokens[0]
            i = 1
        while i < len(tokens):
            if tokens[i] == "--model":
                if i + 1 >= len(tokens):
                    self._line("[red]--model needs a value[/red]")
                    return
                model_override = tokens[i + 1]
                i += 2
            else:
                self._line(f"[red]unknown flag:[/red] {tokens[i]}")
                return
        if policy_name not in list_policies():
            self._line(f"[red]unknown policy:[/red] {policy_name}  (known: {', '.join(list_policies())})")
            return
        if not credentials.get_api_key():
            self._line("[red]chat needs an API key.[/red]  use [cyan]/login <key>[/cyan] first.")
            return
        credentials.inject_env()

        policy = load_policy(policy_name)
        effective_model = model_override or self._session_model
        policy = with_model_override(policy, effective_model)

        ts = time.strftime("%Y%m%d-%H%M%S")
        run_dir = RUNS_DIR / f"chat-{policy_name}-{ts}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        self._chat = start_chat(
            policy=policy,
            log_path=run_dir / "log.jsonl",
            workdir=run_dir / "wd",
            kill_path=run_dir / "kill",
        )
        self._chat_policy_name = policy_name
        self._active_run = run_dir / "log.jsonl"  # so /tree etc see it
        self._active_seen_seq = 0
        self._line(f"[b green]chat started[/b green]   policy=[cyan]{policy_name}[/cyan]  "
                   f"model=[b]{policy.model}[/b]  budget=$[b]{policy.budget_usd}[/b]")
        self._line(f"[dim]   {run_dir}[/dim]")
        self._line("[dim]type a message (no /), or [/dim][cyan]/end[/cyan][dim] to stop.[/dim]")

    def _cmd_end(self, _rest: str) -> None:
        if self._chat is None:
            self._line("[dim]no chat session active.[/dim]")
            return
        try:
            self._chat.close()
        except Exception:
            pass
        used_budget = (self._chat.seat.budget.usd_remaining)
        self._line(f"[b yellow]chat ended[/b yellow]   "
                   f"turns={self._chat.seat.turns_used}  "
                   f"budget_remaining=${used_budget:.4f}")
        self._chat = None
        self._chat_policy_name = None
        self._active_run = None

    def _send_chat(self, user_text: str) -> None:
        if self._is_running:
            self._line("[yellow]a run is in progress; wait for it to finish[/yellow]")
            return
        self._line(f"[b cyan]you[/b cyan]  {user_text}")
        self._is_running = True
        self.run_worker(
            self._do_chat_turn(user_text),
            exclusive=True,
            group="chat",
            thread=True,
        )

    async def _do_chat_turn(self, user_text: str) -> None:
        try:
            chat = self._chat
            if chat is None:
                self.call_from_thread(self._on_chat_reply, "[red]chat ended[/red]")
                return
            reply = chat.send(user_text)
            self.call_from_thread(self._on_chat_reply, reply)
        except Exception:
            self.call_from_thread(
                self._on_chat_reply, f"[red]chat error:[/red]\n{traceback.format_exc()}"
            )

    def _on_chat_reply(self, reply: str) -> None:
        # Stream any tool-call entries that fired during this turn, then the reply.
        self._tail_active_run()
        if reply:
            self._line(f"[b green]agent[/b green]  {reply}")
        self._is_running = False

    # ---- run worker ----------------------------------------------------- #

    async def _do_run(self, policy_name: str, policy, run_dir: Path, message: str) -> None:
        try:
            res = run_forest(
                policy=policy,
                log_path=run_dir / "log.jsonl",
                workdir=run_dir / "wd",
                kill_path=run_dir / "kill",
                user_message=message,
            )
            seat = res.root_seat
            summary = (
                f"[b green]done[/b green]   "
                f"submit={seat.submit_result!r}   "
                f"halt={seat.halt_reason}   "
                f"turns={seat.turns_used}   "
                f"tokens={seat.tokens_prompt}/{seat.tokens_completion}   "
                f"budget_remaining=${seat.budget.usd_remaining:.4f}"
            )
            self.call_from_thread(self._on_run_done, summary)
        except Exception:
            self.call_from_thread(
                self._on_run_done, f"[red]FAILED:[/red]\n{traceback.format_exc()}"
            )

    def _on_run_done(self, summary: str) -> None:
        # Drain any final log entries first.
        self._tail_active_run()
        self._line(summary)
        self._line("")
        self._is_running = False
        self._active_run = None

    # ---- live tail ------------------------------------------------------ #

    def _tail_active_run(self) -> None:
        if self._active_run is None:
            return
        entries = read_log(self._active_run)
        new = [e for e in entries if e.get("seq", 0) > self._active_seen_seq]
        for e in new:
            self._render_entry(e)
            self._active_seen_seq = max(self._active_seen_seq, e.get("seq", 0))

    def _render_entry(self, e: dict) -> None:
        sid = e["seat_id"] or "-"
        self._line(f"  [dim]{e['seq']:>3}[/dim] [cyan]{sid:<7}[/cyan] {summarize_entry(e)}")

    # ---- misc ----------------------------------------------------------- #

    def _resolve_run(self, name: str) -> Optional[Path]:
        name = name.strip()
        if not name:
            self._line("usage: [cyan]/view <run-name>[/cyan]   (see [cyan]/runs[/cyan])")
            return None
        candidate = RUNS_DIR / name
        if not (candidate.exists() and (candidate / "log.jsonl").exists()):
            self._line(f"[red]no such run:[/red] {name}")
            return None
        return candidate

    def _effective_model(self) -> str:
        return self._session_model or DEFAULT_MODEL

    def action_clear(self) -> None:
        self._cmd_clear("")


# --------------------------------------------------------------------------- #
# Tree builder (shared logic with view.py)
# --------------------------------------------------------------------------- #


def build_tree(entries: list[dict]) -> str:
    seats: dict[str, dict] = {}
    parents: dict[str, str] = {}
    for e in entries:
        sid = e["seat_id"]
        if sid and sid not in seats:
            seats[sid] = {"id": sid, "turns": 0, "submit": None, "halt": None}
            if e.get("parent_id"):
                parents[sid] = e["parent_id"]
        if e["type"] == "spawn":
            child = (e.get("payload") or {}).get("child_id")
            if child:
                parents[child] = sid
                seats.setdefault(child, {"id": child, "turns": 0, "submit": None, "halt": None})
        if e["type"] == "model_response":
            seats.setdefault(sid, {"id": sid, "turns": 0, "submit": None, "halt": None})
            seats[sid]["turns"] += 1
        if e["type"] == "submit":
            seats.setdefault(sid, {"id": sid, "turns": 0, "submit": None, "halt": None})
            seats[sid]["submit"] = (e.get("payload") or {}).get("result")
        if e["type"] == "halt":
            seats.setdefault(sid, {"id": sid, "turns": 0, "submit": None, "halt": None})
            seats[sid]["halt"] = (e.get("payload") or {}).get("reason")

    children: dict[str, list] = {}
    roots: list = []
    for sid in seats:
        p = parents.get(sid)
        if p is None:
            roots.append(sid)
        else:
            children.setdefault(p, []).append(sid)

    lines: list[str] = []

    def render(sid: str, depth: int) -> None:
        s = seats[sid]
        indent = "  " * depth
        line = f"{indent}[cyan]{s['id']}[/cyan]  turns={s['turns']}"
        if s["submit"] is not None:
            line += f"  submit=[italic]{s['submit']!r}[/italic]"
        if s["halt"] is not None and s["submit"] is None:
            line += f"  halt=[yellow]{s['halt']}[/yellow]"
        lines.append(line)
        for c in sorted(children.get(sid, [])):
            render(c, depth + 1)

    for r in sorted(roots):
        render(r, 0)
    return "\n".join(lines) or ""


# --------------------------------------------------------------------------- #


def main() -> int:
    HarnessApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
