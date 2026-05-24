#!/usr/bin/env python3
"""Interactive shell for the harness.

Single output area on top, command prompt at the bottom. Slash commands
do everything; plain text without a `/` is sent as a chat message.

First launch:  /login <openrouter-api-key>
After login:   /run <agent> <message>    one-shot task
               /chat [agent]             start a chat
               just type                 auto-starts a chat
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
from harness.types import Agent  # noqa: E402

from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual.suggester import Suggester  # noqa: E402
from textual.widgets import Footer, Header, Input, RichLog  # noqa: E402


class SuggestingInput(Input):
    """Input that accepts the suggester's ghost text on Tab (the default
    keymap binds Tab to focus-next, which is useless when this is the only
    interactive widget). Holds priority=True so it beats the App's default
    Tab handling."""

    BINDINGS = [
        Binding("tab", "cursor_right", "accept suggestion",
                show=False, priority=True),
    ]


RUNS_DIR = ROOT / "runs"
AGENTS_DIR = ROOT / "agents"

DEFAULT_MODEL = "moonshotai/kimi-k2.6"
DEFAULT_CHAT_AGENT = "chat"

COMMANDS = (
    "help", "login", "logout", "status", "model",
    "agents", "skills", "runs", "view", "tree",
    "chat", "end", "run",
    "approve", "deny", "approvals",
    "clear", "quit", "exit",
)

HELP_TEXT = """\
[b]how to use[/b]

  Just type a message — a chat with the [cyan]chat[/cyan] agent starts.
  Type [cyan]/end[/cyan] to end the chat.

[b]commands[/b]

  [cyan]/login[/cyan] [dim]<key>[/dim]         sign in (saves your OpenRouter API key, 0600)
  [cyan]/logout[/cyan]              clear the saved key
  [cyan]/status[/cyan]              auth, default model, chat state
  [cyan]/model[/cyan] [dim]<id|->[/dim]       session model override (or `-` to clear)
  [cyan]/agents[/cyan]              list available agents
  [cyan]/runs[/cyan]                list past runs
  [cyan]/view[/cyan] [dim]<run>[/dim]          replay a run's timeline
  [cyan]/tree[/cyan] [dim]<run>[/dim]          show a run's seat tree
  [cyan]/chat[/cyan] [dim][agent] [flags...][/dim]              start a chat (default: chat)
  [cyan]/end[/cyan]                 end the current chat
  [cyan]/run[/cyan]  [dim]<agent> [flags...] <message...>[/dim]  one-shot task
  [cyan]/approve[/cyan] [dim]<id>[/dim]    approve a pending risky tool call
  [cyan]/deny[/cyan] [dim]<id>[/dim]       deny a pending risky tool call
  [cyan]/approvals[/cyan]           list pending + resolved approvals for this run
  [cyan]/clear[/cyan] · [cyan]/help[/cyan] · [cyan]/quit[/cyan]

[b]keys[/b]

  [yellow]Mouse wheel[/yellow]           scroll the output
  [yellow]PageUp[/yellow] / [yellow]PageDown[/yellow]       scroll the output
  [yellow]Ctrl+Home[/yellow] / [yellow]Ctrl+End[/yellow]    jump to top / bottom (End resumes live tail)
  [yellow]Ctrl+L[/yellow]                clear the screen
  [yellow]Tab[/yellow]                   accept autocomplete suggestion

[b]text selection[/b]

  Hold [yellow]⌥ Option[/yellow] (macOS) or [yellow]Shift[/yellow] (Linux) while dragging to let
  the terminal do native text selection. Without the modifier the
  mouse goes to the app (for scrolling).

[b]flags[/b] (work with [cyan]/run[/cyan] and [cyan]/chat[/cyan])

  [yellow]--model[/yellow] [dim]<id>[/dim]              OpenRouter model id
  [yellow]--tools[/yellow] [dim]a,b,c[/dim]             local tools (code_exec, submit, spawn)
  [yellow]--web[/yellow] [dim]search,fetch[/dim]        enable OpenRouter web tools
  [yellow]--max-turns[/yellow] [dim]N[/dim]             per-seat turn cap
  [yellow]--max-depth[/yellow] [dim]N[/dim]             spawn-recursion depth
  [yellow]--max-children[/yellow] [dim]N[/dim]          siblings per seat
"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def list_agents() -> list[str]:
    return sorted(
        p.stem for p in AGENTS_DIR.glob("*.py") if not p.name.startswith("_")
    )


def list_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted(
        [p for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "log.jsonl").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def load_agent(name: str) -> Agent:
    import importlib
    mod = importlib.import_module(f"agents.{name}")
    return getattr(mod, "AGENT")


def parse_overrides(tokens: list[str]) -> tuple[dict, list[str]]:
    """Pull leading `--flag value` (or `--flag=value`) pairs off `tokens`.
    Returns (overrides_dict, remaining_tokens)."""
    out: dict = {}
    i = 0
    while i < len(tokens) and tokens[i].startswith("--"):
        raw = tokens[i].lstrip("-")
        if "=" in raw:
            key, value = raw.split("=", 1)
            i += 1
        else:
            key = raw
            if i + 1 >= len(tokens):
                raise ValueError(f"missing value for --{key}")
            value = tokens[i + 1]
            i += 2
        out[key] = value
    return out, tokens[i:]


def apply_overrides(agent: Agent, overrides: dict) -> Agent:
    """Build a new Agent with the given flag overrides applied.

    Also patches over an obvious foot-gun: if `spawn` is in the resulting
    tools list but max_children or max_depth are zero (which makes spawn
    instantly fail with breadth_exceeded / depth_exceeded), set them to
    sensible defaults so commands like `/chat --tools spawn` just work.
    The user can still override those with their own --max-children N
    and --max-depth N flags.
    """
    changes: dict = {}
    for k, v in overrides.items():
        if k == "model":
            changes["model"] = v
        elif k == "tools":
            changes["tools"] = tuple(t.strip() for t in v.split(",") if t.strip())
        elif k == "web":
            if v in ("", "-", "none"):
                changes["web"] = ()
            else:
                changes["web"] = tuple(t.strip() for t in v.split(",") if t.strip())
        elif k == "max-turns":
            changes["max_turns"] = int(v)
        elif k == "max-depth":
            changes["max_depth"] = int(v)
        elif k == "max-children":
            changes["max_children"] = int(v)
        else:
            raise ValueError(f"unknown flag: --{k}")
    new_agent = replace(agent, **changes) if changes else agent

    if "spawn" in new_agent.tools:
        fix: dict = {}
        if new_agent.max_children <= 0 and "max-children" not in overrides:
            fix["max_children"] = 3
        if new_agent.max_depth <= 0 and "max-depth" not in overrides:
            fix["max_depth"] = 2
        if fix:
            new_agent = replace(new_agent, **fix)
    return new_agent


def summarize_entry(e: dict) -> str:
    t = e["type"]
    p = e.get("payload", {}) or {}
    if t == "model_request":
        bits = []
        if p.get("tools"):
            bits.append("tools=" + ",".join(p["tools"]))
        if p.get("web_tools"):
            bits.append("web=" + ",".join(s.split(":")[-1] for s in p["web_tools"]))
        return "[dim]model_request[/dim] " + " ".join(bits)
    if t == "model_response":
        tcs = p.get("tool_calls") or []
        u = p.get("usage") or {}
        bits = []
        if u.get("prompt_tokens") is not None or u.get("completion_tokens") is not None:
            bits.append(f"tok={u.get('prompt_tokens') or 0}/{u.get('completion_tokens') or 0}")
        if u.get("usd") is not None:
            bits.append(f"${u['usd']:.5f}")
        cites = p.get("citations") or []
        if cites:
            bits.append(f"+{len(cites)}cites")
        suffix = " [dim](" + " ".join(bits) + ")[/dim]" if bits else ""
        if tcs:
            return "[dim]model_response[/dim] tool_calls=[{}]{}".format(
                ", ".join(tc.get("name", "?") for tc in tcs), suffix
            )
        return f"[dim]model_response[/dim] text={(p.get('text') or '')[:80]!r}{suffix}"
    if t == "tool_call":
        return f"[b]→[/b] {p.get('tool')} [dim]{json.dumps(p.get('args'))[:120]}[/dim]"
    if t == "tool_result":
        glyph = "[green]✓[/green]" if p.get("ok") else "[red]✗[/red]"
        return f"  {glyph} {p.get('tool')} [dim]err={p.get('error')}[/dim]"
    if t == "spawn":
        return f"[b magenta]spawn[/b magenta] → {p.get('child_id')} depth={p.get('depth')}"
    if t == "submit":
        return f"[b green]submit[/b green] [italic]{(p.get('result') or '')[:120]!r}[/italic]"
    if t == "halt":
        return f"[red]halt[/red] reason={p.get('reason')}"
    if t == "denial":
        return f"[yellow]denial[/yellow] reason={p.get('reason')} tool={p.get('tool')}"
    return t


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
                seats.setdefault(
                    child, {"id": child, "turns": 0, "submit": None, "halt": None}
                )
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
# Suggester (context-aware autocomplete)
# --------------------------------------------------------------------------- #


class HarnessSuggester(Suggester):
    def __init__(self) -> None:
        super().__init__(case_sensitive=False, use_cache=False)

    async def get_suggestion(self, value: str) -> Optional[str]:
        if not value or not value.startswith("/"):
            return None
        if " " not in value:
            stem = value[1:].lower()
            if not stem:
                return None
            for name in COMMANDS:
                if name.startswith(stem):
                    return "/" + name
            return None
        head, _, rest = value.partition(" ")
        cmd = head[1:].lower()
        if " " in rest:
            return None
        if cmd in ("run", "chat"):
            for n in list_agents():
                if n.startswith(rest):
                    return f"{head} {n}"
        if cmd in ("view", "tree"):
            for r in list_runs():
                if r.name.startswith(rest):
                    return f"{head} {r.name}"
        return None


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #


CSS = """
Screen { layout: vertical; }
#output { height: 1fr; padding: 0 1; }
#prompt { dock: bottom; }
"""


class HarnessApp(App):
    CSS = CSS
    TITLE = "harness"
    SUB_TITLE = "interactive agent harness"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("ctrl+l", "clear", "Clear"),
        # Scrolling — mouse is off (so terminal copy/paste works) and the
        # Input widget holds focus, so we route these to the output log.
        Binding("pageup",      "scroll_up",     "↑ page"),
        Binding("pagedown",    "scroll_down",   "↓ page"),
        Binding("ctrl+home",   "scroll_top",    "top"),
        Binding("ctrl+end",    "scroll_bottom", "bottom"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._session_model: Optional[str] = None
        self._active_run: Optional[Path] = None
        self._active_seen_seq: int = 0
        self._is_running: bool = False
        self._chat = None
        self._chat_agent_name: Optional[str] = None
        self._run_ctx = None   # set by the run worker before driver starts

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield RichLog(id="output", highlight=True, markup=True, wrap=True)
        yield SuggestingInput(
            placeholder="type a message to chat, or /help for commands  (tab to accept suggestion)",
            id="prompt",
            suggester=HarnessSuggester(),
        )
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
        self.set_interval(0.4, self._check_pending_approvals)

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
            if self._chat is not None:
                self._send_chat(raw)
                return
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
        self._line(
            f"default model: [b]{self._effective_model()}[/b]"
            + ("  [dim](session override)[/dim]" if self._session_model else "")
        )
        if self._chat is not None:
            seat = self._chat.seat
            self._line(
                f"chat: [green]active[/green]   agent=[b]{self._chat_agent_name}[/b]   "
                f"turns=[b]{seat.turns_used}[/b]   "
                f"tokens=[b]{seat.tokens_prompt}[/b]/[b]{seat.tokens_completion}[/b]   "
                f"spent=$[b]{seat.cost_usd:.4f}[/b]   "
                f"searches=[b]{seat.web_searches}[/b]"
            )
        else:
            self._line("chat: [dim]inactive[/dim]")
        self._line(f"runs dir: [dim]{RUNS_DIR}[/dim]")
        self._line(f"credentials: [dim]{credentials.CREDENTIALS_PATH}[/dim]")

    def _cmd_login(self, rest: str) -> None:
        key = rest.strip()
        if not key:
            self._line("usage: [cyan]/login <openrouter-api-key>[/cyan]")
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
            self._line(f"current: [b]{self._effective_model()}[/b]")
            self._line("usage: [cyan]/model moonshotai/kimi-k2.6[/cyan]   (or [cyan]/model -[/cyan] to clear)")
            return
        if rest == "-":
            self._session_model = None
            self._line("session model override cleared")
            return
        self._session_model = rest
        self._line(f"session model set to [b]{rest}[/b]")

    def _cmd_skills(self, _rest: str) -> None:
        from harness.skills import list_skills
        skills = list_skills()
        if not skills:
            self._line("[dim](no skills installed — add markdown files to skills/)[/dim]")
            return
        for s in skills:
            self._line(f"  [cyan]{s.name}[/cyan]  [dim]{s.description}[/dim]")
            if s.when_to_use:
                self._line(f"     [dim]when: {s.when_to_use}[/dim]")

    def _cmd_agents(self, _rest: str) -> None:
        for name in list_agents():
            try:
                a = load_agent(name)
                bits = f"tools={list(a.tools)}"
                if a.web:
                    bits += f"  web={list(a.web)}"
                bits += f"  model={a.model}  max_turns={a.max_turns}"
                if a.max_depth > 0:
                    bits += f"  max_depth={a.max_depth}"
                self._line(f"  [cyan]{name}[/cyan]  [dim]{bits}[/dim]")
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

    # ---- approval gate ------------------------------------------------- #

    def _active_ctx(self):
        """Return the RunCtx for whichever activity is currently running."""
        if self._chat is not None:
            return self._chat.ctx
        return self._run_ctx

    def _check_pending_approvals(self) -> None:
        ctx = self._active_ctx()
        if ctx is None:
            return
        try:
            for req in ctx.pending_undisplayed():
                self._line(
                    f"\n[b yellow on red] ⚠ APPROVAL NEEDED [/]  "
                    f"id=[b cyan]{req.id}[/b cyan]  "
                    f"seat=[cyan]{req.seat_id}[/cyan]  "
                    f"tool=[b]{req.tool_name}[/b]"
                )
                args_text = json.dumps(req.args, indent=2)
                # Indent for readability
                for line in args_text.splitlines():
                    self._line(f"    {line}")
                self._line(
                    f"  [dim]respond with[/dim] [green]/approve {req.id}[/green]  "
                    f"[dim]or[/dim] [red]/deny {req.id}[/red]\n"
                )
        except Exception:
            pass

    def _cmd_approve(self, rest: str) -> None:
        self._resolve_approval(rest.strip(), "approve")

    def _cmd_deny(self, rest: str) -> None:
        self._resolve_approval(rest.strip(), "deny")

    def _resolve_approval(self, aid: str, decision: str) -> None:
        if not aid:
            self._line(f"usage: [cyan]/{decision} <approval-id>[/cyan]")
            return
        ctx = self._active_ctx()
        if ctx is None:
            self._line("[red]no active run with pending approvals[/red]")
            return
        req = ctx.resolve_approval(aid, decision)
        if req is None:
            self._line(f"[red]no such pending approval:[/red] {aid}")
            return
        glyph = "[green]✓ approved[/green]" if decision == "approve" else "[red]✗ denied[/red]"
        self._line(f"  {glyph}  {aid}  ({req.tool_name})")

    def _cmd_approvals(self, _rest: str) -> None:
        ctx = self._active_ctx()
        if ctx is None:
            self._line("[dim](no active run)[/dim]")
            return
        with ctx._approvals_lock:
            items = list(ctx._approvals.values())
        if not items:
            self._line("[dim](no approvals so far)[/dim]")
            return
        for req in items:
            if req.decision is None:
                status = "[yellow]pending[/yellow]"
            elif req.decision == "approve":
                status = "[green]approved[/green]"
            else:
                status = "[red]denied[/red]"
            self._line(
                f"  [cyan]{req.id}[/cyan]  {status}  {req.tool_name}  "
                f"[dim]{json.dumps(req.args)[:120]}[/dim]"
            )

    # ---- /chat ---------------------------------------------------------- #

    def _cmd_chat(self, rest: str) -> None:
        if self._chat is not None:
            self._line(
                f"[yellow]chat already active[/yellow] (agent: {self._chat_agent_name}). "
                "use [cyan]/end[/cyan] to end it first."
            )
            return
        try:
            tokens = shlex.split(rest) if rest else []
        except ValueError as e:
            self._line(f"[red]bad quoting:[/red] {e}")
            return

        agent_name = DEFAULT_CHAT_AGENT
        if tokens and not tokens[0].startswith("--"):
            agent_name = tokens[0]
            tokens = tokens[1:]
        try:
            overrides, leftover = parse_overrides(tokens)
        except ValueError as e:
            self._line(f"[red]{e}[/red]")
            return
        if leftover:
            self._line(f"[red]unexpected arg(s):[/red] {leftover}")
            return

        if agent_name not in list_agents():
            self._line(f"[red]unknown agent:[/red] {agent_name}  (known: {', '.join(list_agents())})")
            return
        if not credentials.get_api_key():
            self._line("[red]chat needs an API key.[/red]  use [cyan]/login <key>[/cyan] first.")
            return
        credentials.inject_env()

        try:
            agent = load_agent(agent_name)
            agent = apply_overrides(agent, overrides)
            if self._session_model and "model" not in overrides:
                agent = replace(agent, model=self._session_model)
        except Exception as e:
            self._line(f"[red]bad agent config:[/red] {e}")
            return

        ts = time.strftime("%Y%m%d-%H%M%S")
        run_dir = RUNS_DIR / f"chat-{agent_name}-{ts}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        self._chat = start_chat(agent=agent, log_path=run_dir / "log.jsonl", workdir=run_dir / "wd")
        self._chat_agent_name = agent_name
        self._active_run = run_dir / "log.jsonl"
        self._active_seen_seq = 0
        self._line(
            f"[b green]chat started[/b green]   agent=[cyan]{agent_name}[/cyan]  "
            f"model=[b]{agent.model}[/b]  tools={list(agent.tools)}  "
            + (f"web={list(agent.web)}" if agent.web else "")
        )
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
        seat = self._chat.seat
        self._line(
            f"[b yellow]chat ended[/b yellow]   turns={seat.turns_used}  "
            f"spent=$[b]{seat.cost_usd:.4f}[/b]"
        )
        self._chat = None
        self._chat_agent_name = None
        self._active_run = None

    def _send_chat(self, user_text: str) -> None:
        if self._is_running:
            self._line("[yellow]busy[/yellow] — wait for the current turn to finish")
            return
        self._line(f"[b cyan]you[/b cyan]  {user_text}")
        self._is_running = True
        self.run_worker(self._do_chat_turn(user_text), exclusive=True, group="chat", thread=True)

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
        self._tail_active_run()
        if reply:
            self._line(f"[b green]agent[/b green]  {reply}")
        self._is_running = False

    # ---- /run ----------------------------------------------------------- #

    def _cmd_run(self, rest: str) -> None:
        if self._is_running:
            self._line("[yellow]busy[/yellow] — wait for the current task to finish")
            return
        if not rest:
            self._line("usage: [cyan]/run <agent> [--flags...] <message>[/cyan]")
            self._line(f"agents: {', '.join(list_agents())}")
            return
        try:
            tokens = shlex.split(rest)
        except ValueError as e:
            self._line(f"[red]bad quoting:[/red] {e}")
            return
        if not tokens:
            return

        agent_name = tokens[0]
        if agent_name not in list_agents():
            self._line(f"[red]unknown agent:[/red] {agent_name}  (known: {', '.join(list_agents())})")
            return
        try:
            overrides, remaining = parse_overrides(tokens[1:])
        except ValueError as e:
            self._line(f"[red]{e}[/red]")
            return
        message = " ".join(remaining).strip()
        if not message:
            self._line("[red]missing user message[/red]")
            return
        if not credentials.get_api_key():
            self._line("[red]a run needs an API key.[/red]  use [cyan]/login <key>[/cyan] first.")
            return
        credentials.inject_env()

        try:
            agent = load_agent(agent_name)
            agent = apply_overrides(agent, overrides)
            if self._session_model and "model" not in overrides:
                agent = replace(agent, model=self._session_model)
        except Exception as e:
            self._line(f"[red]bad agent config:[/red] {e}")
            return

        ts = time.strftime("%Y%m%d-%H%M%S")
        run_dir = RUNS_DIR / f"{agent_name}-{ts}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        self._line(
            f"[b]→ running[/b] [cyan]{agent_name}[/cyan]  model=[b]{agent.model}[/b]  "
            f"tools={list(agent.tools)}"
            + (f"  web={list(agent.web)}" if agent.web else "")
        )
        self._line(f"[dim]   {run_dir}[/dim]")
        self._active_run = run_dir / "log.jsonl"
        self._active_seen_seq = 0
        self._is_running = True
        self.run_worker(self._do_run(agent, run_dir, message), exclusive=True, group="run", thread=True)

    async def _do_run(self, agent: Agent, run_dir: Path, message: str) -> None:
        try:
            def grab_ctx(ctx):
                self._run_ctx = ctx
            res = run_forest(
                agent=agent,
                log_path=run_dir / "log.jsonl",
                workdir=run_dir / "wd",
                user_message=message,
                on_ctx=grab_ctx,
            )
            seat = res.root_seat
            summary = (
                f"[b green]done[/b green]   submit={seat.submit_result!r}   "
                f"halt={seat.halt_reason}   turns={seat.turns_used}   "
                f"tokens={seat.tokens_prompt}/{seat.tokens_completion}   "
                f"spent=$[b]{seat.cost_usd:.4f}[/b]"
            )
            self.call_from_thread(self._on_run_done, summary)
        except Exception:
            self.call_from_thread(
                self._on_run_done, f"[red]FAILED:[/red]\n{traceback.format_exc()}"
            )

    def _on_run_done(self, summary: str) -> None:
        self._tail_active_run()
        self._line(summary)
        self._line("")
        self._is_running = False
        self._run_ctx = None

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

    # ---- scrolling ----------------------------------------------------- #
    # When the user scrolls back through history, pause auto-scroll so
    # incoming live entries don't yank them to the bottom mid-read.
    # Scrolling all the way to the bottom (Ctrl+End) resumes it.

    def action_scroll_up(self) -> None:
        out = self.query_one("#output", RichLog)
        out.auto_scroll = False
        out.scroll_page_up()

    def action_scroll_down(self) -> None:
        out = self.query_one("#output", RichLog)
        out.scroll_page_down()
        # If they paged past the bottom, treat that as "resume live".
        if out.is_vertical_scroll_end:
            out.auto_scroll = True

    def action_scroll_top(self) -> None:
        out = self.query_one("#output", RichLog)
        out.auto_scroll = False
        out.scroll_home()

    def action_scroll_bottom(self) -> None:
        out = self.query_one("#output", RichLog)
        out.scroll_end()
        out.auto_scroll = True


def main() -> int:
    # Mouse capture ON so the scroll wheel works on the output area.
    # Terminal-native text selection still works while holding Option (⌥)
    # on macOS or Shift on Linux — that's the standard "let the terminal
    # see the mouse" modifier in iTerm2, Terminal.app and gnome-terminal.
    HarnessApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
