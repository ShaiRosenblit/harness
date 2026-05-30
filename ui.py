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
import re
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
from textual import events  # noqa: E402
from textual.widgets import Footer, Header, Input, RichLog, Static  # noqa: E402
from rich.markup import escape  # noqa: E402


_PASTE_MARKER_RE = re.compile(r"\[Pasted #(\d+) \+(\d+) lines\]")


class SuggestingInput(Input):
    """Input with shell-like UX:
       - Tab accepts the suggester's ghost-text suggestion.
       - ↑ / ↓ navigate command history (most-recent first on first ↑).
       - Multi-line pastes are stashed under a numbered `[Pasted #N +K lines]`
         marker so the user can still edit around them; on submit the marker
         expands back to the real text.

    History preserves paste markers and their content together — if the
    user re-submits a recalled line with a paste marker in it, the marker
    still expands correctly.
    """

    BINDINGS = [
        Binding("tab",  "cursor_right", "accept suggestion",
                show=False, priority=True),
        # ↑/↓ are dual-purpose: history nav when there is something to
        # navigate, otherwise fall through to the App's scroll action so
        # the mouse wheel (which alt-scroll-mode translates to ↑/↓) can
        # scroll the conversation log. Ctrl+P / Ctrl+N always do history.
        Binding("up",     "history_prev",       "↑ history", show=False, priority=True),
        Binding("down",   "history_next",       "↓ history", show=False, priority=True),
        Binding("ctrl+p", "force_history_prev", show=False, priority=True),
        Binding("ctrl+n", "force_history_next", show=False, priority=True),
    ]

    # Defense window for double-fire pastes (see _on_paste below).
    _PASTE_DEDUP_WINDOW_S = 0.5
    # Below this length we won't collapse "ABCABC" → "ABC" — too risky
    # to mistake a legitimate short repeating paste for a terminal-doubled
    # one. Empirically, real-world doubled pastes are token-length or
    # longer (a Telegram bot token is 46 chars), so 10 is a safe floor.
    _PASTE_HALVES_DEDUP_MIN_LEN = 10

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pastes: dict[int, str] = {}
        # Command history. Each entry is (marker-form text, paste dict).
        # _history_idx == len(history) means "current input" (live draft).
        self._history: list[tuple[str, dict[int, str]]] = []
        self._history_idx: int = 0
        self._draft_value: str = ""
        self._draft_pastes: dict[int, str] = {}
        # Last paste text + monotonic timestamp, for dedup.
        self._last_paste_text: str = ""
        self._last_paste_at: float = 0.0

    def _on_paste(self, event: events.Paste) -> None:
        # Textual dispatches _on_paste through the *entire* MRO (see
        # textual/message_pump.py:_get_dispatch_methods — it iterates
        # cls.__mro__ and yields every matching method). event.stop()
        # only halts bubbling, not parent-class dispatch. Without
        # prevent_default(), Input._on_paste also runs and inserts
        # event.text.splitlines()[0] on top of whatever we inserted —
        # which is why a single paste of "abc" produced "abcabc". Call
        # prevent_default() up front so the parent handler is skipped.
        event.prevent_default()
        text = event.text or ""
        if not text:
            event.stop()
            return
        # Defense against terminals, multiplexers (tmux/screen), or
        # clipboard managers that silently fire the same paste twice
        # within milliseconds. Textual's bracketed-paste parser itself
        # emits one Paste per ESC[200~…ESC[201~ pair (see
        # textual/_xterm_parser.py), so a repeat we see here is coming
        # from below us, not from Textual. Treat any identical paste
        # within _PASTE_DEDUP_WINDOW_S as a duplicate and drop it.
        import time
        now = time.monotonic()
        if (
            text == self._last_paste_text
            and (now - self._last_paste_at) < self._PASTE_DEDUP_WINDOW_S
        ):
            event.stop()
            return
        self._last_paste_text = text
        self._last_paste_at = now
        # Defense against the other doubling mode: a single Paste event
        # whose payload is already concatenated with itself (some terminals
        # / clipboard managers do this inside one bracketed-paste pair, so
        # the rapid-fire dedup above never sees a second event). If text
        # is long enough that two-halves-identical can't be a coincidence,
        # collapse to the first half and tell the user we did.
        n = len(text)
        if n >= self._PASTE_HALVES_DEDUP_MIN_LEN and n % 2 == 0 \
                and text[: n // 2] == text[n // 2:]:
            text = text[: n // 2]
            try:
                app = self.app
                if app is not None and hasattr(app, "_line"):
                    app._line(
                        f"[yellow]⚠ paste looked doubled "
                        f"({n} → {len(text)} chars); kept the first half[/yellow]"
                    )
            except Exception:
                pass
        if "\n" in text:
            paste_id = len(self._pastes) + 1
            self._pastes[paste_id] = text
            line_count = text.count("\n") + 1
            insert = f"[Pasted #{paste_id} +{line_count - 1} lines]"
        else:
            insert = text
        selection = self.selection
        if selection.is_empty:
            self.insert_text_at_cursor(insert)
        else:
            self.replace(insert, *selection)
        event.stop()

    def consume(self) -> str:
        """Return the current value with paste markers expanded back to
        their full text, push the line into history, and reset the input."""
        def sub(m: "re.Match[str]") -> str:
            return self._pastes.get(int(m.group(1)), m.group(0))
        expanded = _PASTE_MARKER_RE.sub(sub, self.value)
        # Save the *marker-form* line + its paste dict so navigating back
        # to it later still resolves correctly.
        if self.value and (not self._history or self._history[-1][0] != self.value):
            self._history.append((self.value, dict(self._pastes)))
        self._history_idx = len(self._history)
        self._draft_value = ""
        self._draft_pastes = {}
        self.value = ""
        self._pastes.clear()
        return expanded

    # ---- history navigation ------------------------------------------- #
    # When the input is empty and we are NOT already navigating history,
    # ↑/↓ defer to the App's scroll-the-log actions instead. That keeps
    # the mouse wheel (which terminal alt-scroll-mode translates into ↑/↓
    # keystrokes) usable for reading back through a long conversation.
    # Ctrl+P / Ctrl+N still force history nav regardless of state.

    def _scroll_fallthrough(self) -> bool:
        return not self.value and self._history_idx == len(self._history)

    def _app_scroll(self, action: str) -> None:
        try:
            handler = getattr(self.app, action, None)
            if callable(handler):
                handler()
        except Exception:
            pass

    def action_history_prev(self) -> None:
        if self._scroll_fallthrough():
            self._app_scroll("action_scroll_one_up")
            return
        self.action_force_history_prev()

    def action_history_next(self) -> None:
        if self._scroll_fallthrough():
            self._app_scroll("action_scroll_one_down")
            return
        self.action_force_history_next()

    def action_force_history_prev(self) -> None:
        if not self._history:
            return
        if self._history_idx == len(self._history):
            # Snapshot the in-progress draft so ↓-back-to-now restores it.
            self._draft_value = self.value
            self._draft_pastes = dict(self._pastes)
        if self._history_idx > 0:
            self._history_idx -= 1
            text, pastes = self._history[self._history_idx]
            self.value = text
            self._pastes = dict(pastes)
            self.cursor_position = len(self.value)

    def action_force_history_next(self) -> None:
        if not self._history:
            return
        if self._history_idx < len(self._history):
            self._history_idx += 1
            if self._history_idx == len(self._history):
                # Back at "now" — restore the live draft.
                self.value = self._draft_value
                self._pastes = dict(self._draft_pastes)
            else:
                text, pastes = self._history[self._history_idx]
                self.value = text
                self._pastes = dict(pastes)
            self.cursor_position = len(self.value)


RUNS_DIR = ROOT / "runs"
AGENTS_DIR = ROOT / "agents"
PROMPTS_DIR = ROOT / "prompts"

DEFAULT_MODEL = "moonshotai/kimi-k2.6"
DEFAULT_CHAT_AGENT = "chat"

COMMANDS = (
    "help", "login", "logout", "status", "model",
    "agents", "skills", "runs", "view", "tree",
    "chat", "end", "run",
    "prompts", "prompt", "p",
    "approve", "deny", "approvals", "auto",
    "telegram",
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
  [cyan]/prompts[/cyan]             list saved prompts (markdown files in [dim]prompts/[/dim])
  [cyan]/prompt[/cyan] [dim]<name> [extra...][/dim]   send a saved prompt as a chat message ([cyan]/p[/cyan] alias)
  [cyan]/approve[/cyan] [dim]<id>[/dim]    approve a pending risky tool call
  [cyan]/deny[/cyan] [dim]<id>[/dim]       deny a pending risky tool call
  [cyan]/approvals[/cyan]           list pending + resolved approvals for this run
  [cyan]/auto[/cyan] [dim]on|off[/dim]      session-wide auto-approve (risky tools run without prompting)
  [cyan]/telegram[/cyan] [dim]status|start|stop[/dim]   mirror the UI on Telegram
  [cyan]/telegram login[/cyan] [dim]<token>[/dim]      save bot token (from @BotFather)
  [cyan]/telegram allow[/cyan] [dim]<id>[,<id>...][/dim]   set allowed numeric chat ids ([dim]-[/dim] to clear)
  [cyan]/telegram test[/cyan]            one-shot getMe call — verify the saved token without polling
  [cyan]/clear[/cyan] · [cyan]/help[/cyan] · [cyan]/quit[/cyan]

[b]keyboard[/b]

  [yellow]↑[/yellow] / [yellow]↓[/yellow]                  shell history (or scroll the log if input is empty)
  [yellow]Ctrl+P[/yellow] / [yellow]Ctrl+N[/yellow]        shell history (always, even with empty input)
  [yellow]Tab[/yellow]                   accept autocomplete suggestion
  [yellow]PageUp[/yellow] / [yellow]PageDown[/yellow]       scroll the output
  [yellow]Ctrl+Home[/yellow] / [yellow]Ctrl+End[/yellow]    top / bottom (End resumes live tail)
  [yellow]Ctrl+L[/yellow]                clear the screen

[b]mouse[/b]

  [yellow]Wheel up/down[/yellow]         scroll the output (via terminal alt-scroll mode)
  Drag to select; [yellow]Cmd+C[/yellow] / [yellow]Ctrl+Shift+C[/yellow] to copy (terminal-native).

[b]flags[/b] (work with [cyan]/run[/cyan] and [cyan]/chat[/cyan])

  [yellow]--model[/yellow] [dim]<id>[/dim]              OpenRouter model id
  [yellow]--tools[/yellow] [dim]a,b,c[/dim]             local tools (code_exec, submit, spawn)
  [yellow]--web[/yellow] [dim]search,fetch[/dim]        enable OpenRouter web tools
  [yellow]--max-turns[/yellow] [dim]N[/dim]             per-seat turn cap
  [yellow]--max-depth[/yellow] [dim]N[/dim]             spawn-recursion depth
  [yellow]--max-children[/yellow] [dim]N[/dim]          siblings per seat
  [yellow]--auto-approve[/yellow]            run risky tools without prompting (per-run)
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


def list_prompts() -> list[str]:
    if not PROMPTS_DIR.exists():
        return []
    return sorted(
        p.stem for p in PROMPTS_DIR.glob("*.md")
        if p.name.lower() != "readme.md"
    )


def load_prompt(name: str) -> Optional[str]:
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


BOOLEAN_FLAGS = {"auto-approve"}


def parse_overrides(tokens: list[str]) -> tuple[dict, list[str]]:
    """Pull leading `--flag value` (or `--flag=value`) pairs off `tokens`.
    Boolean flags in BOOLEAN_FLAGS don't take a value — their presence
    means True. Returns (overrides_dict, remaining_tokens)."""
    out: dict = {}
    i = 0
    while i < len(tokens) and tokens[i].startswith("--"):
        raw = tokens[i].lstrip("-")
        if "=" in raw:
            key, value = raw.split("=", 1)
            i += 1
        elif raw in BOOLEAN_FLAGS:
            key, value = raw, "true"
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


# Friendly icons for tool_call rendering in the live (compact) view.
# Unknown tools fall back to a plain bullet — keep this small and stable.
_TOOL_GLYPH = {
    "code_exec":          "🐍",
    "bash":               "🖥️",
    "spawn":              "🌱",
    "submit":             "✓",
    "use_skill":          "🧠",
    "web_search":         "🔍",
    "web:search":         "🔍",
    "search":             "🔍",
    "web_fetch":          "🌐",
    "web:fetch":          "🌐",
    "fetch":              "🌐",
    "read_file":          "📖",
    "write_file":         "✏️",
}


# Live-view caps. The full detail is always available via `/view`.
_MAX_INPUT_LINES = 40    # code_exec / bash source shown before folding
_MAX_RESULT_LINES = 15   # tool stdout lines shown before folding


def _clip_block(lines: list[str], cap: int) -> list[str]:
    """Escape each content line, cap the count, and append a
    "… +N more lines" footer when there's more. Markup-safe."""
    shown = [escape(ln) for ln in lines[:cap]]
    extra = len(lines) - cap
    if extra > 0:
        shown.append(f"… +{extra} more line{'s' if extra != 1 else ''}")
    return shown


def _tool_input_lines(tool: str, args: dict) -> list[str]:
    """Full multi-line input for code_exec / bash (the whole function /
    command, not just line 1). Empty for tools whose one-line preview in
    _tool_arg_preview already says everything."""
    if not isinstance(args, dict):
        return []
    if tool == "code_exec":
        code = (args.get("code") or "").rstrip()
        return code.splitlines() if code.strip() else []
    if tool == "bash":
        cmd = (args.get("command") or args.get("cmd") or "").rstrip()
        return cmd.splitlines() if cmd.strip() else []
    return []


def _tool_arg_preview(tool: str, args: dict) -> str:
    """One-line, human-readable hint for what a tool call is doing.
    Returns "" when there's nothing useful to show (or when the full
    multi-line input is rendered separately, as for code_exec / bash)."""
    if not isinstance(args, dict):
        return ""
    if tool in ("code_exec", "bash"):
        # The full source is rendered as an indented block right under
        # the tool header; a one-line preview would just duplicate line 1.
        return ""
    if tool == "spawn":
        agent = args.get("agent") or args.get("name") or ""
        msg = (
            args.get("prompt") or args.get("message")
            or args.get("task") or ""
        ).strip().splitlines()[:1]
        msg = msg[0] if msg else ""
        if agent and msg:
            return f"{agent}  {msg[:80]}"
        return (agent or msg)[:100]
    if tool == "submit":
        result = (args.get("result") or args.get("text") or "").strip()
        return result.splitlines()[0][:100] if result else ""
    if tool in ("search", "web_search", "web:search"):
        return (args.get("query") or args.get("q") or "")[:100]
    if tool in ("fetch", "web_fetch", "web:fetch"):
        return (args.get("url") or "")[:100]
    if tool in ("read_file", "write_file"):
        return (args.get("path") or args.get("file") or "")[:100]
    j = json.dumps(args, ensure_ascii=False)
    return j[:100] + ("…" if len(j) > 100 else "")


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
        if cmd in ("prompt", "prompts", "p"):
            for n in list_prompts():
                if n.startswith(rest):
                    return f"{head} {n}"
        return None


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #


CSS = """
Screen { layout: vertical; }
#output { height: 1fr; padding: 0 1; }
/* Status bar flows in the normal vertical layout. Without an explicit
   height it collapses to its content (so it takes 0 rows when idle and
   1 row when active). Docking it would put it on top of the prompt /
   Footer at the same y; flow layout keeps it visible just above the
   prompt. */
#status         { height: auto; padding: 0 1; color: $text-muted; }
#status.busy    { color: $accent; }
#status.stopped { color: $error; text-style: bold; background: $error 10%; }
#status.idle    { color: $text-muted; }
#prompt         { dock: bottom; }
"""

# Animation frames for the "agent is doing something" spinner. Braille
# spinner is one cell wide and renders well in monospace terminals.
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Recessive palette for the "activity" stream — the agent's thinking and
# tool steps. The actual conversation (user messages + the agent's answer)
# renders in the terminal's normal foreground and at a shallower indent, so
# it stays the most prominent thing on screen; activity is dimmer AND
# indented one level deeper so the eye reads it as subordinate detail.
_ACT = "#6b7280"       # grey-500: activity labels, glyphs, gutters
_ACT_TEXT = "#9ca3af"  # grey-400: code / output content (a touch brighter, still legible)
_ACT_IND = "    "      # activity indent (the conversation answer sits at 2 spaces)


class HarnessApp(App):
    CSS = CSS
    TITLE = "harness"
    SUB_TITLE = "interactive agent harness"
    # Textual's built-in command palette is bound to Ctrl+P by default,
    # which would swallow our readline-style Ctrl+P history binding on the
    # prompt. We have our own slash-command system (/help, /chat, …) so the
    # palette is redundant — disable it to free the key.
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("ctrl+l", "clear", "Clear"),
        # ↑/↓ are owned by SuggestingInput for command history when there's
        # something to navigate; SuggestingInput.action_history_prev/next
        # falls through to the actions below when the input is empty, so
        # the mouse wheel (via alt-scroll-mode ↑/↓) keeps scrolling the log.
        Binding("up",        "scroll_one_up",   show=False),
        Binding("down",      "scroll_one_down", show=False),
        Binding("pageup",    "scroll_up",     "↑ page"),
        Binding("pagedown",  "scroll_down",   "↓ page"),
        Binding("ctrl+home", "scroll_top",    "top"),
        Binding("ctrl+end",  "scroll_bottom", "bottom"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._session_model: Optional[str] = None
        self._active_run: Optional[Path] = None
        self._active_seen_seq: int = 0
        # Incremental tail state: byte offset already consumed, plus a
        # carry buffer for a trailing partial (un-newline-terminated) line.
        self._active_offset: int = 0
        self._active_buf: str = ""
        self._is_running: bool = False
        self._chat = None
        self._chat_agent_name: Optional[str] = None
        self._run_ctx = None   # set by the run worker before driver starts
        # Session-wide auto-approve toggle. Off by default; opt in with
        # `/auto on` or a per-run `--auto-approve` flag.
        self._auto_approve: bool = False
        self._telegram = None  # harness.telegram_bridge.TelegramBridge
        # ---- live status bar state ---------------------------------- #
        # _status_label is what the agent is currently doing, e.g.
        # "thinking", "running code_exec". When _status_kind == "busy"
        # the spinner animates next to it; "stopped" sticks a red
        # banner that explains why we halted; "idle" hides the bar.
        self._status_label: str = ""
        self._status_kind: str = "idle"   # "idle" | "busy" | "stopped"
        self._spinner_frame: int = 0
        self._status_clear_at: float = 0.0  # epoch; 0 = no auto-clear

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield RichLog(id="output", highlight=True, markup=True, wrap=True)
        # Live status bar between the log and the prompt. Always
        # present so spinner / "stopped" lines have somewhere to land
        # the instant we have something to say.
        yield Static("", id="status", classes="idle", markup=True)
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
        # Tail fast so tool calls / results land on screen as they happen
        # rather than in half-second clumps. read_log is cheap (only new
        # seq rows are rendered).
        self.set_interval(0.15, self._tail_active_run)
        self.set_interval(0.4, self._check_pending_approvals)
        # Spinner tick — runs every 100ms so the braille frames look
        # alive without flooding the screen. _refresh_status is cheap
        # when nothing's changed.
        self.set_interval(0.1, self._tick_status)
        # ?1007h = enable alternate-scroll-mode: terminal converts wheel
        # events into ↑/↓ arrow keystrokes. Lets the wheel scroll the log
        # while leaving normal mouse clicks/drags to the terminal for
        # native text selection.
        self._enable_alt_scroll()

    # ---- output ---------------------------------------------------------- #

    def _line(self, text: str = "") -> None:
        self.query_one("#output", RichLog).write(text)

    # ---- live status bar ------------------------------------------------ #

    def _set_status(self, kind: str, label: str = "", *, clear_after: float = 0.0) -> None:
        """Update the inline status bar. `kind` is one of:
          - "idle":    hide the bar (label ignored)
          - "busy":    show animated spinner + label
          - "stopped": show red bold stopped-banner with label
        clear_after, if > 0, auto-clears to idle after that many seconds.
        """
        self._status_kind = kind
        self._status_label = label
        self._status_clear_at = time.time() + clear_after if clear_after else 0.0
        self._refresh_status()

    def _tick_status(self) -> None:
        # Auto-clear sticky statuses (e.g. "stopped" / "ready").
        if self._status_clear_at and time.time() >= self._status_clear_at:
            self._status_kind = "idle"
            self._status_label = ""
            self._status_clear_at = 0.0
        if self._status_kind == "busy":
            self._spinner_frame = (self._spinner_frame + 1) % len(_SPINNER_FRAMES)
        self._refresh_status()

    def _refresh_status(self) -> None:
        try:
            bar = self.query_one("#status", Static)
        except Exception:
            return
        bar.remove_class("busy", "stopped", "idle")
        if self._status_kind == "busy":
            bar.add_class("busy")
            frame = _SPINNER_FRAMES[self._spinner_frame]
            bar.update(f"{frame}  {self._status_label}")
        elif self._status_kind == "stopped":
            bar.add_class("stopped")
            bar.update(f"■  stopped · {self._status_label}")
        else:
            bar.add_class("idle")
            bar.update("")

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
        inp = event.input
        if isinstance(inp, SuggestingInput):
            raw = inp.consume().strip()
        else:
            raw = (event.value or "").strip()
            inp.value = ""
        if not raw:
            return
        # Echo slash commands so the user gets confirmation they ran;
        # chat messages render their own "┌─ you" block in _send_chat,
        # so echoing here would duplicate the line.
        if raw.startswith("/"):
            self._echo(raw if "\n" not in raw
                       else raw.split("\n", 1)[0] + f" [+{raw.count(chr(10))} lines]")
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
            auto = " [red](auto-approve ON)[/red]" if self._chat.ctx.auto_approve else ""
            self._line(
                f"chat: [green]active[/green]   agent=[b]{self._chat_agent_name}[/b]   "
                f"turns=[b]{seat.turns_used}[/b]   "
                f"tokens=[b]{seat.tokens_prompt}[/b]/[b]{seat.tokens_completion}[/b]   "
                f"spent=$[b]{seat.cost_usd:.4f}[/b]   "
                f"searches=[b]{seat.web_searches}[/b]{auto}"
            )
        else:
            self._line("chat: [dim]inactive[/dim]")
        auto_state = "[red]ON[/red]" if self._auto_approve else "[dim]off[/dim]"
        self._line(f"auto-approve (session default): {auto_state}")
        tg_state = "[green]running[/green]" if (self._telegram and self._telegram.is_running()) else "[dim]stopped[/dim]"
        self._line(f"telegram bridge: {tg_state}")
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

    def _cmd_auto(self, rest: str) -> None:
        """Toggle session-wide auto-approve: skips approval prompts for
        risky tools (e.g. bash). Off by default. Per-run override:
        `--auto-approve` on /run or /chat. State here applies to NEW
        runs/chats; the active chat's setting is locked in at /chat
        time."""
        rest = rest.strip().lower()
        if rest in ("on", "true", "yes", "1"):
            self._auto_approve = True
            self._line(
                "[red]⚠ session auto-approve ON.[/red] Risky tools (bash, …) "
                "will run WITHOUT prompting on new runs/chats. "
                "Use [cyan]/auto off[/cyan] to require approval again."
            )
        elif rest in ("off", "false", "no", "0"):
            self._auto_approve = False
            self._line("session auto-approve [green]OFF[/green] — risky tools require approval.")
        elif rest == "":
            state = "[red]ON[/red]" if self._auto_approve else "[green]off[/green]"
            self._line(f"session auto-approve: {state}")
            self._line("usage: [cyan]/auto on[/cyan]  ·  [cyan]/auto off[/cyan]")
        else:
            self._line(f"[red]unknown:[/red] /auto {rest!r} — try [cyan]/auto on|off[/cyan]")

    def _cmd_telegram(self, rest: str) -> None:
        """Manage the Telegram bridge. Subcommands:
          status                       (default) running + saved config
          start | stop                 run / shut down the bridge
          login <token>                save bot token (~/.config/harness/credentials.json)
          logout                       clear saved token
          allow <id>[,<id>...] | -     set the numeric chat-id allowlist (`-` clears)
        Env vars TELEGRAM_BOT_TOKEN / TELEGRAM_ALLOWED_CHAT_IDS still win
        if set. The bridge runs in a background thread with its own
        asyncio loop; each authorized Telegram chat gets its own
        ChatSession, independent of the textual UI's local chat."""
        parts = rest.strip().split(None, 1)
        sub = (parts[0].lower() if parts else "status")
        arg = (parts[1].strip() if len(parts) > 1 else "")

        if sub == "status":
            self._telegram_status()
            return
        if sub == "login":
            self._telegram_login(arg)
            return
        if sub == "logout":
            credentials.clear_telegram_token()
            self._line("[yellow]telegram token cleared[/yellow]")
            return
        if sub == "allow":
            self._telegram_allow(arg)
            return
        if sub == "test":
            self._telegram_test()
            return
        if sub == "stop":
            if not (self._telegram and self._telegram.is_running()):
                self._line("[dim]telegram bridge not running[/dim]")
                return
            self._telegram.stop()
            self._telegram = None
            self._line("[yellow]telegram bridge stopped[/yellow]")
            return
        if sub != "start":
            self._line(f"[red]unknown:[/red] /telegram {sub!r} — try start|stop|status|login|logout|allow|test")
            return

        if self._telegram and self._telegram.is_running():
            self._line("[yellow]telegram bridge already running[/yellow]")
            return
        if not credentials.get_api_key():
            self._line("[red]bridge needs an OpenRouter key.[/red]  use [cyan]/login <key>[/cyan] first.")
            return
        credentials.inject_env()
        try:
            from harness import telegram_bridge
        except ImportError as e:
            self._line(f"[red]missing dependency:[/red] {e}")
            self._line("install with: [cyan]pip install python-telegram-bot>=20[/cyan]")
            return
        try:
            token, allowed = telegram_bridge.load_config()
        except ValueError as e:
            self._line(f"[red]{e}[/red]")
            return

        def _on_log(msg: str) -> None:
            # Marshal back to the textual thread.
            self.call_from_thread(self._line, f"[magenta]telegram[/magenta] · {msg}")

        bridge = telegram_bridge.TelegramBridge(
            token=token,
            allowed_ids=allowed,
            runs_dir=RUNS_DIR,
            default_chat_agent=DEFAULT_CHAT_AGENT,
            load_agent=load_agent,
            list_agents=list_agents,
            parse_overrides=parse_overrides,
            apply_overrides=apply_overrides,
            on_log=_on_log,
        )
        try:
            bridge.start_in_thread()
        except Exception as e:
            self._line(f"[red]failed to start bridge:[/red] {e}")
            return
        self._telegram = bridge
        self._line(
            f"[green]telegram bridge starting[/green]   allowed_ids={sorted(allowed)}"
        )

    def _telegram_status(self) -> None:
        running = self._telegram and self._telegram.is_running()
        state = "[green]running[/green]" if running else "[dim]stopped[/dim]"
        self._line(f"telegram: {state}")
        token = credentials.get_telegram_token()
        if token:
            note = ""
            if len(token) < 30:
                note = "  [yellow](short — real tokens are ~46 chars)[/yellow]"
            elif len(token) > 60:
                note = "  [yellow](unusually long — typical is ~46, double check for double-paste)[/yellow]"
            colons = token.count(":")
            colon_note = "" if colons == 1 else f"  [yellow]({colons} colons — expected 1)[/yellow]"
            self._line(
                f"  token: [dim]{credentials.mask(token)}[/dim]  "
                f"[dim]({len(token)} chars)[/dim]{note}{colon_note}"
            )
        else:
            self._line("  token: [yellow]not set[/yellow]  ([cyan]/telegram login <token>[/cyan])")
        ids = credentials.get_telegram_allowed_ids()
        if ids:
            self._line(f"  allowed ids: [dim]{sorted(ids)}[/dim]")
        else:
            self._line("  allowed ids: [yellow]not set[/yellow]  ([cyan]/telegram allow <id>[/cyan])")
        if not running:
            self._line(
                "usage: [cyan]/telegram start[/cyan] · "
                "[cyan]/telegram test[/cyan] (verify token without polling) · "
                "[cyan]/telegram stop[/cyan]"
            )

    def _telegram_test(self) -> None:
        """One-shot getMe call against Telegram — fastest way to verify
        whether the saved token is the problem, without spinning up the
        full polling loop."""
        token = credentials.get_telegram_token()
        if not token:
            self._line("[red]no token saved[/red] — [cyan]/telegram login <token>[/cyan] first")
            return
        try:
            import telegram  # noqa: F401
        except ImportError as e:
            self._line(f"[red]missing dependency:[/red] {e}")
            return
        self._line(f"[dim]testing token {credentials.mask(token)} ({len(token)} chars) against api.telegram.org/getMe …[/dim]")
        self.run_worker(
            self._do_telegram_test(token),
            exclusive=False, group="telegram-test", thread=True,
        )

    async def _do_telegram_test(self, token: str) -> None:
        import asyncio
        try:
            from telegram import Bot
            from telegram.error import InvalidToken, NetworkError, TelegramError

            async def call() -> str:
                bot = Bot(token=token)
                async with bot:
                    me = await bot.get_me()
                return f"[green]✓ token works[/green]   bot=@{me.username}  id={me.id}  name={me.first_name!r}"

            result = asyncio.run(call())
        except InvalidToken as e:
            result = (
                f"[red]✗ Telegram rejected the token[/red]  ({type(e).__name__}: {e})\n"
                f"   stored value is {len(token)} chars; real BotFather tokens are typically ~46 chars\n"
                f"   if length is much longer, you likely pasted the token twice or copied surrounding text"
            )
        except NetworkError as e:
            result = f"[red]✗ network error reaching Telegram[/red]  ({type(e).__name__}: {e})"
        except TelegramError as e:
            result = f"[red]✗ Telegram error[/red]  ({type(e).__name__}: {e})"
        except Exception as e:
            result = f"[red]✗ unexpected error[/red]  ({type(e).__name__}: {e})"
        self.call_from_thread(self._line, result)

    def _telegram_login(self, token: str) -> None:
        if not token:
            self._line("usage: [cyan]/telegram login <bot-token>[/cyan]   (get one from @BotFather)")
            return
        # Don't pre-validate the shape. Token formats vary (test-server
        # tokens, newer issuances, …) and Telegram itself is the source
        # of truth at /telegram start. We just strip whitespace and warn
        # on really-obviously-wrong inputs.
        cleaned = "".join(token.split())
        if len(cleaned) < 20 or ":" not in cleaned:
            self._line(
                f"[red]that doesn't look like a real bot token[/red] "
                f"(got {len(cleaned)} chars, expected `<id>:<secret>`)"
            )
            return
        try:
            info = credentials.save_telegram_token_with_info(cleaned)
        except OSError as e:
            self._line(f"[red]failed to save:[/red] {e}")
            return
        saved = info["stored"]
        self._line(
            f"[green]✓ telegram token saved[/green]   "
            f"token=[dim]{credentials.mask(saved)}[/dim]  "
            f"[dim]({info['cleaned_len']} chars)[/dim]"
        )
        if info["deduped"]:
            self._line(
                f"[yellow]⚠ your paste looked doubled "
                f"({info['raw_len']} → {info['cleaned_len']} chars after dedup) — "
                "kept the first half. run [cyan]/telegram test[/cyan] to verify.[/yellow]"
            )
        self._line(f"[dim]saved to {credentials.CREDENTIALS_PATH}[/dim]")
        if not credentials.get_telegram_allowed_ids():
            self._line(
                "[dim]next:[/dim] [cyan]/telegram allow <your-telegram-id>[/cyan]  "
                "[dim](DM @userinfobot to get it)[/dim]"
            )

    def _telegram_allow(self, arg: str) -> None:
        if not arg:
            ids = credentials.get_telegram_allowed_ids()
            self._line(f"allowed ids: {sorted(ids) if ids else '(none)'}")
            self._line("usage: [cyan]/telegram allow <id>[,<id>...][/cyan]   (or [cyan]-[/cyan] to clear)")
            return
        if arg.strip() == "-":
            credentials.clear_telegram_allowed_ids()
            self._line("[yellow]allowed ids cleared[/yellow]")
            return
        try:
            ids = [int(t.strip()) for t in arg.split(",") if t.strip()]
        except ValueError:
            self._line(f"[red]ids must be integers:[/red] {arg!r}")
            return
        if not ids:
            self._line("[red]no ids parsed[/red]")
            return
        try:
            credentials.save_telegram_allowed_ids(ids)
        except OSError as e:
            self._line(f"[red]failed to save:[/red] {e}")
            return
        self._line(f"[green]✓ allowed ids saved[/green]: {sorted(ids)}")

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

        # Pull auto-approve out before applying overrides to the Agent
        # config (it's a runtime flag, not an Agent field).
        per_run_auto = overrides.pop("auto-approve", None)
        auto_approve = (
            self._auto_approve if per_run_auto is None
            else per_run_auto.lower() in ("1", "true", "yes", "on")
        )

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

        self._chat = start_chat(
            agent=agent,
            log_path=run_dir / "log.jsonl",
            workdir=run_dir / "wd",
            auto_approve=auto_approve,
        )
        self._chat_agent_name = agent_name
        self._active_run = run_dir / "log.jsonl"
        self._active_seen_seq = 0
        self._active_offset = 0
        self._active_buf = ""
        bits = [f"[cyan]{agent_name}[/cyan]", f"[b]{agent.model}[/b]"]
        if agent.tools:
            bits.append("tools: " + ", ".join(agent.tools))
        if agent.web:
            bits.append("web: " + ", ".join(agent.web))
        if auto_approve:
            bits.append("[red]auto-approve[/red]")
        self._line(f"[green]●[/green] chat with " + "  [dim]·[/dim]  ".join(bits))
        self._line(f"  [dim]{run_dir}[/dim]")
        self._line(
            "  [dim]type a message, or [/dim][cyan]/end[/cyan][dim] to stop[/dim]"
        )

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
            f"[yellow]○[/yellow] chat ended  "
            f"[dim]· {seat.turns_used} turns · ${seat.cost_usd:.4f}[/dim]"
        )
        self._chat = None
        self._chat_agent_name = None
        self._set_status("idle")
        self._active_run = None
        self._active_offset = 0
        self._active_buf = ""

    # ---- /prompts, /prompt --------------------------------------------- #

    def _cmd_prompts(self, rest: str) -> None:
        # With an argument, behave like /prompt — easy mistake to make,
        # and tab-completion suggests prompt names after `/prompts ` too.
        if rest.strip():
            self._cmd_prompt(rest)
            return
        prompts = list_prompts()
        if not prompts:
            self._line(
                f"[dim](no prompts yet — add markdown files to {PROMPTS_DIR})[/dim]"
            )
            return
        for n in prompts:
            path = PROMPTS_DIR / f"{n}.md"
            try:
                first = next(
                    (ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
                     if ln.strip()),
                    "",
                )
            except OSError:
                first = ""
            preview = first[:80] + ("…" if len(first) > 80 else "")
            self._line(f"  [cyan]{n}[/cyan]  [dim]{preview}[/dim]")

    def _cmd_prompt(self, rest: str) -> None:
        rest = rest.strip()
        if not rest:
            self._line("usage: [cyan]/prompt <name> [extra text...][/cyan]")
            self._line("see [cyan]/prompts[/cyan] for available prompts")
            return
        parts = rest.split(None, 1)
        name = parts[0]
        extra = parts[1].strip() if len(parts) > 1 else ""
        content = load_prompt(name)
        if content is None:
            self._line(
                f"[red]no such prompt:[/red] {name}  "
                f"(known: {', '.join(list_prompts()) or 'none'})"
            )
            return
        message = f"{content}\n\n{extra}" if extra else content
        if self._chat is None:
            if not credentials.get_api_key():
                self._line(
                    "[red]not signed in.[/red]  use [cyan]/login <key>[/cyan] first."
                )
                return
            self._cmd_chat("")
            if self._chat is None:
                return
        self._send_chat(message)

    def _cmd_p(self, rest: str) -> None:
        self._cmd_prompt(rest)

    def _send_chat(self, user_text: str) -> None:
        if self._is_running:
            self._line("[yellow]busy[/yellow] — wait for the current turn to finish")
            return
        # Asymmetric "activity log" layout: a thin user-turn marker opens
        # the turn; the agent's tool calls / thoughts then stream below it
        # as an indented activity stream; the final answer gets its own
        # timestamped block in _on_chat_reply once it actually lands. We
        # deliberately do NOT pre-print an agent header here — there's no
        # meaningful time to stamp yet, and a bare bar with a pile of tool
        # output dangling under it is exactly what felt broken.
        self._line("")
        self._line(
            f"[b cyan]┌─ you[/b cyan]  [dim]· {time.strftime('%H:%M:%S')}[/dim]"
        )
        for line in user_text.splitlines() or [""]:
            self._line(f"[cyan]│[/cyan] {escape(line)}")
        self._line("")
        self._is_running = True
        self._set_status("busy", "thinking…")
        self.run_worker(self._do_chat_turn(user_text), exclusive=True, group="chat", thread=True)

    async def _do_chat_turn(self, user_text: str) -> None:
        try:
            chat = self._chat
            if chat is None:
                self.call_from_thread(self._on_chat_reply, "[red]chat ended[/red]", True)
                return
            reply = chat.send(user_text)
            self.call_from_thread(self._on_chat_reply, reply, False)
        except Exception:
            self.call_from_thread(
                self._on_chat_reply,
                f"[red]chat error:[/red]\n{traceback.format_exc()}",
                True,
            )

    def _on_chat_reply(self, reply: str, errored: bool = False) -> None:
        self._tail_active_run()
        if reply:
            # The answer block opens HERE — after the activity stream and
            # at the moment the reply lands — so its timestamp marks the top
            # of the answer instead of being buried mid-block. Error replies
            # are pre-formatted with markup, so don't escape those.
            agent = self._chat_agent_name or "agent"
            self._line("")
            self._line(
                f"[b green]● {agent}[/b green]  [dim]· {time.strftime('%H:%M:%S')}[/dim]"
            )
            lines = reply.splitlines() or [""]
            for line in lines:
                self._line(f"  {line if errored else escape(line)}")
        self._is_running = False
        if errored:
            self._set_status("stopped", "error during turn", clear_after=8.0)
        elif self._status_kind != "stopped":
            # Don't trample a "stopped" already set by halt/denial.
            self._set_status("idle")

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

        per_run_auto = overrides.pop("auto-approve", None)
        auto_approve = (
            self._auto_approve if per_run_auto is None
            else per_run_auto.lower() in ("1", "true", "yes", "on")
        )

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

        bits = [f"[cyan]{agent_name}[/cyan]", f"[b]{agent.model}[/b]"]
        if agent.tools:
            bits.append("tools: " + ", ".join(agent.tools))
        if agent.web:
            bits.append("web: " + ", ".join(agent.web))
        if auto_approve:
            bits.append("[red]auto-approve[/red]")
        self._line(f"[green]▶[/green] running " + "  [dim]·[/dim]  ".join(bits))
        self._line(f"  [dim]{run_dir}[/dim]")
        self._active_run = run_dir / "log.jsonl"
        self._active_seen_seq = 0
        self._active_offset = 0
        self._active_buf = ""
        self._is_running = True
        self._set_status("busy", "thinking…")
        self.run_worker(
            self._do_run(agent, run_dir, message, auto_approve),
            exclusive=True, group="run", thread=True,
        )

    async def _do_run(self, agent: Agent, run_dir: Path, message: str, auto_approve: bool = False) -> None:
        try:
            def grab_ctx(ctx):
                self._run_ctx = ctx
            res = run_forest(
                agent=agent,
                log_path=run_dir / "log.jsonl",
                workdir=run_dir / "wd",
                user_message=message,
                on_ctx=grab_ctx,
                auto_approve=auto_approve,
            )
            seat = res.root_seat
            if seat.submit_result is not None:
                head = f"[green]✓ done[/green]  [italic]{seat.submit_result!r}[/italic]"
                final_status = ("idle", "")
            elif seat.halt_reason:
                head = f"[yellow]○ halt[/yellow]  [dim]{seat.halt_reason}[/dim]"
                final_status = ("stopped", seat.halt_reason or "halt")
            else:
                head = "[green]✓ done[/green]"
                final_status = ("idle", "")
            summary = (
                f"{head}\n"
                f"  [dim]{seat.turns_used} turns · "
                f"{seat.tokens_prompt}/{seat.tokens_completion} tok · "
                f"${seat.cost_usd:.4f}[/dim]"
            )
            self.call_from_thread(self._on_run_done, summary, final_status)
        except Exception:
            self.call_from_thread(
                self._on_run_done,
                f"[red]FAILED:[/red]\n{traceback.format_exc()}",
                ("stopped", "error during run"),
            )

    def _on_run_done(self, summary: str, final_status: tuple = ("idle", "")) -> None:
        self._tail_active_run()
        self._line(summary)
        self._line("")
        self._is_running = False
        self._run_ctx = None
        kind, label = final_status
        if kind == "stopped":
            self._set_status("stopped", label, clear_after=8.0)
        elif self._status_kind != "stopped":
            self._set_status("idle")

    # ---- live tail ------------------------------------------------------ #

    def _tail_active_run(self) -> None:
        # Incremental: read only the bytes appended since last tick from a
        # stored offset, rather than re-parsing the whole (potentially
        # multi-MB, full-history) log every 0.15s. A trailing partial line
        # is carried in _active_buf until its newline shows up.
        if self._active_run is None:
            return
        path = self._active_run
        try:
            size = path.stat().st_size
        except OSError:
            return
        if size < self._active_offset:
            # File was truncated/replaced (new run in the same slot) — restart.
            self._active_offset = 0
            self._active_buf = ""
        if size == self._active_offset:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                fh.seek(self._active_offset)
                chunk = fh.read()
                self._active_offset = fh.tell()
        except OSError:
            return
        data = self._active_buf + chunk
        lines = data.split("\n")
        # The last element is whatever follows the final newline — an
        # incomplete line if the writer is mid-append; hold it for next tick.
        self._active_buf = lines.pop()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("seq", 0) <= self._active_seen_seq:
                continue
            self._render_live(e)
            self._active_seen_seq = max(self._active_seen_seq, e.get("seq", 0))

    def _render_entry(self, e: dict) -> None:
        """Verbose, debug-style row used by `/view` for log replay."""
        sid = e["seat_id"] or "-"
        self._line(f"  [dim]{e['seq']:>3}[/dim] [cyan]{sid:<7}[/cyan] {summarize_entry(e)}")

    def _render_live(self, e: dict) -> None:
        """Compact, chat-style row used during /chat and /run live tail.

        Hides bookkeeping events (model_request, approval_decision, ok
        tool_result) and renders the rest with friendly icons instead of
        raw type names. The seq number is dropped; seat id appears only
        when it's not the root seat (i.e. sub-agents in a spawn tree).
        """
        t = e["type"]
        p = e.get("payload", {}) or {}
        sid = e.get("seat_id") or ""
        in_chat = self._chat is not None
        seat_prefix = f"[dim]{sid}[/dim] " if sid and sid != "s0" else ""

        if t == "model_request":
            # Dropped from the log view, but it's the clearest signal
            # that the model is currently working — drive the spinner.
            if self._is_running:
                self._set_status("busy", "thinking…")
            return
        if t == "approval_decision":
            return

        if t == "model_response":
            text = (p.get("text") or "").strip()
            tcs = p.get("tool_calls") or []
            cites = p.get("citations") or []

            # Implicit web search done by the model surfaces as citations.
            # There's no tool_call event for it, so render one here so the
            # user can see search activity actually happened.
            if cites:
                self._line(
                    f"{_ACT_IND}{seat_prefix}[{_ACT}]🔍 web search · "
                    f"{len(cites)} source"
                    f"{'s' if len(cites) != 1 else ''}[/{_ACT}]"
                )

            if not text:
                return

            # In chat mode, text-only model_response IS the final reply —
            # _on_chat_reply / the submit fold below renders it. Skip here
            # to avoid duplication. Text alongside tool_calls is narration
            # ("inner thought"), which is worth showing in both modes.
            is_inner_thought = bool(tcs)
            if in_chat and not is_inner_thought:
                return

            if len(text) > 500:
                text = text[:500].rstrip() + "…"
            for line in text.splitlines():
                self._line(
                    f"{_ACT_IND}{seat_prefix}[italic {_ACT}]💭 {escape(line)}[/italic {_ACT}]"
                )
            return

        if t == "tool_call":
            tool = p.get("tool") or "?"
            args = p.get("args") or {}
            # The dedicated "submit" event renders right after this, so
            # skip the tool_call form to avoid showing the same thing twice.
            if tool == "submit":
                if self._is_running:
                    self._set_status("busy", "finishing…")
                return
            preview = _tool_arg_preview(tool, args)
            glyph = _TOOL_GLYPH.get(tool, "·")
            line = f"{_ACT_IND}{seat_prefix}[{_ACT}]{glyph} {tool}[/{_ACT}]"
            if preview:
                line += f"  [{_ACT}]{escape(preview)}[/{_ACT}]"
            self._line(line)
            # Full multi-line input (the whole function / command) folded
            # under the header with a gutter, so the user sees ALL of it
            # rather than just the first line.
            for bl in _clip_block(_tool_input_lines(tool, args), _MAX_INPUT_LINES):
                self._line(f"{_ACT_IND}{seat_prefix}[{_ACT}]│[/{_ACT}] [{_ACT_TEXT}]{bl}[/{_ACT_TEXT}]")
            if self._is_running:
                self._set_status("busy", f"running {glyph} {tool}…")
            return

        if t == "tool_result":
            tool = p.get("tool") or "?"
            # submit / spawn have dedicated events that already render their
            # outcome; their tool_result would just double it.
            if tool in ("submit", "spawn"):
                if p.get("ok") and self._is_running:
                    self._set_status("busy", "thinking…")
                return
            if p.get("ok"):
                # Show the actual output, compactly. The full content is in
                # the log (and via /view); here we fold to _MAX_RESULT_LINES.
                content = (p.get("content") or "").rstrip()
                body = content.splitlines() if content else []
                if len(body) == 1 and len(body[0]) <= 100:
                    self._line(
                        f"{_ACT_IND}{seat_prefix}[{_ACT}]✓[/{_ACT}] "
                        f"[{_ACT_TEXT}]{escape(body[0])}[/{_ACT_TEXT}]"
                    )
                elif body:
                    self._line(f"{_ACT_IND}{seat_prefix}[{_ACT}]✓ {tool} result[/{_ACT}]")
                    for bl in _clip_block(body, _MAX_RESULT_LINES):
                        self._line(f"{_ACT_IND}{seat_prefix}[{_ACT}]│[/{_ACT}] [{_ACT_TEXT}]{bl}[/{_ACT_TEXT}]")
                else:
                    self._line(f"{_ACT_IND}{seat_prefix}[{_ACT}]✓ {tool}[/{_ACT}]")
                # Once a tool finishes, the model gets the result and starts
                # thinking about the next move — keep the spinner honest.
                if self._is_running:
                    self._set_status("busy", "thinking…")
                return
            err = (p.get("error") or "failed").splitlines()[0][:120]
            self._line(
                f"{_ACT_IND}{seat_prefix}[red]✗[/red] [{_ACT_TEXT}]{tool}[/{_ACT_TEXT}] "
                f"[red dim]{escape(err)}[/red dim]"
            )
            if self._is_running:
                self._set_status("busy", f"recovering from {tool} error…")
            return

        if t == "spawn":
            child = p.get("child_id")
            depth = p.get("depth")
            self._line(
                f"{_ACT_IND}{seat_prefix}[{_ACT}]↳ spawn {child} · depth={depth}[/{_ACT}]"
            )
            return

        if t == "submit":
            # In chat mode the submit text is exactly what _on_chat_reply
            # prints as the agent message — showing it here just doubles
            # the same content with a "✓ submit" label.
            if in_chat:
                return
            # Run-mode submit is the agent's actual answer — render it at
            # conversation prominence (normal text), not as a dim step.
            result = (p.get("result") or "").strip()
            self._line("")
            self._line(f"  {seat_prefix}[b green]✓ submit[/b green]")
            if result:
                if len(result) > 500:
                    result = result[:500].rstrip() + "…"
                for line in result.splitlines():
                    self._line(f"  {escape(line)}")
            return

        if t == "halt":
            reason = p.get("reason") or ""
            # A normal end-of-seat halt always trails a submit event that
            # already conveys completion — second line would be noise.
            if reason == "submit":
                return
            # An abnormal halt deserves a loud, sticky banner on the
            # status bar plus a clearly-formatted log block so the user
            # can't miss that the agent gave up.
            self._line("")
            self._line(
                f"{_ACT_IND}{seat_prefix}[b white on red] ■ STOPPED [/]  "
                f"[red]{reason}[/red]"
            )
            if sid and sid == "s0" or not sid:
                # Only flip the top-level bar for the root seat; a halt
                # in a sub-seat is the parent's problem, not the user's.
                self._set_status("stopped", reason, clear_after=8.0)
            return

        if t == "denial":
            reason = p.get("reason") or ""
            tool = p.get("tool") or ""
            self._line(
                f"{_ACT_IND}{seat_prefix}[yellow]denied[/yellow] "
                f"[yellow dim]{tool}: {reason}[/yellow dim]"
            )
            return

        # Fallback for any future event types — keep it quiet but visible.
        self._line(f"{_ACT_IND}{seat_prefix}[{_ACT}]{t}[/{_ACT}]")

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

    def action_scroll_one_up(self) -> None:
        out = self.query_one("#output", RichLog)
        out.auto_scroll = False
        out.scroll_up()

    def action_scroll_one_down(self) -> None:
        out = self.query_one("#output", RichLog)
        out.scroll_down()
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

    # ---- alt scroll mode (?1007h) -------------------------------------- #
    # Translates terminal mouse wheel events into ↑/↓ arrow keys while
    # leaving clicks/drags to the terminal (so native text selection keeps
    # working). Supported by xterm, iTerm2, gnome-terminal, kitty, alacritty,
    # wezterm. Terminal.app supports it but the default profile may have it
    # off — most users won't need to change anything.

    def _enable_alt_scroll(self) -> None:
        self._write_raw("\x1b[?1007h")

    def _disable_alt_scroll(self) -> None:
        self._write_raw("\x1b[?1007l")

    def _write_raw(self, seq: str) -> None:
        try:
            self._driver.write(seq)
            self._driver.flush()
        except Exception:
            try:
                import sys
                sys.stdout.write(seq)
                sys.stdout.flush()
            except Exception:
                pass

    def on_unmount(self) -> None:
        self._disable_alt_scroll()


def main() -> int:
    # mouse=False so terminal-native text selection (and cmd+c / ctrl+shift+c)
    # works on any terminal without requiring a modifier. Mouse wheel won't
    # scroll the output as a result — use PageUp / PageDown / Ctrl+End for
    # that (also faster than the wheel for the long logs we produce).
    HarnessApp().run(mouse=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
