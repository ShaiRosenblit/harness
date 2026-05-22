# harness

A small, fully-owned harness for running LLM agents — chat with one, or hand
it a task. Built on the official `openai` SDK pointed at
[OpenRouter](https://openrouter.ai), with no agent framework underneath.

Single file UI (`ui.py`). One forest-wide JSONL log per run. Capability
grants, budget caps, and a file-based kill switch are all enforced by the
substrate; what the agent does is controlled by the policy.

## Install

```
pip install -e .
```

Get an OpenRouter API key from <https://openrouter.ai/keys>.

## Run

```
python3 ui.py
```

Then, inside the UI:

```
/login sk-or-v1-...     # one-time; saves the key to ~/.config/harness/credentials.json (0600)
what is 7 ** 11?        # plain text -> auto-starts a chat
/end                    # end the chat
/run task find the 1000th prime, then submit it as a string
/help                   # all commands
```

## Commands

| | |
|---|---|
| `/login <key>` | save your OpenRouter API key |
| `/logout` | clear the saved key |
| `/status` | auth, default model, chat state |
| `/model <id>` | session-wide model override (e.g. `anthropic/claude-haiku-4-5`); `/model -` clears it |
| `/policies` | list available policies |
| `/runs` | list past runs |
| `/view <run>` | replay a run's timeline |
| `/tree <run>` | seat tree for a run |
| `/chat [policy] [--model M]` | start a chat session (default policy: `chat`) |
| `/end` | end the active chat |
| `/run <policy> [--model M] <message...>` | one-shot task (the agent must call `submit` to finish) |
| `/clear` `/help` `/quit` | |

Plain text without a leading `/` is interpreted as a chat message. If no
chat is active, one auto-starts with the default policy.

## Policies

A policy in `policies/<name>.py` controls what an agent can do:

- system prompt
- granted tools (subset of `code_exec`, `submit`, `spawn`)
- model id (any OpenRouter model)
- budget cap (USD) and limits (turns, depth, breadth, concurrent seats)
- for spawn: a `child_policy` template (children's tools/budget are
  attenuated from the parent)

Two policies ship:

- **`chat`** — conversational mode. `code_exec` only (no `submit`); the
  agent answers in text, you keep the conversation going.
- **`task`** — one-shot mode. `code_exec` + `submit` + `spawn`. The agent
  must call `submit(...)` to finish. Sub-agents inherit `code_exec` +
  `submit` only.

To add your own, copy one of these and edit. The UI picks them up
automatically.

## Logs and runs

Each run writes a forest-wide JSONL log to
`runs/<policy>-<timestamp>/log.jsonl`. Inspect inside the UI with
`/view` and `/tree`, or read the JSONL directly.

Every line: `{seq, ts, seat_id, parent_id, type, payload}`. Types:
`model_request`, `model_response`, `tool_call`, `tool_result`, `spawn`,
`submit`, `halt`, `denial`.

## Kill switch

Drop a file at `runs/<name>/kill`:

- file missing → alive
- file present, empty → whole forest halts on its next pre-turn check
- file present, non-empty → each line is a dotted seat-id prefix; matching
  seats and their subtrees halt

## Layout

```
harness/        substrate + driver
  types.py        Seat, Policy, Limits, Budget, LogEntry, ToolSpec, ToolResult
  log.py          single-file JSONL writer
  killswitch.py   file-based kill check
  credentials.py  ~/.config/harness/credentials.json
  model.py        the one OpenRouter call wrapper
  tools.py        adjudicator + built-in tools (code_exec, submit, spawn)
  driver.py       the per-seat turn loop
  forest.py       /run entrypoint
  session.py     /chat entrypoint (persistent seat, yields on text-only)
policies/
  chat.py
  task.py
ui.py
pyproject.toml
```
