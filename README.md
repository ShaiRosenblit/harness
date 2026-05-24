# harness

A small, fully-owned harness for running LLM agents — talk to one, or hand
it a task. Single-pane interactive shell over the official `openai` SDK
pointed at [OpenRouter](https://openrouter.ai). No agent framework
underneath.

```
harness/      substrate + driver  (the whole engine, ~7 files)
agents/       3 agent configs:  chat, task, research
ui.py         interactive shell
pyproject.toml
```

## Install

```
pip install -e .
```

Get an OpenRouter API key at <https://openrouter.ai/keys>.

## Run

```
python3 ui.py
/login sk-or-v1-...        # one-time; saves to ~/.config/harness/credentials.json (0600)
what is 7 ** 11?           # plain text -> auto-starts a chat
/end                       # end the chat
/run task <message>        # one-shot task that must call submit()
/run research <message>    # one-shot research with web access
/help                      # all commands
```

## Commands

| | |
|---|---|
| `/login <key>` · `/logout` | manage the saved OpenRouter API key |
| `/status` | auth, default model, chat state |
| `/model <id>` | session-wide model override (`/model -` clears) |
| `/agents` | list available agent configs |
| `/runs` | list past runs |
| `/view <run>` | replay a run's full timeline |
| `/tree <run>` | seat tree for a run |
| `/chat [agent] [flags]` | start a chat (defaults to `chat` agent) |
| `/end` | end the active chat |
| `/run <agent> [flags] <message>` | one-shot task |
| `/clear` `/help` `/quit` | |

Plain text without a leading `/` is sent as a chat message. If no chat
is active, one auto-starts.

## Flags for `/run` and `/chat`

Override any field of the agent config at launch time:

```
--model <id>             OpenRouter model id (e.g. anthropic/claude-haiku-4-5)
--tools a,b,c            local tools (code_exec, submit, spawn)
--web search,fetch       enable OpenRouter web tools (or --web none)
--max-turns N            per-seat turn cap
--max-depth N            spawn-recursion depth
--max-children N         siblings per seat
```

Example:

```
/run task --max-depth=3 --max-children=4 split 30 numbers into 4 ranges, spawn a sub-agent per range
/chat --web search       chat with web search but no fetch
/run research --model anthropic/claude-haiku-4-5 latest Kubernetes LTS version
```

## Agents

An agent is a single flat dataclass — a kind of agent, with one model,
one prompt, one set of tools, one set of limits. When a seat spawns, the
child gets a copy of the same config (one less depth slot). Capability
attenuation is automatic — child == parent.

Three agents ship:

- **`chat`** — talks to the user turn by turn. `code_exec` + web. No
  `submit` (yields on text-only replies). No `spawn`.
- **`task`** — one-shot. `code_exec` + `submit` + `spawn`. Must call
  `submit()` to finish. Children inherit the same config, so the agent
  can decompose recursively up to `max_depth` deep.
- **`research`** — single seat with web access. `code_exec` + `submit`
  + `web=(search, fetch)`. No `spawn`.

Add your own at `agents/<name>.py` — just one `AGENT = Agent(...)` per
file. The UI picks them up automatically.

## Logs

Each run writes one forest-wide JSONL at `runs/<agent>-<timestamp>/log.jsonl`.
Inspect inside the UI with `/view` and `/tree`, or read directly.

Every line: `{seq, ts, seat_id, parent_id, type, payload}`. Types:
`model_request`, `model_response`, `tool_call`, `tool_result`, `spawn`,
`submit`, `halt`, `denial`.

## What stops a runaway

- **`max_turns`** per seat — hard cap on model calls per agent.
- **`max_depth`** — hard cap on spawn recursion.
- **`max_children`** — hard cap on siblings per seat.
- **Ctrl+C** — kills the process.

Token/cost counters are tracked per seat for observability (`/status`
shows them, run summaries include them) but not enforced. If you want
a hard dollar ceiling, set `max_turns` tightly enough — at typical model
rates, that's the same thing.
