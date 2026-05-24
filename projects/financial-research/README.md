# financial-research

Real, sourced financial research runs driven by the harness.

Each run produces a dated, sourced memo in `memos/` and a full JSONL
transcript under `runs/` at the repo root.

## Layout

```
projects/financial-research/
  README.md          this file
  run.py             runner: launch a research seat, save the memo
  questions/         standing research questions (one per file)
  memos/             one memo per run: YYYY-MM-DD-slug.md
```

The agent itself lives at `agents/finance_research.py` so the harness
UI picks it up automatically.

## Usage

From the repo root:

```bash
python3 projects/financial-research/run.py questions/<slug>.md
```

Or one-off, ad-hoc:

```bash
python3 projects/financial-research/run.py --ask "Q1 2026 hyperscaler capex check"
```

Override the model:

```bash
python3 projects/financial-research/run.py --model anthropic/claude-haiku-4-5 questions/<slug>.md
```

## What the agent does

- Web-searches primary sources first (press releases, 10-Q/10-K, earnings
  call transcripts, official filings) before commentary.
- Cross-checks every number against at least two sources where possible.
- Returns a memo with explicit numbers, dates, and source URLs — no
  unsourced claims.
- Refuses to speculate beyond what sources support.

## What the agent does not do

- Give investment advice or price targets.
- Predict prices.
- Cite a source it didn't actually fetch.
