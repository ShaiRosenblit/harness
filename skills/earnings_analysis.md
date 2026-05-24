---
name: earnings_analysis
description: Analyze a company's quarterly earnings — beat/miss vs estimates, key metric trends, guidance updates, management commentary, and the read-through to the investment thesis. Use after a company reports earnings. Triggers on "analyze $TICKER earnings", "Q$N results", "post-earnings on X", "what happened with X earnings".
when_to_use: A company has just reported earnings and the user wants a tight analyst-style note.
---

# Earnings Analysis

You are writing a post-earnings update. The reader is familiar with the company — focus on **what's new**, not company background. Output is markdown via `submit()`.

## Workflow

### Step 1 — Identify the quarter

Pin down: ticker, quarter (e.g. "Q3 FY2026"), reporting date. Search:
`"$COMPANY Q$N $YEAR earnings"` — include the current year from harness context.

### Step 2 — Pull the headline numbers

For revenue and EPS:
- **Reported actual** value
- **Consensus estimate** (the "Street" number before the print)
- **Beat / miss** in absolute and percent terms — compute with `code_exec`

Search for consensus on Yahoo Finance, Stockanalysis, Seeking Alpha, or earnings-tracker sites. If consensus is not findable, say so explicitly rather than guessing.

### Step 3 — Segment / metric breakdown

Identify the 3-5 metrics the market actually cares about for this company (varies by industry):

- **SaaS:** ARR, NRR, customer count, gross margin, FCF
- **Retail:** Same-store sales, gross margin, inventory
- **Banks:** Net interest margin, loan growth, NPL ratio, return on tangible equity
- **Energy:** Production, realized prices, capex, FCF, ND/EBITDA
- **Ads/media:** DAU/MAU, ARPU, ad pricing, content cost

For each metric, capture: reported value, YoY change, sequential (QoQ) change if relevant. Compute YoY/QoQ with `code_exec`.

### Step 4 — Guidance update

Did the company change forward guidance? For each component:
- Prior guidance
- New guidance
- Direction (raised / maintained / lowered)
- Magnitude of change

This is usually the single most market-moving piece of information.

### Step 5 — Management commentary

Read the earnings press release and (if available) the prepared remarks from the call. Search:
`"$COMPANY Q$N $YEAR earnings call transcript"` — try seekingalpha.com or motleyfool.com.

Capture:
- 2-3 sentences on what management called out as drivers (positive)
- 2-3 sentences on what they cited as headwinds (negative)
- Any major new strategic moves (M&A, restructuring, capex plans)
- Notable Q&A exchanges that exposed something material

### Step 6 — Read-through to thesis

In 3-5 sentences, address:
- Does this print confirm or challenge the bull case?
- Does it move estimates materially? (For revenue and FCF the next year)
- Is the stock reaction (post-print move) sensible vs the fundamentals?

### Step 7 — Submit

```
# $COMPANY ($TICKER) — Q$N FY$YEAR Earnings Update

**Bottom line:** [1-2 sentences: beat/miss + thesis impact]
**Stock reaction:** $XX (+/- Y% after-hours)

## Headline numbers

| | Reported | Consensus | Beat/Miss |
|---|---|---|---|
| Revenue | $X.XB | $X.XB | +X% |
| EPS (Adj.) | $X.XX | $X.XX | +$0.0X |

## Key metrics
- **Metric 1:** $XXX, +X% YoY (vs +Y% prior quarter)
- **Metric 2:** $XXX, ...
- …

## Guidance
| | Prior | New | Change |
|---|---|---|---|
| FY rev | $X | $Y | +/-Z% |
| FY EPS | $X | $Y | +/-Z% |

## What management said
- Drivers (positive): …
- Headwinds (negative): …
- Strategic notes: …

## Read-through
[3-5 sentences on thesis impact]

## Sources
- [Earnings release](url)  ·  [10-Q](url)  ·  [Transcript](url)
```

## Discipline

- **Date the data.** Quarter and fiscal-year framing — `Q3 FY2026 (Oct 2025)` or whatever the company uses.
- **Compute every variance in code.** Beat/miss percentages, YoY growth, all of it.
- **Don't recycle company background.** This is an update for someone who already knows the story.
- **Don't infer guidance from press releases ambiguously** — quote the exact language if possible.
