---
name: company_research
description: End-to-end research on a public company — business model, financials, recent developments, competitive position, and risks. Use when the user asks to research, analyze, or investigate a specific company. Triggers on "research $TICKER", "tell me about Company X", "is X a good investment", "what does X do".
when_to_use: User wants a comprehensive picture of a public company in one pass.
---

# Company Research

You are doing structured research on a specific public company. The output is a markdown report submitted via `submit()`. **Lead with the conclusion. Cite every fact with a URL.**

## Workflow

### Step 1 — Identify the company precisely

Confirm the ticker, exchange, and full legal name. If the user gave an ambiguous reference, search for "$TERM stock ticker site:finance.yahoo.com OR site:sec.gov" and disambiguate.

### Step 2 — Business overview

Search for the company's most recent 10-K filing (annual report) on SEC EDGAR:
`web_search("$COMPANY 10-K $YEAR site:sec.gov")` — use the current year from the harness context.

Extract:
- **What they do** — one sentence, plain English.
- **How they make money** — revenue segments with percentages.
- **Geographic mix** — US vs international.
- **Business model** — recurring vs transactional, B2B vs B2C, capital intensity.

### Step 3 — Financial snapshot

Pull the most recent quarterly results. Search: `"$COMPANY Q$N $YEAR earnings results" OR "$COMPANY 10-Q $YEAR site:sec.gov"`.

Build a small table:

| Metric | Latest | YoY % | TTM |
|---|---|---|---|
| Revenue | | | |
| Gross margin | | | |
| Operating margin | | | |
| Net income / Adjusted EBITDA | | | |
| Free cash flow | | | |

Use `code_exec` for any arithmetic (growth %, margins, TTM aggregation). Never compute in your head.

### Step 4 — Valuation snapshot

Search for current market cap and headline multiples on a finance portal (Yahoo Finance, Stockanalysis, Finviz). Capture:
- Market cap
- P/E (trailing and forward)
- EV/EBITDA
- Price/Sales

If you can find consensus estimates (forward revenue / EPS growth), include them. Note any 52-week range to anchor where the stock trades.

### Step 5 — Recent developments (last 90 days)

Search: `"$COMPANY news $CURRENT_YEAR -$MONTH_AGO"` or similar. Surface:
- Last earnings call major themes
- Any guidance updates
- Material announcements (M&A, leadership change, lawsuits, product launches)
- Notable analyst upgrades/downgrades

### Step 6 — Competitive position

Identify 2-4 direct competitors. For each, note:
- Relative size (market cap, revenue)
- Where they overlap vs differentiate
- Any market-share trends visible from recent commentary

### Step 7 — Risks

List 3-5 specific, concrete risks from the company's own 10-K Risk Factors section plus anything material from recent news. Avoid generic risks ("macro headwinds") — be specific.

### Step 8 — Synthesize and submit

Submit a markdown report with this structure:

```
# $COMPANY ($TICKER) — Research Report

**Bottom line:** [1-2 sentences with your overall read]

## What they do
…

## Financial snapshot
[table]

## Valuation
…

## Recent developments
- …
- …

## Competitive position
- …

## Risks
1. …
2. …

## Sources
- [source 1](url)
- [source 2](url)
```

## Discipline

- **Date everything.** A fact without a date is unusable. Cite the filing date or article date for every number.
- **No invented numbers.** If you can't find a number, say so explicitly rather than guessing.
- **Use code_exec for arithmetic.** Growth percentages, margins, TTM rollups — all in code.
- **Search with the current year.** Stale results are the most common failure mode for time-sensitive research.
