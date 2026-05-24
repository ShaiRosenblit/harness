---
name: sector_overview
description: Industry / sector landscape report — TAM, growth, structure, top players, valuation context, key trends and risks. Use when the user asks about an industry, sector, or market space rather than a single company. Triggers on "research the X sector", "industry overview of X", "what's happening in X", "X market landscape".
when_to_use: User wants the lay of the land for an industry — its size, players, dynamics, and outlook.
---

# Sector Overview

You are producing a sector landscape report. The output is a markdown report submitted via `submit()`. **Lead with the key dynamics. Cite sources for every claim.**

## Workflow

### Step 1 — Scope

Pin down what the user actually means:
- Sector vs sub-sector (e.g. "fintech" vs "B2B payment APIs")
- Geographic scope (US, EU, global)
- Time horizon (current state, 1-year outlook, 5-year outlook)

If the user's request is ambiguous, name the scope you're using in your report's intro.

### Step 2 — Market size & growth

Search for TAM estimates from named research firms:
`"$SECTOR market size $CURRENT_YEAR" OR "$SECTOR TAM forecast"`

Capture:
- Current market size (USD) + the year measured
- Historical growth (CAGR, last 3-5 years if available)
- Forecast growth (CAGR, next 3-5 years if available)
- 1-3 source firms cited (Gartner, McKinsey, IDC, etc.)

If estimates differ wildly across sources, note the range and the most credible methodology.

### Step 3 — Structure

- **Fragmented vs consolidated.** Estimate top 5 market share if visible.
- **Value chain.** Where does value accrue (manufacturers vs platforms vs distributors)?
- **Business model patterns.** Subscription, transactional, ad-supported, licensing.
- **Barriers to entry.** Capital, regulation, network effects, technical depth.

### Step 4 — Top players

Pick 5-10 leading public companies in the sector. For each:
- One-sentence description
- Approximate revenue or market cap
- Distinct positioning (price leader, premium, niche, etc.)

Build a comparison table:

| Company | Market cap | Revenue | YoY growth | Positioning |
|---|---|---|---|---|
| | | | | |

Use `code_exec` to format the table cleanly.

### Step 5 — Key trends

3-5 secular trends shaping the sector. For each:
- What's changing
- Who benefits / who's at risk
- Concrete evidence (a recent deal, a stat, a regulatory move)

### Step 6 — Risks & headwinds

3-5 sector-level risks. Be specific — name regulations, technological disruptors, demand shifts.

### Step 7 — Valuation context

If the sector has identifiable trading patterns:
- Typical EV/Revenue or EV/EBITDA range
- Recent M&A multiples (if any notable transactions)
- Is the sector trading at a premium or discount to its history / the broader market?

### Step 8 — Submit

```
# $SECTOR — Landscape Overview

**Scope:** [what you covered]
**Key takeaway:** [1-2 sentences]

## Market size & growth
$XXB in $YEAR, growing $X% (CAGR), per [source].
[1-2 paragraphs with the data]

## Structure
…

## Top players
[table]

## Trends
1. **Trend name** — what's happening + evidence
2. …

## Risks
1. **Risk name** — why it matters
2. …

## Valuation context
…

## Sources
- [source 1](url)
- …
```

## Discipline

- **Source every market-size number.** "The market is $80B" is worthless without a citation.
- **Date data.** Different reports estimate different years — say which.
- **Distinguish TAM from realistic addressable market.** TAM is often inflated.
- **Use `web_fetch` to read 1-2 industry reports in depth** rather than summarizing only from search snippets.
