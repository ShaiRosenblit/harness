---
name: peer_comps
description: Peer comparable analysis — compare a target company against 3-6 peers on size, growth, profitability, and valuation multiples. Use when the user wants to know how a company stacks up vs its peers, or whether it's cheap/expensive on a relative basis. Triggers on "compare X to its peers", "how does X compare to Y", "peer comps for X", "is X cheap vs peers".
when_to_use: User wants a relative-value or competitive snapshot across a small group of similar companies.
---

# Peer Comparables Analysis

You are producing a peer comparison table and synthesis. Output is markdown via `submit()`. **Same metrics across all peers, same fiscal period.** Use `code_exec` for every calculation.

## Workflow

### Step 1 — Pick the peer set

If the user named peers, use exactly those. If not, propose 3-6 companies that match the target on:
- **Business model** — same revenue stream type
- **Scale** — within ~5x revenue
- **Geography** — comparable market exposure

Search the target's 10-K Risk Factors / Competition section for the company's own list of peers — that's the most authoritative source.

Confirm the peer set in your intro to the report.

### Step 2 — Pull the same metrics for each company

For each peer (and the target), pull:

**Size**
- Market cap (USD)
- Enterprise value (market cap + debt − cash)
- Revenue (most recent TTM)

**Growth**
- Revenue growth YoY (latest quarter, and TTM if available)
- Forward revenue growth (consensus next FY)

**Profitability**
- Gross margin
- Operating margin (or EBITDA margin)
- Net margin
- Free cash flow margin

**Valuation multiples**
- P/E (trailing)
- P/E (forward, on consensus)
- EV/Revenue (TTM)
- EV/EBITDA (TTM)
- Price/Sales

Pull from finance portals: stockanalysis.com, finviz.com, yahoo finance, marketwatch. Cite each.

### Step 3 — Normalize fiscal periods

If peers have different fiscal calendars, align them or flag mismatches explicitly:
- All TTM as of the latest reported quarter
- Forward = next 12 months (NTM) consensus

Use `code_exec` to compute consistent TTM rollups if needed.

### Step 4 — Build the table

Single comparison table, target highlighted:

| Company | Mkt Cap | Rev (TTM) | Rev YoY | EBITDA % | P/E (fwd) | EV/Rev | EV/EBITDA |
|---|---|---|---|---|---|---|---|
| **TARGET** | $XB | $XB | X% | X% | X.Xx | X.Xx | X.Xx |
| Peer 1 | | | | | | | |
| Peer 2 | | | | | | | |
| Peer 3 | | | | | | | |
| **Median** | | | | | | | |
| **Mean** | | | | | | | |

Compute median and mean in `code_exec` — never eyeball.

### Step 5 — Synthesize

Address:
- **Where does TARGET trade vs the median?** Premium / in-line / discount on each multiple.
- **Does the premium/discount reflect fundamentals?** (Faster growth deserves a premium; lower margins justify a discount.)
- **Outlier peers.** Any peer trading way above or below — why?
- **Implied value at peer median multiples.** If TARGET traded at the peer-median EV/EBITDA, what would the stock be worth?

### Step 6 — Submit

```
# Peer Comps — $TARGET vs $N peers

**Peer set:** $T1, $T2, $T3, … ($N total)
**Bottom line:** [1-2 sentences on whether target is cheap, fair, or expensive vs peers]

## Comparison
[the table from Step 4]

## Multiples vs median
- P/E: $TARGET at X.Xx vs median Y.Yx → $Z% premium/discount
- EV/EBITDA: ...
- EV/Rev: ...

## Why the gap (or lack of one)?
- Growth: ...
- Margins: ...
- Other (positioning, cyclicality, leverage): ...

## Implied value at peer-median multiples
- At median EV/EBITDA Y.Yx → $XX.XX/share (X% up/down from current $YY.YY)
- At median P/E Y.Yx → $XX.XX/share (X% up/down)

## Caveats
- [Any metric gaps in the source data]
- [Any peers whose business model differs meaningfully]

## Sources
- [stockanalysis](url)  ·  [yahoo finance](url)  ·  …
```

## Discipline

- **Same metric definition for every peer.** "EBITDA" can mean different things — use the same source format for each.
- **No mental math.** Compute medians, percentages, premiums in code.
- **Date the snapshot.** Multiples change daily — note "as of $DATE".
- **Mark estimates clearly.** Anything that's a forward number is consensus, mark "(E)" or "(NTM)".
