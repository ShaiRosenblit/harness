---
name: sec_filings
description: Locate, fetch, and analyze SEC filings (10-K, 10-Q, 8-K, S-1, proxy) from EDGAR. Use when you need primary-source company data — accounting policies, risk factors, segment disclosures, executive comp, material events. Triggers on "read the 10-K of X", "what does X's S-1 say", "latest 8-K from X", "SEC filing for X".
when_to_use: You need authoritative primary-source data straight from a filing, not from a news summary.
---

# SEC Filings — Lookup and Analyze

You are pulling primary-source data from SEC EDGAR and reading the filing carefully. **Use `web_fetch` to read the actual filing**, not just search snippets. Output is markdown via `submit()`.

## Workflow

### Step 1 — Find the filing index

EDGAR's company page lives at:
`https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=$CIK&type=$FORM&dateb=&owner=include&count=10`

If you don't have the CIK, search:
`web_search("$COMPANY CIK site:sec.gov")` — or use `site:sec.gov/cgi-bin/browse-edgar $COMPANY`.

Common form types:
- **10-K** — annual report (most comprehensive)
- **10-Q** — quarterly report (lighter than 10-K)
- **8-K** — current report (announcements, M&A, leadership change)
- **S-1** — IPO registration
- **DEF 14A** — proxy (governance, exec comp)
- **13F** — institutional holdings (filed by funds, quarterly)

### Step 2 — Fetch the filing

Once you have the filing URL, use `web_fetch`. Filings are long — be deliberate about which sections you ask for:

For **10-K**, the high-value sections are:
- **Item 1: Business** — what they do, segments
- **Item 1A: Risk Factors** — material risks (read these carefully)
- **Item 7: MD&A** — management's discussion of results
- **Item 7A: Quantitative & Qualitative Disclosures About Market Risk**
- **Item 8: Financial Statements & Notes** — the actual financials + accounting policies

For **10-Q**:
- **Item 1: Financial Statements** (the most recent quarter)
- **Item 2: MD&A**

For **8-K**:
- Read the whole thing — they're usually short and material.

For **S-1**:
- **Use of Proceeds**, **Risk Factors**, **Business**, **MD&A**.

For **DEF 14A**:
- Compensation tables, related-party transactions, board composition.

### Step 3 — Extract what the user asked for

Quote exact passages — never paraphrase financial figures. If the user wants risk factors, list them with the filing's own phrasing. If they want a metric, give the exact number, the page reference (if findable), and the filing date.

### Step 4 — Synthesize and submit

```
# $COMPANY — $FORM filed $DATE

**Filing URL:** $url
**Period covered:** [fiscal period]

## [Section the user wanted]

> [Direct quotes from the filing, with exact figures]

## Key takeaways
- …
- …

## Sources
- [$FORM]($url)
- [EDGAR company page]($index_url)
```

## Discipline

- **Direct quotes for material claims.** A 10-K is a legal document — phrasing matters.
- **Cite the page or section** when possible.
- **Note the filing date.** A 10-K from 18 months ago has stale risk factors.
- **Don't summarize whole filings.** Pull only what was asked for. Filings are too long for a one-shot summary.
