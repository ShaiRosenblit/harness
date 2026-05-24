---
name: dcf_valuation
description: Discounted Cash Flow valuation — project free cash flow, compute WACC, discount, derive equity value per share. Use when the user asks for a DCF, intrinsic value, fair value estimate, or "what's $TICKER worth". Triggers on "DCF for X", "value X", "intrinsic value of X", "is X overvalued/undervalued".
when_to_use: User wants a quantitative valuation, not just qualitative analysis.
---

# DCF Valuation

You are producing a DCF model for a public company. **Use `code_exec` for every calculation** — never compute in your head. The output is a markdown report with the model, assumptions, and a sensitivity table, submitted via `submit()`.

## Workflow

### Step 1 — Gather inputs

For the target company, pull from recent 10-K / 10-Q / finance portals:

**Historical (last 3 years if available):**
- Revenue
- EBIT (operating income)
- D&A (depreciation & amortization)
- CapEx (from cash flow statement)
- Change in working capital
- Effective tax rate
- Stock-based compensation (treat as a cash cost in adjusted FCF)

**Capital structure / market:**
- Shares outstanding (diluted)
- Total debt (book value)
- Cash & equivalents
- Current share price
- Beta (from Yahoo Finance, Stockanalysis, etc.)
- Risk-free rate (current 10-year Treasury yield)
- Equity risk premium (use 5-6% unless otherwise sourced)
- Cost of debt (estimate from credit rating or recent bond yields)

Cite the filing or page for every number.

### Step 2 — Project free cash flow (5 years)

State your assumptions explicitly:
- **Revenue growth** by year (start near recent run-rate, decline toward GDP)
- **Operating margin** trajectory (expansion, stable, or compression)
- **D&A as % of revenue** (typically stable)
- **CapEx as % of revenue** (typically stable or trending down for mature firms)
- **Working capital change** as % of revenue change

Then compute, in code:

```python
fcf = ebit * (1 - tax_rate) + da - capex - delta_wc
```

Use `code_exec` to project all 5 years into a table.

### Step 3 — Compute WACC

In code:

```python
re = risk_free + beta * equity_risk_premium     # cost of equity (CAPM)
rd_after_tax = cost_of_debt * (1 - tax_rate)
total = equity_value + debt_value
wacc = (equity_value/total) * re + (debt_value/total) * rd_after_tax
```

Use **market value of equity** (= share price × shares) for the equity weight, not book.

### Step 4 — Terminal value

Pick a terminal growth rate `g` (1.5–3.0% for mature firms; never above the country's long-run GDP growth).

```python
terminal_value = fcf_year_5 * (1 + g) / (wacc - g)   # Gordon growth
```

### Step 5 — Discount and roll up

```python
pv_of_fcf  = sum(fcf[i] / (1 + wacc) ** (i+1) for i in range(5))
pv_terminal = terminal_value / (1 + wacc) ** 5
enterprise_value = pv_of_fcf + pv_terminal
equity_value     = enterprise_value - debt + cash
per_share        = equity_value / shares_diluted
```

Compare `per_share` to current price → % upside or downside.

### Step 6 — Sensitivity

Re-run for 3 WACC values (base ± 1pp) × 3 terminal-growth values (base ± 0.5pp) = a 3×3 table.

```python
for wacc in [base_wacc - 0.01, base_wacc, base_wacc + 0.01]:
    for g in [base_g - 0.005, base_g, base_g + 0.005]:
        # recompute per_share, store in table
```

### Step 7 — Submit

```
# DCF Valuation — $COMPANY ($TICKER)

**Implied fair value:** $XX.XX per share (vs current $YY.YY — Z% upside/downside)
**Base WACC:** X.X%  ·  **Terminal growth:** Y.Y%

## Inputs
[bullet list, every number cited]

## Projection (FCF, $M)
| | Y1 | Y2 | Y3 | Y4 | Y5 |
|---|---|---|---|---|---|
| Revenue | | | | | |
| EBIT | | | | | |
| FCF | | | | | |

## WACC
- Cost of equity (CAPM): X.X%
- After-tax cost of debt: X.X%
- WACC: X.X%

## Valuation
- PV of explicit FCF: $XB
- PV of terminal value: $XB
- Enterprise value: $XB
- Equity value: $XB
- Per share: $XX.XX

## Sensitivity ($ per share)
|   | g=X.X% | g=Y.Y% | g=Z.Z% |
|---|---|---|---|
| WACC=A | | | |
| WACC=B | | | |
| WACC=C | | | |

## Key risks to the valuation
1. …
2. …

## Sources
- [10-K](url)  ·  [Q earnings](url)  ·  [share price source](url)
```

## Discipline

- **Every number computed in `code_exec`.** No mental math, no estimation.
- **State every assumption.** A DCF without explicit assumptions is just numerology.
- **Be honest about uncertainty.** A 5×5 sensitivity table is more informative than a single point estimate.
- **Note when an input is your estimate.** Equity risk premium, terminal growth — these are choices, mark them.
