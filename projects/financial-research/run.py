#!/usr/bin/env python3
"""Runner for the financial-research project.

Loads the finance_research agent, runs one investigation, writes the
submitted memo to memos/YYYY-MM-DD-<slug>.md, and prints the run path.

Usage:
    python3 projects/financial-research/run.py questions/<slug>.md
    python3 projects/financial-research/run.py --ask "<one-line question>"

Flags:
    --model <id>          override the agent's model
    --max-turns N         override per-seat turn cap
    --slug <name>         override the memo slug (default: derived)
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.forest import run_forest  # noqa: E402

AGENTS_DIR = ROOT / "agents"
PROJECT_DIR = ROOT / "projects" / "financial-research"
RUNS_DIR = ROOT / "runs"
MEMOS_DIR = PROJECT_DIR / "memos"


def load_agent(name: str):
    import importlib.util

    path = AGENTS_DIR / f"{name}.py"
    if not path.exists():
        sys.exit(f"agent not found: {path}")
    spec = importlib.util.spec_from_file_location(f"agents.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if not hasattr(mod, "AGENT"):
        sys.exit(f"{path} has no AGENT")
    return mod.AGENT


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:60] or "memo"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("question_file", nargs="?", help="path to a questions/<slug>.md file")
    p.add_argument("--ask", help="one-line question (alternative to question_file)")
    p.add_argument("--model", help="override agent model")
    p.add_argument("--max-turns", type=int, dest="max_turns")
    p.add_argument("--slug", help="memo filename slug")
    p.add_argument("--agent", default="finance_research")
    p.add_argument(
        "--provider",
        help="comma-separated OpenRouter provider slugs (e.g. moonshotai,together)",
    )
    args = p.parse_args()

    if args.question_file:
        qpath = Path(args.question_file)
        if not qpath.is_absolute():
            qpath = (PROJECT_DIR / args.question_file).resolve()
            if not qpath.exists():
                qpath = (Path.cwd() / args.question_file).resolve()
        if not qpath.exists():
            sys.exit(f"question file not found: {args.question_file}")
        question_text = qpath.read_text().strip()
        default_slug = qpath.stem
    elif args.ask:
        question_text = args.ask.strip()
        default_slug = slugify(args.ask)
    else:
        sys.exit("provide a question_file or --ask")

    agent = load_agent(args.agent)
    if args.model:
        agent = replace(agent, model=args.model)
    if args.max_turns:
        agent = replace(agent, max_turns=args.max_turns)
    if args.provider:
        agent = replace(
            agent,
            provider=tuple(s.strip() for s in args.provider.split(",") if s.strip()),
        )

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    date = datetime.now().strftime("%Y-%m-%d")
    slug = args.slug or default_slug
    run_dir = RUNS_DIR / f"finance-{slug}-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"

    print(f"agent:    {args.agent} ({agent.model})")
    print(f"question: {question_text[:120]}{'...' if len(question_text) > 120 else ''}")
    print(f"log:      {log_path}")
    print()

    result = run_forest(
        agent=agent,
        log_path=log_path,
        workdir=run_dir,
        user_message=question_text,
    )

    seat = result.root_seat
    final = result.final

    print()
    print(f"turns:   {seat.turns_used}")
    print(f"tokens:  prompt={seat.tokens_prompt} completion={seat.tokens_completion}")
    print(f"cost:    ${seat.cost_usd:.4f}")
    print(f"halted:  {seat.halt_reason or '—'}")

    memo = (final.content or "").strip()
    if not memo:
        print("\nno memo submitted.")
        return 1

    MEMOS_DIR.mkdir(parents=True, exist_ok=True)
    memo_path = MEMOS_DIR / f"{date}-{slug}.md"
    if memo_path.exists():
        memo_path = MEMOS_DIR / f"{date}-{slug}-{ts}.md"

    header = (
        f"<!-- run: {run_dir.name}  model: {agent.model}  "
        f"turns: {seat.turns_used}  cost: ${seat.cost_usd:.4f} -->\n\n"
    )
    memo_path.write_text(header + memo + "\n")
    print(f"\nmemo:    {memo_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
