"""Skills — reusable agent workflows loaded on demand.

A skill is a markdown file at `skills/<name>.md` with simple YAML-ish
front matter. The agent sees the name + description of every available
skill in its system prompt; when it judges a skill is relevant, it
calls `use_skill(name)` to load the full body into context.

Front matter format (keys delimited by leading and trailing `---` lines):

    ---
    name: deep_research
    description: Multi-step research with cross-verification.
    when_to_use: When one source isn't enough for a factual claim.
    ---

    <skill body — the procedure the agent should follow>

Only `description` is required; `name` defaults to the filename stem.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    when_to_use: str
    body: str
    path: Path


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse a leading `---`-delimited frontmatter block. Returns (meta, body)."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    _, fm_text, body = parts
    meta: dict = {}
    for line in fm_text.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, body.lstrip("\n")


def list_skills() -> List[Skill]:
    """All skills in `skills/<name>.md`, sorted by name."""
    if not SKILLS_DIR.exists():
        return []
    out: List[Skill] = []
    for p in sorted(SKILLS_DIR.glob("*.md")):
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        meta, body = _parse_frontmatter(text)
        out.append(
            Skill(
                name=meta.get("name", p.stem),
                description=meta.get("description", ""),
                when_to_use=meta.get("when_to_use", ""),
                body=body,
                path=p,
            )
        )
    return out


def load_skill(name: str) -> Optional[Skill]:
    for s in list_skills():
        if s.name == name:
            return s
    return None
