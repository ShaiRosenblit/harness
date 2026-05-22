from __future__ import annotations

from pathlib import Path
from typing import Optional

from .types import Seat


def _is_dotted_prefix(prefix: str, sid: str) -> bool:
    if prefix == sid:
        return True
    return sid.startswith(prefix + ".")


def check(seat: Seat, path: Path) -> Optional[str]:
    """Return a halt reason if this seat (or its subtree, by lineage prefix)
    should die, else None.

    Convention:
      - file missing -> alive
      - file present, empty/whitespace only -> whole forest killed
      - file present with non-blank lines -> each line is a dotted seat-id
        prefix; this seat is killed iff any line is a prefix of seat.id.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "kill_forest"
    for ln in lines:
        if _is_dotted_prefix(ln, seat.id):
            return f"kill_seat:{ln}"
    return None
