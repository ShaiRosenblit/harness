from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

from .types import LogEntry, Seat


class Log:
    """Single forest-wide append-only JSONL log. Brief rule #3.

    Thread-safe: writes are guarded by an internal lock so concurrent
    seats (parallel-within-turn spawn) can't interleave JSON inside one
    line and the `seq` counter stays monotonic.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", buffering=1, encoding="utf-8")
        self._seq = 0
        self._write_lock = threading.Lock()
        if self.path.stat().st_size > 0:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._seq = max(self._seq, json.loads(line).get("seq", 0))

    def write(
        self,
        seat: Optional[Seat],
        type_: str,
        payload: dict,
    ) -> int:
        with self._write_lock:
            self._seq += 1
            seq = self._seq
            entry = {
                "seq": seq,
                "ts": time.time(),
                "seat_id": seat.id if seat is not None else "",
                "parent_id": (seat.parent_id if seat is not None else None),
                "type": type_,
                "payload": payload,
            }
            self._fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            return seq

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

    def stream(self) -> Iterator[LogEntry]:
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                yield LogEntry(
                    seq=d["seq"],
                    ts=d["ts"],
                    seat_id=d["seat_id"],
                    parent_id=d.get("parent_id"),
                    type=d["type"],
                    payload=d.get("payload", {}),
                )
