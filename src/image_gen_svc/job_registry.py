"""In-memory job store for SSE event distribution.

Each job has a list of buffered events plus a list of asyncio.Queues for live
subscribers. On `publish`, the event is appended to the buffer AND fanned out
to every queue. On `subscribe`, the consumer first receives all buffered
events, then awaits the queue for new ones. The terminal events
(`job_completed`, `job_failed`) close the queue, ending iteration.

After a job receives a terminal event, it remains queryable for
`retain_seconds` so a late subscriber can still replay history. `gc()` clears
expired jobs; the service can call this on a timer or per-request.

This module has no awareness of HTTP — `routes/events.py` adapts these events
to SSE wire format."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field

_TERMINAL_TYPES = {"job_completed", "job_failed", "error"}


@dataclass(frozen=True)
class JobEvent:
    type: str
    data: dict


@dataclass
class _JobState:
    buffer: list[JobEvent] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    completed_at: float | None = None


class JobRegistry:
    def __init__(self, retain_seconds: float = 300.0):
        self._jobs: dict[str, _JobState] = {}
        self._retain = retain_seconds

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = _JobState()
        return job_id

    def register_with_id(self, job_id: str) -> None:
        """Register a caller-supplied job_id. No-op if already registered."""
        if job_id not in self._jobs:
            self._jobs[job_id] = _JobState()

    def publish(self, job_id: str, event: JobEvent) -> None:
        st = self._jobs.get(job_id)
        if st is None:
            return  # silently drop — caller used a stale id
        st.buffer.append(event)
        for q in list(st.subscribers):
            q.put_nowait(event)
        if event.type in _TERMINAL_TYPES:
            st.completed_at = time.monotonic()
            for q in list(st.subscribers):
                q.put_nowait(None)  # sentinel: stream ends

    async def subscribe(self, job_id: str):
        st = self._jobs.get(job_id)
        if st is None:
            yield JobEvent(type="error", data={"message": f"unknown job: {job_id}"})
            return

        # Replay buffered history.
        terminal_seen = False
        for ev in list(st.buffer):
            yield ev
            if ev.type in _TERMINAL_TYPES:
                terminal_seen = True
        if terminal_seen:
            return

        # Stream live events.
        q: asyncio.Queue = asyncio.Queue()
        st.subscribers.append(q)
        try:
            while True:
                ev = await q.get()
                if ev is None:
                    return
                yield ev
                if ev.type in _TERMINAL_TYPES:
                    return
        finally:
            try:
                st.subscribers.remove(q)
            except ValueError:
                pass

    def gc(self) -> None:
        now = time.monotonic()
        expired = [
            jid
            for jid, st in self._jobs.items()
            if st.completed_at is not None and now - st.completed_at >= self._retain
        ]
        for jid in expired:
            del self._jobs[jid]
