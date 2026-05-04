from __future__ import annotations

import json

import pytest

from image_gen_svc.job_registry import JobEvent


def _parse_sse_chunk(raw: str) -> list[tuple[str, dict]]:
    """Split SSE chunks into (event, data-as-dict) pairs."""
    events: list[tuple[str, dict]] = []
    for block in raw.strip().split("\n\n"):
        ev_type = ""
        ev_data: dict = {}
        for line in block.splitlines():
            if line.startswith("event: "):
                ev_type = line[len("event: ") :]
            elif line.startswith("data: "):
                ev_data = json.loads(line[len("data: ") :])
        if ev_type:
            events.append((ev_type, ev_data))
    return events


@pytest.mark.asyncio
async def test_events_replay_history_for_completed_job(client, job_reg):
    job_id = job_reg.create_job()
    job_reg.publish(job_id, JobEvent("job_started", {"model": "chroma-1-hd"}))
    job_reg.publish(job_id, JobEvent("step_progress", {"step": 5, "total_steps": 10}))
    job_reg.publish(job_id, JobEvent("job_completed", {"model_used": "chroma-1-hd", "seed": 1}))

    async with client.stream("GET", f"/events/{job_id}") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = ""
        async for chunk in r.aiter_text():
            body += chunk

    parsed = _parse_sse_chunk(body)
    assert [t for t, _ in parsed] == ["job_started", "step_progress", "job_completed"]
    assert parsed[1][1]["step"] == 5
    assert parsed[1][1]["total_steps"] == 10


@pytest.mark.asyncio
async def test_events_unknown_job_emits_error_and_closes(client):
    async with client.stream("GET", "/events/nonexistent-job") as r:
        body = ""
        async for chunk in r.aiter_text():
            body += chunk

    parsed = _parse_sse_chunk(body)
    assert len(parsed) == 1
    assert parsed[0][0] == "error"


@pytest.mark.asyncio
async def test_events_stream_during_real_render(client):
    """Subscribe to a job_id, then trigger a render with that id, and verify
    the SSE stream surfaces queued → started → progress → completed."""
    body = {
        "prompt": "x",
        "seed": 1,
        "width": 32,
        "height": 32,
        "steps": 1,
        "guidance": 1.0,
        "job_id": "stream-test-job",
    }

    # Trigger render first so events are buffered, then read the SSE stream.
    r = await client.post("/render", json=body)
    assert r.status_code == 200

    async with client.stream("GET", "/events/stream-test-job") as resp:
        text = ""
        async for chunk in resp.aiter_text():
            text += chunk

    parsed = _parse_sse_chunk(text)
    types = [t for t, _ in parsed]
    assert types[0] == "job_queued"
    assert "job_started" in types
    assert types[-1] == "job_completed"
    assert "step_progress" in types
