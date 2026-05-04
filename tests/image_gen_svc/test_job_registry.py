from __future__ import annotations

import asyncio

import pytest

from image_gen_svc.job_registry import JobEvent, JobRegistry


@pytest.mark.asyncio
async def test_create_returns_unique_ids():
    reg = JobRegistry()

    j1 = reg.create_job()
    j2 = reg.create_job()

    assert j1 != j2
    assert isinstance(j1, str)


@pytest.mark.asyncio
async def test_publish_then_subscribe_replays_history():
    reg = JobRegistry()
    job = reg.create_job()

    reg.publish(job, JobEvent(type="job_started", data={"model": "chroma-1-hd"}))
    reg.publish(job, JobEvent(type="step_progress", data={"step": 5, "total_steps": 20}))
    reg.publish(job, JobEvent(type="job_completed", data={"model_used": "chroma-1-hd"}))

    received: list[JobEvent] = []
    async for ev in reg.subscribe(job):
        received.append(ev)

    assert [e.type for e in received] == [
        "job_started",
        "step_progress",
        "job_completed",
    ]


@pytest.mark.asyncio
async def test_subscriber_receives_live_events_and_terminates_on_complete():
    reg = JobRegistry()
    job = reg.create_job()

    received: list[JobEvent] = []

    async def consume():
        async for ev in reg.subscribe(job):
            received.append(ev)

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)

    reg.publish(job, JobEvent(type="job_started", data={}))
    await asyncio.sleep(0)
    reg.publish(job, JobEvent(type="step_progress", data={"step": 1, "total_steps": 2}))
    await asyncio.sleep(0)
    reg.publish(job, JobEvent(type="job_completed", data={"model_used": "x"}))

    await asyncio.wait_for(consumer, timeout=1.0)

    assert [e.type for e in received] == [
        "job_started",
        "step_progress",
        "job_completed",
    ]


@pytest.mark.asyncio
async def test_failed_event_terminates_subscriber():
    reg = JobRegistry()
    job = reg.create_job()
    reg.publish(job, JobEvent(type="job_failed", data={"error": "generation_failed"}))

    received: list[JobEvent] = []
    async for ev in reg.subscribe(job):
        received.append(ev)

    assert [e.type for e in received] == ["job_failed"]


@pytest.mark.asyncio
async def test_unknown_job_yields_error_and_closes():
    reg = JobRegistry()

    received: list[JobEvent] = []
    async for ev in reg.subscribe("never-existed"):
        received.append(ev)

    assert len(received) == 1
    assert received[0].type == "error"
    assert "unknown job" in received[0].data["message"].lower()


@pytest.mark.asyncio
async def test_multiple_concurrent_subscribers_each_get_full_history():
    reg = JobRegistry()
    job = reg.create_job()

    async def consume():
        return [ev async for ev in reg.subscribe(job)]

    t1 = asyncio.create_task(consume())
    t2 = asyncio.create_task(consume())
    await asyncio.sleep(0)

    reg.publish(job, JobEvent(type="job_started", data={}))
    reg.publish(job, JobEvent(type="job_completed", data={}))

    r1 = await asyncio.wait_for(t1, timeout=1.0)
    r2 = await asyncio.wait_for(t2, timeout=1.0)

    assert [e.type for e in r1] == ["job_started", "job_completed"]
    assert [e.type for e in r2] == ["job_started", "job_completed"]


@pytest.mark.asyncio
async def test_jobs_garbage_collected_after_retain_window():
    reg = JobRegistry(retain_seconds=0.05)
    job = reg.create_job()
    reg.publish(job, JobEvent(type="job_completed", data={}))

    await asyncio.sleep(0.1)
    reg.gc()

    received: list[JobEvent] = []
    async for ev in reg.subscribe(job):
        received.append(ev)

    assert received[0].type == "error"
