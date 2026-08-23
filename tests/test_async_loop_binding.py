"""Event-loop binding regressions for the two async queues.

On Python 3.9, asyncio primitives bind the event loop they were CONSTRUCTED
on. Both JobRunner/JobStore (built in create_app(), before uvicorn starts its
loop) and TaskQueue (built before asyncio.run()) used to create their
Queue/Lock eagerly: on 3.9 the workers then died with "Future attached to a
different loop" — server jobs sat in 'pending' forever behind a 200 /health,
and `batch` crashed at queue.join() after doing the work. These tests build
the objects OUTSIDE any loop, exactly like the real entry points do.
"""

import asyncio
from pathlib import Path

import pytest

from subtitle_forge.core.queue import run_batch_sync
from subtitle_forge.models.task import VideoTask
from subtitle_forge.server.jobs import Job, JobRunner, JobStore


def test_job_runner_survives_an_empty_queue_wait():
    store = JobStore()
    processed = []

    def processor(job):
        processed.append(job.job_id)
        return [{"language": "zh", "path": "out.srt"}]

    runner = JobRunner(store, processor, max_workers=1)

    async def scenario():
        await runner.start()
        job = Job(job_id="j1", video_path="v.mp4", target_languages=["zh"])
        await runner.submit(job)
        for _ in range(200):
            if job.status in ("completed", "failed"):
                break
            await asyncio.sleep(0.01)
        # Give the worker time to loop back into queue.get() on an EMPTY
        # queue — that wait is where the 3.9 loop-binding bug killed it.
        await asyncio.sleep(0.1)
        alive = runner.alive_workers
        await runner.stop()
        return job, alive

    job, alive_after_drain = asyncio.run(scenario())
    assert job.status == "completed"
    assert processed == ["j1"]
    assert alive_after_drain == 1


def test_job_runner_submit_before_start_is_an_error():
    runner = JobRunner(JobStore(), lambda job: [], max_workers=1)
    job = Job(job_id="x", video_path="v.mp4", target_languages=["zh"])
    with pytest.raises(RuntimeError):
        asyncio.run(runner.submit(job))


def test_run_batch_sync_completes_and_returns_statuses():
    tasks = [
        VideoTask(video_path=Path("a.mp4"), target_langs=["zh"], output_dir=Path(".")),
        VideoTask(video_path=Path("b.mp4"), target_langs=["zh"], output_dir=Path(".")),
    ]
    processed = []
    results = run_batch_sync(
        tasks, lambda t: processed.append(t.video_path.name), max_workers=2
    )
    assert [t.status.value for t in results] == ["completed", "completed"]
    assert sorted(processed) == ["a.mp4", "b.mp4"]


def test_run_batch_sync_marks_failures():
    tasks = [
        VideoTask(video_path=Path("a.mp4"), target_langs=["zh"], output_dir=Path(".")),
    ]

    def boom(task):
        raise RuntimeError("ffmpeg exploded")

    results = run_batch_sync(tasks, boom, max_workers=1)
    assert results[0].status.value == "failed"
    assert "ffmpeg exploded" in results[0].error
