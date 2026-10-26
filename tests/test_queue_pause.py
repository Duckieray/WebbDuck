#!/usr/bin/env python3
"""Tests for queue pause/resume worker behavior."""

import asyncio

from core import worker as workermod
from server import queue_control


def test_pause_gate_holds_job_until_resume(monkeypatch):
    """A paused worker must not start the next job, and must requeue it."""

    def fake_run_selected_model(settings, cancel_event, progress_cb):
        raise workermod.GenerationCancelledError("test stub")

    monkeypatch.setattr(workermod, "run_selected_model", fake_run_selected_model)

    async def scenario():
        queue = asyncio.Queue()
        started = []
        future = asyncio.get_running_loop().create_future()
        job = {
            "type": "generation",
            "job_id": "paused-job",
            "settings": {},
            "future": future,
            "on_start": lambda j: started.append(j["job_id"]),
        }
        await queue.put(job)

        queue_control.set_queue_paused(True)
        try:
            task = asyncio.create_task(workermod.gpu_worker(queue))
            await asyncio.sleep(0.2)

            assert started == []
            assert queue.qsize() == 1

            queue_control.set_queue_paused(False)
            await asyncio.sleep(0.3)
            assert started == ["paused-job"]

            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        finally:
            queue_control.set_queue_paused(False)

    asyncio.run(scenario())


def test_release_gpu_for_pause_is_best_effort():
    # Should never raise even when no models are resident.
    queue_control.release_gpu_for_pause()
