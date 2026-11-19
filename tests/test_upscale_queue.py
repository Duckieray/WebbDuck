#!/usr/bin/env python3
"""Tests for upscale queue priority and queue metadata."""

import asyncio

from server import app as appmod
from server.queue_control import put_job_front


def test_put_job_front_runs_before_existing_jobs():
    async def scenario():
        queue = asyncio.Queue()
        await queue.put({"job_id": "generation-1"})
        await queue.put({"job_id": "generation-2"})

        put_job_front(queue, {"job_id": "upscale"})

        assert queue.qsize() == 3
        assert queue._unfinished_tasks == 3
        assert (await queue.get())["job_id"] == "upscale"
        assert (await queue.get())["job_id"] == "generation-1"
        assert (await queue.get())["job_id"] == "generation-2"

    asyncio.run(scenario())


def test_summarize_upscale_settings_has_thumbnail_and_label(tmp_path, monkeypatch):
    monkeypatch.setattr(appmod, "BASE", tmp_path)
    source = tmp_path / "renders" / "portrait.png"
    source.parent.mkdir()
    source.write_bytes(b"not-a-real-image")

    settings = appmod.summarize_upscale_settings(source, 2)

    assert settings["is_upscale"] is True
    assert settings["mode"] == "upscale"
    assert settings["prompt"] == "Upscaling image"
    assert settings["scale"] == 2
    assert settings["input_image_url"] == "/outputs/renders/portrait.png"
    assert settings["steps"] is None
    assert settings["base_model"] is None
