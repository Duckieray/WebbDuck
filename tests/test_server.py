"""Tests for the FastAPI server endpoints."""

import pytest
import threading




class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_returns_ok(self, client):
        """Health endpoint should return status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert "cuda_available" in data
        assert "models" in data
        assert "embeddings" in data

    def test_health_includes_pipeline_info(self, client):
        """Health should include pipeline information."""
        response = client.get("/health")
        data = response.json()
        
        assert "pipeline" in data
        assert "loaded" in data["pipeline"]


class TestModelEndpoints:
    """Test model listing endpoints."""

    def test_list_models(self, client):
        """Should list available models."""
        response = client.get("/models")
        assert response.status_code == 200
        
        models = response.json()
        assert isinstance(models, list)

    def test_list_second_pass_models(self, client):
        """Should list second pass models."""
        response = client.get("/second_pass_models")
        assert response.status_code == 200
        
        models = response.json()
        assert isinstance(models, list)

    def test_list_schedulers(self, client):
        """Should list available schedulers."""
        response = client.get("/schedulers")
        assert response.status_code == 200
        
        schedulers = response.json()
        assert isinstance(schedulers, list)
        assert len(schedulers) > 0

    def test_list_model_embeddings(self, client, first_available_model):
        """Should list embeddings for a model."""
        response = client.get(f"/models/{first_available_model}/embeddings")
        assert response.status_code == 200
        embeddings = response.json()
        assert isinstance(embeddings, list)


class TestUIEndpoint:
    """Test UI serving endpoint."""

    def test_ui_returns_html(self, client):
        """Root should return HTML page."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestGalleryEndpoint:
    """Test gallery endpoint."""

    def test_gallery_returns_list(self, client):
        """Gallery should return list of runs."""
        response = client.get("/gallery")
        assert response.status_code == 200
        
        gallery = response.json()
        assert isinstance(gallery, list)


class TestQueueControls:
    """Tests for queue cancellation behavior."""

    def test_cancel_running_job_requests_cancellation(self, client):
        from server import app as appmod

        job_id = "test-running-job"
        cancel_event = threading.Event()
        appmod.job_registry[job_id] = {"job_id": job_id, "status": "running"}
        appmod.active_job_id = job_id
        appmod.active_job = {"job_id": job_id, "cancel_event": cancel_event}

        try:
            response = client.post("/queue/cancel", data={"job_id": job_id})
            assert response.status_code == 200
            payload = response.json()
            assert payload["status"] == "cancelling"
            assert payload["job_id"] == job_id
            assert cancel_event.is_set()
            assert appmod.job_registry[job_id]["status"] == "cancelling"
        finally:
            appmod.job_registry.pop(job_id, None)
            appmod.active_job_id = None
            appmod.active_job = None


class TestApiJobErrors:
    """Tests for third-party API error reporting."""

    def test_generate_returns_json_error_when_worker_fails(self, client, first_available_model, monkeypatch):
        from server import app as appmod

        async def fail_put(job):
            job["future"].set_exception(RuntimeError("model load exploded"))

        monkeypatch.setattr(appmod.generation_queue, "put", fail_put)

        response = client.post("/generate", data={
            "prompt": "a cat",
            "base_model": first_available_model,
            "steps": 5,
            "num_images": 1,
            "width": 512,
            "height": 512,
        })

        assert response.status_code == 500
        payload = response.json()
        assert payload["status"] == "failed"
        assert payload["error"] == "model load exploded"
        assert isinstance(payload["job_id"], str)

        meta = appmod.job_registry[payload["job_id"]]
        assert meta["status"] == "failed"
        assert meta["error"] == "model load exploded"

        appmod.job_registry.pop(payload["job_id"], None)

    def test_queue_job_endpoint_returns_failed_job_error(self, client, first_available_model, monkeypatch):
        from server import app as appmod

        async def hold_put(job):
            return None

        monkeypatch.setattr(appmod.generation_queue, "put", hold_put)

        response = client.post("/generate", data={
            "prompt": "a cat",
            "base_model": first_available_model,
            "steps": 5,
            "num_images": 1,
            "width": 512,
            "height": 512,
            "wait_for_result": "false",
        })

        assert response.status_code == 200
        payload = response.json()
        job_id = payload["job_id"]

        appmod.job_registry[job_id]["status"] = "failed"
        appmod.job_registry[job_id]["error"] = "scheduler mismatch"

        status_response = client.get(f"/queue/{job_id}")
        assert status_response.status_code == 200
        status_payload = status_response.json()
        assert status_payload["job_id"] == job_id
        assert status_payload["status"] == "failed"
        assert status_payload["error"] == "scheduler mismatch"

        appmod.job_registry.pop(job_id, None)


class TestResolutionNormalization:
    """Tests for width/height normalization helpers."""

    def test_normalize_dimensions_rounds_to_multiple_of_8(self):
        from server.app import normalize_dimensions

        width, height = normalize_dimensions(724, 1060)
        assert width % 8 == 0
        assert height % 8 == 0
        assert (width, height) == (720, 1056)


@pytest.mark.slow
class TestGenerationEndpoints:
    """Test generation endpoints (requires GPU)."""

    def test_generate_requires_base_model(self, client):
        """Generate should require base_model parameter."""
        response = client.post("/generate", data={
            "prompt": "test",
        })
        # Should fail without base_model
        assert response.status_code == 422

    def test_generate_basic(self, client, first_available_model):
        """Test basic generation endpoint."""
        response = client.post("/generate", data={
            "prompt": "a cat",
            "base_model": first_available_model,
            "steps": 5,  # Minimal steps for faster test
            "num_images": 1,
            "width": 512,
            "height": 512,
        })
        
        assert response.status_code == 200
        data = response.json()
        
        if "error" not in data:
            assert "images" in data
            assert "seed" in data

class TestGenerateIdempotency:
    """client_request_id idempotency on POST /generate."""

    def test_omitted_seed_returns_same_job(self, client):
        """Same client_request_id with omitted seed returns same job (deterministic fingerprint)."""
        data = {"prompt": "a cat", "num_images": 1, "client_request_id": "idem-seed-1", "wait_for_result": False}
        r1 = client.post("/generate", data=data)
        assert r1.status_code == 200, r1.text
        j1 = r1.json()

        r2 = client.post("/generate", data=data)
        assert r2.status_code == 200, r2.text
        j2 = r2.json()
        assert j2["job_id"] == j1["job_id"]

    def test_same_explicit_seed_returns_same_job(self, client):
        """Same client_request_id + same explicit seed returns same job."""
        data = {"prompt": "a cat", "num_images": 1, "seed": 42, "client_request_id": "idem-seed-2", "wait_for_result": False}
        r1 = client.post("/generate", data=data)
        assert r1.status_code == 200, r1.text
        j1 = r1.json()

        r2 = client.post("/generate", data=data)
        assert r2.status_code == 200, r2.text
        j2 = r2.json()
        assert j2["job_id"] == j1["job_id"]

    def test_different_seed_returns_409(self, client):
        """Same client_request_id + different explicit seed returns 409."""
        data = {"prompt": "a cat", "num_images": 1, "seed": 42, "client_request_id": "idem-seed-3", "wait_for_result": False}
        r1 = client.post("/generate", data=data)
        assert r1.status_code == 200, r1.text

        data["seed"] = 99
        r2 = client.post("/generate", data=data)
        assert r2.status_code == 409, r2.text
        err = r2.json()
        assert "existing_job_id" in err

    def test_different_identity_adapter_returns_409(self, client):
        """Same client_request_id + different identity_adapter returns 409."""
        data = {"prompt": "a cat", "num_images": 1, "seed": 42, "client_request_id": "idem-adapter-4", "wait_for_result": False}
        r1 = client.post("/generate", data=data)
        assert r1.status_code == 200, r1.text

        data["identity_adapter"] = '{"reference_images": ["ref1.jpg"]}'
        r2 = client.post("/generate", data=data)
        assert r2.status_code == 409, r2.text

    def test_no_client_id_creates_separate_jobs(self, client):
        """Two requests without client_request_id produce separate jobs."""
        data = {"prompt": "a cat", "num_images": 1, "wait_for_result": False}
        r1 = client.post("/generate", data=data)
        assert r1.status_code == 200
        r2 = client.post("/generate", data=data)
        assert r2.status_code == 200
        assert r1.json()["job_id"] != r2.json()["job_id"]

    def test_generate_no_client_request_id(self, client):
        """Test the /test endpoint for single image generation."""
        response = client.post("/test", data={
            "prompt": "a dog",
            "base_model": first_available_model,
            "steps": 5,
            "width": 512,
            "height": 512,
        })
        
        assert response.status_code == 200
