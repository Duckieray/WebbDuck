"""Tests for the FastAPI server endpoints."""

import pytest
from pathlib import Path


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    from fastapi.testclient import TestClient
    from webbduck.server.app import app
    return TestClient(app)


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

    def test_test_endpoint(self, client, first_available_model):
        """Test the /test endpoint for single image generation."""
        response = client.post("/test", data={
            "prompt": "a dog",
            "base_model": first_available_model,
            "steps": 5,
            "width": 512,
            "height": 512,
        })
        
        assert response.status_code == 200
