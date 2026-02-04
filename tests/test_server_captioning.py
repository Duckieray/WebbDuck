"""Tests for captioning server endpoints."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestCaptioningEndpoints:
    """Test captioning API endpoints."""

    def test_get_captioners_endpoint_unavailable(self, client):
        """Test /captioners when unavailable."""
        with patch("webbduck.server.app.is_captioning_available", return_value=False):
            with patch("webbduck.server.app.list_captioners", return_value=[]):
                response = client.get("/captioners")
                assert response.status_code == 200
                data = response.json()
                assert data["available"] is False
                assert data["captioners"] == []

    def test_get_captioners_endpoint_available(self, client):
        """Test /captioners when available."""
        with patch("webbduck.server.app.is_captioning_available", return_value=True):
            with patch("webbduck.server.app.list_captioners", return_value=["test_captioner"]):
                response = client.get("/captioners")
                assert response.status_code == 200
                data = response.json()
                assert data["available"] is True
                assert data["captioners"] == ["test_captioner"]

    def test_get_caption_styles(self, client):
        """Test /caption_styles endpoint."""
        response = client.get("/caption_styles")
        assert response.status_code == 200
        styles = response.json()
        assert "detailed" in styles
        assert "short" in styles

    def test_caption_image_no_plugins(self, client, test_image_path):
        """Test /caption endpoint returns 503 when no plugins."""
        with patch("webbduck.server.app.is_captioning_available", return_value=False):
            files = {"image": open(test_image_path, "rb")}
            response = client.post("/caption", files=files)
            
            assert response.status_code == 503
            assert "error" in response.json()

    def test_caption_image_success(self, client, test_image_path):
        """Test successful caption generation."""
        # Mock dependencies in server/app.py
        with patch("webbduck.server.app.is_captioning_available", return_value=True), \
             patch("webbduck.server.app.generate_caption", return_value="A photo of a cat"), \
             patch("webbduck.server.app.gpu_worker"), \
             patch("webbduck.server.app.broadcast_state"), \
             patch("webbduck.core.pipeline.pipeline_manager.pipe", None):  # Mock pipeline manager
                
            files = {"image": open(test_image_path, "rb")}
            data = {"style": "short", "max_tokens": 100}
            
            response = client.post("/caption", files=files, data=data)
            
            assert response.status_code == 200
            result = response.json()
            assert result["caption"] == "A photo of a cat"
            assert result["style"] == "short"

    def test_caption_image_handles_pipeline_offload_safe(self, client, test_image_path):
        """Test captioning offload logic safely handles missing pipeline."""
        with patch("webbduck.server.app.is_captioning_available", return_value=True), \
             patch("webbduck.server.app.generate_caption", return_value="Caption"), \
             patch("webbduck.server.app.broadcast_state"):
                 
            # Don't mock pipeline_manager completely, let it error or be None safely
            # Ideally mock just what we need
             with patch("webbduck.core.pipeline.pipeline_manager") as mock_pm:
                mock_pm.pipe = None  # Ensure no pipe to offload
                
                files = {"image": open(test_image_path, "rb")}
                response = client.post("/caption", files=files)
                
                assert response.status_code == 200
                assert response.json()["caption"] == "Caption"
