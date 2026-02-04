"""Tests for the captioning module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import os


class TestCaptioningConfig:
    """Tests for captioning_config.py."""

    @patch("webbduck.core.captioning_config.LOCAL_PLUGINS_DIR", Path("/nonexistent/local"))
    def test_get_plugins_dirs_default(self):
        """Test default plugins directory path."""
        from webbduck.core.captioning_config import get_plugins_dirs
        
        # Without env var, should use home directory (and skip local if mocked to non-exist)
        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing WEBBDUCK_PLUGINS_DIR
            if 'WEBBDUCK_PLUGINS_DIR' in os.environ:
                del os.environ['WEBBDUCK_PLUGINS_DIR']
            
            result = get_plugins_dirs()
            expected = [Path.home() / ".webbduck" / "plugins"]
            assert result == expected

    @patch("webbduck.core.captioning_config.LOCAL_PLUGINS_DIR", Path("/nonexistent/local"))
    def test_get_plugins_dirs_env(self):
        """Test plugins directory from environment variable."""
        from webbduck.core.captioning_config import get_plugins_dirs
        
        custom_path = "/custom/plugins/path"
        with patch.dict(os.environ, {"WEBBDUCK_PLUGINS_DIR": custom_path}):
            result = get_plugins_dirs()
            # Contains env path AND default home path
            assert Path(custom_path) in result
            assert (Path.home() / ".webbduck" / "plugins") in result

    @patch("webbduck.core.captioning_config.LOCAL_PLUGINS_DIR", Path("/nonexistent/local"))
    def test_get_captioners_dirs(self):
        """Test captioners subdirectory path."""
        from webbduck.core.captioning_config import get_captioners_dirs, get_plugins_dirs
        
        captioners_dirs = get_captioners_dirs()
        plugins_dirs = get_plugins_dirs()
        
        assert len(captioners_dirs) == len(plugins_dirs)
        for i, p_dir in enumerate(plugins_dirs):
            assert captioners_dirs[i] == p_dir / "captioners"

    @patch("webbduck.core.captioning_config.LOCAL_PLUGINS_DIR", Path("/nonexistent/local"))
    def test_list_available_captioners_empty(self):
        """Test listing captioners when directory doesn't exist."""
        from webbduck.core.captioning_config import list_available_captioners
        
        with patch.dict(os.environ, {"WEBBDUCK_PLUGINS_DIR": "/nonexistent/path"}):
            result = list_available_captioners()
            assert result == []

    def test_list_available_captioners_with_plugins(self):
        """Test listing captioners when valid plugins exist."""
        from webbduck.core.captioning_config import list_available_captioners
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create captioners directory structure
            captioners_dir = Path(tmpdir) / "captioners"
            captioners_dir.mkdir(parents=True)
            
            # Create a valid captioner
            valid_captioner = captioners_dir / "test_captioner"
            valid_captioner.mkdir()
            (valid_captioner / "captioner.py").write_text("# mock captioner")
            
            # Create an invalid directory (no captioner.py)
            invalid_dir = captioners_dir / "invalid"
            invalid_dir.mkdir()
            
            with patch.dict(os.environ, {"WEBBDUCK_PLUGINS_DIR": tmpdir}):
                result = list_available_captioners()
                assert "test_captioner" in result
                assert "invalid" not in result


class TestCaptionerModule:
    """Tests for captioner.py."""

    @patch("webbduck.core.captioning_config.LOCAL_PLUGINS_DIR", Path("/nonexistent/local"))
    def test_is_captioning_available_no_plugins(self):
        """Test availability check with no plugins."""
        from webbduck.core.captioner import is_captioning_available
        
        with patch.dict(os.environ, {"WEBBDUCK_PLUGINS_DIR": "/nonexistent/path"}):
            result = is_captioning_available()
            assert result is False

    @patch("webbduck.core.captioning_config.LOCAL_PLUGINS_DIR", Path("/nonexistent/local"))
    def test_list_captioners_empty(self):
        """Test list_captioners returns empty list when no plugins."""
        from webbduck.core.captioner import list_captioners
        
        with patch.dict(os.environ, {"WEBBDUCK_PLUGINS_DIR": "/nonexistent/path"}):
            result = list_captioners()
            assert result == []

    def test_get_caption_styles(self):
        """Test that caption styles are returned correctly."""
        from webbduck.core.captioner import get_caption_styles
        
        styles = get_caption_styles()
        
        assert "detailed" in styles
        assert "short" in styles
        assert "sd_prompt" in styles
        assert isinstance(styles["detailed"], str)

    def test_caption_prompts_content(self):
        """Test that caption prompts have meaningful content."""
        from webbduck.core.captioner import CAPTION_PROMPTS
        
        for style, prompt in CAPTION_PROMPTS.items():
            assert len(prompt) > 10, f"Prompt for {style} is too short"
            assert "image" in prompt.lower() or "description" in prompt.lower()


class TestCaptionerManager:
    """Tests for CaptionerManager class."""

    @patch("webbduck.core.captioning_config.LOCAL_PLUGINS_DIR", Path("/nonexistent/local"))
    def test_get_available_empty(self):
        """Test get_available returns empty list with no plugins."""
        from webbduck.core.captioner import CaptionerManager
        
        manager = CaptionerManager()
        
        with patch.dict(os.environ, {"WEBBDUCK_PLUGINS_DIR": "/nonexistent/path"}):
            result = manager.get_available()
            assert result == []

    @patch("webbduck.core.captioning_config.LOCAL_PLUGINS_DIR", Path("/nonexistent/local"))
    def test_is_available_false(self):
        """Test is_available returns False when no plugins."""
        from webbduck.core.captioner import CaptionerManager
        
        manager = CaptionerManager()
        
        with patch.dict(os.environ, {"WEBBDUCK_PLUGINS_DIR": "/nonexistent/path"}):
            assert manager.is_available() is False

    @patch("webbduck.core.captioning_config.LOCAL_PLUGINS_DIR", Path("/nonexistent/local"))
    def test_generate_caption_no_captioner(self):
        """Test generate_caption raises error when no captioner available."""
        from webbduck.core.captioner import CaptionerManager
        
        manager = CaptionerManager()
        
        with patch.dict(os.environ, {"WEBBDUCK_PLUGINS_DIR": "/nonexistent/path"}):
            with pytest.raises(ValueError, match="No captioner plugins available"):
                manager.generate_caption(Path("test.jpg"))
