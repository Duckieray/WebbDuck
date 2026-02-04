"""Configuration for the captioning plugin system."""

import os
from pathlib import Path
from typing import Optional

# Project-local plugins directory
LOCAL_PLUGINS_DIR = Path(__file__).parent.parent / "plugins"


def get_plugins_dirs() -> list[Path]:
    """Get all plugins directories to search, in priority order.
    
    Returns:
        List of paths to check for plugins:
        1. Environment variable WEBBDUCK_PLUGINS_DIR (if set)
        2. Local webbduck/plugins/ folder  
        3. ~/.webbduck/plugins/
    """
    dirs = []
    
    # Environment variable takes highest priority
    env_path = os.environ.get("WEBBDUCK_PLUGINS_DIR")
    if env_path:
        dirs.append(Path(env_path))
    
    # Project-local plugins folder
    if LOCAL_PLUGINS_DIR.exists():
        dirs.append(LOCAL_PLUGINS_DIR)
    
    # User home directory
    dirs.append(Path.home() / ".webbduck" / "plugins")
    
    return dirs


def get_captioners_dirs() -> list[Path]:
    """Get all captioner directories to search."""
    return [d / "captioners" for d in get_plugins_dirs()]


def list_available_captioners() -> list[str]:
    """List all available captioner plugins.
    
    A valid captioner must have:
    - A folder in one of the captioners directories
    - A captioner.py file with a `generate_caption` function
    """
    available = []
    
    for captioners_dir in get_captioners_dirs():
        if not captioners_dir.exists():
            continue
        
        for item in captioners_dir.iterdir():
            if item.is_dir():
                captioner_file = item / "captioner.py"
                if captioner_file.exists() and item.name not in available:
                    available.append(item.name)
    
    return available


def get_captioner_module_path(name: str) -> Optional[Path]:
    """Get the path to a captioner's module file."""
    for captioners_dir in get_captioners_dirs():
        captioner_dir = captioners_dir / name
        captioner_file = captioner_dir / "captioner.py"
        
        if captioner_file.exists():
            return captioner_file
    
    return None

