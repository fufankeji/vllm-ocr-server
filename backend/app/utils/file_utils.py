#!/usr/bin/env python3
"""
File Utilities
Helper functions for file operations
"""

import os
import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)

def ensure_directories():
    """Ensure required directories exist"""

    directories = [
        os.getenv("UPLOAD_DIR", "./uploads"),
        os.getenv("EXPORT_DIR", "./exports"),
        os.getenv("TEMP_DIR", "./temp"),
        "./static"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def cleanup_file(file_path: str):
    """Safely remove a file"""

    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to cleanup file {file_path}: {str(e)}")

def cleanup_directory(directory_path: str, max_age_hours: int = 24):
    """Clean up old files in a directory"""

    try:
        directory = Path(directory_path)
        if not directory.exists():
            return

        current_time = Path.stat(directory).st_mtime

        for file_path in directory.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_hours * 3600:  # Convert hours to seconds
                    file_path.unlink()
                    logger.info(f"Cleaned up old file: {file_path}")

    except Exception as e:
        logger.error(f"Failed to cleanup directory {directory_path}: {str(e)}")

def get_file_info(file_path: str) -> dict:
    """Get file information"""

    try:
        path = Path(file_path)
        if not path.exists():
            return None

        stat = path.stat()

        return {
            "name": path.name,
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "extension": path.suffix,
            "is_file": path.is_file(),
            "is_directory": path.is_dir()
        }

    except Exception as e:
        logger.error(f"Failed to get file info for {file_path}: {str(e)}")
        return None

def safe_move_file(source: str, destination: str) -> bool:
    """Safely move a file from source to destination"""

    try:
        # Ensure destination directory exists
        Path(destination).parent.mkdir(parents=True, exist_ok=True)

        # If destination exists, generate unique name
        if Path(destination).exists():
            base, ext = os.path.splitext(destination)
            counter = 1
            while Path(f"{base}_{counter}{ext}").exists():
                counter += 1
            destination = f"{base}_{counter}{ext}"

        shutil.move(source, destination)
        logger.info(f"Moved file from {source} to {destination}")
        return True

    except Exception as e:
        logger.error(f"Failed to move file from {source} to {destination}: {str(e)}")
        return False