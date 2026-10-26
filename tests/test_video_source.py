"""Tests for VideoSource: file missing, invalid file, and config behavior.

Webcam-open tests cannot assert hardware presence in CI; they only verify the
error path raises VideoReadError instead of crashing.
"""
import pytest

from src.video_source import VideoReadError, VideoSource


def test_missing_file_raises():
    vs = VideoSource("definitely_not_here_12345.mp4")
    with pytest.raises(VideoReadError):
        vs.open()


def test_webcam_unavailable_raises_cleanly():
    # Camera index 99 is almost never present; must raise, not crash.
    vs = VideoSource(99)
    with pytest.raises(VideoReadError):
        vs.open()


def test_release_is_idempotent():
    vs = VideoSource(99)
    vs.release()
    vs.release()  # no error
