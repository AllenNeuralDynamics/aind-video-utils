"""Example test module."""

import aind_video_utils


def test_version():
    """Test that version is defined."""
    assert aind_video_utils.__version__ is not None
    assert isinstance(aind_video_utils.__version__, str)
