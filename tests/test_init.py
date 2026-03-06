"""Test top-level package imports."""

import subprocess
import sys

import aind_video_utils


def test_version():
    """Test that version is defined."""
    assert aind_video_utils.__version__ is not None
    assert isinstance(aind_video_utils.__version__, str)


_OPTIONAL_DEPS = ["matplotlib", "cv2", "pydantic_settings", "rich"]

_IMPORT_WITHOUT_OPTIONAL = f"""\
import sys

class _BlockOptional:
    BLOCKED = {_OPTIONAL_DEPS!r}
    def find_module(self, name, path=None):
        if any(name == b or name.startswith(b + ".") for b in self.BLOCKED):
            return self
    def load_module(self, name):
        raise ImportError(f"blocked by test: {{name}}")

sys.meta_path.insert(0, _BlockOptional())

import aind_video_utils
print("OK")
"""


def test_import_without_optional_deps():
    """Importing aind_video_utils must not require optional dependencies."""
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_WITHOUT_OPTIONAL],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Import failed without optional deps:\n{result.stderr}"
