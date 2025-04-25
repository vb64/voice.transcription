"""Module init.py tests.

make test T=test_init.py
"""
import torch
from . import TestBase


class TestInit(TestBase):
    """Module __init__.py."""

    def test_torch_types(self):
        """Check torch types."""
        # print(repr(torch.float16))
        assert torch.float16

    def test_list_mp3(self):
        """Check list_mp3 function."""
        from voice_transcription import list_mp3

        assert list_mp3(self.fixture())
