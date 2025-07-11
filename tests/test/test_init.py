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

    def test_get_tasks(self):
        """Check get_tasks function."""
        from voice_transcription import get_tasks

        assert len(get_tasks(self.fixture('short.mp3'), self.build(), 5)) == 4
        assert len(get_tasks(self.fixture('short.mp3'), self.build(), 1000)) == 1
