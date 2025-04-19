"""Module init.py tests.

make test T=test_init.py
"""
import os
from pathlib import Path
import torch

from . import TestBase


class TestInit(TestBase):
    """Module __init__.py."""

    def test_torch_types(self):
        """Check torch types."""
        # print(repr(torch.float16))
        assert torch.float16

    def test_add_log(self):
        """Check add_log function."""
        from voice_transcription import add_log

        assert add_log(None, 'xxx', None)

    def test_cleanup(self):
        """Check cleanup function."""
        from voice_transcription import cleanup

        os.makedirs(self.build('tmp'), exist_ok=True)
        Path(self.build('tmp', 'file1.txt')).touch(exist_ok=True)
        Path(self.build('tmp', 'file2.txt')).touch(exist_ok=True)
        assert cleanup(self.build('tmp', 'file1.txt')) is None
        assert cleanup(self.build('tmp')) is None
        assert cleanup(self.build('not_exist')) is None

    def test_list_mp3(self):
        """Check list_mp3 function."""
        from voice_transcription import list_mp3

        assert list_mp3(self.fixture())
