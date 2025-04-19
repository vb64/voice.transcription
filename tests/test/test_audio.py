"""Module audio tests.

make test T=test_audio.py
"""
import pytest
from . import TestBase


class TestAudio(TestBase):
    """Module audio."""

    def test_main(self):
        """Check main function."""
        from voice_transcription.audio import main

        assert main(self.fixture('short.mp3')) is None

    @pytest.mark.longrunning
    def test_convert_audio(self):
        """Check convert_audio function."""
        from voice_transcription.audio import convert_audio

        inp_file = self.fixture('short.mp3')
        out_file = self.build('short.wav')
        assert convert_audio(inp_file, out_file) is None
