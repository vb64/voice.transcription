"""Module audio tests.

make test T=test_audio.py
"""
import pytest
from . import TestBase


class TestAudio(TestBase):
    """Module audio."""

    @pytest.mark.longrunning
    def test_split_audio(self):
        """Check split_audio function."""
        from voice_transcription.audio import split_audio

        assert split_audio(self.fixture('short.mp3'), self.build('short'), 10 * 1000 * 60, 'mp3') is None

    @pytest.mark.longrunning
    def test_convert_audio(self):
        """Check convert_audio function."""
        from voice_transcription.audio import convert_audio

        inp_file = self.fixture('short.mp3')
        out_file = self.build('short.wav')
        assert convert_audio(inp_file, out_file) is None
