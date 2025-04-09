"""Module audio_splitter tests.

make test T=test_audio_splitter.py
"""
from . import TestBase


class TestAudioSplitter(TestBase):
    """Module audio_splitter."""

    def test_main(self):
        """Check main function."""
        from voice_transcription.audio_splitter import main

        assert main(self.fixture('short.mp3')) is None
