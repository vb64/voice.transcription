"""Module audio_splitter tests.

make test T=test_audio_splitter.py
"""
import os
from . import TestBase


class TestAudioSplitter(TestBase):
    """Module audio_splitter."""

    def test_main(self):
        """Check main function."""
        from voice_transcription.audio_splitter import main

        assert main(self.fixture('short.mp3')) is None

    def test_convert_audio(self):
        """Check convert_audio function."""
        from voice_transcription.audio_splitter import convert_audio

        inp_file = os.path.join('D:\\', 'tmp', 'April_6_Session_1.aac')
        out_file = self.build('April_6_Session_1.mp3')
        assert convert_audio(inp_file, out_file) is None
