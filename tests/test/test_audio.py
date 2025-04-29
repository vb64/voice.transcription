"""Module audio tests.

make test T=test_audio.py
"""
import pytest

from pydub import AudioSegment
from pydub.silence import split_on_silence

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

    def test_merge_short_chunks(self):
        """Check merge_short_chunks function."""
        from voice_transcription.audio import merge_short_chunks

        audio = AudioSegment.from_file(self.fixture('short.mp3'))
        chunks = split_on_silence(
          audio,
          min_silence_len=100,  # Minimum length of silence in milliseconds
          silence_thresh=-40  # Silence threshold in dB
        )
        assert len(chunks) == 14

        merged = merge_short_chunks(chunks, 2 * 1000)  # by 2 sec min
        assert len(merged) == 8

        merged = merge_short_chunks(chunks, 3 * 1000)  # by 3 sec min
        assert len(merged) == 5
