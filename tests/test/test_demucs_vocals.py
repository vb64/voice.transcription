"""Module demucs_vocals.py tests.

make test T=test_demucs_vocals.py
"""
import pytest
from . import TestBase


class TestDemucsVocals(TestBase):
    """Module demucs_vocals."""

    def test_isolate_vocals(self):
        """Check isolate_vocals function."""
        from voice_transcription import demucs_vocals, Device

        demucs_separate = demucs_vocals.demucs_separate
        demucs_vocals.demucs_separate = lambda args: "xxx.wav"

        assert 'vocals.wav' in demucs_vocals.isolate_vocals(
          self.fixture('short.mp3'),
          'build',
          Device.Cpu
        )

        demucs_vocals.demucs_separate = demucs_separate

    @pytest.mark.longrunning
    def test_isolate_vocals_real(self):
        """Check real call for isolate_vocals function."""
        from voice_transcription import demucs_vocals, Device

        assert 'vocals.wav' in demucs_vocals.isolate_vocals(self.fixture('short.mp3'), 'build', Device.Cpu)
