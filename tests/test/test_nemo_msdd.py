"""Module nemo_msdd.py tests.

make test T=test_nemo_msdd.py
"""
# import pytest

from . import TestBase


class TestNemo(TestBase):
    """Module nemo_msdd."""

    def setUp(self):
        """Set defaults."""
        super().setUp()
        self.wav = self.fixture('mono.wav')

    def test_create_config(self):
        """Check create_config function."""
        from voice_transcription.nemo_msdd import create_config

        assert create_config(self.wav, 'build', num_speakers=0)
