"""Module nemo_msdd.py tests.

make test T=test_nemo_msdd.py
"""
import os
import pytest

from . import TestBase


class TestNemo(TestBase):
    """Module nemo_msdd."""

    def setUp(self):
        """Set defaults."""
        super().setUp()
        self.wav = self.fixture('mono.wav')
        self.config_path = os.path.join('nemo.config', 'diar_infer_telephonic.yaml')

    def test_create_config(self):
        """Check create_config function."""
        from voice_transcription import nemo_msdd

        assert nemo_msdd.create_config(self.wav, 'build', self.config_path, num_speakers=0)

    @pytest.mark.longrunning
    def test_diarize(self):
        """Check diarize function."""
        from voice_transcription.nemo_msdd import diarize
        from voice_transcription import Device

        call_log = []
        assert diarize(call_log, self.wav, Device.Cpu, 2, 'build', self.config_path) == 0
