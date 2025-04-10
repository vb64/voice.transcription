"""Module nemo_msdd.py tests.

make test T=test_nemo_msdd.py
"""
import pytest

from . import TestBase


class TestNemo(TestBase):
    """Module nemo_msdd."""

    def setUp(self):
        """Set defaults."""
        super().setUp()

        from voice_transcription import nemo_msdd

        self.wav = self.fixture('mono.wav')
        self.model_config_path = nemo_msdd.MODEL_CONFIG_PATH
        nemo_msdd.MODEL_CONFIG_PATH = self.fixture('nemo_msdd_configs', 'diar_infer_telephonic.yaml')

    def tearDown(self):
        """Restore defaults."""
        from voice_transcription import nemo_msdd

        nemo_msdd.MODEL_CONFIG_PATH = self.model_config_path
        super().tearDown()

    def test_create_config(self):
        """Check create_config function."""
        from voice_transcription import nemo_msdd

        assert nemo_msdd.create_config(self.wav, 'build', num_speakers=0)

    @pytest.mark.longrunning
    def test_diarize(self):
        """Check diarize function."""
        from voice_transcription.nemo_msdd import diarize
        from voice_transcription import Device

        call_log = []
        assert diarize(call_log, self.wav, Device.Cpu, 2, 'build') == 0
