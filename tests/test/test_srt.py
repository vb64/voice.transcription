"""Module srt.py tests.

make test T=test_srt.py
"""
import pytest
from . import TestBase


class TestSrt(TestBase):
    """Module srt."""

    def test_format_timestamp(self):
        """Check format_timestamp function."""
        from voice_transcription.srt import format_timestamp

        with pytest.raises(ValueError) as err:
            format_timestamp(-1)
        assert 'Non-negative timestamp expected' in str(err.value)
