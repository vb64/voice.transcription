"""Module speaker_mapping.py tests.

make test T=test_speaker_mapping.py
"""
from . import TestBase


class TestSpeakerMapping(TestBase):
    """Module speaker_mapping."""

    def test_get_word_ts_anchor(self):
        """Check get_word_ts_anchor function."""
        from voice_transcription.speaker_mapping import get_word_ts_anchor

        assert get_word_ts_anchor(2, 6, option='mid') == 4
