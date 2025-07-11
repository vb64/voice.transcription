"""Module whisper.py tests.

make test T=test_whisper.py
"""
from datetime import datetime
import faster_whisper
from . import TestBase


class TestWhisper(TestBase):
    """Module to_words.py."""

    def setUp(self):
        """Set up tests."""
        super().setUp()

        from voice_transcription import Model, Device, MTYPES

        self.model = faster_whisper.WhisperModel(
          Model.Large,
          device=Device.Cpu,
          compute_type=MTYPES[Device.Cpu]
        )

    def test_segments_to_json(self):
        """Check segments_to_json function."""
        from voice_transcription.whisper import segments_to_json, msec
        from voice_transcription import progress_bar

        segments, info = self.model.transcribe(
          faster_whisper.decode_audio(self.fixture('short.mp3')),
          'ru',
          suppress_tokens=[-1],
          vad_filter=True,
          word_timestamps=True
        )
        #  multilingual=False,
        #  max_new_tokens=None,
        #  hotwords=None
        assert int(info.duration * 1000) == 19592
        assert int(info.duration_after_vad * 1000) == 11736
        # rttm = NemoRttm.from_file(rttm_file, int(info.duration * 1000))
        # first = rttm.rows[0]
        # last = rttm.rows[-1]
        # print("# rttm", last.start + last.length - first.start)

        data = segments_to_json(segments, msec(info.duration_after_vad), progress_bar, datetime.utcnow())
        assert len(data) == 5

    def test_make_json(self):
        """Check make_json function."""
        from voice_transcription.whisper import make_json
        from voice_transcription import progress_bar

        data = make_json(self.model, self.fixture('short.mp3'), progress_bar)
        assert len(data) == 5
