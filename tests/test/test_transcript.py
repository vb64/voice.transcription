"""Module transcript.py tests.

make test T=test_transcript.py
"""
import pytest

from . import TestBase


class TestTranscript(TestBase):
    """Module transcript."""

    def setUp(self):
        """Set defaults."""
        super().setUp()

        from voice_transcription.cli_options import PARSER
        self.options = PARSER.parse_args(args=[self.fixture('short.mp3')])

    def test_main(self):
        """Check main function."""
        from voice_transcription import transcript

        isolate_vocals = transcript.isolate_vocals
        transcript.isolate_vocals = lambda input_file, folder: "xxx.wav"

        transcribe = transcript.transcribe
        transcript.transcribe = lambda model_name, device, vocal_target, lang: (None, None, None)

        forced_alignment = transcript.forced_alignment
        transcript.forced_alignment = lambda device, segments, info, waveform: None

        assert transcript.main(self.options) == 0

        transcript.isolate_vocals = isolate_vocals
        transcript.transcribe = transcribe
        transcript.forced_alignment = forced_alignment

    def test_isolate_vocals(self):
        """Check isolate_vocals function."""
        from voice_transcription import transcript

        demucs_separate = transcript.demucs_separate
        transcript.demucs_separate = lambda args: "xxx.wav"

        assert 'vocals.wav' in transcript.isolate_vocals(self.fixture('short.mp3'), 'build')

        transcript.demucs_separate = demucs_separate

    @pytest.mark.longrunning
    def test_isolate_vocals_real(self):
        """Check real call for isolate_vocals function."""
        from voice_transcription import transcript

        assert 'vocals.wav' in transcript.isolate_vocals(self.fixture('short.mp3'), 'build')

    @pytest.mark.longrunning
    def test_transcribe(self):
        """Check transcribe function."""
        from voice_transcription import transcript

        segments, info, waveform = transcript.transcribe(
          transcript.MODEL,
          transcript.DEVICE,
          self.fixture('vocals.wav'),
          'ru'
        )

        # print(segments)
        # print(info)
        word_timestamps = transcript.forced_alignment(transcript.DEVICE, segments, info, waveform)
        assert len(word_timestamps) > 1
