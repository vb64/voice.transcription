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
        transcript.isolate_vocals = lambda call_log, input_file, folder: "xxx.wav"

        transcribe = transcript.transcribe
        transcript.transcribe = lambda call_log, model_name, device, vocal_target, lang: (None, None, None)

        forced_alignment = transcript.forced_alignment
        transcript.forced_alignment = lambda call_log, device, segments, info, waveform: None

        to_mono = transcript.to_mono
        transcript.to_mono = lambda call_log, waveform, file_name: None

        assert transcript.main(self.options) == 0

        transcript.isolate_vocals = isolate_vocals
        transcript.transcribe = transcribe
        transcript.forced_alignment = forced_alignment
        transcript.to_mono = to_mono

    def test_isolate_vocals(self):
        """Check isolate_vocals function."""
        from voice_transcription import transcript

        demucs_separate = transcript.demucs_separate
        transcript.demucs_separate = lambda args: "xxx.wav"

        call_log = []
        assert 'vocals.wav' in transcript.isolate_vocals(call_log, self.fixture('short.mp3'), 'build')

        transcript.demucs_separate = demucs_separate

    @pytest.mark.longrunning
    def test_isolate_vocals_real(self):
        """Check real call for isolate_vocals function."""
        from voice_transcription import transcript

        call_log = []
        assert 'vocals.wav' in transcript.isolate_vocals(call_log, self.fixture('short.mp3'), 'build')

    @pytest.mark.longrunning
    def test_transcribe(self):
        """Check transcribe function."""
        from voice_transcription import transcript

        call_log = []

        segments, info, waveform = transcript.transcribe(
          call_log,
          transcript.MODEL,
          transcript.DEVICE,
          self.fixture('vocals.wav'),
          'ru'
        )

        # print(segments)
        # print(info)
        word_timestamps = transcript.forced_alignment(call_log, transcript.DEVICE, segments, info, waveform)
        assert len(word_timestamps) > 1

        assert transcript.to_mono(call_log, waveform, self.build('mono.wav')) is None
