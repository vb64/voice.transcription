"""Module transcript.py tests.

make test T=test_transcript.py
"""
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
        transcript.isolate_vocals = lambda input_file: "xxx.wav"

        assert transcript.main(self.options) == 0

        transcript.isolate_vocals = isolate_vocals
