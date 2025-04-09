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
        from voice_transcription.transcript import main

        assert main(self.options) == 0
