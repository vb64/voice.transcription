"""Module transcript.py tests.

make test T=test_transcript.py
"""
import pytest

import faster_whisper
from ctc_forced_aligner import load_alignment_model

from . import TestBase


class TestTranscript(TestBase):
    """Module transcript."""

    def test_isolate_vocals(self):
        """Check isolate_vocals function."""
        from voice_transcription import transcript, Device

        demucs_separate = transcript.demucs_separate
        transcript.demucs_separate = lambda args: "xxx.wav"

        assert 'vocals.wav' in transcript.isolate_vocals(
          self.fixture('short.mp3'),
          'build',
          Device.Cpu
        )

        transcript.demucs_separate = demucs_separate

    @pytest.mark.longrunning
    def test_isolate_vocals_real(self):
        """Check real call for isolate_vocals function."""
        from voice_transcription import transcript, Device

        assert 'vocals.wav' in transcript.isolate_vocals(self.fixture('short.mp3'), 'build', Device.Cpu)

    @pytest.mark.longrunning
    def test_transcribe(self):
        """Check transcribe function."""
        from voice_transcription import transcript, Model, Device, TTYPES, MTYPES

        alignment_model, alignment_tokenizer = load_alignment_model(
          Device.Cpu,
          dtype=TTYPES[Device.Cpu]
        )
        whisper_model = faster_whisper.WhisperModel(
          Model.Large,
          device=Device.Cpu,
          compute_type=MTYPES[Device.Cpu]
        )
        whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)

        assert '.srt' in transcript.transcribe(
          self.fixture('vocals.wav'),
          whisper_pipeline,
          whisper_model,
          alignment_model,
          alignment_tokenizer,
          [-1],
          0,
          0,
          'ru'
        )

        assert '.srt' in transcript.transcribe(
          self.fixture('vocals.wav'),
          whisper_pipeline,
          whisper_model,
          alignment_model,
          alignment_tokenizer,
          [-1],
          2,
          0,
          'ru'
        )
