"""Transcibe mp3 to srt."""
import argparse
import os
import sys
import time

import torch
import faster_whisper
from ctc_forced_aligner import (
  generate_emissions,
  get_alignments,
  get_spans,
  load_alignment_model,
  postprocess_results,
  preprocess_text,
)

sys.path.insert(1, '.')
from voice_transcription import list_mp3, Model, Device, MTYPES, TTYPES
from voice_transcription.speaker_mapping import map_speakers
from voice_transcription.srt import write_srt
from voice_transcription.transcript import transcribe

VERSION = '1.0'
COPYRIGHTS = 'Copyrights by Vitaly Bogomolov 2025'
LANGUAGE = 'ru'
PARSER = argparse.ArgumentParser(description='Whisper transcribe tool.')

PARSER.add_argument(
  "input_folder",
  help="Folder with mp3 files."
)
PARSER.add_argument(
  "--whisper_batch",
  type=int,
  default=0,
  help="Batch size for whisper batched inference. Default 0 (original whisper longform inference).",
)
PARSER.add_argument(
  "--torch_batch",
  type=int,
  default=0,
  help="Torch batch size. Default 0 (disabled).",
)


def main(options):
    print("Whisper transcribe tool v.{}. {}".format(VERSION, COPYRIGHTS))

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

    for i in list_mp3(options.input_folder):
        stime = time.time()
        print(transcribe(
          i,
          whisper_pipeline,
          whisper_model,
          alignment_model,
          alignment_tokenizer,
          [-1],
          options.whisper_batch,
          options.torch_batch,
          LANGUAGE
        ), int(time.time() - stime), 'sec')


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(PARSER.parse_args()))
