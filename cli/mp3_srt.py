"""Transcibe mp3 to srt."""
import argparse
import os
import sys

sys.path.insert(1, '.')
from voice_transcription import list_mp3

VERSION = '1.0'
COPYRIGHTS = 'Copyrights by Vitaly Bogomolov 2025'
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

    for i in list_mp3(options.input_folder):
        print(i)


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(PARSER.parse_args()))
