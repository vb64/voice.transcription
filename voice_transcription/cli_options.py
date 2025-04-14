"""CLI options."""
import argparse

VERSION = '1.0'
COPYRIGHTS = 'Copyrights by Vitaly Bogomolov 2025'
PARSER = argparse.ArgumentParser(description='Transcript audio to text.')

PARSER.add_argument(
  "input_file",
  help="Input file name."
)

PARSER.add_argument(
  "--num_speakers",
  type=int,
  default=0,
  help="Forcing the number of speakers, default 0 (auto detection)",
)
PARSER.add_argument(
  "--batch_size",
  type=int,
  default=0,
  help="Batch size for batched inference, set to 8 for 16Gb RAM. Default 0 (original whisper longform inference).",
)
