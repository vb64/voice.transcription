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
PARSER.add_argument(
    "--extract_vocals",
    action="store_true",
    help="Filter out music and other non voices from audio. Requires more time for processing."
)
