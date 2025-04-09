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
  help="forcing the number of speakers, default 0 (auto detection)",
)
