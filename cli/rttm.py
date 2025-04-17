"""CLI options."""
import argparse
import os
import sys

sys.path.insert(1, '.')
from voice_transcription.nemo_msdd import create_config, diarize

VERSION = '1.0'
COPYRIGHTS = 'Copyrights by Vitaly Bogomolov 2025'
PARSER = argparse.ArgumentParser(description='Nemo diarize tool.')

PARSER.add_argument(
  "input_folder",
  help="Folder with mp3 files."
)
PARSER.add_argument(
  "--num_speakers",
  type=int,
  default=0,
  help="Forcing the number of speakers, default 0 (auto detection)",
)
PARSER.add_argument(
  "--temp_folder",
  default='temp',
  help="Folder for temp files.",
)
PARSER.add_argument(
  "--config",
  default='nemo.cfg',
  help="Path to Nemo config file.",
)


def list_mp3(folder):
    result = []
    for item in [os.path.join(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))]:
        if os.path.splitext(item)[1] == '.mp3':
            result.append(item)
    return result


def main(options):
    print("Nemo diarization tool v.{}. {}".format(VERSION, COPYRIGHTS))
    mp3_files = list_mp3(options.input_folder)
    for i in mp3_files:
        print(i)


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(PARSER.parse_args()))
