"""Transcibe mp3 to json with word data."""
import os
import time
import argparse
import sys
from datetime import datetime
import json
import shutil

import faster_whisper

sys.path.insert(1, '.')
from voice_transcription import Model, Device, MTYPES, progress_bar, get_tasks
from voice_transcription.whisper import make_json, segments_to_json, join_jsons, msec

VERSION = '1.0'
COPYRIGHTS = 'Copyrights by Vitaly Bogomolov 2025'
PARSER = argparse.ArgumentParser(description='Whisper to json transcribe tool.')

PARSER.add_argument(
  "mp3_file",
  help="Audio file for transcribe."
)
PARSER.add_argument(
  "out_file",
  help="Json file for output."
)
PARSER.add_argument(
  "--temp_folder",
  default='temp',
  help="Folder for temp files.",
)
PARSER.add_argument(
  "--max_length",
  type=int,
  default=90 * 60,
  help="Split the input file into parts if duration exceeds the parameter value. Default is 5400 (90 minutes).",
)


def main(options):
    """Entry point."""
    print("Whisper to json transcribe tool v.{}. {}".format(VERSION, COPYRIGHTS))
    stime = time.time()
    tasks = get_tasks(options.mp3_file, options.temp_folder, options.max_length)

    print("Loading Whisper model...")
    whisper_model = faster_whisper.WhisperModel(
      Model.Large,
      device=Device.Cpu,
      compute_type=MTYPES[Device.Cpu]
    )

    os.makedirs(options.temp_folder, exist_ok=True)
    data = join_jsons([
      make_json(whisper_model, i, progress_bar)
      for i in tasks
    ])

    print("\nSaving json {}...".format(options.out_file))

    with open(options.out_file, "wt", encoding="utf8") as out:
        out.write(json.dumps(data, indent=4, ensure_ascii=False).encode('utf8').decode())

    shutil.rmtree(options.temp_folder)

    print(options.out_file, "{} sec".format(int(time.time() - stime)))
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(PARSER.parse_args()))
