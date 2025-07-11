"""Transcibe mp3 to json with word data."""
import time
import argparse
import sys
from datetime import datetime
import json

import faster_whisper

sys.path.insert(1, '.')
from voice_transcription import Model, Device, MTYPES, progress_bar
from voice_transcription.whisper import segments_to_json, msec

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


def main(options):  # pylint: disable=too-many-locals
    """Entry point."""
    print("Whisper to json transcribe tool v.{}. {}".format(VERSION, COPYRIGHTS))
    stime = time.time()

    print("Loading model...")
    whisper_model = faster_whisper.WhisperModel(
      Model.Large,
      device=Device.Cpu,
      compute_type=MTYPES[Device.Cpu]
    )
    print("Decode wave...")
    waveform = faster_whisper.decode_audio(options.mp3_file)

    print("Creating segments...")
    segments, info = whisper_model.transcribe(
      waveform, 'ru', suppress_tokens=[-1],
      vad_filter=True,
      word_timestamps=True
    )
    duration = msec(info.duration_after_vad)
    print("Duration", duration, "msec")

    now = datetime.utcnow()
    data = segments_to_json(segments, duration, progress_bar, now)
    progress_bar(duration, duration, now)
    print("\nSaving json {}...".format(options.out_file))

    with open(options.out_file, "wt", encoding="utf8") as out:
        out.write(json.dumps(data, indent=4, ensure_ascii=False).encode('utf8').decode())

    print(options.out_file, "{} sec".format(int(time.time() - stime)))
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(PARSER.parse_args()))
