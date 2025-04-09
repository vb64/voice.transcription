"""Voice transcription stuff."""
import os
import sys
import time

from demucs.separate import main as demucs_separate

from .cli_options import PARSER, VERSION, COPYRIGHTS
from .language import process_language_arg
from . import Model, Device

LANGUAGE = 'ru'
TEMP_DIR = "temp_outputs"
DEVICE = Device.Cpu


def isolate_vocals(input_file):
    """Isolate vocals from the rest of the audio."""
    demucs_separate([
      "-n", "htdemucs",
      "--two-stems", "vocals",
      "-o", TEMP_DIR,
      "--device", DEVICE,
      input_file
    ])

    return os.path.join(
        TEMP_DIR,
        "htdemucs",
        os.path.splitext(os.path.basename(input_file))[0],
        "vocals.wav",
    )


def main(options):
    """Entry point."""
    start_time = time.time()
    print("Voice to text tool v.{}. {}".format(VERSION, COPYRIGHTS))

    lang = process_language_arg(LANGUAGE, Model.Large)

    print("File: '{}' speakers: {} language: {}".format(
      options.input_file,
      options.num_speakers if options.num_speakers > 0 else 'auto',
      lang
    ))

    vocal_target = isolate_vocals(options.input_file)
    print("Vocal target: {}".format(vocal_target))

    print("\nTotal: {} sec".format(int(time.time() - start_time)))

    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(PARSER.parse_args()))
