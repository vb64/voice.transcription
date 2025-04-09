"""Voice transcription stuff."""
import os
import sys
import time

import faster_whisper
from demucs.separate import main as demucs_separate

from .cli_options import PARSER, VERSION, COPYRIGHTS
from .language import process_language_arg
from . import Model, Device, MTYPES

LANGUAGE = 'ru'
TEMP_DIR = "temp_outputs"
DEVICE = Device.Cpu
MODEL = Model.Large


def isolate_vocals(input_file, folder):
    """Isolate vocals from the rest of the audio."""
    demucs_separate([
      "-n", "htdemucs",
      "--two-stems", "vocals",
      "-o", folder,
      "--device", DEVICE,
      input_file
    ])

    return os.path.join(
        folder,
        "htdemucs",
        os.path.splitext(os.path.basename(input_file))[0],
        "vocals.wav",
    )


def transcribe(model_name, device):
    """Transcribe the audio file."""
    whisper_model = faster_whisper.WhisperModel(
      model_name,
      device=device,
      compute_type=MTYPES[device]
    )

    print("#", whisper_model)


def main(options):
    """Entry point."""
    start_time = time.time()
    print("Voice to text tool v.{}. {}".format(VERSION, COPYRIGHTS))

    lang = process_language_arg(LANGUAGE, MODEL)

    print("File: '{}' speakers: {} language: {}".format(
      options.input_file,
      options.num_speakers if options.num_speakers > 0 else 'auto',
      lang
    ))

    vocal_target = isolate_vocals(options.input_file, TEMP_DIR)
    print("Vocal target: {}".format(vocal_target))

    transcribe(MODEL, DEVICE)

    print("\nTotal: {} sec".format(int(time.time() - start_time)))

    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(PARSER.parse_args()))
