"""Voice transcription stuff."""
import sys
import time

from .cli_options import PARSER, VERSION, COPYRIGHTS
from .language import process_language_arg
from . import Model

LANGUAGE = 'ru'


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

    print("\nTotal: {} sec".format(int(time.time() - start_time)))

    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(PARSER.parse_args()))
