"""Voice transcription stuff."""
import sys
import time

from .cli_options import PARSER, VERSION, COPYRIGHTS


def main(options):
    """Entry point."""
    print("Voice to text tool v.{}. {}".format(VERSION, COPYRIGHTS))
    print("File: '{}' speakers: {}".format(
      options.input_file,
      options.num_speakers if options.num_speakers > 0 else 'auto'
    ))
    start_time = time.time()
    print("\nTotal: {} sec".format(int(time.time() - start_time)))

    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(PARSER.parse_args()))
