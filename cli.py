import sys
from voice_transcription.transcript import main
from voice_transcription.cli_options import PARSER

if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(PARSER.parse_args()))
