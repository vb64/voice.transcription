import os
import sys
from pydub import AudioSegment

sys.path.insert(1, '.')
from voice_transcription.audio import cut_part


def main(input_file, start_seconds, end_seconds, output_file):
    print("Cut", input_file, "seconds", start_seconds, "<-->", end_seconds, "to", output_file)

    inp_format = os.path.splitext(input_file)[1][1:]
    out_format = os.path.splitext(output_file)[1][1:]

    audio = cut_part(
      AudioSegment.from_file(input_file, inp_format),
      int(start_seconds) * 1000,  # milliseconds
      int(end_seconds) * 1000
    )
    audio.export(output_file, format=out_format)


if __name__ == '__main__':  # pragma: no cover
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
