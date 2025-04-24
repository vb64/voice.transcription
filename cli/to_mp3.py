import os
import sys

sys.path.insert(1, '.')
from voice_transcription.audio import convert_audio


def main(folder):
    for item in [os.path.join(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))]:
        out = os.path.splitext(item)[0] + '.mp3'
        convert_audio(item, out)
        print(item, '->', out)


if __name__ == '__main__':  # pragma: no cover
    main(sys.argv[1])
