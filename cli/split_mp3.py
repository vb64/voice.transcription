import os
import sys
import time

sys.path.insert(1, '.')
from voice_transcription.audio import split_audio


def main(mp3_file, chunk_minutes):
    start_time = time.time()
    out_file_mask = os.path.splitext(mp3_file)[0]
    chunk_size = int(chunk_minutes)

    print("Split", mp3_file , "by", int(chunk_minutes), "minutes")
    split_audio(mp3_file, out_file_mask, chunk_size * 1000 * 60, 'mp3')

    print("Total: {} sec".format(int(time.time() - start_time)))


if __name__ == '__main__':  # pragma: no cover
    main(sys.argv[1], sys.argv[2])
