"""Split audio file stuff."""

import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
from pydub.silence import split_on_silence

CHUNK_LENGTH = 1000 * 60 * 20  # milliseconds (20 min)
MIN_SILENCE_LEN = 100  # Minimum length of silence in milliseconds
SILENCE_THRESHOLD = -40  # Silence threshold in dB


class Format:
    """Audio formats."""

    Wav = 'wav'
    Aac = 'aac'
    Mp3 = 'mp3'


def save_chunk(chunk, start_time, output_dir, output_format):
    """Save chunk to file."""
    name = "chunk_{}.{}".format(start_time, output_format)
    chunk.export(os.path.join(output_dir, name), format=output_format)


# https://github.com/jiaaro/pydub/issues/256
def merge_short_chunks(chunks, min_chunk_length_ms):
    """Merge short chunks up to given length in milliseconds."""
    merged_chunks = []
    current_chunk = chunks[0]

    for chunk in chunks[1:]:
        if len(current_chunk) + len(chunk) < min_chunk_length_ms:
            current_chunk += chunk
        else:
            merged_chunks.append(current_chunk)
            current_chunk = chunk

    merged_chunks.append(current_chunk)
    return merged_chunks


def split_audio(input_file, output_dir, chunk_length_ms, output_format):
    """Split input audio file by chunks with given size."""
    start_time = time.time()
    total_time = start_time

    # Load the input audio file using Pydub
    audio = AudioSegment.from_file(input_file)
    print("Load audio: {} sec".format(int(time.time() - start_time)))
    start_time = time.time()

    # Split the audio file based on silence
    chunks = split_on_silence(audio, min_silence_len=MIN_SILENCE_LEN, silence_thresh=SILENCE_THRESHOLD)
    print("Split on silence: {} sec".format(int(time.time() - start_time)))
    start_time = time.time()

    # Merge adjacent chunks shorter than the specified length
    chunks = merge_short_chunks(chunks, chunk_length_ms)
    print("Merge short chunks: {} sec".format(int(time.time() - start_time)))
    start_time = time.time()

    # Save chunks in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        for i, chunk in enumerate(chunks):
            executor.submit(save_chunk, chunk, i, output_dir, output_format)

    print("Saving chunks: {} sec".format(int(time.time() - start_time)))
    print("\nTotal: {} sec".format(int(time.time() - total_time)))


def main(file_name):
    """Entry point."""
    split_audio(file_name, 'build', CHUNK_LENGTH, 'mp3')


def convert_audio(input_file, output_file):
    """Convert audio according file extensions."""
    inp_format = os.path.splitext(input_file)[1][1:]
    out_format = os.path.splitext(output_file)[1][1:]

    audio = AudioSegment.from_file(input_file, inp_format)
    # print(inp_format, '->', out_format, audio.duration_seconds, 'sec')
    audio.export(output_file, format=out_format)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1])
