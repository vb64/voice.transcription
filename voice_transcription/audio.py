"""Split audio file stuff."""
import os
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
from pydub.silence import split_on_silence

MIN_SILENCE_LEN = 100  # Minimum length of silence in milliseconds
SILENCE_THRESHOLD = -40  # Silence threshold in dB


class Format:
    """Audio formats."""

    Wav = 'wav'
    Aac = 'aac'
    Mp3 = 'mp3'


def save_chunk(chunk, name, output_format):
    """Save chunk to file."""
    chunk.export(name, format=output_format)


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


def split_audio(input_file, out_file_mask, chunk_length_ms, output_format):
    """Split input audio file by chunks with given size."""
    audio = AudioSegment.from_file(input_file)
    # Split the audio file based on silence
    chunks = split_on_silence(audio, min_silence_len=MIN_SILENCE_LEN, silence_thresh=SILENCE_THRESHOLD)
    # Merge adjacent chunks shorter than the specified length
    chunks = merge_short_chunks(chunks, chunk_length_ms)
    # Save chunks in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        for i, chunk in enumerate(chunks):
            executor.submit(
              save_chunk,
              chunk,
              "{}_{}.{}".format(out_file_mask, i, output_format),
              output_format
            )


def convert_audio(input_file, output_file):
    """Convert audio according file extensions."""
    inp_format = os.path.splitext(input_file)[1][1:]
    out_format = os.path.splitext(output_file)[1][1:]

    audio = AudioSegment.from_file(input_file, inp_format)
    # print(inp_format, '->', out_format, audio.duration_seconds, 'sec')
    audio.export(output_file, format=out_format)
