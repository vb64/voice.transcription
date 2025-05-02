"""Split audio file stuff."""
import os
import array
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
from pydub.silence import detect_silence
from pydub.utils import db_to_float

MIN_SILENCE_LEN = 100  # Minimum length of silence in milliseconds
SILENCE_THRESHOLD = -40  # Silence threshold in dB


class Format:
    """Audio formats."""

    Wav = 'wav'
    Aac = 'aac'
    Mp3 = 'mp3'


# https://github.com/jiaaro/pydub/issues/256
class Mixer:
    """Chunk joiner."""

    def __init__(self):
        """Make empty."""
        self.parts = []

    def overlay(self, sound, position=0):
        """Add chunk to given position."""
        self.parts.append((position, sound))
        return self

    def _sync(self):
        """Sync parts."""
        positions, segs = zip(*self.parts)
        frame_rate = segs[0].frame_rate
        offsets = [int(frame_rate * pos / 1000.0) for pos in positions]
        segs = AudioSegment.empty()._sync(*segs)  # pylint: disable=protected-access
        return list(zip(offsets, segs))

    def __len__(self):
        """Return length."""
        parts = self._sync()
        seg = parts[0][1]
        frame_count = max(
            offset + seg.frame_count()
            for offset, seg in parts
        )
        return int(1000.0 * frame_count / seg.frame_rate)

    def append(self, sound):
        """Add chunk."""
        self.overlay(sound, position=len(self))

    def to_audio_segment(self, gain=0):
        """Make joined chunk."""
        samp_multiplier = db_to_float(gain)
        parts = self._sync()
        seg = parts[0][1]
        channels = seg.channels

        frame_count = max(
            offset + seg.frame_count()
            for offset, seg in parts
        )
        sample_count = int(frame_count * seg.channels)

        output = array.array(seg.array_type, [0]*sample_count)
        for offset, seg in parts:
            sample_offset = offset * channels
            samples = seg.get_array_of_samples()
            for i, sample in enumerate(samples):
                output[i+sample_offset] += int(sample * samp_multiplier)

        return seg._spawn(output)  # pylint: disable=protected-access


def save_chunk(chunk, name, output_format):
    """Save chunk to file."""
    chunk.export(name, format=output_format)


def join_chunks(chunk_list):
    """Return joined chunks from given list."""
    mixer = Mixer()
    mixer.overlay(chunk_list[0])
    for i in chunk_list[1:]:
        mixer.append(i)

    return mixer.to_audio_segment()


def merge_short_chunks(chunks, min_chunk_length_ms):
    """Merge short chunks up to given length in milliseconds."""
    merged_chunks = []
    current_chunk = [chunks[0]]
    current_chunk_length = len(current_chunk[0])

    for chunk in chunks[1:]:
        if current_chunk_length + len(chunk) < min_chunk_length_ms:
            current_chunk.append(chunk)
            current_chunk_length += len(chunk)
        else:
            merged_chunks.append(join_chunks(current_chunk))
            current_chunk = [chunk]
            current_chunk_length = len(chunk)

    merged_chunks.append(join_chunks(current_chunk))

    return merged_chunks


def get_cut_positions(silence_chunks, chunk_length_ms):
    """Return list of cut positions for given minimal chunk length."""
    offset = 0
    cuts = []
    for start, end in silence_chunks:
        pos = start + int((end - start) / 2)
        if (pos - offset) >= chunk_length_ms:
            cuts.append(pos)
            offset = pos

    return cuts


def split_on_silence_min_length(
    audio,
    min_silence_len=1000, silence_thresh=-16, seek_step=1,
    min_chunk_length=0
):
    """Return list of audio segments from splitting audio_segment on silent sections.

    audio_segment - original pydub.AudioSegment() object

    min_silence_len - (in ms) minimum length of a silence to be used for
        a split. default: 1000ms

    silence_thresh - (in dBFS) anything quieter than this will be
        considered silence. default: -16dBFS

    seek_step - step size for interating over the segment in ms

    min_chunk_length - minimal chunk length (in ms).
        default: 0 (no minimal chunk length, split by middle of silence segments)
    """
    silence_chunks = detect_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        seek_step=seek_step
    )
    chunks = []
    start = 0
    for i in get_cut_positions(silence_chunks, min_chunk_length):
        chunks.append(audio[start:i])
        start = i
    chunks.append(audio[start:])

    return chunks


def split_audio(input_file, out_file_mask, chunk_length_ms, output_format):
    """Split input audio file by chunks with given size."""
    chunks = split_on_silence_min_length(
      AudioSegment.from_file(input_file),
      min_silence_len=MIN_SILENCE_LEN,
      silence_thresh=SILENCE_THRESHOLD,
      min_chunk_length=chunk_length_ms
    )
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
