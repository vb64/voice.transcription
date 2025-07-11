"""Voice transcription stuff."""
from datetime import datetime, timedelta
import os
import wave
import contextlib

import torch
from pydub import AudioSegment
import audioread


class Model:
    """Whisper model to use."""

    Large = 'large'


class Device:
    """Device to use."""

    Gpu = 'cuda'
    Cpu = 'cpu'


MTYPES = {
  Device.Cpu: "int8",
  Device.Gpu: "float16"
}

TTYPES = {
  Device.Cpu: torch.float32,
  Device.Gpu: torch.float16,
}


def list_mp3(folder):
    """Return list of the mp3 files in given folder."""
    result = []
    for item in [os.path.join(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))]:
        if os.path.splitext(item)[1] == '.mp3':
            result.append(item)
    return result


# https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def progress_bar(iteration, total, utc_time_start, length=100, fill='█'):
    """Call in a loop to create terminal progress bar."""
    decimals = 1
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    body = fill * filled_length + '-' * (length - filled_length)
    sec_done = int(round((datetime.utcnow() - utc_time_start).total_seconds()))
    sec_estimated = 0
    if iteration > 0:
        sec_estimated = int(round(sec_done * total / iteration))

    print("\rDecode |{}| {}% {} / {}".format(
      body, percent,
      timedelta(seconds=sec_done),
      timedelta(seconds=sec_estimated)
    ), end="\r")


def get_tasks(mp3_file, temp_folder, max_length_sec):
    """Create task list for given mp3."""
    wav_name = os.path.join(temp_folder, 'long.wav')

    with audioread.audio_open(mp3_file) as audio_file:
        print("File", mp3_file, audio_file.duration, "sec")
        print('Backend:', str(type(audio_file).__module__).split('.')[1])

        if audio_file.duration <= max_length_sec:
            print("Single file.")
            return [mp3_file]

        print("Converting to", wav_name, "...")

        with contextlib.closing(wave.open(wav_name, 'w')) as out:
            out.setnchannels(audio_file.channels)
            out.setframerate(audio_file.samplerate)
            out.setsampwidth(2)

            for buf in audio_file:
                out.writeframes(buf)

    audio = AudioSegment.from_wav(wav_name)

    from .audio import split_on_silence_min_length

    chunks = split_on_silence_min_length(
      audio, min_silence_len=100, silence_thresh=-40, min_chunk_length=max_length_sec * 1000
    )
    print("Split to chunks:", len(chunks))

    name = os.path.join(temp_folder, "nemo_chunk_")
    names = []
    for i, chunk in enumerate(chunks):
        names.append(name + str(i) + ".mp3")
        chunk.export(names[-1], format='mp3')

    return names
