"""Voice transcription stuff."""
import os
import time
import shutil
import torch


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


def add_log(call_log, name, start_time):
    """Add record to call log."""
    now = time.time()
    seconds = None

    if start_time is not None:
        seconds = int(now - start_time)
    #     print("{}: {} sec".format(name, seconds))
    # else:
    #     print("# Call: {}".format(name))

    if call_log is not None:
        call_log.append((name, seconds))

    return now


def cleanup(path: str):
    """Remove path could either be relative or absolute."""
    # check if file or directory exists
    if os.path.isfile(path) or os.path.islink(path):
        # remove file
        os.remove(path)
    elif os.path.isdir(path):
        # remove directory and all its content
        shutil.rmtree(path)


def list_mp3(folder):
    """Return list of the mp3 files in given folder."""
    result = []
    for item in [os.path.join(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))]:
        if os.path.splitext(item)[1] == '.mp3':
            result.append(item)
    return result
