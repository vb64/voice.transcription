"""Voice transcription stuff."""
import os
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


def list_mp3(folder):
    """Return list of the mp3 files in given folder."""
    result = []
    for item in [os.path.join(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))]:
        if os.path.splitext(item)[1] == '.mp3':
            result.append(item)
    return result
