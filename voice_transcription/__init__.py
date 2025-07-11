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


# https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def progress_bar(iteration, total, length=100, fill='█'):
    """Call in a loop to create terminal progress bar."""
    decimals = 1
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    body = fill * filled_length + '-' * (length - filled_length)

    print("\rDecode |{}| {}%".format(body, percent), end="\r")
