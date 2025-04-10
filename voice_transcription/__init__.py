"""Voice transcription stuff."""
import time
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
    call_log.append((name, seconds))

    return now
