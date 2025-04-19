"""Make Nemo rttm files."""
import argparse
import os
import sys
import time
import shutil

import torch
import torchaudio
import faster_whisper

sys.path.insert(1, '.')
from voice_transcription.nemo_msdd import create_config, diarize
from voice_transcription import Device, cleanup, list_mp3

VERSION = '1.0'
COPYRIGHTS = 'Copyrights by Vitaly Bogomolov 2025'
PARSER = argparse.ArgumentParser(description='Nemo diarize tool.')

PARSER.add_argument(
  "input_folder",
  help="Folder with mp3 files."
)
PARSER.add_argument(
  "--num_speakers",
  type=int,
  default=0,
  help="Forcing the number of speakers, default 0 (auto detection)",
)
PARSER.add_argument(
  "--temp_folder",
  default='temp',
  help="Folder for temp files.",
)
PARSER.add_argument(
  "--config",
  default='nemo.cfg',
  help="Path to Nemo config file.",
)


def main(options):
    print("Nemo diarization tool v.{}. {}".format(VERSION, COPYRIGHTS))
    mp3_files = list_mp3(options.input_folder)
    wav_file = os.path.join(options.temp_folder, "mono_file.wav")
    os.makedirs(options.temp_folder, exist_ok=True)

    for i in mp3_files:
        start_time = time.time()
        waveform = faster_whisper.decode_audio(i)
        torchaudio.save(
          wav_file,
          torch.from_numpy(waveform).unsqueeze(0).float(),
          16000,
          channels_first=True
        )
        diarize(
          None,
          wav_file,
          Device.Cpu,
          options.num_speakers,
          options.temp_folder,
          options.config
        )

        rttm_file = os.path.join(options.temp_folder, "pred_rttms", "mono_file.rttm")
        dest = os.path.splitext(i)[0] + '.rttm'
        shutil.copyfile(rttm_file, dest)
        print(dest, "{} sec".format(int(time.time() - start_time)))

    cleanup(options.temp_folder)


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(PARSER.parse_args()))
