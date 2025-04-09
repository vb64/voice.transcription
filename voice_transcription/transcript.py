"""Voice transcription stuff."""
import os
import sys
import time

import torch
import faster_whisper
from demucs.separate import main as demucs_separate
from ctc_forced_aligner import (
  generate_emissions,
  # get_alignments,
  # get_spans,
  load_alignment_model,
  # postprocess_results,
  # preprocess_text,
)

from .cli_options import PARSER, VERSION, COPYRIGHTS
from .language import process_language_arg
from . import Model, Device, MTYPES, TTYPES

LANGUAGE = 'ru'
TEMP_DIR = "temp_outputs"
DEVICE = Device.Cpu
MODEL = Model.Large

# Batch size for batched inference, reduce if you run out of memory,
# set to 0 for original whisper longform inference
BATCH_SIZE = 8


def isolate_vocals(input_file, folder):
    """Isolate vocals from the rest of the audio."""
    demucs_separate([
      "-n", "htdemucs",
      "--two-stems", "vocals",
      "-o", folder,
      "--device", DEVICE,
      input_file
    ])

    return os.path.join(
        folder,
        "htdemucs",
        os.path.splitext(os.path.basename(input_file))[0],
        "vocals.wav",
    )


def transcribe(model_name, device, vocal_target, language):
    """Transcribe the audio file."""
    start_time = time.time()

    whisper_model = faster_whisper.WhisperModel(
      model_name,
      device=device,
      compute_type=MTYPES[device]
    )
    print("\nWhisperModel: {} sec".format(int(time.time() - start_time)))
    start_time = time.time()

    whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
    print("BatchedInferencePipeline: {} sec".format(int(time.time() - start_time)))
    start_time = time.time()

    audio_waveform = faster_whisper.decode_audio(vocal_target)
    print("Decode_audio: {} sec".format(int(time.time() - start_time)))
    start_time = time.time()

    # args.suppress_numerals == False
    suppress_tokens = [-1]

    if BATCH_SIZE > 0:
        transcript_segments, info = whisper_pipeline.transcribe(
          audio_waveform,
          language,
          suppress_tokens=suppress_tokens,
          batch_size=BATCH_SIZE,
        )
    else:
        transcript_segments, info = whisper_model.transcribe(
          audio_waveform,
          language,
          suppress_tokens=suppress_tokens,
          vad_filter=True,
        )
    print("Transcribe: {} sec".format(int(time.time() - start_time)))

    # clear gpu vram
    # del whisper_model, whisper_pipeline
    # torch.cuda.empty_cache()
    # print("clear gpu vram: {} sec".format(int(time.time() - start_time)))
    # start_time = time.time()

    return (transcript_segments, info, audio_waveform)


def forced_alignment(device, _segments, _info, waveform):
    """Force alignment."""
    start_time = time.time()
    # full_transcript = "".join(segment.text for segment in transcript_segments)
    # print("full_transcript: {} sec".format(int(time.time() - start_time)))
    # start_time = time.time()

    alignment_model, alignment_tokenizer = load_alignment_model(
      device,
      dtype=TTYPES[device]
    )
    print("load_alignment_model: {} sec".format(int(time.time() - start_time)))

    emissions, stride = generate_emissions(
      alignment_model,
      torch.from_numpy(waveform)
      .to(alignment_model.dtype)
      .to(alignment_model.device),
      batch_size=BATCH_SIZE,
    )
    start_time = time.time()
    print("generate_emissions: {} sec".format(int(time.time() - start_time)))

    print(type(alignment_tokenizer))
    print(type(emissions))
    print(type(stride))


def main(options):
    """Entry point."""
    start_time = time.time()
    print("Voice to text tool v.{}. {}".format(VERSION, COPYRIGHTS))

    lang = process_language_arg(LANGUAGE, MODEL)

    print("File: '{}' speakers: {} language: {}".format(
      options.input_file,
      options.num_speakers if options.num_speakers > 0 else 'auto',
      lang
    ))

    vocal_target = isolate_vocals(options.input_file, TEMP_DIR)
    segments, info, waveform = transcribe(MODEL, DEVICE, vocal_target, lang)
    forced_alignment(DEVICE, segments, info, waveform)

    print("\nTotal: {} sec".format(int(time.time() - start_time)))

    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(PARSER.parse_args()))
