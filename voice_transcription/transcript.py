"""Voice transcription stuff."""
import os
import time
import shutil

import torch
import torchaudio
import faster_whisper
from demucs.separate import main as demucs_separate
from ctc_forced_aligner import (
  generate_emissions,
  get_alignments,
  get_spans,
  load_alignment_model,
  postprocess_results,
  preprocess_text,
)

from .cli_options import VERSION, COPYRIGHTS
from .language import process_language_arg, LANGS_TO_ISO
from .nemo_msdd import diarize
from .speaker_mapping import map_speakers
from .srt import write_srt

from . import Model, Device, MTYPES, TTYPES, add_log

LANGUAGE = 'ru'
TEMP_DIR = "temp_outputs"
DEVICE = Device.Cpu
MODEL = Model.Large


def isolate_vocals(call_log, input_file, folder):
    """Isolate vocals from the rest of the audio."""
    start_time = add_log(call_log, "isolate_vocals", None)
    demucs_separate([
      "-n", "htdemucs",
      "--two-stems", "vocals",
      "-o", folder,
      "--device", DEVICE,
      input_file
    ])
    add_log(call_log, "demucs_separate", start_time)

    return os.path.join(
        folder,
        "htdemucs",
        os.path.splitext(os.path.basename(input_file))[0],
        "vocals.wav",
    )


def transcribe(call_log, model_name, device, vocal_target, language, batch_size):
    """Transcribe the audio file."""
    start_time = add_log(call_log, "Transcribe", None)

    whisper_model = faster_whisper.WhisperModel(
      model_name,
      device=device,
      compute_type=MTYPES[device]
    )
    start_time = add_log(call_log, "WhisperModel", start_time)

    audio_waveform = faster_whisper.decode_audio(vocal_target)
    start_time = add_log(call_log, "decode_audio", start_time)

    # args.suppress_numerals == False
    suppress_tokens = [-1]

    if batch_size > 0:
        whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
        transcript_segments, info = whisper_pipeline.transcribe(
          audio_waveform,
          language,
          suppress_tokens=suppress_tokens,
          batch_size=batch_size,
        )
    else:
        transcript_segments, info = whisper_model.transcribe(
          audio_waveform,
          language,
          suppress_tokens=suppress_tokens,
          vad_filter=True,
        )
    add_log(call_log, "transcribe", start_time)

    return (transcript_segments, info, audio_waveform)


def forced_alignment(call_log, device, segments, info, waveform):  # pylint: disable=too-many-locals
    """Force alignment."""
    start_time = add_log(call_log, "forced_alignment", None)

    alignment_model, alignment_tokenizer = load_alignment_model(
      device,
      dtype=TTYPES[device]
    )
    start_time = add_log(call_log, "load_alignment_model", start_time)

    emissions, stride = generate_emissions(
      alignment_model,
      torch.from_numpy(waveform)
      .to(alignment_model.dtype)
      .to(alignment_model.device)
    )
    start_time = add_log(call_log, "generate_emissions", start_time)

    full_transcript = "".join(segment.text for segment in segments)
    start_time = add_log(call_log, "full_transcript", start_time)

    tokens_starred, text_starred = preprocess_text(
      full_transcript,
      romanize=True,
      language=LANGS_TO_ISO[info.language],
    )
    start_time = add_log(call_log, "preprocess_text", start_time)

    segments, scores, blank_token = get_alignments(
      emissions,
      tokens_starred,
      alignment_tokenizer,
    )
    start_time = add_log(call_log, "get_alignments", start_time)

    spans = get_spans(tokens_starred, segments, blank_token)
    start_time = add_log(call_log, "get_spans", start_time)

    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    add_log(call_log, "postprocess_results", start_time)

    return word_timestamps


def to_mono(call_log, waveform, file_name):
    """Convert audio to mono for NeMo combatibility."""
    start_time = add_log(call_log, "to_mono", None)
    torchaudio.save(
      file_name,
      torch.from_numpy(waveform).unsqueeze(0).float(),
      16000,
      channels_first=True
    )
    add_log(call_log, "to_mono", start_time)


def dump_log(call_log):
    """Print out call log."""
    for name, seconds in call_log:
        if seconds is None:
            print("# Call: {}".format(name))
        else:
            print("{}: {} sec".format(name, seconds))


def cleanup(path: str):
    """Remove path could either be relative or absolute."""
    # check if file or directory exists
    if os.path.isfile(path) or os.path.islink(path):
        # remove file
        os.remove(path)
    elif os.path.isdir(path):
        # remove directory and all its content
        shutil.rmtree(path)


def main(options):
    """Entry point."""
    start_time = time.time()
    print("Voice to text tool v.{}. {}".format(VERSION, COPYRIGHTS))
    call_log = []

    lang = process_language_arg(LANGUAGE, MODEL)

    print("File: '{}' speakers: {} language: {}".format(
      options.input_file,
      options.num_speakers if options.num_speakers > 0 else 'auto',
      lang
    ))

    vocal_target = isolate_vocals(call_log, options.input_file, TEMP_DIR)
    segments, info, waveform = transcribe(call_log, MODEL, DEVICE, vocal_target, lang, options.batch_size)
    word_timestamps = forced_alignment(call_log, DEVICE, segments, info, waveform)
    wav_file = os.path.join(TEMP_DIR, "mono_file.wav")
    to_mono(call_log, waveform, wav_file)
    diarize(call_log, wav_file, DEVICE, options.num_speakers, TEMP_DIR)
    rttm_file = os.path.join(TEMP_DIR, "pred_rttms", "mono_file.rttm")
    ssm = map_speakers(call_log, rttm_file, word_timestamps)
    srt_file = os.path.splitext(options.input_file)[0] + '.srt'
    write_srt(call_log, ssm, srt_file)

    cleanup(TEMP_DIR)
    dump_log(call_log)
    print("\nTotal: {} sec".format(int(time.time() - start_time)))

    return 0
