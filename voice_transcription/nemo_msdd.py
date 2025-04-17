"""Nemo model."""
import os
import json
import wget

from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from omegaconf import OmegaConf

from . import add_log

DOMAIN_TYPE = "telephonic"
CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"
CONFIG_URL = "{}/main/examples/speaker_tasks/diarization/conf/inference/{}".format(
  "https://raw.githubusercontent.com/NVIDIA/NeMo",
  CONFIG_FILE_NAME
)


def create_config(wav_file, output_dir, config_path, num_speakers=0):
    """Make Nemo config."""
    if not os.path.exists(config_path):
        config_path = wget.download(CONFIG_URL, config_path)

    config = OmegaConf.load(config_path)

    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    meta = {
        "audio_filepath": wav_file,
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    if num_speakers > 0:
        meta["num_speakers"] = num_speakers

    with open(os.path.join(data_dir, "input_manifest.json"), "w", encoding='utf-8') as fp:
        json.dump(meta, fp)
        fp.write("\n")

    pretrained_vad = "vad_multilingual_marblenet"
    pretrained_speaker_model = "titanet_large"
    config.num_workers = 0
    config.diarizer.manifest_filepath = os.path.join(data_dir, "input_manifest.json")
    config.diarizer.out_dir = (
        output_dir  # Directory to store intermediate files and prediction outputs
    )

    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.oracle_vad = (
        False  # compute VAD provided with model_path to vad config
    )
    config.diarizer.clustering.parameters.oracle_num_speakers = (num_speakers > 0)

    # Here, we use our in-house pretrained NeMo VAD model
    config.diarizer.vad.model_path = pretrained_vad
    config.diarizer.vad.parameters.onset = 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = -0.05
    config.diarizer.msdd_model.model_path = (
        "diar_msdd_telephonic"  # Telephonic speaker diarization model
    )

    return config


def diarize(call_log, wav_file, device, num_speakers, temp_path, config_path):
    """Initialize NeMo MSDD diarization model."""
    start_time = add_log(call_log, "Nemo", None)
    config = create_config(wav_file, temp_path, config_path, num_speakers=num_speakers)
    model = NeuralDiarizer(cfg=config).to(device)
    start_time = add_log(call_log, "NeuralDiarizer", start_time)

    model.diarize()
    add_log(call_log, "diarize", start_time)

    return 0
