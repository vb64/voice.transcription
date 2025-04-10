"""Nemo model."""
import os
import json
import wget

from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from omegaconf import OmegaConf

from . import add_log


def create_config(output_dir, num_speakers=0):
    """Make Nemo config."""
    DOMAIN_TYPE = "telephonic"
    CONFIG_LOCAL_DIRECTORY = "nemo_msdd_configs"
    CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"
    MODEL_CONFIG_PATH = os.path.join(CONFIG_LOCAL_DIRECTORY, CONFIG_FILE_NAME)
    if not os.path.exists(MODEL_CONFIG_PATH):
        os.makedirs(CONFIG_LOCAL_DIRECTORY, exist_ok=True)
        CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"
        MODEL_CONFIG_PATH = wget.download(CONFIG_URL, MODEL_CONFIG_PATH)

    config = OmegaConf.load(MODEL_CONFIG_PATH)

    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    meta = {
        "audio_filepath": os.path.join(output_dir, "mono_file.wav"),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    if num_speakers > 0:
        meta["num_speakers"] = num_speakers

    with open(os.path.join(data_dir, "input_manifest.json"), "w") as fp:
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


def diarize(call_log, device, num_speakers, temp_path):
    """Initialize NeMo MSDD diarization model."""
    start_time = add_log(call_log, "Nemo", None)
    model = NeuralDiarizer(cfg=create_config(temp_path, num_speakers=num_speakers)).to(device)
    start_time = add_log(call_log, "NeuralDiarizer", start_time)

    model.diarize()
    add_log(call_log, "diarize", start_time)

    return 0
