# Настройки whisper-diarization

## Проблема с вызовом perl в venv.

В файле `venv\Lib\site-packages\ctc_forced_aligner\text_utils.py` в функции `get_uroman_tokens` строку 

```python
cmd = ["perl", os.path.join(UROMAN_PATH, "uroman.pl")]
```
заменить на

```python
cmd = ["D:/Perl/perl/bin/perl.exe", os.path.join(UROMAN_PATH, "uroman.pl")]
```

## Проблема с вызовом python в venv.

[Request](https://github.com/MahmoudAshraf97/whisper-diarization/issues/317) for --python_bin command line option.

[Места](https://github.com/MahmoudAshraf97/whisper-diarization/commit/f35796c0a141c41903c120e8bae4d8adb622f224), где нужно заменить.

## Проблема с неверным распознаванием спикеров.

https://github.com/MahmoudAshraf97/whisper-diarization/issues/245#issuecomment-2566592343

Try forcing the number of speakers for NeMo MSDD if you know how many speakers there are. You'll need to do two things:

First -

Find create_config function under the helper functions;
Find the meta dict within it, add this parameter to it: "num_speakers": [number of speakers];
So before, it looks like this:

```python
meta = {
"audio_filepath": os.path.join(output_dir, "mono_file.wav"),
"offset": 0,
"duration": None,
"label": "infer",
"text": "-",
"rttm_filepath": None,
"uem_filepath": None,
}
```

Now it looks like this:

```python
meta = {
"audio_filepath": os.path.join(output_dir, "mono_file.wav"),
"offset": 0,
"duration": None,
"label": "infer",
"text": "-",
"num_speakers": [number of speakers], # force to X speakers
"rttm_filepath": None,
"uem_filepath": None,
}
```

Second -

Under the same function, find and change config.diarizer.clustering.parameters.oracle_num_speakers = False to config.diarizer.clustering.parameters.oracle_num_speakers = True. This will ask the model to use the number of speakers you specified.

---

Also:

- https://github.com/MahmoudAshraf97/whisper-diarization/issues/191
- https://github.com/MahmoudAshraf97/whisper-diarization/issues/303
- https://github.com/MahmoudAshraf97/whisper-diarization/issues/310

## Проблема с Ubuntu.

Ubuntu 22.04
https://github.com/MahmoudAshraf97/whisper-diarization/issues/293

I think the Ubuntu I have has sox 1.5.0 pre installed. I think it wants numpy 2.2., but we need numpy less than 2 for the bigger install.

I removed it and
pip3 install sox==1.2.5

## Альтернативы:

https://github.com/google/uis-rnn
