https://github.com/MahmoudAshraf97/whisper-diarization/issues/317

Request for --python_bin command line option.

First of all, thank you very much for this project, it is really useful for a lot of people.

I use several versions of python for my task and this line of code calls the wrong version of python.

https://github.com/MahmoudAshraf97/whisper-diarization/blob/6c9047dd7334c48acde62042890fd357117a9f55/diarize.py#L99

If change this line like this 

'''python
f'{args.python_bin} -m demucs.separate -n htdemucs --two-stems=vocals "{args.audio}" -o temp_outputs --device "{args.device}"'
'''

and add support for the corresponding command line option (with default value 'python' for backward compatibility), it would save me from having to patch the diarize.py code.

If you pre-approve this change, I can prepare a pull request with the necessary changes.
