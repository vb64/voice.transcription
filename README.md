# Преобразование mp3 в текст с распознаванием спикеров

Предварительно установить следующие программы.

- [Python3](https://www.python.org/downloads/release/python-3810/)
- GNU [Unix Utils](http://unxutils.sourceforge.net/) для операций через makefile
- [Git for Windows](https://git-scm.com/download/win) для доступа к репозитарию исходных кодов.

Затем:

```bash
git clone https://github.com/vb64/voice.transcription.git
cd voice.transcription
make setup PYTHON_BIN=C:\полный\путь\на\python.exe
```
