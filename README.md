# Преобразование mp3 в текст с распознаванием спикеров

## Настройка под Windows.

Предварительно установить следующие программы.

- GNU [Unix Utils](http://unxutils.sourceforge.net/) для операций через makefile
- [Git for Windows](https://git-scm.com/download/win) для доступа к репозитарию исходных кодов.
- [Python3.10.11](https://www.python.org/downloads/release/python-31011/)
- [ffmpeg](https://ffmpeg.org/download.html) (распаковать архив и добавить в PATH)
- [Perl](https://strawberryperl.com/)
- build tools by installing [Microsoft Visual Studio](https://visualstudio.microsoft.com/downloads/)

Затем:

```bash
git clone https://github.com/vb64/voice.transcription.git
cd voice.transcription
make setup PYTHON_BIN=C:\полный\путь\на\3.10\python.exe
```

## Настройка под Linux.

```
sudo apt update
sudo apt-get install build-essential python3.10-venv python3-pip ffmpeg perl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
