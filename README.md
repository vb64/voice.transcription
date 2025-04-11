# Преобразование mp3 в текст с распознаванием спикеров

## Настройка под Windows.

Предварительно установить следующие программы.

- GNU [Unix Utils](http://unxutils.sourceforge.net/) для операций через makefile
- [Git for Windows](https://git-scm.com/download/win) для доступа к репозитарию исходных кодов.
- [Python3.10.11](https://www.python.org/downloads/release/python-31011/)
- [ffmpeg](https://ffmpeg.org/download.html) (распаковать архив и добавить в PATH)
- [Perl](https://strawberryperl.com/)
- build tools by installing [Microsoft Visual Studio](https://visualstudio.microsoft.com/downloads/)

## Настройка под Ubuntu. 22.04

```
ssh -l vit -i C:\Users\vit\.ssh\ed25519 158.160.yyy.xxx
sudo apt update
sudo apt-get install build-essential python3.10-venv python3-pip ffmpeg
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Установка программы

Затем:

```bash
git clone https://github.com/vb64/voice.transcription.git
cd voice.transcription
make setup PYTHON_BIN=python3
```

## screen

```bash
screen -S transcript
Ctrl+a и d
screen -ls
screen -r
```

## Первоначальная загрузка моделей

В screen:

```bash
make
```

## Загрузка файлов

```bash
scp -i ~/.ssh/ed25519 filname vit@158.160.yyy.xxx:~/voice.transcription/build/
```

## Конвертация

Прописать в makefile нужные файлы в таргете `mp3`.

В screen:

```bash
make mp3
```

## Выгрузка srt

```bash
scp -i ~/.ssh/ed25519 vit@158.160.yyy.xxx:~/voice.transcription/build/*.srt ./
```

## Цитаты

```bibtex
@unpublished{hassouna2024whisperdiarization,
  title={Whisper Diarization: Speaker Diarization Using OpenAI Whisper},
  author={Ashraf, Mahmoud},
  year={2024}
}
```
