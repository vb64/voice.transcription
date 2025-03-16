# Использование whisper-diarization

Настройка под Windows.

Перечень средств разработки, необходимых для настройки хоста. Скачать и установить:

- [Python3.10.11](https://www.python.org/downloads/release/python-31011/)
- [ffmpeg](https://ffmpeg.org/download.html) (распаковать архив и добавить в PATH)
- [Perl](https://strawberryperl.com/)

Затем в окне консоли:

```
git clone git@github.com:MahmoudAshraf97/whisper-diarization.git
cd whisper-diarization
make setup PYTHON_BIN=D:\python\3.10\python.exe
make short
```
