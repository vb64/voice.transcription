# Использование whisper-diarization

Настройка под Windows.

Перечень средств разработки, необходимых для настройки хоста. Скачать и установить:

- [Python3.10.11](https://www.python.org/downloads/release/python-31011/)
- [ffmpeg](https://ffmpeg.org/download.html) (распаковать архив и добавить в PATH)
- [Perl](https://strawberryperl.com/)

Затем в окне консоли:

```
git clone https://github.com/MahmoudAshraf97/whisper-diarization.git
cd whisper-diarization
make setup PYTHON_BIN=D:\python\3.10\python.exe
```

В файле `venv\Lib\site-packages\ctc_forced_aligner\text_utils.py` в функции `get_uroman_tokens` строку 

```python
cmd = ["perl", os.path.join(UROMAN_PATH, "uroman.pl")]
```
заменить на

```python
cmd = ["D:/Perl/perl/bin/perl.exe", os.path.join(UROMAN_PATH, "uroman.pl")]
```

Затем:

```
make short
```

В каталоге `fixtures` будут созданы файлы

- short.srt
- short.txt

Настройка под Linux.

```
sudo apt update
sudo apt-get install build-essential python3.10-venv python3-pip ffmpeg perl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
