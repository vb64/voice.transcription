.PHONY: all setup
# make >debug.log 2>&1
# git remote prune origin
ifeq ($(OS),Windows_NT)
PYTHON = venv/Scripts/python.exe
WHISPER = venv/Scripts/whisper.exe
PIP = $(PYTHON) -m pip install
else
PYTHON = ./venv/bin/python
PIP = $(PYTHON) -m pip3 install
WHISPER = ./venv/bin/whisper
endif


whisper:
	$(WHISPER) fixtures/short.mp3 --language ru --model large --output_dir transcrip --threads 5

setup: setup_python setup_pip

setup_pip:
	$(PIP) --upgrade pip
	$(PIP) -r requirements.txt

setup_python:
	$(PYTHON_BIN) -m venv ./venv
