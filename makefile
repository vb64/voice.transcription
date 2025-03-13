.PHONY: all setup
# make >debug.log 2>&1
# git remote prune origin
ifeq ($(OS),Windows_NT)
PYTHON = venv/Scripts/python.exe
WHISPER = venv/Scripts/whisper.exe
else
PYTHON = ./venv/bin/python
WHISPER = ./venv/bin/whisper
endif

PIP = $(PYTHON) -m pip install

whisper:
	$(WHISPER) fixtures/short.mp3 --language ru --model medium

setup: setup_python setup_pip

setup_pip:
	$(PIP) --upgrade pip
	$(PIP) -r requirements.txt

setup_python:
	$(PYTHON_BIN) -m venv ./venv
