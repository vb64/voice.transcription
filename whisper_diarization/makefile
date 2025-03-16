.PHONY: all setup
# make >debug.log 2>&1
# git remote prune origin
ifeq ($(OS),Windows_NT)
PYTHON = venv/Scripts/python.exe
else
PYTHON = ./venv/bin/python
endif

PIP = $(PYTHON) -m pip install

short:
	$(PYTHON) diarize.py -a fixtures/short.mp3 --whisper-model large --language ru

setup: setup_python setup_pip

setup_pip:
	$(PIP) --upgrade pip
	$(PIP) cython
	$(PIP) -c constraints.txt -r requirements.txt

setup_python:
	$(PYTHON_BIN) -m venv ./venv
