.PHONY: all setup
# make >debug.log 2>&1
# git remote prune origin
ifeq ($(OS),Windows_NT)
PYTHON = venv/Scripts/python.exe
PTEST = venv/Scripts/pytest.exe
COVERAGE = venv/Scripts/coverage.exe
else
PYTHON = ./venv/bin/python
PTEST = ./venv/bin/pytest
COVERAGE = ./venv/bin/coverage
endif

SOURCE = voice_transcription
TESTS = tests

FLAKE8 = $(PYTHON) -m flake8
PYLINT = $(PYTHON) -m pylint
PYTEST = $(PTEST) --cov=$(SOURCE) --cov-report term:skip-covered
PIP = $(PYTHON) -m pip install


all:
	$(PYTHON) cli.py fixtures/short.mp3

aac:
	$(PYTHON) cli/aac2mp3.py build/aac

mp3:
	$(PYTHON) cli.py --whisper_batch 8 --torch_batch 4 build/xxx.mp3

test:
	$(PTEST) -s $(TESTS)/test/$(T)

flake8:
	$(FLAKE8) $(TESTS)/test
	$(FLAKE8) $(SOURCE)

lint:
	$(PYLINT) $(TESTS)/test
	$(PYLINT) $(SOURCE)

pep257:
	$(PYTHON) -m pydocstyle $(TESTS)/test
	$(PYTHON) -m pydocstyle $(SOURCE)

tests: flake8 pep257 lint
	$(PYTEST) -m "not longrunning" --durations=5 $(TESTS)

cover: flake8 pep257 lint
	$(PYTEST) --durations=5 $(TESTS)
	$(COVERAGE) html --skip-covered

setup: setup_python setup_pip

setup_pip:
	$(PIP) --upgrade pip
	$(PIP) cython
	$(PIP) -c constraints.txt -r requirements.txt
	$(PIP) -r $(TESTS)/requirements.txt

setup_python:
	$(PYTHON_BIN) -m venv ./venv
