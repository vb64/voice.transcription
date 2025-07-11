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


all: tests

mp3:
	$(PYTHON) cli/to_mp3.py build

cut:
	$(PYTHON) cli/cut.py fixtures/short.mp3 10 15 build/short_10_15.mp3

split:
	$(PYTHON) cli/split_mp3.py build/xxx.mp3 90

duration:
	$(PYTHON) cli/duration.py build/xxx.mp3

json:
	$(PYTHON) cli/to_json.py build/xxx.mp3 build/xxx.json --temp_folder build/temp

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
	$(PIP) -r requirements.txt
	$(PIP) -r $(TESTS)/requirements.txt

setup_python:
	$(PYTHON_BIN) -m venv ./venv
