VENV_NAME = venv
PYTHON = $(VENV_NAME)/bin/python

.PHONY: all setup install train infer clean

build: setup install

setup:
	python3 -m venv $(VENV_NAME)

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements1.txt

train:
	$(PYTHON) train.py

test:
	$(PYTHON) test.py

infer:
	uvicorn backend.main:app --reload


clean:
	rm -rf $(VENV_NAME)
	rm -rf __pycache__
