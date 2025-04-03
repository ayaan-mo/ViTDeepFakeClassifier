VENV_NAME = venv
PYTHON = $(VENV_NAME)/bin/python

.PHONY: all setup install train infer clean

build: setup install

setup:
	python3 -m venv $(VENV_NAME)

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) train.py

infer:
	$(PYTHON) inference.py

test:
	$(PYTHON) test.py

clean:
	rm -rf $(VENV_NAME)
	rm -rf __pycache__
