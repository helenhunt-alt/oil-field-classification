PYTHON ?= python

.PHONY: train predict analyze all

train:
	$(PYTHON) -m src.train

predict:
	$(PYTHON) -m src.predict

analyze:
	$(PYTHON) -m src.analyze

all:
	$(PYTHON) -m src.main