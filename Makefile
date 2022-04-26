.PHONY: lint


PYTHON = python3
LINTER = flake8


lint:
	$(PYTHON) -m $(LINTER) *.py
