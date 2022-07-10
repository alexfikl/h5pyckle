PYTHON?=python
PYTEST_ADDOPTS?=

all: flake8 pylint mypy-strict

black:
	$(PYTHON) -m black h5pyckle tests examples

flake8:
	$(PYTHON) -m flake8 h5pyckle tests examples docs
	@echo -e "\e[1;32mflake8 clean!\e[0m"

pylint:
	PYTHONWARNINGS=ignore $(PYTHON) -m pylint h5pyckle tests/*.py examples/*.py
	@echo -e "\e[1;32mpylint clean!\e[0m"

mypy:
	$(PYTHON) -m mypy --show-error-codes h5pyckle tests examples
	@echo -e "\e[1;32mmypy clean!\e[0m"

mypy-strict:
	$(PYTHON) -m mypy --strict --show-error-codes h5pyckle tests examples
	@echo -e "\e[1;32mmypy clean!\e[0m"

test:
	$(PYTHON) -m pytest -rswx --durations=25 -v -s $(PYTEST_ADDOPTS)

pip-install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e '.[dev,unittest]'

tags:
	ctags -R

.PHONY: all black flake8 mypy pip-install pylint test
