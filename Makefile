PYTHON?=python -X dev
PYTEST_ADDOPTS?=

all: flake8 pylint mypy-strict

# {{{ linting

black:
	$(PYTHON) -m black --safe --target-version py38 h5pyckle tests examples

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

pyright:
	pyright --stats h5pyckle tests examples

codespell:
	@codespell --summary \
		--skip _build \
		--ignore-words .codespell-ignore \
		h5pyckle tests examples docs

reuse:
	@reuse lint
	@echo -e "\e[1;32mREUSE compliant!\e[0m"

# }}}

# {{{ testing

pin:
	$(PYTHON) -m piptools compile \
		--resolver=backtracking \
		--extra dev --extra fancy --upgrade \
		-o requirements.txt setup.cfg

pip-install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt -e .

test:
	$(PYTHON) -m pytest -rswx --durations=25 -v -s $(PYTEST_ADDOPTS)

run-examples:
	@for ex in $$(find examples -name "*.py"); do \
		echo -e "\x1b[1;32m===> \x1b[97mRunning $${ex}\x1b[0m"; \
		$(PYTHON) "$${ex}"; \
		sleep 1; \
	done

# }}}

ctags:
	ctags --recurse=yes \
		--tag-relative=yes \
		--exclude=.git \
		--exclude=docs \
		--python-kinds=-i \
		--language-force=python

.PHONY: black flake8 pylint mypy mypy-strict pyright reuse pin pip-install test run-examples ctags
