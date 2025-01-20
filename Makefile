PYTHON?=python -X dev

all: help

help: 			## Show this help
	@echo -e "Specify a command. The choices are:\n"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[0;36m%-12s\033[m %s\n", $$1, $$2}'
	@echo ""
.PHONY: help

# {{{ linting

format: black isort pyproject					## Run all formatting scripts
.PHONY: format

fmt: format
.PHONY: fmt

pyproject:		## Run pyproject-fmt over the configuration
	$(PYTHON) -m pyproject_fmt --indent 4 pyproject.toml
	@echo -e "\e[1;32mpyproject clean!\e[0m"
.PHONY: pyproject

black:			## Run ruff format over the source code
	ruff format src tests examples docs
	@echo -e "\e[1;32mruff format clean!\e[0m"
.PHONY: black

isort:			## Run ruff isort fixes over the source code
	ruff check --fix --select=I src tests examples docs
	ruff check --fix --select=RUF022 src
	@echo -e "\e[1;32mruff isort clean!\e[0m"
.PHONY: isort

lint: ruff mypy doc8 typos reuse 				## Run linting checks
.PHONY: lint

ruff:			## Run ruff checks over the source code
	ruff check src tests examples docs
	@echo -e "\e[1;32mruff lint clean!\e[0m"
.PHONY: ruff

mypy:			## Run mypy checks over the source code
	$(PYTHON) -m mypy src tests examples
	@echo -e "\e[1;32mmypy clean!\e[0m"
.PHONY: mypy

doc8:			## Run doc8 checks over the source code
	$(PYTHON) -m doc8 src docs
	@echo -e "\e[1;32mdoc8 clean!\e[0m"
.PHONY: doc8

typos:			## Run typos over the source code and documentation
	typos
	@echo -e "\e[1;32mtypos clean!\e[0m"
.PHONY: typos

reuse:			## Check REUSE license compliance
	$(PYTHON) -m reuse lint
	@echo -e "\e[1;32mREUSE compliant!\e[0m"
.PHONY: reuse

# }}}

# {{{ testing

REQUIREMENTS=\
	requirements-test.txt \
	requirements.txt

requirements-test.txt: pyproject.toml
	uv pip compile --upgrade --universal --python-version '3.10' \
		--extra test \
		-o $@ $<
.PHONY: requirements-test.txt

requirements.txt: pyproject.toml
	uv pip compile --upgrade --universal --python-version '3.10' \
		-o $@ $<
.PHONY: requirements.txt

pin: $(REQUIREMENTS)	## Pin dependency versions to requirements.txt
.PHONY: pin

pip-install:	## Install pinned dependencies from requirements.txt
	$(PYTHON) -m pip install --upgrade pip hatchling wheel
	$(PYTHON) -m pip install -r requirements-test.txt -e .
.PHONY: pip-install

test:			## Run pytest tests
	$(PYTHON) -m pytest -rswx --durations=25 -v -s
.PHONY: test

run-examples:	## Run examples with default options
	@for ex in $$(find examples -name "*.py"); do \
		echo -e "\x1b[1;32m===> \x1b[97mRunning $${ex}\x1b[0m"; \
		$(PYTHON) "$${ex}"; \
		sleep 1; \
	done
.PHONY: run-examples

# }}}

ctags:			## Regenerate ctags
	ctags --recurse=yes \
		--tag-relative=yes \
		--exclude=.git \
		--exclude=docs \
		--python-kinds=-i \
		--language-force=python
.PHONY: ctags

clean:			## Remove various build artifacts
	rm -rf build dist
	rm -rf docs/_build
.PHONY: clean

purge: clean	## Remove various temporary files
	rm -rf .ruff_cache .pytest_cache .mypy_cache
.PHONY: purge
