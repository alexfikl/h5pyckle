PYTHON := 'python -X dev'

_default:
    @just --list

# {{{ formatting

alias fmt: format

[doc('Reformat all source code')]
format: isort black pyproject justfmt

[doc('Run ruff isort fixes over the source code')]
isort:
    ruff check --fix --select=I src tests examples docs
    ruff check --fix --select=RUF022 src
    @echo -e "\e[1;32mruff isort clean!\e[0m"

[doc('Run ruff format over the source code')]
black:
    ruff format src tests examples docs
    @echo -e "\e[1;32mruff format clean!\e[0m"

[doc('Run pyproject-fmt over the configuration')]
pyproject:
    {{ PYTHON }} -m pyproject_fmt --indent 4 pyproject.toml
    @echo -e "\e[1;32mpyproject clean!\e[0m"

[doc('Run just --fmt over the justfile')]
justfmt:
    just --unstable --fmt
    just -f docs/justfile --unstable --fmt
    @echo -e "\e[1;32mjust --fmt clean!\e[0m"

# }}}
# {{{ linting

[doc('Run all linting checks over the source code')]
lint: typos reuse ruff pyright

[doc('Run typos over the source code and documentation')]
typos:
    typos --sort
    @echo -e "\e[1;32mtypos clean!\e[0m"

[doc('Check REUSE license compliance')]
reuse:
    {{ PYTHON }} -m reuse lint
    @echo -e "\e[1;32mREUSE compliant!\e[0m"

[doc('Run ruff checks over the source code')]
ruff:
    ruff check src tests examples docs
    @echo -e "\e[1;32mruff clean!\e[0m"

[doc("Run pyright checks over the source code")]
pyright:
    basedpyright src tests examples
    @echo -e "\e[1;32mpyright clean!\e[0m"

# }}}
# {{{ pin

[private]
requirements_meshmode_txt:
    uv pip compile --upgrade --universal --python-version "3.10" \
        -o .ci/requirements-meshmode.txt .ci/requirements-meshmode.in

[private]
requirements_build_txt:
    uv pip compile --upgrade --universal --python-version "3.10" \
        -o .ci/requirements-build.txt .ci/requirements-build.in

[private]
requirements_test_txt:
    uv pip compile --upgrade --universal --python-version '3.10' \
        --group test \
        -o .ci/requirements-test.txt pyproject.toml

[private]
requirements_txt:
    uv pip compile --upgrade --universal --python-version '3.10' \
        -o requirements.txt pyproject.toml

[doc('Pin dependency versions to requirements.txt')]
pin: requirements_txt requirements_test_txt requirements_build_txt requirements_meshmode_txt

# }}}
# {{{ develop

[private]
pyright-ignore-meshmode:
    #!/usr/bin/env bash
    cat << 'EOF' | sed 's/^    //' >> pyproject.toml

        ignore = [
            'tests/test_meshmode.py',
            'src/h5pyckle/interop_meshmode.py',
        ]
    EOF

[doc('Install project in editable mode')]
develop: clean
    {{ PYTHON }} -m pip install \
        --verbose \
        --no-build-isolation \
        --editable .

[doc("Editable install using pinned dependencies from requirements-test.txt")]
pip-install:
    {{ PYTHON }} -m pip install --requirement .ci/requirements-build.txt
    {{ PYTHON }} -m pip install \
        --verbose \
        --requirement .ci/requirements-test.txt \
        --no-build-isolation \
        --editable .

[doc("Remove various build artifacts")]
clean:
    rm -rf build dist
    rm -rf docs/build.sphinx

[doc("Remove various temporary files")]
purge: clean
    rm -rf .ruff_cache .pytest_cache tags

[doc("Regenerate ctags")]
ctags:
    ctags --recurse=yes \
        --tag-relative=yes \
        --exclude=.git \
        --exclude=docs \
        --python-kinds=-i \
        --language-force=python

# }}}
# {{{ tests

[doc("Run pytest tests")]
test *PYTEST_ADDOPTS:
    {{ PYTHON }} -m pytest \
        --junit-xml=pytest-results.xml \
        -rswx --durations=25 -v -s \
        {{ PYTEST_ADDOPTS }}

[doc("Run examples with default options")]
examples:
    for ex in `find examples -name "*.py"`; do \
        echo "::group::Running ${ex}"; \
        {{ PYTHON }} "${ex}"; \
        echo "::endgroup::"; \
    done

# }}}
