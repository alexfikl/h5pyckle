# https://www.sphinx-doc.org/en/master/usage/configuration.html
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
from importlib import metadata

# {{{ project information

m = metadata.metadata("h5pyckle")
project = m["Name"]
author = m["Author-email"]
copyright = f"2021 {author}"  # noqa: A001
version = m["Version"]
release = version

# }}}

# {{{ general configuration

# needed extensions
extensions = [
    "autoapi.extension",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

try:
    import sphinxcontrib.spelling  # noqa: F401

    extensions.append("sphinxcontrib.spelling")
except ImportError:
    pass

# extension for source files
source_suffix = ".rst"
# name of the main (master) document
master_doc = "index"
# min sphinx version
needs_sphinx = "6.0"
# files to ignore
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# highlighting
pygments_style = "sphinx"

# }}}

# {{{ internationalization

language = "en"

# sphinxcontrib.spelling options
spelling_lang = "en_US"
tokenizer_lang = "en_US"
spelling_word_list_filename = "wordlist_en.txt"

# }}

# {{{ output

# html
html_theme = "sphinx_book_theme"
html_title = "h5pyckle"
html_theme_options = {
    "show_toc_level": 2,
    "use_source_button": True,
    "use_repository_button": True,
    "repository_url": "https://github.com/alexfikl/h5pyckle",
    "repository_branch": "main",
    "icon_links": [
        {
            "name": "Release",
            "url": "https://github.com/alexfikl/h5pyckle/releases",
            "icon": "https://img.shields.io/github/v/release/alexfikl/h5pyckle",
            "type": "url",
        },
        {
            "name": "License",
            "url": "https://github.com/alexfikl/h5pyckle/tree/main/LICENSES",
            "icon": "https://img.shields.io/badge/License-MIT-blue.svg",
            "type": "url",
        },
        {
            "name": "CI",
            "url": "https://github.com/alexfikl/h5pyckle",
            "icon": "https://github.com/alexfikl/h5pyckle/workflows/CI/badge.svg",
            "type": "url",
        },
        {
            "name": "Issues",
            "url": "https://github.com/alexfikl/h5pyckle/issues",
            "icon": "https://img.shields.io/github/issues/alexfikl/h5pyckle",
            "type": "url",
        },
    ],
}

# }}}

# {{{ extension settings

autoapi_type = "python"
autoapi_dirs = ["."]
autoapi_keep_files = True
autoapi_add_toctree_entry = False

autoapi_python_class_content = "class"
autoapi_member_order = "bysource"
autoapi_options = [
    "show-inheritance",
]
suppress_warnings = ["autoapi"]

# }}}

# {{{ links

intersphinx_mapping = {
    "arraycontext": ("https://documen.tician.de/arraycontext", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
    "h5pyapi": ("https://api.h5py.org", None),
    "meshmode": ("https://documen.tician.de/meshmode", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pyopencl": ("https://documen.tician.de/pyopencl", None),
    "python": ("https://docs.python.org/3", None),
}

# }}}
