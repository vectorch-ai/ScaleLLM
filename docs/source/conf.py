# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import datetime
from typing import Any, Dict

year = datetime.datetime.now().year

project = "ScaleLLM"
copyright = f"{year}, ScaleLLM Team."
author = "ScaleLLM Team"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']

exclude_patterns = []
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx_tabs.tabs",
    "sphinx_copybutton",
    "myst_parser",
]


pygments_style = "sphinx"

todo_include_todos = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"  # "sphinx_rtd_theme"
html_theme_options = {
    "source_repository": "https://github.com/vectorch-ai/ScaleLLM",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

html_theme_options["top_of_page_buttons"] = ["view"]

announcement = """
📢 ScaleLLM is distributed on <a href="https://pypi.org/project/scalellm/">PyPI</a> 🎉.
If you like this project, <a href='https://github.com/vectorch-ai/ScaleLLM' target='_blank'>give us a star ⭐️! </a>
"""
# html_theme_options["announcement"] = announcement
# html_static_path = ['_static']

# show On GitHub link in footer
html_context: Dict[str, Any] = {}
html_context["display_github"] = True
html_context["github_user"] = "vectorch-ai"
html_context["github_repo"] = "ScaleLLM"
html_context["github_version"] = "main"


# https://sphinx-copybutton.readthedocs.io/en/latest/use.html
# strip the dollar prompt when copying code
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
# honor linecontinuation characters when copying multline snippets
copybutton_line_continuation_character = "\\"
