# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are on another path,
# add these directories to sys.path here. If the directory is relative to the
# source, use os.path.abspath to make it absolute, like shown here.

import os
import sys

# Add the parent directory to the path for autodoc
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'FuzzyART'
copyright = '2024, FuzzyART Contributors'
author = 'FuzzyART Contributors'

# The full version, including alpha/beta/rc tags
release = '1.0.0'
version = '1.0.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (and in sphinx_contrib.*)

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',  # For markdown support
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that should be ignored when building
# the documentation. This supports file patterns matched by fnmatch as well as Unix
# shell-style wildcards.

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The reST default role (used for this markup: `text`) to use for all
# documents. Set to the name of a role.

default_role = None

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the themes
# documentation for a list of builtin themes.

html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ['_static']

# Theme options are theme-specific and are used to customize the look and feel of a
# specific theme. See the theme documentation for more details.

html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': 'blob',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# Custom sidebar (uncomment if you want a custom sidebar)
# html_sidebars = {}

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'fuzzyartdoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    'papersize': 'letterpaper',
    # The font size for LaTeX TeX ('10pt', '11pt', '12pt')
    'pointsize': '11pt',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target file name, toctree if True).
latex_documents = [
    ('index', 'fuzzyart.tex', 'FuzzyART Documentation',
     'FuzzyART Contributors', 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, section).
man_pages = [
    ('index', 'fuzzyart', 'FuzzyART Documentation',
     ['FuzzyART Contributors'], 1)
]

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target file name, toctree if True).
texinfo_documents = [
    ('index', 'fuzzyart', 'FuzzyART Documentation',
     'FuzzyART Contributors', 'fuzzyart', 'One line description of project.',
     'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
}

# -- Options for autodoc extension -------------------------------------------

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': False,
    'show-inheritance': True,
}

autosummary_generate = True

# -- Options for napoleon extension ------------------------------------------

# Use Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for myst_parser -------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "linkify",
    "smartquotes",
]

# -- Suppress warnings -------------------------------------------------------

# Suppress warnings that are not critical
suppress_warnings = []

# -- Build configuration -----------------------------------------------------

# If true, `todo` and `todoNote` produce output, else they are hidden.
todo_include_todos = False

# -- Setup function for additional roles and directives ----

def setup(app):
    """
    Setup function for Sphinx extensions.
    
    This is called by Sphinx during initialization. You can use it to
    register custom directives, roles, or connect to Sphinx events.
    """
    # Custom setup can be added here
    pass
