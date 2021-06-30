# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# print(sys.path)
# sys.path.insert(0, os.path.abspath('../pulse'))


import shutil
import sys

# -- Project information -----------------------------------------------------
from pathlib import Path
from textwrap import dedent
from unittest import mock

import sphinx_rtd_theme  # noqa: E401

sys.modules["dolfin"] = mock.MagicMock()

import pulse  # noqa: E402

HERE = Path(__file__).absolute().parent

project = "pulse"
copyright = "2020, Henrik Finsberg"
author = "Henrik Finsberg"


# The short X.Y version
version = pulse.__version__
# The full version, including alpha/beta/rc tags
release = pulse.__version__


demo_dir = HERE.joinpath("../../demo")

demoes = [
    demo_dir.joinpath("benchmark").joinpath("problem1").joinpath("problem1.ipynb"),
    demo_dir.joinpath("benchmark").joinpath("problem2").joinpath("problem2.ipynb"),
    demo_dir.joinpath("benchmark").joinpath("problem3").joinpath("problem3.ipynb"),
    demo_dir.joinpath("biaxial_stress_test").joinpath("biaxial_stress_test.ipynb"),
    demo_dir.joinpath("shear_experiment").joinpath("shear_experiment.ipynb"),
    demo_dir.joinpath("compressible_model").joinpath("compressible_model.ipynb"),
    demo_dir.joinpath("compute_stress_strain").joinpath("compute_stress_strain.ipynb"),
    demo_dir.joinpath("from_xml").joinpath("from_xml.ipynb"),
    demo_dir.joinpath("klotz_curve").joinpath("klotz_curve.ipynb"),
    demo_dir.joinpath("simple_ellipsoid").joinpath("simple_ellipsoid.ipynb"),
    demo_dir.joinpath("unit_cube").joinpath("unit_cube_demo.ipynb"),
    demo_dir.joinpath("unloading").joinpath("demo_fixedpointunloader.ipynb"),
    demo_dir.joinpath("rigid_motion").joinpath("rigid_motion.ipynb"),
]


demo_docs = HERE.joinpath("demos")
demo_docs.mkdir(exist_ok=True, parents=True)

for f in demo_docs.iterdir():
    if f.suffix == ".ipynb":
        f.unlink()

for notebook in demoes:
    src = notebook
    dst = notebook.name

    shutil.copy2(src, demo_docs.joinpath(dst))

with open(demo_docs.joinpath("demos.rst"), "w+") as f:
    f.write(
        dedent(
            """
    .. _demos


    Demos
    =====

    Here you will find all the demos. These are all found in the main
    repository in the `demo folder <https://github.com/finsberg/pulse/tree/master/demo>`_

    .. toctree::
       :titlesonly:
       :maxdepth: 1

    """
        )
    )
    for i in demoes:
        f.write("   " + i.stem + "\n")


# for demo in demoes:


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.graphviz",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx_gallery.load_style",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.graphviz",  # Dependency diagrams
    # "myst_parser",
]

# # Hoverxref Extension
# hoverxref_auto_ref = True
# hoverxref_mathjax = True
# hoverxref_domains = ["py"]
# hoverxref_role_types = {
#     "hoverxref": "modal",
#     "ref": "modal",  # for hoverxref_auto_ref config
#     "confval": "tooltip",  # for custom object
#     "mod": "tooltip",  # for Python Sphinx Domain
#     "class": "tooltip",  # for Python Sphinx Domain
#     "meth": "tooltip",
#     "obj": "tooltip",
# }

try:
    import matplotlib.sphinxext.plot_directive  # noqa: F401

    extensions.append("matplotlib.sphinxext.plot_directive")
except ImportError:
    pass

# https://stackoverflow.com/questions/46269345/embed-plotly-graph-in-a-sphinx-doc


# Enable plotly figure in the docs
# nbsphinx_prolog = r"""
# .. raw:: html

#     <script src='https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js'></script>
#     <script>require=requirejs;</script>
#     <script src="https://cdn.plot.ly/plotly-1.2.0.min.js"></script>

# """
nbsphinx_timeout = -1


# nbsphinx_execute = "always"
nbsphinx_execute = "never"
nbsphinx_allow_errors = True


def setup(app):
    # https://docs.readthedocs.io/en/latest/guides/adding-custom-css.html
    # https://www.sphinx-doc.org/en/master/extdev/appapi.html#sphinx.application.Sphinx.add_js_file
    app.add_js_file(
        "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"
    )


autosummary_generate = True
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "dolfin": ("https://fenicsproject.org/olddocs/dolfin/latest/python", None),
    "ufl": ("https://fenics.readthedocs.io/projects/ufl/en/latest/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org", None),
}
inheritance_node_attrs = dict(
    shape="ellipse", fontsize=12, color="orange", style="filled"
)

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "dist", "**.ipynb_checkpoints"]


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

todo_include_todos = False
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "pulsedoc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "pulse.tex", "pulse Documentation", "Henrik Finsberg", "manual"),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "pulse", "pulse Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "pulse",
        "pulse Documentation",
        author,
        "pulse",
        "One line description of project.",
        "Miscellaneous",
    ),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]


# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {"https://docs.python.org/": None}
