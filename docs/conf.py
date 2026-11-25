"""Sphinx configuration for the inv_emulator documentation."""

project = "inv_emulator"
author = "inv_emulator developers"
copyright = "2026, inv_emulator developers"

extensions = [
    "myst_parser",           # Markdown sources (MyST)
    "sphinx.ext.mathjax",    # LaTeX math
    "sphinx.ext.githubpages",  # .nojekyll for GitHub Pages
]

source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST: $...$ / $$...$$ math, ```{note} admonitions, (label)= targets,
# GitHub-style anchors for headings up to level 4 (links like page.md#section)
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 4
myst_dmath_double_inline = True

html_theme = "furo"
html_title = "inv_emulator"
html_static_path = []
pygments_style = "default"
