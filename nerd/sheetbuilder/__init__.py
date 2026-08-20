"""Build nerd `create` inputs from sample names and lab metadata.

This package is pure Python with no web dependency: it holds the sample
sheet model, the naming-pattern engine, the column fillers, the entity
catalog, validation, and the exporters. `nerd.webui` is one frontend over
it; a headless CLI path can reuse the same pieces.
"""
