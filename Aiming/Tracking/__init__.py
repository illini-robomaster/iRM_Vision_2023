"""Aggregate all the tracking algorithms into one module."""

# Basic tracker: simple linear extrapolation + Hungarian algorithm on
# image plane
from .basic_tracker import basic_tracker
