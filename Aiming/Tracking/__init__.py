"""Aggregate all the tracking algorithms into one module."""

# Basic tracker: simple linear extrapolation + Hungarian algorithm on
# image plane
from .KF_tracker import KalmanTracker, KF_tracker
