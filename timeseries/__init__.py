"""
The :mod:`timeseries` module includes learning methods wrappers for timeseries.
"""

from .timeseriesclassifier import TimeSeriesClassifier
from .timeseriesregressor import TimeSeriesRegressor

__all__ = [ "TimeSeriesClassifier", "TimeSeriesRegressor"]