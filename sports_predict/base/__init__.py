"""Base abstract classes for the sports prediction framework."""

from .feature_source import FeatureSource, StatefulFeatureSource
from .feature_pipeline import BaseFeaturePipeline
from .pipeline import BaseSportsPipeline
from .rating import GenericEloSystem
from .ensemble import GenericEnsemble

__all__ = [
    "FeatureSource",
    "StatefulFeatureSource",
    "BaseFeaturePipeline",
    "BaseSportsPipeline",
    "GenericEloSystem",
    "GenericEnsemble",
]
