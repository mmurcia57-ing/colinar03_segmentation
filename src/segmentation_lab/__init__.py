"""Implementaciones reproducibles para el laboratorio de segmentación."""

from .algorithms import kmeans_segment, otsu_threshold
from .metrics import binary_metrics, confusion_matrix
from .region_growing import region_growing, select_seeds
from .watershed import build_markers, marker_watershed

__all__ = [
    "otsu_threshold",
    "kmeans_segment",
    "region_growing",
    "select_seeds",
    "build_markers",
    "marker_watershed",
    "binary_metrics",
    "confusion_matrix",
]
