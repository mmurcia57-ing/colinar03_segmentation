"""Implementaciones reproducibles para el laboratorio de segmentación."""

from .algorithms import kmeans_segment, otsu_threshold
from .metrics import binary_metrics, confusion_matrix

__all__ = ["otsu_threshold", "kmeans_segment", "binary_metrics", "confusion_matrix"]
