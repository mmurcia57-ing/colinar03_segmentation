"""Algoritmos de segmentación binaria implementados sin llamadas equivalentes."""

from __future__ import annotations

import numpy as np


def _as_gray_float(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., :3].mean(axis=2)
    if array.ndim != 2:
        raise ValueError("La imagen debe ser una matriz 2-D o RGB.")
    return array.astype(np.float64, copy=False)


def otsu_threshold(image: np.ndarray, bins: int = 256) -> tuple[int, np.ndarray]:
    """Calcula umbral de Otsu recorriendo el histograma y devuelve (umbral, máscara)."""
    gray = _as_gray_float(image)
    if bins != 256:
        raise ValueError("Esta implementación espera bins=256 para intensidades uint8.")
    values = np.clip(np.rint(gray), 0, 255).astype(np.uint8)
    histogram = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    total = histogram.sum()
    if total == 0:
        raise ValueError("La imagen no contiene píxeles.")
    probabilities = histogram / total
    cumulative_probability = np.cumsum(probabilities)
    cumulative_mean = np.cumsum(probabilities * np.arange(256))
    global_mean = cumulative_mean[-1]
    denominator = cumulative_probability * (1.0 - cumulative_probability)
    between = np.zeros(256, dtype=np.float64)
    valid = denominator > 0
    between[valid] = (global_mean * cumulative_probability[valid] - cumulative_mean[valid]) ** 2 / denominator[valid]
    threshold = int(np.argmax(between))
    return threshold, values > threshold


def kmeans_segment(
    image: np.ndarray,
    k: int = 2,
    max_iter: int = 100,
    tolerance: float = 1e-4,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Segmenta por K-means 1-D propio y devuelve (centroides, máscara del grupo claro)."""
    if k != 2:
        raise ValueError("El laboratorio requiere segmentación binaria (k=2).")
    gray = _as_gray_float(image)
    pixels = gray.ravel()
    if pixels.size == 0:
        raise ValueError("La imagen no contiene píxeles.")
    rng = np.random.default_rng(seed)
    centroids = np.array([pixels.min(), pixels.max()], dtype=np.float64)
    if np.isclose(centroids[0], centroids[1]):
        centroids += np.array([-0.5, 0.5])
    for _ in range(max_iter):
        distances = np.abs(pixels[:, None] - centroids[None, :])
        labels = np.argmin(distances, axis=1)
        updated = centroids.copy()
        for cluster in range(k):
            members = pixels[labels == cluster]
            if members.size:
                updated[cluster] = members.mean()
            else:
                updated[cluster] = pixels[rng.integers(0, pixels.size)]
        if np.max(np.abs(updated - centroids)) <= tolerance:
            centroids = updated
            break
        centroids = updated
    distances = np.abs(pixels[:, None] - centroids[None, :])
    labels = np.argmin(distances, axis=1).reshape(gray.shape)
    bright_cluster = int(np.argmax(centroids))
    return centroids, labels == bright_cluster
