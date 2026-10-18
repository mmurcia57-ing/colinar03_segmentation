"""Crecimiento de regiones basado en semillas (Tema 10, sección 10.4), implementación propia."""
from __future__ import annotations

from collections import deque

import numpy as np

NEIGHBORS_8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def select_seeds(intensity: np.ndarray, percentile: float) -> np.ndarray:
    """Semillas automáticas: píxeles con intensidad >= percentil dado (objetivo = claro)."""
    return np.asarray(intensity) >= np.percentile(intensity, percentile)


def region_growing(
    intensity: np.ndarray,
    seeds: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Hace crecer una región desde las semillas con vecindad 8.

    Función de similitud: un vecino p se agrega si |I(p) - mu_R| <= tolerance, donde
    mu_R es la media actual de la región (se actualiza en cada inserción).
    Devuelve (máscara, orden) donde orden[p] es la iteración BFS en que p entró a la
    región (0 = semilla, -1 = nunca), útil para mostrar la evolución del crecimiento.
    """
    values = np.asarray(intensity, dtype=np.float64)
    seed_mask = np.asarray(seeds, dtype=bool)
    if values.shape != seed_mask.shape or values.ndim != 2:
        raise ValueError("intensidad y semillas deben ser matrices 2-D de igual forma.")
    if not seed_mask.any():
        raise ValueError("Se requiere al menos una semilla.")
    height, width = values.shape
    order = np.full(values.shape, -1, dtype=np.int32)
    order[seed_mask] = 0
    total = float(values[seed_mask].sum())
    count = int(seed_mask.sum())
    queue = deque((int(y), int(x)) for y, x in zip(*np.nonzero(seed_mask)))
    while queue:
        y, x = queue.popleft()
        step = order[y, x] + 1
        for dy, dx in NEIGHBORS_8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and order[ny, nx] < 0:
                if abs(values[ny, nx] - total / count) <= tolerance:
                    order[ny, nx] = step
                    total += values[ny, nx]
                    count += 1
                    queue.append((ny, nx))
    return order >= 0, order
