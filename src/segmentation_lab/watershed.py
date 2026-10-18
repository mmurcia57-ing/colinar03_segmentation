"""Transformada Watershed con marcadores (Tema 10, sección 10.7), implementación propia.

La inundación sigue el algoritmo de cola de prioridad de Meyer: los píxeles se
etiquetan en orden creciente de gradiente a partir de los marcadores. Solo la
transformada de distancia euclídea usada para colocar los marcadores de fondo se
toma de SciPy (scipy.ndimage.distance_transform_edt).
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy.ndimage import distance_transform_edt

BACKGROUND, FOREGROUND = 1, 2
NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def build_markers(
    intensity: np.ndarray,
    candidate_threshold: float,
    seed_percentile: float,
    background_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construye marcadores de objeto y de fondo.

    - Candidatos a objeto: intensidad > candidate_threshold.
    - Marcador de objeto (2): intensidad >= percentil seed_percentile.
    - Mapa de distancias: distancia euclídea de cada píxel al candidato más cercano.
    - Marcador de fondo (1): píxeles a más de background_distance de cualquier candidato.
    Devuelve (marcadores, mapa_de_distancias).
    """
    values = np.asarray(intensity, dtype=np.float64)
    candidates = values > candidate_threshold
    distance = distance_transform_edt(~candidates)
    markers = np.zeros(values.shape, dtype=np.int32)
    markers[distance > background_distance] = BACKGROUND
    markers[values >= np.percentile(values, seed_percentile)] = FOREGROUND
    return markers, distance


def marker_watershed(gradient: np.ndarray, markers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inunda la imagen de gradiente desde los marcadores (vecindad 4).

    Devuelve (etiquetas, líneas): etiquetas asigna 1/2 a cada píxel y líneas marca
    los píxeles donde se encuentran cuencas distintas (diques).
    """
    relief = np.asarray(gradient, dtype=np.float64)
    labels = np.asarray(markers, dtype=np.int32).copy()
    if relief.shape != labels.shape or relief.ndim != 2:
        raise ValueError("gradiente y marcadores deben ser matrices 2-D de igual forma.")
    if not (labels > 0).any():
        raise ValueError("Se requiere al menos un marcador.")
    height, width = labels.shape
    heap: list[tuple[float, int, int, int, int]] = []
    counter = 0
    queued = labels > 0
    for y, x in zip(*np.nonzero(labels > 0)):
        for dy, dx in NEIGHBORS_4:
            ny, nx = int(y) + dy, int(x) + dx
            if 0 <= ny < height and 0 <= nx < width and not queued[ny, nx]:
                queued[ny, nx] = True
                heapq.heappush(heap, (relief[ny, nx], counter, ny, nx, int(labels[y, x])))
                counter += 1
    while heap:
        level, _, y, x, label = heapq.heappop(heap)
        labels[y, x] = label
        for dy, dx in NEIGHBORS_4:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and not queued[ny, nx]:
                queued[ny, nx] = True
                heapq.heappush(heap, (max(level, relief[ny, nx]), counter, ny, nx, label))
                counter += 1
    lines = np.zeros(labels.shape, dtype=bool)
    lines[:-1, :] |= labels[:-1, :] != labels[1:, :]
    lines[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return labels, lines
