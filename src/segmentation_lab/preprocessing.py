"""Preprocesamiento común: escala de grises, suavizado gaussiano y gradiente de Sobel."""
from __future__ import annotations

import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convierte RGB a la media aritmética de sus canales, en float64."""
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 3:
        array = array[..., :3].mean(axis=2)
    if array.ndim != 2:
        raise ValueError("La imagen debe ser una matriz 2-D o RGB.")
    return array


def gaussian_kernel(sigma: float) -> np.ndarray:
    """Núcleo gaussiano 1-D normalizado con radio ceil(3*sigma)."""
    if sigma <= 0:
        raise ValueError("sigma debe ser positivo.")
    radius = int(np.ceil(3 * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _convolve_axis(array: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = kernel.size // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (radius, radius)
    padded = np.pad(array, pad, mode="reflect")
    output = np.zeros_like(array, dtype=np.float64)
    length = array.shape[axis]
    for index, weight in enumerate(kernel):
        window = [slice(None), slice(None)]
        window[axis] = slice(index, index + length)
        output += weight * padded[tuple(window)]
    return output


def gaussian_blur(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Suavizado gaussiano separable (filas y columnas) con borde reflejado."""
    kernel = gaussian_kernel(sigma)
    return _convolve_axis(_convolve_axis(np.asarray(gray, dtype=np.float64), kernel, 0), kernel, 1)


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Magnitud del gradiente de Sobel, sqrt(Gx^2 + Gy^2)."""
    padded = np.pad(np.asarray(gray, dtype=np.float64), 1, mode="reflect")
    height, width = gray.shape

    def shifted(dy: int, dx: int) -> np.ndarray:
        return padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]

    gx = (shifted(-1, 1) + 2 * shifted(0, 1) + shifted(1, 1)) - (shifted(-1, -1) + 2 * shifted(0, -1) + shifted(1, -1))
    gy = (shifted(1, -1) + 2 * shifted(1, 0) + shifted(1, 1)) - (shifted(-1, -1) + 2 * shifted(-1, 0) + shifted(-1, 1))
    return np.hypot(gx, gy)


def subtract_background(intensity: np.ndarray, sigma: float, gain: float, offset: float = 128.0) -> np.ndarray:
    """Corrige iluminación no uniforme: I' = gain * (I - G_sigma * I) + offset.

    El fondo se estima con un gaussiano ancho; las estructuras finas (grietas)
    sobreviven a la resta mientras que los cambios lentos de iluminación se anulan.
    """
    values = np.asarray(intensity, dtype=np.float64)
    return gain * (values - gaussian_blur(values, sigma)) + offset


def to_uint8(array: np.ndarray) -> np.ndarray:
    """Redondea y recorta al rango [0, 255]."""
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)
