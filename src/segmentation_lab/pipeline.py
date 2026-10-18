"""Pipelines fijos por escenario: preprocesamiento común y ejecución de los cuatro segmentadores.

Todos los métodos de un escenario reciben exactamente la misma imagen preprocesada y
usan parámetros fijos para todas sus imágenes (sin ajuste por imagen). En ambos
escenarios el objeto a segmentar corresponde a intensidades altas de la entrada.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .algorithms import kmeans_segment, otsu_threshold
from .preprocessing import gaussian_blur, sobel_magnitude, subtract_background, to_gray, to_uint8
from .region_growing import region_growing, select_seeds
from .watershed import FOREGROUND, build_markers, marker_watershed

METHODS = ("Otsu", "K-means", "Region Growing", "Watershed")

CFD_PARAMS = {
    "smoothing_sigma": 1.5,
    "background_sigma": 15.0,
    "contrast_gain": 4.0,
    "kmeans_seed": 7,
    "rg_seed_percentile": 99.8,
    "rg_tolerance": 30.0,
    "ws_seed_percentile": 99.5,
    "ws_background_distance": 3.0,
}

CANDELARIA_PARAMS = {
    "working_max_side": 1024,
    "smoothing_sigma": 1.5,
    "kmeans_seed": 7,
    "rg_seed_percentile": 70.0,
    "rg_tolerance": 25.0,
    "ws_seed_percentile": 80.0,
    "ws_background_distance": 10.0,
}


def preprocess_cfd(image: np.ndarray, params: dict = CFD_PARAMS) -> dict[str, np.ndarray]:
    """Gris -> suavizado gaussiano -> inversión (grieta oscura pasa a ser clara) -> resta de fondo."""
    gray = to_gray(image)
    smoothed = gaussian_blur(gray, params["smoothing_sigma"])
    inverted = 255.0 - smoothed
    corrected = to_uint8(subtract_background(inverted, params["background_sigma"], params["contrast_gain"]))
    return {"gray": gray, "smoothed": smoothed, "inverted": inverted, "input": corrected}


def resize_to_working(image: np.ndarray, max_side: int) -> np.ndarray:
    """Reduce la imagen para que su lado mayor sea max_side (interpolación bilineal)."""
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    scale = min(1.0, max_side / max(pil.size))
    if scale == 1.0:
        return np.asarray(pil)
    size = (round(pil.size[0] * scale), round(pil.size[1] * scale))
    return np.asarray(pil.resize(size, Image.BILINEAR))


def upscale_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Devuelve la máscara a la resolución original por vecino más cercano."""
    if mask.shape == shape:
        return mask
    pil = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(pil.resize((shape[1], shape[0]), Image.NEAREST)) > 0


def preprocess_candelaria(image: np.ndarray, params: dict = CANDELARIA_PARAMS) -> dict[str, np.ndarray]:
    """Reducción a resolución de trabajo -> gris -> suavizado gaussiano."""
    working = resize_to_working(image, params["working_max_side"])
    gray = to_gray(working)
    smoothed = gaussian_blur(gray, params["smoothing_sigma"])
    return {"working_rgb": working, "gray": gray, "smoothed": smoothed, "input": to_uint8(smoothed)}


def run_methods(intensity: np.ndarray, params: dict) -> dict[str, dict]:
    """Ejecuta Otsu, K-means, Region Growing y Watershed sobre la misma entrada uint8.

    Para cada método devuelve la máscara, una descripción de parámetros y los
    resultados intermedios necesarios para visualizar los pasos principales.
    """
    threshold, otsu_mask = otsu_threshold(intensity)
    centroids, kmeans_mask = kmeans_segment(intensity, seed=params["kmeans_seed"])

    seeds = select_seeds(intensity, params["rg_seed_percentile"])
    rg_mask, growth_order = region_growing(intensity, seeds, params["rg_tolerance"])

    gradient = sobel_magnitude(intensity)
    markers, distance = build_markers(
        intensity, threshold, params["ws_seed_percentile"], params["ws_background_distance"]
    )
    labels, lines = marker_watershed(gradient, markers)

    return {
        "Otsu": {
            "mask": otsu_mask,
            "parameter": f"threshold={threshold}; bins=256",
            "threshold": threshold,
        },
        "K-means": {
            "mask": kmeans_mask,
            "parameter": f"k=2; seed={params['kmeans_seed']}; max_iter=100; tolerance=1e-4; "
            f"centroids={np.round(centroids, 2).tolist()}",
            "centroids": centroids,
        },
        "Region Growing": {
            "mask": rg_mask,
            "parameter": f"seeds>=P{params['rg_seed_percentile']:g} ({int(seeds.sum())} px); "
            f"|I-mu_R|<={params['rg_tolerance']:g}; 8-neighborhood",
            "seeds": seeds,
            "growth_order": growth_order,
        },
        "Watershed": {
            "mask": labels == FOREGROUND,
            "parameter": f"object markers>=P{params['ws_seed_percentile']:g}; background markers: "
            f"distance>{params['ws_background_distance']:g} px from I>{threshold}; Sobel relief; 4-neighborhood",
            "gradient": gradient,
            "distance": distance,
            "markers": markers,
            "lines": lines,
        },
    }
