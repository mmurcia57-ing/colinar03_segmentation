"""Figuras de pasos intermedios y de errores, compartidas por el script, el notebook y el informe."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .pipeline import METHODS
from .watershed import BACKGROUND, FOREGROUND


def error_map(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """TN negro, FP rojo, FN azul y TP verde."""
    view = np.zeros((*truth.shape, 3), dtype=np.uint8)
    view[prediction & ~truth] = [220, 40, 40]
    view[~prediction & truth] = [40, 90, 220]
    view[prediction & truth] = [40, 170, 70]
    return view


def _clean(axis, title: str) -> None:
    axis.set_title(title, fontsize=9, fontweight="bold")
    axis.set_xticks([])
    axis.set_yticks([])


def preprocessing_figure(steps: dict[str, np.ndarray], labels: list[tuple[str, str]]):
    """Muestra en fila los pasos de preprocesamiento indicados como (clave, título)."""
    figure, axes = plt.subplots(1, len(labels), figsize=(3.6 * len(labels), 3.2), dpi=130)
    for axis, (key, title) in zip(np.atleast_1d(axes), labels):
        axis.imshow(steps[key], cmap="gray")
        _clean(axis, title)
    figure.tight_layout()
    return figure


def thresholds_figure(intensity: np.ndarray, outputs: dict[str, dict]):
    """Histograma con el umbral de Otsu y la frontera de K-means (punto medio de centroides)."""
    threshold = outputs["Otsu"]["threshold"]
    centroids = np.sort(outputs["K-means"]["centroids"])
    boundary = centroids.mean()
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.2), dpi=130)
    axes[0].bar(np.arange(256), np.bincount(intensity.ravel(), minlength=256), width=1.0, color="#6b7c86")
    axes[0].axvline(threshold, color="#c0392b", lw=2, label=f"Otsu t={threshold}")
    axes[0].axvline(boundary, color="#2471a3", lw=2, ls="--", label=f"K-means frontera={boundary:.1f}")
    for centroid in centroids:
        axes[0].axvline(centroid, color="#2471a3", lw=1, ls=":")
    axes[0].set_title("Histograma de la entrada", fontsize=9, fontweight="bold")
    axes[0].legend(fontsize=7)
    axes[0].set_yscale("log")
    axes[1].imshow(outputs["Otsu"]["mask"], cmap="gray")
    _clean(axes[1], "Otsu: máscara")
    axes[2].imshow(outputs["K-means"]["mask"], cmap="gray")
    _clean(axes[2], "K-means: máscara")
    figure.tight_layout()
    return figure


def region_growing_figure(intensity: np.ndarray, outputs: dict[str, dict]):
    """Semillas, mapa de orden de crecimiento, instantáneas de la evolución y máscara final."""
    data = outputs["Region Growing"]
    order = data["growth_order"]
    final_step = int(order.max())
    snapshots = [max(1, final_step // 10), max(1, final_step // 3)]
    figure, axes = plt.subplots(1, 5, figsize=(17, 3.3), dpi=130)
    overlay = np.dstack([intensity] * 3).astype(np.uint8)
    seed_view = overlay.copy()
    seed_view[data["seeds"]] = [255, 210, 0]
    axes[0].imshow(seed_view)
    _clean(axes[0], f"Semillas ({int(data['seeds'].sum())} px)")
    for axis, step in zip(axes[1:3], snapshots):
        view = overlay.copy()
        view[(order >= 0) & (order <= step)] = [230, 60, 60]
        view[data["seeds"]] = [255, 210, 0]
        axis.imshow(view)
        _clean(axis, f"Región tras {step} iteraciones")
    grown = np.where(order >= 0, order, np.nan)
    axes[3].imshow(intensity, cmap="gray")
    axes[3].imshow(grown, cmap="viridis")
    _clean(axes[3], f"Orden de crecimiento (0–{final_step})")
    axes[4].imshow(data["mask"], cmap="gray")
    _clean(axes[4], "Máscara final")
    figure.tight_layout()
    return figure


def watershed_figure(intensity: np.ndarray, outputs: dict[str, dict]):
    """Gradiente, mapa de distancias, marcadores, líneas de cuenca y máscara final."""
    data = outputs["Watershed"]
    figure, axes = plt.subplots(1, 5, figsize=(17, 3.3), dpi=130)
    axes[0].imshow(data["gradient"], cmap="magma")
    _clean(axes[0], "Gradiente de Sobel (relieve)")
    axes[1].imshow(data["distance"], cmap="viridis")
    _clean(axes[1], "Mapa de distancias a candidatos")
    marker_view = np.dstack([intensity] * 3).astype(np.uint8) // 2
    marker_view[data["markers"] == BACKGROUND] = [40, 90, 220]
    marker_view[data["markers"] == FOREGROUND] = [255, 210, 0]
    axes[2].imshow(marker_view)
    _clean(axes[2], "Marcadores: objeto (amarillo) / fondo (azul)")
    line_view = np.dstack([intensity] * 3).astype(np.uint8)
    line_view[data["lines"]] = [255, 40, 40]
    axes[3].imshow(line_view)
    _clean(axes[3], "Líneas de cuenca (diques)")
    axes[4].imshow(data["mask"], cmap="gray")
    _clean(axes[4], "Máscara final")
    figure.tight_layout()
    return figure


def errors_figure(truth: np.ndarray, predictions: dict[str, np.ndarray], title: str = ""):
    """Referencia y mapa FP/FN/TP de cada método para una imagen."""
    figure, axes = plt.subplots(1, len(METHODS) + 1, figsize=(3.4 * (len(METHODS) + 1), 3.0), dpi=130)
    axes[0].imshow(truth, cmap="gray")
    _clean(axes[0], "Ground truth")
    for axis, method in zip(axes[1:], METHODS):
        axis.imshow(error_map(truth, predictions[method]))
        _clean(axis, f"{method}: FP rojo / FN azul / TP verde")
    if title:
        figure.suptitle(title, fontsize=11, fontweight="bold")
    figure.tight_layout()
    return figure


def overview_figure(records: list[dict], heading: str):
    """Rejilla: original, referencia y mapa de errores de cada método por imagen."""
    columns = 2 + len(METHODS)
    figure, axes = plt.subplots(len(records), columns, figsize=(2.9 * columns, 2.25 * len(records)), dpi=170)
    for row, record in enumerate(records):
        panels = [(record["image"], "Original", None), (record["truth"], "Ground truth", "gray")]
        panels += [(error_map(record["truth"], record["masks"][method]), method, None) for method in METHODS]
        for column, (content, title, cmap) in enumerate(panels):
            axis = axes[row, column]
            axis.imshow(content, cmap=cmap)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(title, fontsize=11, fontweight="bold")
            if column == 0:
                axis.set_ylabel(record["id"], fontsize=10)
    figure.suptitle(heading, fontsize=13, fontweight="bold")
    figure.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.4, w_pad=0.3)
    return figure


def steps_figure(steps: dict[str, np.ndarray], outputs: dict[str, dict], truth: np.ndarray, identifier: str):
    """Figura compacta para el informe: preprocesamiento y pasos clave de RG y Watershed."""
    intensity = steps["input"]
    rg = outputs["Region Growing"]
    ws = outputs["Watershed"]
    figure, axes = plt.subplots(2, 4, figsize=(15, 6.2), dpi=170)
    axes[0, 0].imshow(steps["gray"], cmap="gray")
    _clean(axes[0, 0], f"CFD {identifier}: gris original")
    axes[0, 1].imshow(intensity, cmap="gray")
    _clean(axes[0, 1], "Entrada común (suavizado, invertido, sin fondo)")
    seed_view = np.dstack([intensity] * 3).astype(np.uint8)
    seed_view[rg["seeds"]] = [255, 210, 0]
    axes[0, 2].imshow(seed_view)
    _clean(axes[0, 2], "Region Growing: semillas")
    axes[0, 3].imshow(intensity, cmap="gray")
    axes[0, 3].imshow(np.where(rg["growth_order"] >= 0, rg["growth_order"], np.nan), cmap="viridis")
    _clean(axes[0, 3], "Region Growing: orden de crecimiento")
    axes[1, 0].imshow(ws["gradient"], cmap="magma")
    _clean(axes[1, 0], "Watershed: gradiente de Sobel")
    axes[1, 1].imshow(ws["distance"], cmap="viridis")
    _clean(axes[1, 1], "Watershed: mapa de distancias")
    marker_view = np.dstack([intensity] * 3).astype(np.uint8) // 2
    marker_view[ws["markers"] == BACKGROUND] = [40, 90, 220]
    marker_view[ws["markers"] == FOREGROUND] = [255, 210, 0]
    axes[1, 2].imshow(marker_view)
    _clean(axes[1, 2], "Watershed: marcadores")
    line_view = np.dstack([intensity] * 3).astype(np.uint8)
    line_view[ws["lines"]] = [255, 40, 40]
    axes[1, 3].imshow(line_view)
    _clean(axes[1, 3], "Watershed: líneas de cuenca")
    figure.tight_layout()
    return figure
