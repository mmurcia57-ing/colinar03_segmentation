"""Evaluación canónica de los cuatro segmentadores sobre CFD (principal) y La Candelaria (secundario).

Otsu y K-means son líneas base; Region Growing y Watershed con marcadores son los
métodos del Tema 10. Las métricas se calculan con el evaluador propio de
segmentation_lab.metrics. El notebook y el informe consumen deliverables/results.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from segmentation_lab import binary_metrics, confusion_matrix, otsu_threshold  # noqa: E402
from segmentation_lab.pipeline import (  # noqa: E402
    CANDELARIA_PARAMS,
    CFD_PARAMS,
    METHODS,
    preprocess_candelaria,
    preprocess_cfd,
    run_methods,
    upscale_mask,
)
from segmentation_lab.region_growing import region_growing, select_seeds  # noqa: E402
from segmentation_lab.preprocessing import sobel_magnitude  # noqa: E402
from segmentation_lab.watershed import FOREGROUND, build_markers, marker_watershed  # noqa: E402
from segmentation_lab import visualization  # noqa: E402

TITLE = (
    "Evaluación de segmentadores por umbralización y crecimiento de regiones "
    "frente a ground truth: CrackForest Dataset y escenas urbanas de La Candelaria"
)
REAL_IMAGES = ROOT / "data" / "real" / "images"
REAL_MASKS = ROOT / "data" / "real" / "masks"
CFD_ROOT = ROOT / "data" / "benchmarks" / "cfd5"
FIGURES = ROOT / "deliverables" / "figures"
RESULTS_PATH = ROOT / "deliverables" / "results.json"
CFD_RESULTS_PATH = ROOT / "deliverables" / "cfd_results.json"
STEPS_IMAGE_ID = "001"
RG_TOLERANCES = (20.0, 25.0, 30.0, 35.0, 40.0)
WS_DISTANCES = (3.0, 5.0, 8.0)


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _summarize(records: list[dict]) -> dict:
    """Métricas por imagen, matrices, promedio por imagen y agregado ponderado por píxel."""
    rows: list[dict] = []
    matrices: dict[str, dict[str, list[list[int]]]] = {}
    for record in records:
        matrices[record["id"]] = {}
        for method in METHODS:
            prediction = record["masks"][method]
            matrices[record["id"]][method] = confusion_matrix(record["truth"], prediction).tolist()
            rows.append(
                {
                    "image": record["id"],
                    "method": method,
                    "parameter": record["parameters"][method],
                    **binary_metrics(record["truth"], prediction),
                }
            )
    mean_per_image = {}
    aggregate = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        mean_per_image[method] = {
            field: float(np.mean([row[field] for row in method_rows]))
            for field in ("iou", "dice", "precision", "recall", "accuracy")
        }
        truth = np.concatenate([record["truth"].ravel() for record in records])
        prediction = np.concatenate([record["masks"][method].ravel() for record in records])
        aggregate[method] = {
            "metrics": binary_metrics(truth, prediction),
            "confusion_matrix": confusion_matrix(truth, prediction).tolist(),
        }
    return {
        "rows": rows,
        "confusion_matrices": matrices,
        "mean_per_image": mean_per_image,
        "aggregate_pixel_weighted": aggregate,
    }


def _save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=figure.dpi, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def load_cfd() -> list[dict]:
    """Carga las parejas imagen/ground truth del manifiesto CFD."""
    manifest_path = CFD_ROOT / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Ejecute primero scripts/prepare_cfd_benchmark.py --prepare.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs = []
    for item in manifest["files"]:
        image = np.asarray(Image.open(ROOT / item["image"]).convert("RGB"))
        truth = np.asarray(Image.open(ROOT / item["binary_reference"]).convert("L")) > 0
        pairs.append({"id": item["id"], "image": image, "truth": truth})
    return pairs


def load_candelaria() -> list[dict]:
    """Carga las cinco fotografías y sus referencias manuales preliminares."""
    pairs = []
    for image_path in sorted(REAL_IMAGES.glob("*.jpg")):
        mask_path = REAL_MASKS / f"{image_path.stem}_mask.png"
        image = np.asarray(Image.open(image_path).convert("RGB"))
        truth = np.asarray(Image.open(mask_path).convert("L")) > 0
        if image.shape[:2] != truth.shape:
            raise ValueError(f"Dimensiones distintas en {image_path.name}")
        pairs.append({"id": image_path.stem, "image": image, "truth": truth})
    return pairs


def cfd_sensitivity(pairs: list[dict]) -> dict:
    """Dice/IoU medios al variar la tolerancia de RG y la distancia de fondo de Watershed."""
    inputs = [(preprocess_cfd(pair["image"])["input"], pair["truth"]) for pair in pairs]
    region = []
    for tolerance in RG_TOLERANCES:
        scores = [
            binary_metrics(truth, region_growing(x, select_seeds(x, CFD_PARAMS["rg_seed_percentile"]), tolerance)[0])
            for x, truth in inputs
        ]
        region.append(
            {
                "tolerance": tolerance,
                "mean_dice": float(np.mean([s["dice"] for s in scores])),
                "mean_iou": float(np.mean([s["iou"] for s in scores])),
                "mean_recall": float(np.mean([s["recall"] for s in scores])),
                "mean_precision": float(np.mean([s["precision"] for s in scores])),
            }
        )
    watershed = []
    for distance in WS_DISTANCES:
        scores = []
        for x, truth in inputs:
            threshold, _ = otsu_threshold(x)
            markers, _ = build_markers(x, threshold, CFD_PARAMS["ws_seed_percentile"], distance)
            labels, _ = marker_watershed(sobel_magnitude(x), markers)
            scores.append(binary_metrics(truth, labels == FOREGROUND))
        watershed.append(
            {
                "background_distance": distance,
                "mean_dice": float(np.mean([s["dice"] for s in scores])),
                "mean_iou": float(np.mean([s["iou"] for s in scores])),
                "mean_recall": float(np.mean([s["recall"] for s in scores])),
                "mean_precision": float(np.mean([s["precision"] for s in scores])),
            }
        )
    return {"region_growing_tolerance": region, "watershed_background_distance": watershed}


def evaluate_cfd(create_figures: bool = True) -> dict:
    records = []
    for pair in load_cfd():
        steps = preprocess_cfd(pair["image"])
        outputs = run_methods(steps["input"], CFD_PARAMS)
        records.append(
            {
                **pair,
                "masks": {method: outputs[method]["mask"] for method in METHODS},
                "parameters": {method: outputs[method]["parameter"] for method in METHODS},
                "steps": steps,
                "outputs": outputs,
            }
        )
    overview_path = FIGURES / "cfd_overview.png"
    steps_path = FIGURES / "cfd_pasos.png"
    if create_figures:
        _save(
            visualization.overview_figure(records, "CrackForest: ground truth y errores (FP rojo, FN azul, TP verde)"),
            overview_path,
        )
        chosen = next(record for record in records if record["id"] == STEPS_IMAGE_ID)
        _save(
            visualization.steps_figure(chosen["steps"], chosen["outputs"], chosen["truth"], STEPS_IMAGE_ID),
            steps_path,
        )
    return {
        "id": "cfd5",
        "role": "evaluación cuantitativa principal",
        "benchmark_manifest": "data/benchmarks/cfd5/manifest.json",
        "reference_type": "Anotación oficial CrackForest groundTruth.Segmentation; etiqueta 2 = grieta (objeto).",
        "preprocessing": "gris = media RGB; suavizado gaussiano sigma=1.5; inversión 255-I; resta de fondo "
        "I' = 4*(I - G15*I) + 128; la misma entrada para los cuatro métodos.",
        "parameters": CFD_PARAMS,
        **_summarize(records),
        "sensitivity": cfd_sensitivity(records),
        "overview_figure": _relative(overview_path),
        "steps_figure": _relative(steps_path),
    }


def evaluate_la_candelaria(create_figures: bool = True) -> dict:
    records = []
    for pair in load_candelaria():
        steps = preprocess_candelaria(pair["image"])
        outputs = run_methods(steps["input"], CANDELARIA_PARAMS)
        shape = pair["truth"].shape
        records.append(
            {
                **pair,
                "masks": {method: upscale_mask(outputs[method]["mask"], shape) for method in METHODS},
                "parameters": {method: outputs[method]["parameter"] for method in METHODS},
            }
        )
    overview_path = FIGURES / "la_candelaria_overview.png"
    if create_figures:
        preview = []
        for record in records:
            image = Image.fromarray(record["image"])
            scale = 480 / max(image.size)
            size = (round(image.size[0] * scale), round(image.size[1] * scale))
            preview.append({
                "id": record["id"],
                "image": np.asarray(image.resize(size, Image.BILINEAR)),
                "truth": np.asarray(Image.fromarray(record["truth"]).resize(size, Image.NEAREST)),
                "masks": {
                    method: np.asarray(Image.fromarray(record["masks"][method]).resize(size, Image.NEAREST))
                    for method in METHODS
                },
            })
        _save(
            visualization.overview_figure(
                preview, "La Candelaria: referencia manual preliminar y errores (FP rojo, FN azul, TP verde)"
            ),
            overview_path,
        )
    return {
        "id": "la_candelaria",
        "role": "aplicación secundaria con métricas orientativas",
        "reference_type": "Máscaras manuales propias de urban_built_surface (fachadas, cubiertas y pavimento), "
        "elaboradas sin usar los segmentadores; no son una anotación experta validada.",
        "preprocessing": "reducción a lado máximo 1024 px; gris = media RGB; suavizado gaussiano sigma=1.5; "
        "las máscaras se reescalan por vecino más cercano a la resolución original antes de evaluar.",
        "parameters": CANDELARIA_PARAMS,
        **_summarize(records),
        "overview_figure": _relative(overview_path),
    }


def evaluate_all(write_outputs: bool = True, create_figures: bool = True) -> dict:
    result = {
        "schema": "segmentation-evaluation-v2",
        "title": TITLE,
        "algorithms": {
            "Otsu": "Línea base: umbral global que maximiza la varianza entre clases (histograma de 256 niveles).",
            "K-means": "Línea base: K-means 1-D propio, k=2, inicialización mín./máx., semilla 7; grupo claro = objeto.",
            "Region Growing": "Crecimiento de regiones por semillas (Tema 10.4), propio: semillas por percentil, "
            "vecindad 8, criterio |I(p) - media de la región| <= T.",
            "Watershed": "Watershed con marcadores (Tema 10.7), inundación propia por cola de prioridad sobre el "
            "gradiente de Sobel; marcadores de fondo por mapa de distancias.",
            "metrics": "Evaluador propio: IoU, Dice, precisión, recall y exactitud; matrices [[TN, FP], [FN, TP]].",
        },
        "cfd": evaluate_cfd(create_figures=create_figures),
        "la_candelaria": evaluate_la_candelaria(create_figures=create_figures),
    }
    if write_outputs:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        CFD_RESULTS_PATH.write_text(json.dumps(result["cfd"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    plt.switch_backend("Agg")
    result = evaluate_all(write_outputs=True, create_figures=True)
    for scenario in ("cfd", "la_candelaria"):
        print(scenario)
        for method, metrics in result[scenario]["mean_per_image"].items():
            print(f"  {method:15s} IoU={metrics['iou']:.4f} Dice={metrics['dice']:.4f}")
    print(f"results={RESULTS_PATH}")


if __name__ == "__main__":
    main()
