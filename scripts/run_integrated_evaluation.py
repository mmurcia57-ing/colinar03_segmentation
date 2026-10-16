"""Run the single academic evaluation for La Candelaria and CFD.

The existing Otsu, K-means and metric implementations are imported unchanged.
La Candelaria remains the main case study, with preliminary references. CFD is a
complementary benchmark with external official annotations.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from segmentation_lab import binary_metrics, confusion_matrix, kmeans_segment, otsu_threshold  # noqa: E402


REAL_IMAGES = ROOT / "data" / "real" / "images"
REAL_MASKS = ROOT / "data" / "real" / "masks"
CFD_ROOT = ROOT / "data" / "benchmarks" / "cfd5"
OUTPUT = ROOT / "artifacts" / "integrated_evaluation"
RESULTS_PATH = ROOT / "deliverables" / "results.json"
CFD_RESULTS_PATH = ROOT / "deliverables" / "cfd_results.json"


def fixed_cfd_input(image: np.ndarray) -> np.ndarray:
    """Apply the approved fixed input transform equally to both segmenters."""
    gray = np.rint(np.asarray(image, dtype=np.float64)[..., :3].mean(axis=2)).astype(np.uint8)
    return (255 - gray).astype(np.uint8)


def error_map(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Render TN black, FP red, FN blue and TP green."""
    view = np.zeros((*truth.shape, 3), dtype=np.uint8)
    view[(prediction == 1) & (truth == 0)] = [220, 40, 40]
    view[(prediction == 0) & (truth == 1)] = [40, 90, 220]
    view[(prediction == 1) & (truth == 1)] = [40, 170, 70]
    return view


def _summarize(records: list[dict]) -> tuple[list[dict], dict, dict]:
    rows: list[dict] = []
    matrices: dict[str, dict[str, list[list[int]]]] = {}
    reference_chunks: dict[str, list[np.ndarray]] = {"Otsu": [], "K-means": []}
    prediction_chunks: dict[str, list[np.ndarray]] = {"Otsu": [], "K-means": []}
    for record in records:
        identifier = record["id"]
        truth = record["truth"]
        matrices[identifier] = {}
        for method, prediction, parameter in record["predictions"]:
            matrix = confusion_matrix(truth, prediction).tolist()
            matrices[identifier][method] = matrix
            rows.append(
                {
                    "image": identifier,
                    "method": method,
                    "parameter": parameter,
                    **binary_metrics(truth, prediction),
                }
            )
            reference_chunks[method].append(truth.ravel())
            prediction_chunks[method].append(prediction.ravel())
    aggregate = {}
    for method in ("Otsu", "K-means"):
        truth = np.concatenate(reference_chunks[method])
        prediction = np.concatenate(prediction_chunks[method])
        aggregate[method] = {
            "metrics": binary_metrics(truth, prediction),
            "confusion_matrix": confusion_matrix(truth, prediction).tolist(),
        }
    return rows, matrices, aggregate


def _save_overview(records: list[dict], path: Path, scenario: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(5, 4, figsize=(15.5, 11), dpi=180)
    for row_index, record in enumerate(records):
        otsu = record["predictions"][0][1]
        kmeans = record["predictions"][1][1]
        if scenario == "la_candelaria":
            panels = [
                (record["image"], "Original", None),
                (record["truth"], "Referencia preliminar", "gray"),
                (otsu, "Otsu", "gray"),
                (kmeans, "K-means", "gray"),
            ]
        else:
            panels = [
                (record["image"], "Original", None),
                (record["truth"], "Referencia oficial", "gray"),
                (error_map(record["truth"], otsu), "Otsu: FP/FN/TP", None),
                (error_map(record["truth"], kmeans), "K-means: FP/FN/TP", None),
            ]
        for column_index, (content, title, cmap) in enumerate(panels):
            axis = axes[row_index, column_index]
            axis.imshow(content, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
            if row_index == 0:
                axis.set_title(title, fontsize=11, fontweight="bold")
            if column_index == 0:
                axis.set_ylabel(record["id"], fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
    heading = (
        "La Candelaria: comparación visual (referencias preliminares)"
        if scenario == "la_candelaria"
        else "CrackForest: referencia oficial y errores (FP rojo, FN azul, TP verde)"
    )
    figure.suptitle(heading, fontsize=15, fontweight="bold")
    figure.tight_layout(rect=[0, 0, 1, 0.965], h_pad=0.5, w_pad=0.35)
    figure.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def evaluate_la_candelaria(create_figures: bool = True) -> dict:
    records: list[dict] = []
    for image_path in sorted(REAL_IMAGES.glob("*.jpg")):
        mask_path = REAL_MASKS / f"{image_path.stem}_mask.png"
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        image = np.asarray(Image.open(image_path).convert("RGB"))
        truth = np.asarray(Image.open(mask_path).convert("L")) > 0
        if image.shape[:2] != truth.shape:
            raise ValueError(f"Dimension mismatch for {image_path.name}")
        threshold, otsu = otsu_threshold(image)
        centroids, kmeans = kmeans_segment(image, seed=7)
        records.append(
            {
                "id": image_path.stem,
                "image": image,
                "truth": truth,
                "predictions": [
                    ("Otsu", otsu, f"threshold={threshold}; RGB mean; bins=256"),
                    (
                        "K-means",
                        kmeans,
                        "k=2; seed=7; max_iter=100; tolerance=1e-4; "
                        f"centroids={np.round(centroids, 2).tolist()}; RGB mean",
                    ),
                ],
            }
        )
    rows, matrices, aggregate = _summarize(records)
    figure_path = OUTPUT / "la_candelaria_overview.png"
    if create_figures:
        _save_overview(records, figure_path, "la_candelaria")
    return {
        "id": "la_candelaria",
        "role": "main case study",
        "status": "exploratory; manual references are preliminary and failed complete visual validation",
        "reference_type": "Independent manual polygon masks for urban_built_surface; not validated ground truth.",
        "preprocessing": "Existing implementations convert RGB to the arithmetic mean of R, G and B; no image-specific tuning.",
        "rows": rows,
        "confusion_matrices": matrices,
        "aggregate_pixel_weighted": aggregate,
        "overview_figure": str(figure_path.relative_to(ROOT)).replace("\\", "/"),
    }


def evaluate_cfd(create_figures: bool = True) -> dict:
    manifest_path = CFD_ROOT / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run scripts/prepare_cfd_benchmark.py --prepare first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[dict] = []
    for item in manifest["files"]:
        image = np.asarray(Image.open(ROOT / item["image"]).convert("RGB"))
        truth = np.asarray(Image.open(ROOT / item["binary_reference"]).convert("L")) > 0
        transformed = fixed_cfd_input(image)
        threshold, otsu = otsu_threshold(transformed)
        centroids, kmeans = kmeans_segment(transformed, seed=7)
        records.append(
            {
                "id": item["id"],
                "image": image,
                "truth": truth,
                "predictions": [
                    ("Otsu", otsu, f"threshold={threshold}; input=255-gray; bins=256"),
                    (
                        "K-means",
                        kmeans,
                        "k=2; seed=7; max_iter=100; tolerance=1e-4; "
                        f"centroids={np.round(centroids, 2).tolist()}; input=255-gray",
                    ),
                ],
            }
        )
    rows, matrices, aggregate = _summarize(records)
    figure_path = OUTPUT / "cfd_overview.png"
    if create_figures:
        _save_overview(records, figure_path, "cfd")
    return {
        "id": "cfd5",
        "role": "complementary benchmark",
        "status": "official references; five selected pairs passed technical alignment and polarity review",
        "benchmark_manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "reference_type": "Official CrackForest groundTruth.Segmentation; label 2 is crack/foreground.",
        "preprocessing": "gray = round(mean(R,G,B)); both methods receive 255 - gray; no per-image tuning.",
        "rows": rows,
        "confusion_matrices": matrices,
        "aggregate_pixel_weighted": aggregate,
        "overview_figure": str(figure_path.relative_to(ROOT)).replace("\\", "/"),
    }


def evaluate_all(write_outputs: bool = True, create_figures: bool = True) -> dict:
    result = {
        "schema": "integrated-segmentation-evaluation-v1",
        "generated_on": date.today().isoformat(),
        "title": "Evaluación de métodos de segmentación en imágenes urbanas de La Candelaria, Bogotá, con validación complementaria mediante CrackForest Dataset",
        "algorithms": {
            "Otsu": "256-bin global threshold; foreground is intensity > threshold.",
            "K-means": "Own 1-D k=2 implementation; min/max initialization, seed 7, max_iter 100, tolerance 1e-4; brighter cluster is foreground.",
            "metrics": "IoU, Dice, precision, recall and accuracy; confusion matrices use [[TN, FP], [FN, TP]].",
        },
        "case_study": evaluate_la_candelaria(create_figures=create_figures),
        "complementary_benchmark": evaluate_cfd(create_figures=create_figures),
        "quality_gates": {
            "cfd_pair_integrity": "passed",
            "cfd_visual_alignment_and_polarity": "passed by technical visual inspection of five review sheets",
            "la_candelaria_reference_validation": "pending; preliminary masks contain known visual errors",
            "qg1": "not approved",
        },
    }
    if write_outputs:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(result, indent=2, ensure_ascii=False)
        RESULTS_PATH.write_text(encoded + "\n", encoding="utf-8")
        cfd = result["complementary_benchmark"]
        CFD_RESULTS_PATH.write_text(json.dumps(cfd, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    result = evaluate_all(write_outputs=True, create_figures=True)
    print(f"La Candelaria rows={len(result['case_study']['rows'])}")
    print(f"CFD rows={len(result['complementary_benchmark']['rows'])}")
    print(f"results={RESULTS_PATH}")


if __name__ == "__main__":
    main()
